# level4/model/losses.py
"""
Loss functions for Level 4 symbolically supervised training.

Total loss:
    L = intent_loss + α * entity_loss + β * domain_loss + λ * constraint_loss

Where constraint_loss is a *differentiable* soft penalty computed over
predicted probability distributions — not a hard post-hoc correction.

Constraint loss formulation:
    For each disallowed (intent_i, entity_j) pair with penalty weight w_ij:
        contribution = w_ij * mean(p_intent[:, i] * p_entity[:, j])

    Total constraint_loss = sum over all disallowed pairs

This is differentiable because it operates on softmax probabilities, which
propagate gradients back through the classification heads and shared trunk.

This is the key Level 4 property: symbolic rules shape learning,
not inference-time output.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path

from level4.model.dataset import INTENT_TO_IDX, ENTITY_TYPE_TO_IDX


def load_constraint_rules(rules_path: str = None) -> list[dict]:
    """Load disallowed pairs from constraint_rules.json."""
    if rules_path is None:
        rules_path = Path(__file__).parent.parent / "ontology" / "constraint_rules.json"
    with open(rules_path) as f:
        rules = json.load(f)
    return rules["disallowed_intent_entity_pairs"]


class SymbolicConstraintLoss(nn.Module):
    """
    Differentiable constraint violation penalty.

    For each disallowed (intent, entity_type) pair, penalizes the model
    when its predicted probabilities jointly assign high mass to both.

    penalty = sum_over_disallowed_pairs [ w_ij * mean(p_i * p_j) ]

    This teaches the model to avoid those combinations during training —
    not via a runtime guard, but via the gradient signal.
    """

    def __init__(self, rules_path: str = None):
        super().__init__()
        raw_rules = load_constraint_rules(rules_path)

        # Precompute index pairs and weights as registered buffers
        intent_idxs, entity_idxs, weights = [], [], []
        skipped = []
        for rule in raw_rules:
            intent = rule["intent"]
            entity = rule["entity_type"]
            if intent not in INTENT_TO_IDX:
                skipped.append(f"unknown intent '{intent}'")
                continue
            if entity not in ENTITY_TYPE_TO_IDX:
                skipped.append(f"unknown entity_type '{entity}'")
                continue
            intent_idxs.append(INTENT_TO_IDX[intent])
            entity_idxs.append(ENTITY_TYPE_TO_IDX[entity])
            weights.append(rule["penalty_weight"])

        if skipped:
            import warnings
            warnings.warn(f"SymbolicConstraintLoss skipped rules: {skipped}")

        self.register_buffer("intent_idxs", torch.tensor(intent_idxs, dtype=torch.long))
        self.register_buffer("entity_idxs", torch.tensor(entity_idxs, dtype=torch.long))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float))

    def forward(
        self,
        intent_logits: torch.Tensor,   # [B, num_intents]
        entity_logits: torch.Tensor,   # [B, num_entity_types]
    ) -> torch.Tensor:
        """
        Returns scalar constraint loss — mean penalty over the batch.
        """
        p_intent = torch.softmax(intent_logits, dim=-1)   # [B, num_intents]
        p_entity = torch.softmax(entity_logits, dim=-1)   # [B, num_entity_types]

        total = torch.tensor(0.0, device=intent_logits.device)
        for k in range(len(self.intent_idxs)):
            i = self.intent_idxs[k]
            j = self.entity_idxs[k]
            w = self.weights[k]
            # Joint probability of predicting this disallowed combination
            joint = p_intent[:, i] * p_entity[:, j]  # [B]
            total = total + w * joint.mean()

        return total


class Level4Loss(nn.Module):
    """
    Combined training loss for Level 4.

    L = intent_loss + α * entity_loss + β * domain_loss + λ * constraint_loss

    α, β, λ are hyperparameters:
        α, β  control task loss balance (default 1.0 each)
        λ = 0 → pure neural baseline (Experiment A)
        λ > 0 → symbolically supervised Level 4 (Experiment C)
        λ values for ablation: [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    """

    def __init__(
        self,
        lam: float = 0.5,    # constraint loss weight (λ)
        alpha: float = 1.0,  # entity loss weight (α)
        beta: float = 1.0,   # domain loss weight (β)
        rules_path: str = None,
    ):
        super().__init__()
        self.lam = lam
        self.alpha = alpha
        self.beta = beta

        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.entity_loss_fn = nn.CrossEntropyLoss()
        self.domain_loss_fn = nn.BCEWithLogitsLoss()
        self.constraint_loss_fn = SymbolicConstraintLoss(rules_path)

    def forward(
        self,
        intent_logits: torch.Tensor,    # [B, 4]
        entity_logits: torch.Tensor,    # [B, 7]
        domain_logits: torch.Tensor,    # [B]
        intent_targets: torch.Tensor,   # [B] long
        entity_targets: torch.Tensor,   # [B] long
        domain_targets: torch.Tensor,   # [B] float
    ) -> dict:
        """
        Returns dict with total loss and each component for logging.
        """
        l_intent = self.intent_loss_fn(intent_logits, intent_targets)
        l_entity = self.entity_loss_fn(entity_logits, entity_targets)
        l_domain = self.domain_loss_fn(domain_logits, domain_targets)
        l_constraint = self.constraint_loss_fn(intent_logits, entity_logits)

        total = (
            l_intent
            + self.alpha * l_entity
            + self.beta * l_domain
            + self.lam * l_constraint
        )

        return {
            "loss": total,
            "intent_loss": l_intent.item(),
            "entity_loss": l_entity.item(),
            "domain_loss": l_domain.item(),
            "constraint_loss": l_constraint.item(),
        }
