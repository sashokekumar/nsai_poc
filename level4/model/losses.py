# level4/model/losses.py
"""
Loss functions for Level 4 symbolically supervised training.

Total loss (v2 — enhanced with hierarchical, causal, and temporal constraint families):

    L = L_intent + α·L_entity + β·L_domain + λ·L_constraint + γ·L_causal

Where:
    L_constraint : SymbolicConstraintLoss — differentiable soft penalty over 12
                   disallowed (intent, entity_type) pairs spanning four constraint
                   families:
                     TYPE_A (×6, w=1.0)   — hierarchical false rejection
                     TYPE_B (×2, w=0.75)  — false execution on observational entities
                     TYPE_C (×2, w=0.5)   — ungrounded SRE intent
                     TYPE_D (×1, w=1.25)  — hierarchical: execution on unknown target
                     TYPE_F (×1, w=0.4)   — temporal: summarization on metric entity

    L_causal     : DomainIntentCausalLoss — cross-head causal consistency penalty
                     TYPE_E               — penalizes predicting out_of_scope when the
                                           domain head is confident the utterance is
                                           SRE-domain (causal contradiction)
                                           penalty = mean(sigmoid(domain_logits) × p_oos)

All constraint losses are differentiable — they operate on softmax/sigmoid probability
distributions and propagate gradients back through all classification heads and the
shared trunk. This is the key Level 4 property: symbolic rules shape learning, not
inference-time output.
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
    Differentiable constraint violation penalty for (intent, entity_type) pairs.

    For each disallowed (intent, entity_type) pair, penalizes the model
    when its predicted probabilities jointly assign high mass to both.

    penalty = sum_over_disallowed_pairs [ w_ij * mean(p_i * p_j) ]

    Covers TYPE_A, TYPE_B, TYPE_C, TYPE_D, and TYPE_F constraints.
    TYPE_E (domain-intent causal) is handled separately by DomainIntentCausalLoss.
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


class DomainIntentCausalLoss(nn.Module):
    """
    TYPE_E: Causal domain-intent consistency penalty.

    Penalises the model when it simultaneously predicts:
        - domain head: high confidence that the utterance IS in the SRE domain
        - intent head: out_of_scope (the utterance should be REJECTED)

    This is a causal contradiction: you cannot coherently believe an utterance is
    SRE-relevant and simultaneously reject it as out-of-scope.

    Constraint formulation:
        L_causal = gamma * mean(sigmoid(domain_logits) * softmax(intent_logits)[:, oos_idx])

    This is the first Level 4 constraint that crosses two prediction heads (domain ×
    intent) rather than operating on a single (intent, entity) pair. It enforces
    causal logical consistency between the two heads at training time.

    Args:
        out_of_scope_idx : index of 'out_of_scope' in INTENT_LABELS (default 3)
        gamma            : loss weight for this causal term (default 0.75)
    """

    def __init__(self, out_of_scope_idx: int = 3, gamma: float = 0.75):
        super().__init__()
        self.out_of_scope_idx = out_of_scope_idx
        self.gamma = gamma

    def forward(
        self,
        domain_logits: torch.Tensor,   # [B] — raw domain head output
        intent_logits: torch.Tensor,   # [B, num_intents]
    ) -> torch.Tensor:
        """
        Returns scalar causal consistency loss.
        """
        p_domain = torch.sigmoid(domain_logits)                          # [B] in [0,1]
        p_intent = torch.softmax(intent_logits, dim=-1)                  # [B, num_intents]
        p_oos    = p_intent[:, self.out_of_scope_idx]                    # [B]

        # Causal contradiction: high domain confidence AND high out_of_scope probability
        return self.gamma * (p_domain * p_oos).mean()


class Level4Loss(nn.Module):
    """
    Combined training loss for Level 4 (v2 — enhanced constraint families).

    L = L_intent + α·L_entity + β·L_domain + λ·L_constraint + γ·L_causal

    Parameters:
        α, β    — task head balance weights (default 1.0 each)
        λ       — weight for SymbolicConstraintLoss (intent×entity pair constraints)
                  λ=0 → pure neural baseline (Experiment A)
                  λ>0 → symbolically supervised Level 4 (Experiments B–F)
                  sweet spot: λ=1.0 (−30% violation rate, zero accuracy cost)
        γ       — weight for DomainIntentCausalLoss (TYPE_E cross-head causal)
                  γ=0 → disables TYPE_E
                  default: γ=0.75 (matches TYPE_B severity)

    Constraint families active under λ:
        TYPE_A (×6, w=1.0)   — out_of_scope + any SRE entity (false rejection)
        TYPE_B (×2, w=0.75)  — execution + incident/metric (false execution)
        TYPE_C (×2, w=0.5)   — investigate/summarization + unknown (ungrounded SRE)
        TYPE_D (×1, w=1.25)  — execution + unknown (hierarchical safety violation)
        TYPE_F (×1, w=0.4)   — summarization + metric (temporal phase mismatch)

    Active under γ:
        TYPE_E               — domain confidence × p(out_of_scope) (causal consistency)
    """

    def __init__(
        self,
        lam: float = 0.5,    # constraint loss weight (λ)
        alpha: float = 1.0,  # entity loss weight (α)
        beta: float = 1.0,   # domain loss weight (β)
        gamma: float = 0.75, # causal loss weight (γ) — TYPE_E
        rules_path: str = None,
        out_of_scope_idx: int = 3,
    ):
        super().__init__()
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.intent_loss_fn    = nn.CrossEntropyLoss()
        self.entity_loss_fn    = nn.CrossEntropyLoss()
        self.domain_loss_fn    = nn.BCEWithLogitsLoss()
        self.constraint_loss_fn = SymbolicConstraintLoss(rules_path)
        self.causal_loss_fn     = DomainIntentCausalLoss(
            out_of_scope_idx=out_of_scope_idx, gamma=gamma
        )

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
        l_intent     = self.intent_loss_fn(intent_logits, intent_targets)
        l_entity     = self.entity_loss_fn(entity_logits, entity_targets)
        l_domain     = self.domain_loss_fn(domain_logits, domain_targets)
        l_constraint = self.constraint_loss_fn(intent_logits, entity_logits)
        l_causal     = self.causal_loss_fn(domain_logits, intent_logits)

        total = (
            l_intent
            + self.alpha  * l_entity
            + self.beta   * l_domain
            + self.lam    * l_constraint
            + l_causal          # gamma is already baked into DomainIntentCausalLoss
        )

        return {
            "loss":            total,
            "intent_loss":     l_intent.item(),
            "entity_loss":     l_entity.item(),
            "domain_loss":     l_domain.item(),
            "constraint_loss": l_constraint.item(),
            "causal_loss":     l_causal.item(),
        }
