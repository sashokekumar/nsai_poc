# level5/model/level5_model.py
"""
Level 5 neural model: symbolic rules compiled into a differentiable rule layer.

Architecture:
    utterance
        ↓
    SentenceTransformer encoder   [384-dim, frozen]
        ↓
    Shared dense trunk            [384 → 256, ReLU, Dropout]
        ↓
    Predicate heads               [256 → 11, sigmoid] — one per predicate
        ↓
    DifferentiableRuleLayer       [11 predicates → 4 rule activations]
        (RuleCompiler: product t-norm logic over predicate confidence scores)
        (learnable rule_strength per rule)
        ↓
    Intent logit blend
        rule_score  = rule_layer.scatter_to_intents(rule_activations)  [B, 4]
        trunk_score = intent_head(trunk)                                [B, 4]
        rule_logits = rule_score_projection(intent_rule_scores)  [B, 4]  ← projects [0,1] rule scores to logit scale
    final_logit = rule_weight * rule_logits + (1 - rule_weight) * trunk_logits
        ↓
    softmax → 4 intent classes

Key Level 5 property:
    Symbolic rules are STRUCTURAL — they live in the forward pass and
    gradients flow back through the rule activations, updating the shared
    trunk and predicate heads. The rule_strength parameters are also
    differentiable, so the network learns how much to trust each rule.

    Contrast with:
      Level 4  — rules shape training loss only; no rules at inference
      Level 4.5 — rules applied post-hoc after softmax at runtime
"""

from pathlib import Path

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from level5.model.dataset import (
    INTENT_LABELS, PREDICATE_COLS,
    IDX_TO_INTENT,
)
from level5.model.rule_compiler import RuleCompiler

ENCODER_DIM  = 384
HIDDEN_DIM   = 256
NUM_INTENTS  = len(INTENT_LABELS)   # 4
NUM_PREDS    = len(PREDICATE_COLS)  # 11

_DEFAULT_RULE_BASE = (
    Path(__file__).parent.parent / "data" / "rule_base.json"
)


class Level5IntentModel(nn.Module):
    """
    Neuro-symbolic intent classifier with rules compiled into the architecture.

    Args:
        encoder_name    : HuggingFace model name for SentenceTransformer
        rule_base_path  : path to rule_base.json (defaults to level5/data/rule_base.json)
        dropout         : dropout rate for shared trunk
        rule_weight     : initial blend weight — fraction of final logit from rule layer.
                          Learnable if rule_weight_learnable=True.
        rule_weight_learnable : if True, rule_weight is a trainable sigmoid parameter
    """

    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        rule_base_path: str = None,
        dropout: float = 0.2,
        rule_weight: float = 0.5,
        rule_weight_learnable: bool = True,
    ):
        super().__init__()

        rule_base_path = rule_base_path or str(_DEFAULT_RULE_BASE)

        # ------------------------------------------------------------------
        # Frozen sentence encoder
        # ------------------------------------------------------------------
        self._encoder = SentenceTransformer(encoder_name)
        for param in self._encoder.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------------
        # Shared dense trunk
        # ------------------------------------------------------------------
        self.shared = nn.Sequential(
            nn.Linear(ENCODER_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # Predicate heads — 11 binary heads (sigmoid outputs)
        # ------------------------------------------------------------------
        self.predicate_head = nn.Linear(HIDDEN_DIM, NUM_PREDS)

        # ------------------------------------------------------------------
        # Differentiable rule layer (symbolic rules compiled in)
        # ------------------------------------------------------------------
        self.rule_layer = RuleCompiler(
            rule_base_path=rule_base_path,
            predicate_cols=PREDICATE_COLS,
            intent_labels=INTENT_LABELS,
        )

        # ------------------------------------------------------------------
        # Residual intent head (trunk → intent logits, no symbolic knowledge)
        # ------------------------------------------------------------------
        self.intent_head = nn.Linear(HIDDEN_DIM, NUM_INTENTS)

        # ------------------------------------------------------------------
        # Rule score projection — maps rule scores from [0,1] to logit scale
        # so they are on the same scale as trunk_logits before blending.
        # Initialized as identity-like (no bias) so early training is stable.
        # ------------------------------------------------------------------
        self.rule_score_projection = nn.Linear(NUM_INTENTS, NUM_INTENTS, bias=True)

        # ------------------------------------------------------------------
        # Blend weight: fraction of final logit from rule layer
        # Stored as a logit so sigmoid keeps it in [0, 1] during training
        # ------------------------------------------------------------------
        import math
        p = max(min(rule_weight, 0.9999), 0.0001)
        rw_logit = math.log(p / (1.0 - p))
        if rule_weight_learnable:
            self.rule_weight_logit = nn.Parameter(torch.tensor(rw_logit))
        else:
            self.register_buffer(
                "rule_weight_logit", torch.tensor(rw_logit)
            )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, utterances: list[str], device: torch.device) -> torch.Tensor:
        """Encode utterances → [B, 384]."""
        embeddings = self._encoder.encode(
            utterances,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device,
        )
        return embeddings.to(device).clone()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, utterances: list[str], device: torch.device) -> dict:
        """
        Full forward pass through the neuro-symbolic architecture.

        Returns dict with:
            intent_logits     : [B, 4]  — final blended intent logits
            predicate_probs   : [B, 11] — sigmoid predicate activations
            rule_activations  : [B, n_rules] — per-rule weighted activation scores
            intent_rule_scores: [B, 4]  — raw rule activations scattered to intent dims
            rule_logits       : [B, 4]  — rule scores projected to logit scale
            trunk_logits      : [B, 4]  — residual trunk-only intent logits
            rule_weight       : scalar  — current blend weight (sigmoid)
        """
        embeddings = self.encode(utterances, device)         # [B, 384]
        trunk      = self.shared(embeddings)                 # [B, 256]

        # Predicate head — sigmoid for [0,1] confidence scores
        predicate_probs = torch.sigmoid(self.predicate_head(trunk))  # [B, 11]

        # Rule layer — compile symbolic rules over predicate activations
        rule_activations   = self.rule_layer(predicate_probs)         # [B, n_rules]
        intent_rule_scores = self.rule_layer.scatter_to_intents(rule_activations)  # [B, 4]

        # Residual trunk intent logits
        trunk_logits = self.intent_head(trunk)               # [B, 4]

        # Project rule scores from [0,1] to logit scale, then blend
        # with trunk logits. Both are now on the same unbounded scale.
        rule_logits  = self.rule_score_projection(intent_rule_scores)  # [B, 4]
        rule_weight  = torch.sigmoid(self.rule_weight_logit)
        intent_logits = (
            rule_weight       * rule_logits
            + (1 - rule_weight) * trunk_logits
        )                                                    # [B, 4]

        return {
            "intent_logits":      intent_logits,
            "predicate_probs":    predicate_probs,
            "rule_activations":   rule_activations,
            "intent_rule_scores": intent_rule_scores,
            "rule_logits":        rule_logits,
            "trunk_logits":       trunk_logits,
            "rule_weight":        rule_weight,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        utterances: list[str],
        device: torch.device = None,
    ) -> list[dict]:
        """
        Clean inference — no symbolic post-processing beyond what the rule
        layer contributes structurally.

        Returns one dict per utterance with:
            intent, intent_prob, predicate_activations, rule_activations,
            rule_weight
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        self.eval()
        with torch.no_grad():
            out   = self.forward(utterances, device)
            probs = torch.softmax(out["intent_logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)
            preds_cpu = preds.cpu().tolist()
            probs_cpu = probs.cpu().tolist()
            pred_probs_cpu = out["predicate_probs"].cpu().tolist()
            rule_act_cpu   = out["rule_activations"].cpu().tolist()
            rw = float(out["rule_weight"].cpu())

        results = []
        for i, utt in enumerate(utterances):
            results.append({
                "utterance":            utt,
                "intent":               IDX_TO_INTENT[preds_cpu[i]],
                "intent_prob":          round(probs_cpu[i][preds_cpu[i]], 4),
                "predicate_activations": {
                    col: round(pred_probs_cpu[i][j], 4)
                    for j, col in enumerate(PREDICATE_COLS)
                },
                "rule_activations": {
                    rule["name"]: round(rule_act_cpu[i][r], 4)
                    for r, rule in enumerate(self.rule_layer.rules)
                },
                "rule_weight": round(rw, 4),
            })
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def rule_strength_dict(self) -> dict:
        """Current learned rule strengths (for logging/checkpointing)."""
        return self.rule_layer.rule_strength_dict()

    def blend_weight(self) -> float:
        """Current learned blend weight α (fraction from rule layer)."""
        return float(torch.sigmoid(self.rule_weight_logit).detach().cpu())
