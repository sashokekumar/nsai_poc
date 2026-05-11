# level5/model/level5_model.py
"""
Level 5 neural model: neural predicate estimator + compiled differentiable symbolic rules.

Architecture (clean L5):
    utterance
        ↓
    SentenceTransformer encoder   [384-dim, frozen]
        ↓
    Shared dense trunk            [384 → 256, ReLU, Dropout]
        ↓
    Predicate heads               [256 → 11, sigmoid] — neural network learns symbolic facts
        ↓
    DifferentiableRuleLayer       [11 predicates → 4 rule activations]
        (RuleCompiler: product t-norm logic over predicate confidence scores)
        (L5A: rule_strength learnable per rule)
        (L5B: rule_strength fixed at 1.0 — hard compiled symbolic)
        ↓
    scatter_to_intents            max rule activation per intent  [B, 4]
        ↓
    torch.logit(score.clamp(ε))   deterministic logit transform (no learned remapping)
        ↓
    softmax → 4 intent classes

Key Level 5 property:
    The symbolic rule layer is the SOLE decision path — no residual neural
    intent head, no blend weight. The neural trunk's only job is to estimate
    symbolic predicate probabilities; the compiled rule layer determines intent.

    Two modes:
      L5A (soft) — rule_strength learnable; rules are compiled but their
                   strength adapts during training
      L5B (hard) — rule_strength fixed at 1.0; rules are strictly structural
                   and cannot be weakened by training (strongest Kautz L5 claim)

    Contrast with:
      Level 4    — rules shape training loss only; no rules at inference
      Level 4.5  — rules applied post-hoc after softmax at runtime
      Old L5     — neural intent_head blended with rule output (not pure symbolic)
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
    Neuro-symbolic intent classifier — neural predicate learner + compiled symbolic rules.

    The neural trunk estimates symbolic predicate probabilities. The compiled
    differentiable rule layer is the sole decision path to intent logits.
    No residual neural intent classifier is present.

    Args:
        encoder_name   : HuggingFace model name for SentenceTransformer
        rule_base_path : path to rule_base.json (defaults to level5/data/rule_base.json)
        dropout        : dropout rate for shared trunk
        hard_rules     : if True (L5B), rule_strength fixed at 1.0 with no grad;
                         if False (L5A), rule_strength_logits remain learnable
    """

    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        rule_base_path: str = None,
        dropout: float = 0.2,
        hard_rules: bool = False,
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
        # Hard rules mode (L5B): fix rule_strength = 1.0, no grad.
        # Soft rules mode (L5A): rule_strength_logits remain learnable.
        # ------------------------------------------------------------------
        if hard_rules:
            with torch.no_grad():
                self.rule_layer.rule_strength_logits.fill_(10.0)  # sigmoid(10) ≈ 1.0
            self.rule_layer.rule_strength_logits.requires_grad_(False)
        self.hard_rules = hard_rules

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
        Full forward pass: neural predicate estimator → compiled symbolic rule layer.

        The rule layer is the sole decision path. No residual intent head, no blend.

        Returns dict with:
            intent_logits     : [B, 4]       — final intent logits (from rules only)
            predicate_probs   : [B, 11]      — sigmoid predicate activations
            rule_activations  : [B, n_rules] — per-rule weighted activation scores
            intent_rule_scores: [B, 4]       — max rule activation per intent
        """
        embeddings = self.encode(utterances, device)                               # [B, 384]
        trunk      = self.shared(embeddings)                                       # [B, 256]

        # Predicate heads — neural network learns symbolic predicate probabilities
        predicate_probs = torch.sigmoid(self.predicate_head(trunk))                # [B, 11]

        # Compiled symbolic rule layer — sole decision path
        rule_activations   = self.rule_layer(predicate_probs)                      # [B, n_rules]
        intent_rule_scores = self.rule_layer.scatter_to_intents(rule_activations)  # [B, 4]

        # Deterministic logit transform — no learned remapping across intents
        eps = 1e-6
        intent_logits = torch.logit(intent_rule_scores.clamp(eps, 1.0 - eps))     # [B, 4]

        return {
            "intent_logits":      intent_logits,
            "predicate_probs":    predicate_probs,
            "rule_activations":   rule_activations,
            "intent_rule_scores": intent_rule_scores,
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
        Inference through the compiled symbolic architecture.

        Returns one dict per utterance with:
            intent, intent_prob, predicate_activations, rule_activations,
            intent_rule_scores
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
            pred_probs_cpu     = out["predicate_probs"].cpu().tolist()
            rule_act_cpu       = out["rule_activations"].cpu().tolist()
            intent_scores_cpu  = out["intent_rule_scores"].cpu().tolist()

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
                "intent_rule_scores": {
                    INTENT_LABELS[k]: round(intent_scores_cpu[i][k], 4)
                    for k in range(len(INTENT_LABELS))
                },
            })
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def rule_strength_dict(self) -> dict:
        """Current rule strengths (for logging/checkpointing)."""
        return self.rule_layer.rule_strength_dict()
