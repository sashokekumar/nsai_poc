# level6/reasoning_state.py
"""
ReasoningState — the formal tensor-backed state structure for Level 6.

Every Level 6 component (FailureCollector, SymbolCluster, EvolutionEngine)
operates on ReasoningState objects. This replaces the conceptual JSON representation
of a reasoning state with a concrete structure anchored in Level 5 model outputs.

Field origins
-------------
The following fields are populated directly from a Level5IntentModel forward pass:

    trunk_repr       [256]   shared trunk activation (Linear(384→256) + ReLU)
    predicate_probs  [11]    sigmoid predicate head outputs — named symbolic grounding
    rule_activations [4]     DifferentiableRuleLayer outputs (R1..R4 compiled scores)
    intent_dist      [4]     softmax over intent logits — final intent probability dist

The predicate_probs vector is the primary Level 6 substrate. It lives in an
11-dimensional named space (one dimension per predicate), making cluster centroids
directly interpretable without an LLM or manual annotation.

Predicate dimension index → name mapping (must stay in sync with level5/model/dataset.py):
    0  is_infrastructure
    1  is_service
    2  is_metric
    3  is_incident
    4  is_job
    5  is_pipeline
    6  is_unknown
    7  is_sre_domain
    8  has_runbook
    9  is_known_incident
    10 is_metric_query

Rule dimension index → name mapping (must stay in sync with level5/data/rule_base.json):
    0  R1_metric_investigate
    1  R2_runbook_execution
    2  R3_incident_summarization
    3  R4_unknown_out_of_scope

Intent dimension index → name mapping (must stay in sync with level5/model/dataset.py):
    0  investigate
    1  summarization
    2  execution
    3  out_of_scope
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Predicate / rule / intent label maps — mirrors level5/model/dataset.py
# ---------------------------------------------------------------------------

PREDICATE_COLS = [
    "is_infrastructure", "is_service", "is_metric",
    "is_incident", "is_job", "is_pipeline", "is_unknown",
    "is_sre_domain", "has_runbook", "is_known_incident", "is_metric_query",
]

RULE_NAMES = [
    "R1_metric_investigate",
    "R2_runbook_execution",
    "R3_incident_summarization",
    "R4_unknown_out_of_scope",
]

INTENT_LABELS = ["investigate", "summarization", "execution", "out_of_scope"]

# Confidence threshold below which a correct prediction is still flagged as a failure
LOW_CONFIDENCE_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# ReasoningState
# ---------------------------------------------------------------------------

@dataclass
class ReasoningState:
    """
    Full cognitive state of a single Level 5 inference pass.

    Tensor fields are stored as plain Python lists internally for JSON
    serializability, but are exposed as tensors via properties. This avoids
    dataclass-with-tensor serialization issues.

    Construction
    ------------
    Use `ReasoningState.from_model_output()` when creating from a live model.
    Use `ReasoningState.from_dict()` when loading from saved JSONL.
    """

    # ------------------------------------------------------------------
    # Core inference fields (required)
    # ------------------------------------------------------------------
    utterance: str

    # Stored as lists; returned as tensors via properties
    _trunk_repr: list[float]        # [256]
    _predicate_probs: list[float]   # [11]
    _rule_activations: list[float]  # [4]
    _intent_dist: list[float]       # [4]

    max_confidence: float
    predicted_intent: str

    # ------------------------------------------------------------------
    # Failure classification (set by FailureCollector)
    # ------------------------------------------------------------------
    is_failure: bool = False
    is_misclassification: bool = False
    is_low_confidence: bool = False
    gold_intent: Optional[str] = None

    # ------------------------------------------------------------------
    # Symbol match fields (set by EvolutionEngine)
    # ------------------------------------------------------------------
    symbol_matches: list[str] = field(default_factory=list)
    _symbol_scores: list[float] = field(default_factory=list)  # [K]

    # ------------------------------------------------------------------
    # Tensor properties
    # ------------------------------------------------------------------

    @property
    def trunk_repr(self) -> torch.Tensor:
        return torch.tensor(self._trunk_repr, dtype=torch.float32)

    @property
    def predicate_probs(self) -> torch.Tensor:
        return torch.tensor(self._predicate_probs, dtype=torch.float32)

    @property
    def rule_activations(self) -> torch.Tensor:
        return torch.tensor(self._rule_activations, dtype=torch.float32)

    @property
    def intent_dist(self) -> torch.Tensor:
        return torch.tensor(self._intent_dist, dtype=torch.float32)

    @property
    def symbol_scores(self) -> torch.Tensor:
        return torch.tensor(self._symbol_scores, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Named accessors
    # ------------------------------------------------------------------

    def predicate_dict(self) -> dict[str, float]:
        """Predicate name → confidence score mapping."""
        return dict(zip(PREDICATE_COLS, self._predicate_probs))

    def rule_dict(self) -> dict[str, float]:
        """Rule name → activation score mapping."""
        return dict(zip(RULE_NAMES, self._rule_activations))

    def intent_dict(self) -> dict[str, float]:
        """Intent label → probability mapping."""
        return dict(zip(INTENT_LABELS, self._intent_dist))

    def present_predicates(self, threshold: float = 0.70) -> list[str]:
        """Predicates with confidence above threshold (stably active)."""
        return [
            name for name, val in self.predicate_dict().items()
            if val >= threshold
        ]

    def absent_predicates(self, threshold: float = 0.30) -> list[str]:
        """Predicates with confidence below threshold (stably inactive)."""
        return [
            name for name, val in self.predicate_dict().items()
            if val <= threshold
        ]

    def uncertain_predicates(
        self,
        low: float = 0.30,
        high: float = 0.70,
    ) -> list[str]:
        """Predicates in the uncertain band [low, high]."""
        return [
            name for name, val in self.predicate_dict().items()
            if low < val < high
        ]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_model_output(
        cls,
        utterance: str,
        trunk_repr: torch.Tensor,
        predicate_probs: torch.Tensor,
        rule_activations: torch.Tensor,
        intent_dist: torch.Tensor,
        gold_intent: Optional[str] = None,
    ) -> "ReasoningState":
        """
        Build a ReasoningState from raw Level 5 model tensors.

        The `is_failure`, `is_misclassification`, and `is_low_confidence` flags
        are computed here automatically if `gold_intent` is provided.

        Args:
            utterance        : raw input text
            trunk_repr       : [256] shared trunk activation tensor
            predicate_probs  : [11]  predicate head sigmoid outputs
            rule_activations : [4]   compiled rule activation scores
            intent_dist      : [4]   softmax intent probability distribution
            gold_intent      : ground-truth intent label (None at inference time)
        """
        max_conf = float(intent_dist.max().item())
        pred_idx = int(intent_dist.argmax().item())
        predicted_intent = INTENT_LABELS[pred_idx]

        is_misclassification = (
            gold_intent is not None and predicted_intent != gold_intent
        )
        is_low_conf = max_conf < LOW_CONFIDENCE_THRESHOLD
        is_failure = is_misclassification or is_low_conf

        return cls(
            utterance=utterance,
            _trunk_repr=trunk_repr.detach().cpu().tolist(),
            _predicate_probs=predicate_probs.detach().cpu().tolist(),
            _rule_activations=rule_activations.detach().cpu().tolist(),
            _intent_dist=intent_dist.detach().cpu().tolist(),
            max_confidence=max_conf,
            predicted_intent=predicted_intent,
            is_failure=is_failure,
            is_misclassification=is_misclassification,
            is_low_confidence=is_low_conf,
            gold_intent=gold_intent,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serializable dict. Tensors are stored as lists."""
        return {
            "utterance": self.utterance,
            "trunk_repr": self._trunk_repr,
            "predicate_probs": self._predicate_probs,
            "rule_activations": self._rule_activations,
            "intent_dist": self._intent_dist,
            "max_confidence": self.max_confidence,
            "predicted_intent": self.predicted_intent,
            "is_failure": self.is_failure,
            "is_misclassification": self.is_misclassification,
            "is_low_confidence": self.is_low_confidence,
            "gold_intent": self.gold_intent,
            "symbol_matches": self.symbol_matches,
            "symbol_scores": self._symbol_scores,
            # Named views for readability (redundant with raw lists, useful for inspection)
            "_predicate_named": self.predicate_dict(),
            "_rule_named": self.rule_dict(),
            "_intent_named": self.intent_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "ReasoningState":
        """Reconstruct from a dict produced by `to_dict()`."""
        return cls(
            utterance=d["utterance"],
            _trunk_repr=d["trunk_repr"],
            _predicate_probs=d["predicate_probs"],
            _rule_activations=d["rule_activations"],
            _intent_dist=d["intent_dist"],
            max_confidence=d["max_confidence"],
            predicted_intent=d["predicted_intent"],
            is_failure=d.get("is_failure", False),
            is_misclassification=d.get("is_misclassification", False),
            is_low_confidence=d.get("is_low_confidence", False),
            gold_intent=d.get("gold_intent"),
            symbol_matches=d.get("symbol_matches", []),
            _symbol_scores=d.get("symbol_scores", []),
        )

    @classmethod
    def from_json(cls, s: str) -> "ReasoningState":
        return cls.from_dict(json.loads(s))

    def __repr__(self) -> str:
        pred_top = sorted(
            self.predicate_dict().items(), key=lambda x: x[1], reverse=True
        )[:3]
        pred_str = ", ".join(f"{k}={v:.2f}" for k, v in pred_top)
        failure_tag = ""
        if self.is_misclassification:
            failure_tag = f" [MISCLASSIFIED: gold={self.gold_intent}]"
        elif self.is_low_confidence:
            failure_tag = f" [LOW_CONF: {self.max_confidence:.2f}]"
        return (
            f"ReasoningState(intent={self.predicted_intent}"
            f" conf={self.max_confidence:.3f}"
            f" top_preds=[{pred_str}]{failure_tag})"
        )
