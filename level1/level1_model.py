"""
Level 1: Symbol-Aligned Neuro-Symbolic Intent Classification

Architecture:
Neural Detectors → Symbolization Layer → Pure Symbolic Rule Engine

Key Principle:
- Neural layer outputs numeric scores internally
- Symbolization layer converts scores + simple linguistic evidence → symbolic predicates
- Rule engine operates ONLY on predicates (no numeric thresholds in rules)

Revisions in this version:
1) Hybrid evidence for "insufficient input":
   - RAW_TOKEN_COUNT_SUFFICIENT / INSUFFICIENT
   - UNIQUE_TOKEN_COUNT_SUFFICIENT / INSUFFICIENT
   - MODEL_TOKEN_COUNT_SUFFICIENT / INSUFFICIENT (TF-IDF active features)
   R_INSUFFICIENT_INPUT fires only if ALL are insufficient.

2) Externalized symbolic rules:
   - rules live in a JSON file (default: <model_dir>/rules.json)
   - python applies rules at runtime (rules are data)

"""

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np


# ============================================================
# Configuration (NUMERIC thresholds live ONLY here)
# ============================================================

CONFIG = {
    "BASE_MIN_SCORE": 0.30,
    "HIGH_CONFIDENCE_SCORE": 0.85,
    "AMBIGUITY_MARGIN": 0.10,

    # Hybrid input evidence thresholds
    "RAW_MIN_TOKENS": 2,
    "MODEL_MIN_TOKENS": 2,
    "UNIQUE_MIN_TOKENS": 2,
}

INTENTS = ["investigate", "execution", "summarization", "out_of_scope"]


# ============================================================
# Default Rules (used only if rules.json not found)
# Rules are SYMBOLIC. No numeric thresholds.
# ============================================================

DEFAULT_RULES = [
    {
        "rule_id": "R_INSUFFICIENT_INPUT",
        "priority": 100,
        "when_all": [
            "RAW_TOKEN_COUNT_INSUFFICIENT",
            "UNIQUE_TOKEN_COUNT_INSUFFICIENT",
            "MODEL_TOKEN_COUNT_INSUFFICIENT"
        ],
        "action": {
            "predicted_intent": "out_of_scope",
            "decision_state": "blocked",
            "decision_reason": "R_INSUFFICIENT_INPUT"
        }
    },
    {
        "rule_id": "R_NO_CANDIDATE_INTENT",
        "priority": 95,
        "when_all": ["NO_CANDIDATE_INTENT"],
        "action": {
            "predicted_intent": "out_of_scope",
            "decision_state": "blocked",
            "decision_reason": "R_NO_CANDIDATE_INTENT"
        }
    },
    {
        "rule_id": "R_EXECUTION_LOW_CONFIDENCE",
        "priority": 90,
        "when_all": ["CANDIDATE_EXECUTION", "NOT_HIGH_CONFIDENCE_EXECUTION"],
        "action": {
            "predicted_intent": "execution",
            "decision_state": "needs_clarification",
            "decision_reason": "R_EXECUTION_LOW_CONFIDENCE"
        },
        "overrides": [
            {
                "when_all": ["CANDIDATE_INVESTIGATE"],
                "action": {
                    "predicted_intent": "investigate",
                    "decision_state": "accepted",
                    "decision_reason": "R_EXECUTION_DOWNGRADED"
                }
            }
        ]
    },
    {
        "rule_id": "R_AMBIGUOUS",
        "priority": 50,
        "when_all": ["AMBIGUOUS"],
        "action": {
            "predicted_intent": "__TOP_INTENT__",
            "decision_state": "needs_clarification",
            "decision_reason": "R_AMBIGUOUS"
        }
    },
    {
        "rule_id": "R_DEFAULT",
        "priority": 0,
        "when_all": [],
        "action": {
            "predicted_intent": "__TOP_INTENT__",
            "decision_state": "accepted",
            "decision_reason": "R_DEFAULT"
        }
    }
]


# ============================================================
# Helpers
# ============================================================

_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9\-_.]*\b")


def _tokenize_raw(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ============================================================
# Level 1 Classifier
# ============================================================

class Level1Classifier:
    """
    Level 1 Symbol-Aligned Neuro-Symbolic Classifier
    """

    def __init__(
        self,
        detectors: Optional[Dict[str, Any]] = None,
        intents: Optional[List[str]] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.detectors = detectors or {}
        self.intents = intents or INTENTS
        self.config = config or CONFIG

        # rules are data, not code
        self.rules = sorted((rules or DEFAULT_RULES), key=lambda r: r.get("priority", 0), reverse=True)

    # --------------------------------------------------------
    # Step 1 — Neural Layer (numeric, internal only)
    # --------------------------------------------------------

    def _extract_numeric_signals(self, text: str) -> Dict[str, Any]:
        detector_scores: Dict[str, float] = {}

        for intent, detector in self.detectors.items():
            proba = detector.predict_proba([text])[0]
            detector_scores[intent] = float(proba[1])

        sorted_detectors = sorted(detector_scores.items(), key=lambda x: x[1], reverse=True)
        top_intent = sorted_detectors[0][0]
        top_score = sorted_detectors[0][1]
        second_score = sorted_detectors[1][1]
        margin = float(top_score - second_score)

        # MODEL token count (TF-IDF active features) — evidence from model vocabulary
        first_detector = self.detectors[self.intents[0]]
        tfidf_vec = first_detector.named_steps["tfidf"].transform([text])
        active_features = tfidf_vec.toarray()[0]
        model_tokens = int(np.sum(active_features > 0))

        return {
            "detector_scores": detector_scores,
            "top_intent": top_intent,
            "margin": margin,
            "model_tokens": model_tokens
        }

    # --------------------------------------------------------
    # Step 2 — Symbolization Layer (ALL thresholds live here)
    # --------------------------------------------------------

    def _symbolize(self, text: str, numeric_signals: Dict[str, Any]) -> Set[str]:
        symbols: Set[str] = set()

        scores = numeric_signals["detector_scores"]
        margin = numeric_signals["margin"]
        model_tokens = numeric_signals["model_tokens"]

        raw_tokens_list = _tokenize_raw(text)
        raw_token_count = len(raw_tokens_list)
        unique_token_count = len(set(raw_tokens_list))

        # --- Raw / Unique / Model token predicates ---
        if raw_token_count >= self.config["RAW_MIN_TOKENS"]:
            symbols.add("RAW_TOKEN_COUNT_SUFFICIENT")
        else:
            symbols.add("RAW_TOKEN_COUNT_INSUFFICIENT")

        if unique_token_count >= self.config["UNIQUE_MIN_TOKENS"]:
            symbols.add("UNIQUE_TOKEN_COUNT_SUFFICIENT")
        else:
            symbols.add("UNIQUE_TOKEN_COUNT_INSUFFICIENT")

        if model_tokens >= self.config["MODEL_MIN_TOKENS"]:
            symbols.add("MODEL_TOKEN_COUNT_SUFFICIENT")
        else:
            symbols.add("MODEL_TOKEN_COUNT_INSUFFICIENT")

        # Optional: ultra-short marker (still symbolic)
        if raw_token_count <= 1:
            symbols.add("VERY_SHORT_UTTERANCE")

        # --- Detector predicates ---
        for intent, score in scores.items():
            if score >= self.config["HIGH_CONFIDENCE_SCORE"]:
                symbols.add(f"HIGH_CONFIDENCE_{intent.upper()}")
            if score >= self.config["BASE_MIN_SCORE"]:
                symbols.add(f"CANDIDATE_{intent.upper()}")

        # "No candidate intent" predicate (purely symbolic)
        if not any(s.startswith("CANDIDATE_") for s in symbols):
            symbols.add("NO_CANDIDATE_INTENT")

        # Convenience negations for rules (still symbols)
        if "HIGH_CONFIDENCE_EXECUTION" not in symbols:
            symbols.add("NOT_HIGH_CONFIDENCE_EXECUTION")

        # --- Ambiguity predicate ---
        if margin < self.config["AMBIGUITY_MARGIN"]:
            symbols.add("AMBIGUOUS")

        return symbols

    # --------------------------------------------------------
    # Step 3 — Pure Symbolic Rule Engine (rules loaded from JSON)
    # --------------------------------------------------------

    def _rule_matches(self, rule: Dict[str, Any], symbols: Set[str]) -> bool:
        required = rule.get("when_all", [])
        return all(req in symbols for req in required)

    def _resolve_action(self, action: Dict[str, Any], top_intent: str) -> Dict[str, Any]:
        out = dict(action)
        if out.get("predicted_intent") == "__TOP_INTENT__":
            out["predicted_intent"] = top_intent
        return out

    def _apply_symbolic_rules(self, symbols: Set[str], top_intent: str) -> Dict[str, Any]:
        triggered_rules: List[str] = []

        for rule in self.rules:
            if not self._rule_matches(rule, symbols):
                continue

            rule_id = rule.get("rule_id", "UNKNOWN_RULE")
            triggered_rules.append(rule_id)

            # Base action
            action = self._resolve_action(rule["action"], top_intent)

            # Optional overrides (still symbolic)
            overrides = rule.get("overrides", [])
            for ov in overrides:
                if self._rule_matches(ov, symbols):
                    action = self._resolve_action(ov["action"], top_intent)
                    triggered_rules.append(f"{rule_id}::override")

            return {
                "predicted_intent": action["predicted_intent"],
                "decision_state": action["decision_state"],
                "decision_reason": action["decision_reason"],
                "triggered_rules": triggered_rules
            }

        # Should never happen because DEFAULT_RULES includes R_DEFAULT
        return {
            "predicted_intent": top_intent,
            "decision_state": "accepted",
            "decision_reason": "FALLBACK_DEFAULT",
            "triggered_rules": triggered_rules + ["FALLBACK_DEFAULT"]
        }

    # --------------------------------------------------------
    # Public Inference Method
    # --------------------------------------------------------

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.detectors:
            raise ValueError("No detectors loaded.")

        numeric_signals = self._extract_numeric_signals(text)
        symbols = self._symbolize(text, numeric_signals)

        rule_output = self._apply_symbolic_rules(symbols, numeric_signals["top_intent"])

        return {
            "utterance": text,
            "symbols": sorted(list(symbols)),
            "predicted_intent": rule_output["predicted_intent"],
            "decision_state": rule_output["decision_state"],
            "decision_reason": rule_output["decision_reason"],
            "triggered_rules": rule_output["triggered_rules"],

            # Keeping numeric scores optional for debugging/analysis,
            # but NOT used by rules (rules only see symbols).
            "detector_scores": numeric_signals["detector_scores"]
        }

    # --------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------

    def save(self, model_dir: str) -> None:
        model_dir_path = Path(model_dir)
        model_dir_path.mkdir(parents=True, exist_ok=True)

        for intent, detector in self.detectors.items():
            with open(model_dir_path / f"detector_{intent}.pkl", "wb") as f:
                pickle.dump(detector, f)

        # Save config and intents (symbolization thresholds)
        with open(model_dir_path / "config.json", "w", encoding="utf-8") as f:
            json.dump({
                "intents": self.intents,
                "config": self.config
            }, f, indent=2)

        # Save rules as external artifact (data-driven rule engine)
        with open(model_dir_path / "rules.json", "w", encoding="utf-8") as f:
            json.dump({
                "rules": self.rules
            }, f, indent=2)

        print(f"✓ Saved Level1 model to {model_dir_path}")

    @classmethod
    def load(cls, model_dir: str) -> "Level1Classifier":
        model_dir_path = Path(model_dir)

        with open(model_dir_path / "config.json", "r", encoding="utf-8") as f:
            config_data = json.load(f)

        intents = config_data["intents"]
        config = config_data.get("config", CONFIG)

        rules_payload = _load_json_if_exists(model_dir_path / "rules.json")
        rules = (rules_payload or {}).get("rules", DEFAULT_RULES)

        detectors: Dict[str, Any] = {}
        for intent in intents:
            with open(model_dir_path / f"detector_{intent}.pkl", "rb") as f:
                detectors[intent] = pickle.load(f)

        print(f"✓ Loaded Level1 model from {model_dir_path}")
        return cls(detectors=detectors, intents=intents, rules=rules, config=config)