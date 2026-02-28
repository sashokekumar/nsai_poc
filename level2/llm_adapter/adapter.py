"""
Level-2 "Neural" Adapter (stub but structured)

Goal (Type-2 alignment):
- Adapter is REQUIRED (not optional).
- Adapter returns structured candidates + confidence per clause.
- Adapter can accept a feedback signal from symbolic validation to refine outputs.

This remains deterministic by default so your pipeline stays testable,
but the interface is designed to be swapped with a real LLM/NN later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple


CLAUSES = [
    "entity",
    "operation",
    "metric",
    "time_window",
    "environment",
    "condition",
    "constraint",
    "output_format",
]


# ----------------------------
# Types
# ----------------------------

@dataclass(frozen=True)
class AdapterFeedback:
    """
    Feedback from symbolic validator to help the adapter refine outputs.

    Examples:
      - focus_clause="operation", reason="conflicting_candidates"
      - focus_clause="environment", reason="missing_clause", hints=["prod","staging"]
    """
    focus_clause: Optional[str] = None
    reason: Optional[str] = None
    hints: Optional[List[str]] = None


@dataclass(frozen=True)
class AdapterResult:
    """
    Structured adapter output:
      - clauses: clause -> list of candidate strings (ordered)
      - confidence: clause -> float in [0, 1]
      - notes: debug/audit strings
    """
    clauses: Dict[str, List[str]]
    confidence: Dict[str, float]
    notes: List[str]

    def to_legacy_clause_map(self) -> Dict[str, List[str]]:
        """Backward-friendly helper if you ever need the old format."""
        return self.clauses


# ----------------------------
# Deterministic "NN-like" heuristics
# ----------------------------

_OP_SYNONYMS: Dict[str, List[str]] = {
    "restart": ["restart", "reboot", "bounce"],
    "start": ["start", "bring up"],
    "stop": ["stop", "shutdown", "shut down", "halt"],
    "deploy": ["deploy", "release", "roll out", "rollout"],
    "delete": ["delete", "remove", "destroy", "terminate"],
    "scale": ["scale", "resize", "increase capacity", "decrease capacity"],
    "backup": ["backup", "snapshot"],
}

_METRICS = ["cpu", "memory", "latency", "errors", "throughput"]
_ENV_ALIASES = {
    "production": "prod",
    "prod": "prod",
    "staging": "staging",
    "stage": "staging",
    "development": "dev",
    "dev": "dev",
    "test": "test",
}

_TIME_WINDOW_PAT = re.compile(
    r"\b(?:last|past)\s*(\d+)\s*(m|min|mins|h|hr|hrs|hour|hours|d|day|days)\b",
    flags=re.I,
)

_ENTITY_PAT = re.compile(r"\b[a-zA-Z0-9_-]+-\d+\b")


def _iso_duration(qty: str, unit: str) -> str:
    u = unit.lower()
    if u.startswith("m"):
        return f"PT{qty}M"
    if u.startswith("h"):
        return f"PT{qty}H"
    if u.startswith("d"):
        return f"P{qty}D"
    return f"PT{qty}{u}"


def _find_entities(text: str) -> List[str]:
    return list(dict.fromkeys(_ENTITY_PAT.findall(text)))


def _find_envs(text: str) -> List[str]:
    found = []
    for token in _ENV_ALIASES.keys():
        if re.search(r"\b" + re.escape(token) + r"\b", text, flags=re.I):
            found.append(_ENV_ALIASES[token])
    return list(dict.fromkeys(found))


def _find_time_windows(text: str) -> List[str]:
    out = []
    for qty, unit in _TIME_WINDOW_PAT.findall(text):
        out.append(_iso_duration(qty, unit))
    return list(dict.fromkeys(out))


def _find_metrics(text: str) -> List[str]:
    out = []
    for m in _METRICS:
        if re.search(r"\b" + re.escape(m) + r"\b", text, flags=re.I):
            out.append(m)
    return out


def _find_conditions(text: str) -> List[str]:
    # crude: "if <...>" or "when <...>"
    matches = re.findall(r"\b(?:if|when)\b\s+([^.,;\n]+)", text, flags=re.I)
    return [m.strip() for m in matches]


def _find_constraints(text: str) -> List[str]:
    cons = []
    if re.search(r"requires approval|needs approval|approval required", text, flags=re.I):
        cons.append("requires_approval")
    return cons


def _score_confidence(cands: List[str], base: float, max_bonus: float = 0.25) -> float:
    """
    Simple heuristic:
      - No candidates => 0
      - 1 candidate => base + bonus
      - >1 candidates => base (ambiguity lowers effective confidence)
    """
    if not cands:
        return 0.0
    if len(cands) == 1:
        return min(1.0, base + max_bonus)
    return max(0.1, min(0.85, base))


def _find_operations(text: str) -> List[str]:
    low = text.lower()
    hits: List[str] = []
    for canonical, synonyms in _OP_SYNONYMS.items():
        for s in synonyms:
            if s in low:
                hits.append(canonical)
                break

    # keep order unique
    hits = list(dict.fromkeys(hits))

    # a small heuristic: "restart" often implies "start" (but this causes ambiguity)
    # Keep it, but let validator request refinement.
    if "restart" in hits and "start" not in hits:
        hits.append("start")

    return hits


def _apply_feedback_boost(
    clause: str,
    candidates: List[str],
    feedback: Optional[AdapterFeedback],
) -> Tuple[List[str], List[str]]:
    """
    If validator says "focus on this clause" and provides hints, try to reorder
    candidates so the hinted value comes first.
    """
    notes: List[str] = []
    if not feedback:
        return candidates, notes

    if feedback.focus_clause != clause:
        return candidates, notes

    hints = feedback.hints or []
    if not hints:
        return candidates, notes

    # reorder candidates by hints priority
    reordered = []
    remaining = candidates[:]
    for h in hints:
        for c in candidates:
            if c.lower() == h.lower() and c in remaining:
                reordered.append(c)
                remaining.remove(c)

    reordered.extend(remaining)

    if reordered != candidates:
        notes.append(f"feedback_reorder:{clause}:{hints}")

    return reordered, notes


# ----------------------------
# Public API
# ----------------------------

def extract_clauses(
    utterance: str,
    timeout_s: float = 2.0,
    feedback: Optional[Dict[str, Any]] = None,
) -> AdapterResult:
    """
    REQUIRED adapter call for Type-2 alignment.

    Parameters
    - utterance: raw user text
    - timeout_s: placeholder for real LLM timeouts (unused in deterministic stub)
    - feedback: dict from validator, e.g.
        {"focus_clause":"operation", "reason":"conflicting_candidates", "hints":["restart"]}

    Returns AdapterResult with:
      - clauses: clause->candidates
      - confidence: clause->float
      - notes: audit notes
    """
    text = utterance or ""
    fb = None
    if feedback:
        fb = AdapterFeedback(
            focus_clause=feedback.get("focus_clause"),
            reason=feedback.get("reason"),
            hints=feedback.get("hints") or None,
        )

    notes: List[str] = []
    clauses: Dict[str, List[str]] = {k: [] for k in CLAUSES}
    conf: Dict[str, float] = {k: 0.0 for k in CLAUSES}

    # Extract
    clauses["entity"] = _find_entities(text)
    clauses["operation"] = _find_operations(text)
    clauses["environment"] = _find_envs(text)
    clauses["time_window"] = _find_time_windows(text)
    clauses["metric"] = _find_metrics(text)
    clauses["condition"] = _find_conditions(text)
    clauses["constraint"] = _find_constraints(text)

    # feedback reorder/boost for iterative refinement
    for k in CLAUSES:
        clauses[k], fb_notes = _apply_feedback_boost(k, clauses[k], fb)
        notes.extend(fb_notes)

    # Confidence (heuristic; replace with model scores later)
    conf["entity"] = _score_confidence(clauses["entity"], base=0.65)
    conf["operation"] = _score_confidence(clauses["operation"], base=0.60)
    conf["environment"] = _score_confidence(clauses["environment"], base=0.60)
    conf["time_window"] = _score_confidence(clauses["time_window"], base=0.60)
    conf["metric"] = _score_confidence(clauses["metric"], base=0.60)
    conf["condition"] = _score_confidence(clauses["condition"], base=0.50)
    conf["constraint"] = _score_confidence(clauses["constraint"], base=0.55)
    conf["output_format"] = _score_confidence(clauses["output_format"], base=0.40)

    if fb and fb.focus_clause:
        notes.append(f"feedback_used:{fb.focus_clause}:{fb.reason}")

    return AdapterResult(clauses=clauses, confidence=conf, notes=notes)