from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any, Optional

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


def _find_entities(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9_-]+-\d+\b", text)


def _find_operations(text: str) -> List[str]:
    ops = ["restart", "reboot", "start", "stop", "deploy", "delete", "restart-service", "scale", "backup"]
    found = []
    low = text.lower()
    for op in ops:
        if op in low:
            found.append(op if op != "reboot" else "restart")
    return list(dict.fromkeys(found))


def _find_environments(text: str) -> List[str]:
    envs = ["prod", "production", "staging", "stage", "dev", "development", "test"]
    found = [e for e in envs if re.search(r"\b" + re.escape(e) + r"\b", text, flags=re.I)]
    out = []
    for e in found:
        if e in ("production",):
            out.append("prod")
        elif e in ("stage",):
            out.append("staging")
        elif e in ("development",):
            out.append("dev")
        else:
            out.append(e)
    return list(dict.fromkeys(out))


def _find_time_windows(text: str) -> List[str]:
    m = re.findall(r"\blast\s*(\d+)\s*(m|min|mins|h|hr|hrs|hour|hours|d|day|days)\b", text, flags=re.I)
    out = []
    for qty, unit in m:
        u = unit.lower()
        if u.startswith("m"):
            out.append(f"PT{qty}M")
        elif u.startswith("h"):
            out.append(f"PT{qty}H")
        elif u.startswith("d"):
            out.append(f"P{qty}D")
        else:
            out.append(f"PT{qty}{u}")
    return out


def _find_metrics(text: str) -> List[str]:
    metrics = ["cpu", "memory", "latency", "errors", "throughput"]
    return [m for m in metrics if re.search(r"\b" + re.escape(m) + r"\b", text, flags=re.I)]


def _find_conditions(text: str) -> List[str]:
    matches = re.findall(r"\b(?:if|when)\b\s+([^.,;\n]+)", text, flags=re.I)
    return [m.strip() for m in matches]


def _find_constraints(text: str) -> List[str]:
    constraints = []
    if re.search(r"requires approval|requires_approval|needs approval|approval required", text, flags=re.I):
        constraints.append("requires_approval")
    return constraints


def _merge_unique(existing: List[str], incoming: List[str]) -> List[str]:
    out = list(existing)
    for item in incoming:
        if item not in out:
            out.append(item)
    return out


def extract_candidates(
    utterance: str,
    adapter_result: Dict[str, Any],
) -> Tuple[Dict[str, List[str]], List[str], Dict[str, Any]]:
    """
    Extract candidate clause values using:
      1) REQUIRED adapter output (NN/LLM)
      2) deterministic detectors (regex/keywords)

    Returns:
      - clauses: Dict[str, List[str]]
      - detectors_fired: List[str]
      - adapter_meta: Dict[str, Any] (confidence, notes)
    """
    detectors: List[str] = []
    text = utterance or ""

    # Start with empty structure
    clauses: Dict[str, List[str]] = {k: [] for k in CLAUSES}

    # --------------------------
    # 1) REQUIRED adapter merge
    # --------------------------
    adapter_clauses = (adapter_result or {}).get("clauses") or {}
    adapter_conf = (adapter_result or {}).get("confidence") or {}
    adapter_notes = (adapter_result or {}).get("notes") or []

    detectors.append("adapter")
    for k in CLAUSES:
        if k in adapter_clauses and adapter_clauses.get(k):
            clauses[k] = _merge_unique(clauses[k], adapter_clauses[k])

    adapter_meta = {
        "confidence": adapter_conf,
        "notes": adapter_notes,
    }

    # --------------------------------
    # 2) deterministic extractor merge
    # --------------------------------
    ents = _find_entities(text)
    if ents:
        clauses["entity"] = _merge_unique(clauses["entity"], ents)
        detectors.append("regex_entity")

    ops = _find_operations(text)
    if ops:
        clauses["operation"] = _merge_unique(clauses["operation"], ops)
        detectors.append("keyword_operation")

    envs = _find_environments(text)
    if envs:
        clauses["environment"] = _merge_unique(clauses["environment"], envs)
        detectors.append("keyword_environment")

    tws = _find_time_windows(text)
    if tws:
        clauses["time_window"] = _merge_unique(clauses["time_window"], tws)
        detectors.append("regex_time_window")

    metrics = _find_metrics(text)
    if metrics:
        clauses["metric"] = _merge_unique(clauses["metric"], metrics)
        detectors.append("keyword_metric")

    conds = _find_conditions(text)
    if conds:
        clauses["condition"] = _merge_unique(clauses["condition"], conds)
        detectors.append("regex_condition")

    cons = _find_constraints(text)
    if cons:
        clauses["constraint"] = _merge_unique(clauses["constraint"], cons)
        detectors.append("keyword_constraint")

    # --------------------------------------------------
    # 3) Feedback override — collapse ambiguous candidates
    #    When the adapter used feedback, trust its ordering
    #    and keep only the top candidate for the focus clause.
    # --------------------------------------------------
    if adapter_meta.get("confidence") and adapter_result:
        notes = adapter_meta.get("notes", [])
        if any("feedback_used" in n for n in notes):
            focus = None
            for n in notes:
                if n.startswith("feedback_used:"):
                    parts = n.split(":")
                    if len(parts) >= 2:
                        focus = parts[1]
            if focus and clauses.get(focus):
                clauses[focus] = [clauses[focus][0]]

    return clauses, detectors, adapter_meta