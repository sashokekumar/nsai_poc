import re
from typing import Dict, List, Tuple

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
    # common host/resource patterns: name-01, svc-123, db-primary
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
    # normalize aliases
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
    # matches like "last 5m", "last 2 hours", "for 24h"
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
    # crude capture: "if <something>", "when <something>"
    matches = re.findall(r"\b(?:if|when)\b\s+([^.,;\n]+)", text, flags=re.I)
    return [m.strip() for m in matches]


def _find_constraints(text: str) -> List[str]:
    constraints = []
    if re.search(r"requires approval|requires_approval|needs approval|approval required", text, flags=re.I):
        constraints.append("requires_approval")
    return constraints


def extract_candidates(utterance: str, use_adapter: bool = False, adapter_result: Dict[str, List[str]] = None) -> Tuple[Dict[str, List[str]], List[str]]:
    """Extract candidate clause values using deterministic detectors.

    Returns (clauses, detectors_fired)
    """
    detectors = []
    text = utterance or ""
    clauses: Dict[str, List[str]] = {k: [] for k in CLAUSES}

    ents = _find_entities(text)
    if ents:
        clauses["entity"] = ents
        detectors.append("regex_entity")

    ops = _find_operations(text)
    if ops:
        clauses["operation"] = ops
        detectors.append("keyword_operation")

    envs = _find_environments(text)
    if envs:
        clauses["environment"] = envs
        detectors.append("keyword_environment")

    tws = _find_time_windows(text)
    if tws:
        clauses["time_window"] = tws
        detectors.append("regex_time_window")

    metrics = _find_metrics(text)
    if metrics:
        clauses["metric"] = metrics
        detectors.append("keyword_metric")

    conds = _find_conditions(text)
    if conds:
        clauses["condition"] = conds
        detectors.append("regex_condition")

    cons = _find_constraints(text)
    if cons:
        clauses["constraint"] = cons
        detectors.append("keyword_constraint")

    # adapter results can be merged (adapter is opt-in and kept separate in trace)
    if use_adapter and adapter_result:
        detectors.append("adapter")
        for k, v in (adapter_result or {}).items():
            if k in clauses and v:
                # extend while keeping order unique
                existing = clauses[k]
                for item in v:
                    if item not in existing:
                        existing.append(item)
                clauses[k] = existing

    return clauses, detectors
