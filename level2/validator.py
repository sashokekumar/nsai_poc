from typing import Dict, List, Tuple


def validate(clauses: Dict[str, List[str]], intent: str) -> Tuple[str, Dict[str, List[str]], Dict]:
    """Validate clauses for a given intent.

    Returns (decision_state, details, ambiguity_report)
    - decision_state: 'accepted' | 'needs_clarification' | 'blocked'
    - details: dict with 'hard_rules_failed' and 'soft_rules_passed'
    - ambiguity_report: dict describing missing/conflicting candidates
    """
    hard_failed = []
    soft_passed = []
    ambiguity = {"missing_clauses": [], "conflicting_candidates": {}, "policy_conflicts": []}

    # execute: needs entity and operation
    if intent == "execute":
        if not clauses.get("entity"):
            hard_failed.append("missing_entity")
            ambiguity["missing_clauses"].append("entity")
        if not clauses.get("operation"):
            hard_failed.append("missing_operation")
            ambiguity["missing_clauses"].append("operation")
        # operation must be unambiguous (single candidate)
        if clauses.get("operation") and len(clauses.get("operation", [])) > 1:
            ambiguity["conflicting_candidates"]["operation"] = clauses.get("operation")

    if intent == "investigate":
        if not (clauses.get("metric") or clauses.get("condition")):
            hard_failed.append("missing_metric_or_condition")
            ambiguity["missing_clauses"].append("metric_or_condition")

    # forbidden combo example
    if ("delete" in (c.lower() for c in clauses.get("operation", []))) and ("prod" in (c.lower() for c in clauses.get("environment", []))):
        if "requires_approval" not in [c.lower() for c in clauses.get("constraint", [])]:
            hard_failed.append("forbidden_delete_in_prod_without_approval")
            ambiguity["policy_conflicts"].append("delete_in_prod_requires_approval")

    # heuristics for soft rules
    if clauses.get("time_window"):
        soft_passed.append("has_time_window")

    if not hard_failed:
        state = "accepted"
    elif hard_failed and ambiguity["missing_clauses"] and not ambiguity["policy_conflicts"]:
        state = "needs_clarification"
    else:
        state = "blocked"

    details = {"hard_rules_failed": hard_failed, "soft_rules_passed": soft_passed}
    return state, details, ambiguity
