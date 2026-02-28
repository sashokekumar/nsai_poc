from __future__ import annotations

from typing import Dict, List, Tuple, Any


def validate(clauses: Dict[str, List[str]], intent: str) -> Tuple[str, Dict[str, List[str]], Dict[str, Any], Dict[str, Any]]:
    """
    Validate clauses for a given intent.

    Returns:
      (decision_state, details, ambiguity_report, feedback)

    decision_state:
      - 'accepted' | 'needs_clarification' | 'blocked'

    feedback:
      - signals to the adapter what to refine in next pass:
          {
            "focus_clause": "operation",
            "reason": "conflicting_candidates",
            "hints": ["restart"]
          }
    """
    hard_failed: List[str] = []
    soft_passed: List[str] = []

    ambiguity: Dict[str, Any] = {
        "missing_clauses": [],
        "conflicting_candidates": {},
        "policy_conflicts": [],
        "needs_iteration": False,
    }

    feedback: Dict[str, Any] = {
        "focus_clause": None,
        "reason": None,
        "hints": [],
    }

    # --------------------------
    # Intent: execute
    # --------------------------
    if intent == "execute":
        if not clauses.get("entity"):
            hard_failed.append("missing_entity")
            ambiguity["missing_clauses"].append("entity")

        if not clauses.get("operation"):
            hard_failed.append("missing_operation")
            ambiguity["missing_clauses"].append("operation")

        # operation must be unambiguous (single candidate)
        ops = clauses.get("operation") or []
        if ops and len(ops) > 1:
            ambiguity["conflicting_candidates"]["operation"] = ops
            ambiguity["needs_iteration"] = True
            feedback["focus_clause"] = "operation"
            feedback["reason"] = "conflicting_candidates"
            # heuristic hint: prefer "restart" over "start" if both exist
            if "restart" in [o.lower() for o in ops]:
                feedback["hints"] = ["restart"]
            else:
                feedback["hints"] = [ops[0]]

    # --------------------------
    # Intent: investigate
    # --------------------------
    if intent == "investigate":
        if not (clauses.get("metric") or clauses.get("condition")):
            hard_failed.append("missing_metric_or_condition")
            ambiguity["missing_clauses"].append("metric_or_condition")
            ambiguity["needs_iteration"] = True
            feedback["focus_clause"] = "metric"
            feedback["reason"] = "missing_clause"
            feedback["hints"] = ["cpu", "memory", "latency", "errors", "throughput"]

    # --------------------------
    # Policy gate example:
    # delete in prod requires approval
    # --------------------------
    ops_lower = [c.lower() for c in (clauses.get("operation") or [])]
    env_lower = [c.lower() for c in (clauses.get("environment") or [])]
    cons_lower = [c.lower() for c in (clauses.get("constraint") or [])]

    if ("delete" in ops_lower) and ("prod" in env_lower):
        if "requires_approval" not in cons_lower:
            hard_failed.append("forbidden_delete_in_prod_without_approval")
            ambiguity["policy_conflicts"].append("delete_in_prod_requires_approval")
            # This is not "needs clarification" — it's a block unless approval is present
            feedback["focus_clause"] = "constraint"
            feedback["reason"] = "policy_requires_constraint"
            feedback["hints"] = ["requires_approval"]

    # --------------------------
    # Soft rules
    # --------------------------
    if clauses.get("time_window"):
        soft_passed.append("has_time_window")

    # --------------------------
    # Decide state
    # --------------------------
    if not hard_failed and not ambiguity["conflicting_candidates"]:
        state = "accepted"
        feedback = {"focus_clause": None, "reason": None, "hints": []}
        ambiguity["needs_iteration"] = False
    elif ambiguity["conflicting_candidates"]:
        state = "needs_clarification"
        ambiguity["needs_iteration"] = True
    else:
        # If only missing clauses (and no policy conflicts), ask for clarification / iteration
        if ambiguity["missing_clauses"] and not ambiguity["policy_conflicts"]:
            state = "needs_clarification"
            ambiguity["needs_iteration"] = True
            if not feedback["focus_clause"]:
                # pick first missing clause for refinement
                first_missing = ambiguity["missing_clauses"][0]
                feedback["focus_clause"] = first_missing
                feedback["reason"] = "missing_clause"
        else:
            state = "blocked"
            # blocked can still provide feedback (e.g., requires approval)

    details = {
        "hard_rules_failed": hard_failed,
        "soft_rules_passed": soft_passed,
    }
    return state, details, ambiguity, feedback