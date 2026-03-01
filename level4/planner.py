# level4/planner.py

from typing import Dict, Any, List

def create_plan(reasoning: Dict[str, Any]) -> Dict[str, Any]:
    mode = reasoning.get("mode")
    steps: List[str] = []

    if mode == "reject":
        steps = ["inform_user_out_of_scope"]

    elif mode == "reporting":
        steps = ["fetch_recent_data", "aggregate", "generate_summary"]

    elif mode == "action":
        steps = ["validate_permissions", "confirm_action", "execute_operation", "audit_log"]

    elif mode == "diagnostic":
        steps = [
            "collect_metrics",
            "collect_logs",
            "check_recent_changes",
            "analyze_correlations",
            "summarize_findings"
        ]

    else:
        steps = ["request_clarification"]

    return {"steps": steps}