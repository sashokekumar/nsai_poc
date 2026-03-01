# level4/responder.py

from typing import Dict, Any

def generate_response(frame, reasoning: Dict[str, Any], plan: Dict[str, Any]) -> str:
    if reasoning.get("mode") == "reject":
        return "Out of scope: I can help with SRE/operations requests (summarize/investigate/execute)."

    return (
        f"Intent: {frame.intent}\n"
        f"Entity: {frame.entity}\n"
        f"Symptom: {frame.symptom}\n"
        f"Time Context: {frame.time_context}\n"
        f"Planned Steps: {plan.get('steps')}"
    )