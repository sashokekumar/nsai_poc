# level3_5/reasoner.py

from typing import Dict, Any
from level3_5.ontology import ENTITIES

def reason(frame) -> Dict[str, Any]:
    """
    Level-4 symbolic reasoning:
    - Interprets the structured frame
    - Produces a deterministic reasoning record
    """
    out: Dict[str, Any] = {}

    if frame.intent == "out_of_scope":
        out["mode"] = "reject"
        out["reason"] = "non_sre_request"
        return out

    if frame.intent == "summarization":
        out["mode"] = "reporting"
        out["target"] = frame.entity or "general"
        out["report_type"] = "summary"
        return out

    if frame.intent == "execution":
        out["mode"] = "action"
        out["target"] = frame.entity or "unspecified"
        out["requires_confirmation"] = True
        return out

    if frame.intent == "investigate":
        out["mode"] = "diagnostic"
        out["target"] = frame.entity or "unspecified"
        out["entity_type"] = ENTITIES.get(frame.entity, {}).get("type") if frame.entity else None
        out["symptom_focus"] = frame.symptom
        out["time_context"] = frame.time_context
        return out

    # fallback
    out["mode"] = "unknown"
    return out