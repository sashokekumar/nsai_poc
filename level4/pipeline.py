# level4/pipeline.py

from typing import Dict, Any
from level4.semantic_parser import parse_utterance
from level4.reasoner import reason
from level4.planner import create_plan
from level4.responder import generate_response

def run_pipeline(utterance: str) -> Dict[str, Any]:
    frame = parse_utterance(utterance)
    reasoning = reason(frame)
    plan = create_plan(reasoning)
    response = generate_response(frame, reasoning, plan)

    return {
        "frame": frame.to_dict(),
        "reasoning": reasoning,
        "plan": plan,
        "response": response
    }