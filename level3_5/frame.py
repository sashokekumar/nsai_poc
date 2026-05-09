# level3_5/frame.py

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class IntentFrame:
    intent: Optional[str] = None
    entity: Optional[str] = None
    symptom: Optional[str] = None
    time_context: Optional[str] = None
    confidence: Optional[Dict[str, Any]] = None

    def to_dict(self):
        d = asdict(self)
        if d["confidence"] is None:
            d["confidence"] = {}
        return d