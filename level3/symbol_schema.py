# level3/symbol_schema.py

from dataclasses import dataclass
from level3.intent_constants import ALLOWED_INTENTS


@dataclass
class IntentSymbol:
    intent: str

    def __post_init__(self):
        if self.intent not in ALLOWED_INTENTS:
            raise ValueError(f"Invalid intent symbol: {self.intent}")