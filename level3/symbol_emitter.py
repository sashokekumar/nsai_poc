# level3/symbol_emitter.py

from level3.symbol_schema import IntentSymbol
from level3.intent_constants import (
    SUMMARIZATION,
    EXECUTION,
    INVESTIGATE,
    OUT_OF_SCOPE,
)


class SymbolEmitter:
    def __init__(self):
        self.index_to_intent = {
            0: SUMMARIZATION,
            1: EXECUTION,
            2: INVESTIGATE,
            3: OUT_OF_SCOPE,
        }

    def emit(self, class_index: int) -> IntentSymbol:
        if class_index not in self.index_to_intent:
            raise ValueError(f"Unknown class index: {class_index}")

        intent = self.index_to_intent[class_index]
        return IntentSymbol(intent=intent)