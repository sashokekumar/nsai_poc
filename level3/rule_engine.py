# level3/rule_engine.py

from level3.intent_constants import (
    SUMMARIZATION,
    EXECUTION,
    INVESTIGATE,
    OUT_OF_SCOPE,
)


class RuleEngine:
    def route(self, symbol):
        """
        symbol: IntentSymbol
        Returns a structured action decision.
        """

        intent = symbol.intent

        if intent == SUMMARIZATION:
            return self._handle_summarization()

        elif intent == EXECUTION:
            return self._handle_execution()

        elif intent == INVESTIGATE:
            return self._handle_investigation()

        elif intent == OUT_OF_SCOPE:
            return self._handle_out_of_scope()

        else:
            raise ValueError(f"Unhandled intent: {intent}")

    def _handle_summarization(self):
        return {
            "action": "call_summarization_pipeline",
            "requires_approval": False,
        }

    def _handle_execution(self):
        return {
            "action": "route_to_executor",
            "requires_approval": True,
        }

    def _handle_investigation(self):
        return {
            "action": "trigger_diagnostics",
            "requires_approval": False,
        }

    def _handle_out_of_scope(self):
        return {
            "action": "return_safe_fallback",
            "requires_approval": False,
        }