from typing import Optional
from level4.frame import IntentFrame
from level4.intent_model import IntentClassifier
from level4.entity_matcher import EntityMatcher
from level4.ontology import SYMPTOM_ALIASES, TIME_CONTEXT_PATTERNS, INTENTS

# Singletons
_intent_model = IntentClassifier()
_entity_matcher = EntityMatcher()

_MODEL_READY = False
try:
    _intent_model.load()
    _MODEL_READY = True
except Exception:
    _MODEL_READY = False


def _extract_symptom(utterance: str) -> Optional[str]:
    u = utterance.lower()
    for canonical, aliases in SYMPTOM_ALIASES.items():
        for a in aliases:
            if a in u:
                return canonical
    return None


def _extract_time_context(utterance: str) -> Optional[str]:
    u = utterance.lower()
    for label, patterns in TIME_CONTEXT_PATTERNS:
        for p in patterns:
            if p in u:
                return label
    return None


def parse_utterance(utterance: str) -> IntentFrame:
    u = str(utterance)

    # --------------------
    # A. Intent prediction
    # --------------------
    if _MODEL_READY:
        intent, conf = _intent_model.predict(u)
    else:
        intent, conf = "out_of_scope", 0.0

    if intent not in INTENTS:
        intent = "out_of_scope"

    # --------------------
    # B. Structured extraction
    # --------------------
    entity = _entity_matcher.match(u)
    symptom = _extract_symptom(u)
    time_context = _extract_time_context(u)

    # --------------------
    # C. Semantic domain guard
    # --------------------
    # If model says SRE intent but no entity and no symptom detected,
    # treat as out_of_scope
    if intent != "out_of_scope":
        if entity is None and symptom is None:
            intent = "out_of_scope"

    return IntentFrame(
        intent=intent,
        entity=entity,
        symptom=symptom,
        time_context=time_context,
        confidence={"intent": conf},
    )