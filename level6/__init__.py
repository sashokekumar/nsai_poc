# level6/__init__.py
"""
Level 6 — Self-Evolving Neuro-Symbolic Cognitive Architecture

Kautz Typology Position
-----------------------
Level 5 (Kautz Type 5) compiles symbolic rules into the neural architecture as a
differentiable rule layer. The rule vocabulary is static: a human authors rule_base.json
and it does not change after training.

Kautz's Type 6 describes neural systems that dynamically use symbolic reasoning
internally — the direction is established in the typology but the standard is not
fully formalised at the implementation level.

This PoC implements a practical extension toward that direction: a Type 5 architecture
with a self-modifying symbolic substrate. The symbolic rule vocabulary is no longer
static. Failures in the Level 5 model drive discovery of new predicate-space symbols,
which generate candidate rules, which are validated and promoted into the rule base.
The neural layer remains unchanged between evolution cycles. The evolution operates
on the symbolic layer only, while preserving the Level 5 differentiable rule architecture.

PoC Scope
---------
This implementation is scoped to Symbol Vocabulary Evolution only:

    L5 inference → failure collection → predicate-space clustering
    → symbol birth → candidate rule generation → validation
    → lifecycle promotion → rule_base update → L5 inference (next cycle)

Out of scope for this PoC (future research directions):
    - Symbol operator evolution
    - Reasoning strategy evolution
    - Differentiable symbolic transforms
    - Ontology relationship evolution

Core Components (implementation order)
---------------------------------------
    ReasoningState      (reasoning_state.py)  — formal tensor-backed state structure
    FailureCollector    (failure_collector.py) — extracts failures from L5 inference
    SymbolCluster       (symbol_cluster.py)    — predicate-space clustering → symbol birth
    SymbolRegistry      (symbol_registry.py)   — lifecycle state machine
    RuleCandidateGen    (rule_candidate_gen.py)— symbol → candidate rule_base entry
    RuleValidator       (rule_validator.py)    — no-retrain + retrain validation
    EvolutionEngine     (evolution_engine.py)  — orchestrates one full evolution cycle
"""

from level6.reasoning_state import ReasoningState
from level6.lifecycle import SymbolStatus, PROPOSED_TO_EXPERIMENTAL, EXPERIMENTAL_TO_ACTIVE

__all__ = [
    "ReasoningState",
    "SymbolStatus",
    "PROPOSED_TO_EXPERIMENTAL",
    "EXPERIMENTAL_TO_ACTIVE",
]
