# level6/lifecycle.py
"""
Symbol lifecycle state machine for Level 6 symbolic evolution.

Every symbol in the SymbolRegistry has a lifecycle status that controls
whether it influences cognition (i.e. whether its candidate rule is active
in the Level 5 rule_base). Promotion and deprecation are gated by
measurable criteria derived from validation experiments.

Lifecycle states
----------------

    Proposed
        The symbol has just been born from a failure cluster. Its candidate
        rule exists but has not been validated. It does NOT yet influence
        the Level 5 rule_base.

    Experimental
        The symbol has passed no-retrain validation (Task 11): its injected
        rule improves accuracy on the failure cluster above the threshold
        without introducing excessive false positives. The candidate rule is
        included in a temporary rule_base extension for evaluation purposes,
        but the base model weights are not yet updated.

    Active
        The symbol has additionally passed retrain validation (Task 11B):
        fine-tuning the Level 5 model with the promoted rule produces a
        meaningful accuracy improvement over the no-retrain injection baseline.
        This is the evidence of genuine neuro-symbolic co-evolution rather
        than runtime rule patching. The rule is permanently promoted into
        rule_base.json.

    Weakening
        An Active symbol whose accuracy_delta has fallen below the minimum
        threshold in two consecutive evaluation cycles. The symbol remains
        in rule_base but is flagged. If it continues to weaken it will be
        deprecated.

    Deprecated
        A symbol that has been Weakening for the minimum required number of
        cycles. Its rule is removed from rule_base.json. The symbol entry is
        retained in the registry for the garbage collection grace period.

Promotion criteria (two-stage gate)
------------------------------------
Stage 1 — Proposed → Experimental  (Task 11: no-retrain forward-pass validation)

    Criteria                                   Threshold
    coverage (failures in cluster)             ≥ MIN_COVERAGE
    cohesion (mean cosine sim to centroid)     ≥ MIN_COHESION
    accuracy_delta on failure cluster          ≥ MIN_ACCURACY_DELTA_NORETRAIN
    false_positive_rate on non-failure set     < MAX_FALSE_POSITIVE_RATE
    grounding_quality score                    ≥ MIN_GROUNDING_QUALITY

Stage 2 — Experimental → Active  (Task 11B: retrain/fine-tune validation)

    Criteria                                                     Threshold
    retrain_delta_over_noretrain                                 ≥ MIN_RETRAIN_DELTA_OVER_NORETRAIN
      (fine-tune accuracy_delta improvement over no-retrain      (proves neural-symbolic co-evolution;
       injection baseline, measured on the full eval set)         not just runtime rule patching)

Weakening / Deprecation
-----------------------
    A symbol enters Weakening when accuracy_delta on the monitored failure
    set drops below WEAKENING_ACCURACY_DELTA_THRESHOLD for
    WEAKENING_CONSECUTIVE_CYCLES consecutive evaluation cycles.

    A Weakening symbol is Deprecated after DEPRECATION_WEAKENING_CYCLES
    consecutive Weakening cycles.

    Deprecated symbols are garbage-collected (removed from registry entirely)
    after GARBAGE_COLLECTION_AGE_CYCLES cycles in the Deprecated state.
"""

from enum import Enum


# ---------------------------------------------------------------------------
# Lifecycle states
# ---------------------------------------------------------------------------

class SymbolStatus(str, Enum):
    """
    Lifecycle state for a symbol in the SymbolRegistry.

    Inherits from str so that instances serialize naturally to JSON as strings
    (e.g. "proposed") without needing a custom encoder.
    """
    PROPOSED     = "proposed"
    EXPERIMENTAL = "experimental"
    ACTIVE       = "active"
    WEAKENING    = "weakening"
    DEPRECATED   = "deprecated"


# ---------------------------------------------------------------------------
# Promotion criteria — Proposed → Experimental (Task 11: no-retrain)
# ---------------------------------------------------------------------------

PROPOSED_TO_EXPERIMENTAL: dict = {
    # Minimum number of failure-set members in the cluster
    "min_coverage": 10,

    # Minimum within-cluster cohesion (mean cosine similarity to centroid)
    # Ensures the cluster is tight enough to represent a coherent symbolic gap
    "min_cohesion": 0.65,

    # Minimum accuracy improvement on the failure cluster after injecting the
    # candidate rule (no-retrain forward-pass only, Task 11)
    "min_accuracy_delta_noretrain": 0.03,

    # Maximum fraction of non-failure examples that the new rule incorrectly
    # fires on (false positive rate on the held-out non-failure set)
    "max_false_positive_rate": 0.10,

    # Minimum grounding quality score:
    #   cohesion × (n_stable_predicates / 11)
    # where stable = mean > 0.70 or mean < 0.30 (not uncertain)
    # Symbols below this threshold are flagged low_confidence_grounding and
    # not promoted without manual override
    "min_grounding_quality": 0.50,
}


# ---------------------------------------------------------------------------
# Promotion criteria — Experimental → Active (Task 11B: retrain)
# ---------------------------------------------------------------------------

EXPERIMENTAL_TO_ACTIVE: dict = {
    # Minimum improvement in accuracy_delta when fine-tuning the full Level 5
    # model with the promoted rule, over the no-retrain injection baseline.
    # Measured on the full evaluation set (not just the failure cluster).
    # A positive value here is evidence that the neural trunk has adapted to
    # the new symbolic structure — genuine neuro-symbolic co-evolution.
    "min_retrain_delta_over_noretrain": 0.01,
}


# ---------------------------------------------------------------------------
# Weakening detection
# ---------------------------------------------------------------------------

WEAKENING_CRITERIA: dict = {
    # accuracy_delta must fall below this threshold to start a Weakening count
    "max_accuracy_delta": 0.01,

    # Number of consecutive evaluation cycles with accuracy_delta below
    # the threshold before the symbol transitions to Weakening
    "consecutive_cycles_to_weaken": 2,
}


# ---------------------------------------------------------------------------
# Deprecation
# ---------------------------------------------------------------------------

DEPRECATION_CRITERIA: dict = {
    # Number of consecutive Weakening cycles before the symbol is Deprecated
    "weakening_cycles_to_deprecate": 3,
}


# ---------------------------------------------------------------------------
# Garbage collection
# ---------------------------------------------------------------------------

GARBAGE_COLLECTION: dict = {
    # Number of cycles a symbol may remain in Deprecated state before its
    # registry entry is removed entirely. Provides a historical grace period.
    "deprecated_age_cycles": 5,
}


# ---------------------------------------------------------------------------
# Transition helpers
# ---------------------------------------------------------------------------

def can_promote_to_experimental(symbol: dict) -> tuple[bool, list[str]]:
    """
    Check whether a Proposed symbol meets all Stage 1 promotion criteria.

    Args:
        symbol : a symbol dict from SymbolRegistry (must contain validation fields)

    Returns:
        (eligible, reasons) — eligible is True if all criteria are met;
        reasons is a list of human-readable failure messages if not eligible.
    """
    criteria = PROPOSED_TO_EXPERIMENTAL
    reasons: list[str] = []

    coverage = symbol.get("coverage", 0)
    if coverage < criteria["min_coverage"]:
        reasons.append(
            f"coverage {coverage} < min {criteria['min_coverage']}"
        )

    cohesion = symbol.get("cohesion", 0.0)
    if cohesion < criteria["min_cohesion"]:
        reasons.append(
            f"cohesion {cohesion:.3f} < min {criteria['min_cohesion']}"
        )

    acc_delta = symbol.get("accuracy_delta_noretrain", 0.0)
    if acc_delta < criteria["min_accuracy_delta_noretrain"]:
        reasons.append(
            f"accuracy_delta_noretrain {acc_delta:.3f} < min "
            f"{criteria['min_accuracy_delta_noretrain']}"
        )

    fpr = symbol.get("false_positive_rate", 1.0)
    if fpr >= criteria["max_false_positive_rate"]:
        reasons.append(
            f"false_positive_rate {fpr:.3f} ≥ max "
            f"{criteria['max_false_positive_rate']}"
        )

    gq = symbol.get("grounding_quality", 0.0)
    if gq < criteria["min_grounding_quality"]:
        reasons.append(
            f"grounding_quality {gq:.3f} < min "
            f"{criteria['min_grounding_quality']} "
            f"(low_confidence_grounding — manual review required)"
        )

    return (len(reasons) == 0), reasons


def can_promote_to_active(symbol: dict) -> tuple[bool, list[str]]:
    """
    Check whether an Experimental symbol meets all Stage 2 promotion criteria.

    Args:
        symbol : a symbol dict from SymbolRegistry (must contain retrain fields)

    Returns:
        (eligible, reasons)
    """
    criteria = EXPERIMENTAL_TO_ACTIVE
    reasons: list[str] = []

    retrain_delta = symbol.get("retrain_delta_over_noretrain", None)
    if retrain_delta is None:
        reasons.append(
            "retrain_delta_over_noretrain not set — Task 11B validation "
            "has not been run for this symbol"
        )
    elif retrain_delta < criteria["min_retrain_delta_over_noretrain"]:
        reasons.append(
            f"retrain_delta_over_noretrain {retrain_delta:.4f} < min "
            f"{criteria['min_retrain_delta_over_noretrain']} — neural trunk "
            "has not meaningfully adapted to the new rule; this would be "
            "runtime rule patching, not neuro-symbolic co-evolution"
        )

    return (len(reasons) == 0), reasons


def should_weaken(symbol: dict) -> bool:
    """
    Return True if an Active symbol should transition to Weakening.

    Checks both the accuracy_delta threshold and the consecutive cycle count.
    """
    criteria = WEAKENING_CRITERIA
    if symbol.get("status") != SymbolStatus.ACTIVE:
        return False
    acc_delta = symbol.get("accuracy_delta_noretrain", 1.0)
    weak_count = symbol.get("consecutive_weak_cycles", 0)
    return (
        acc_delta < criteria["max_accuracy_delta"]
        and weak_count >= criteria["consecutive_cycles_to_weaken"]
    )


def should_deprecate(symbol: dict) -> bool:
    """Return True if a Weakening symbol should transition to Deprecated."""
    criteria = DEPRECATION_CRITERIA
    if symbol.get("status") != SymbolStatus.WEAKENING:
        return False
    return (
        symbol.get("weakening_count", 0)
        >= criteria["weakening_cycles_to_deprecate"]
    )


def should_garbage_collect(symbol: dict, current_cycle: int) -> bool:
    """Return True if a Deprecated symbol is old enough to be removed."""
    if symbol.get("status") != SymbolStatus.DEPRECATED:
        return False
    deprecated_at = symbol.get("deprecated_cycle", current_cycle)
    age = current_cycle - deprecated_at
    return age >= GARBAGE_COLLECTION["deprecated_age_cycles"]
