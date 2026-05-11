# level6/rule_candidate_gen.py
"""
RuleCandidateGen -- generates candidate rule_base.json entries from registered symbols.

Each Proposed symbol in the SymbolRegistry carries a predicate_profile derived
from its failure cluster centroid.  This module translates that profile into a
rule_base.json-compatible rule entry that the Level 5 RuleCompiler can parse.

Antecedent construction
-----------------------
The rule antecedent is built from the symbol's predicate_profile using the
following priority order, which mirrors how the existing L5 rules are written:

    1. present_predicates  (centroid >= 0.70) -- AND together; these fire positively
    2. absent_predicates   (centroid <= 0.30) -- each wrapped in NOT
    3. uncertain_predicates (0.30 < c < 0.70) -- top-1 only, added as a weak signal
                            if no present predicates exist (the common boundary case)

For uncertainty-boundary symbols (all centroids < 0.70), only the top-1
uncertain predicate is included in the antecedent.  This keeps the rule
conservative: a broad uncertain antecedent would fire too widely and drive
up the false_positive_rate.

Consequent
----------
majority_gold_intent from the cluster -- the intent that most failure members
SHOULD have been classified as.

Rule strength init
------------------
    rule_strength_init = cohesion * 0.8

The 0.8 conservative factor prevents an untested rule from dominating
existing rules.  After Task 11 validation, the final rule_strength used in
the compiled model is the validated value from the forward-pass injection.

Duplicate detection
-------------------
Before writing a candidate, the generator checks the EXISTING rules in
level5/data/rule_base.json for antecedent overlap:
    - Exact predicate set overlap (same set of present predicates) with the
      same consequent intent -> rejected as duplicate
    - Same consequent with strict antecedent subset -> flagged as redundant
      (written but metadata field redundant_with set)
Candidate rules that are rejected are recorded in the output JSON with
status="rejected".

Output
------
Each candidate is written to:
    level6/data/candidate_rules/R_<symbol_id>.json

A manifest of all candidates is written to:
    level6/data/candidate_rules/_manifest.json

The candidate JSON exactly matches the rule_base.json rule schema so it can
be directly injected into a temporary copy for Task 11 validation.

Usage
-----
    # Generate candidates for all Proposed symbols
    python -m level6.rule_candidate_gen

    # Explicit registry / rule_base paths
    python -m level6.rule_candidate_gen \\
        --registry level6/data/symbol_registry.json \\
        --rule-base level5/data/rule_base.json

    # Dry-run: print candidates but do not write files
    python -m level6.rule_candidate_gen --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from level6.symbol_registry import SymbolRegistry  # noqa: E402
from level6.lifecycle import SymbolStatus           # noqa: E402

DEFAULT_REGISTRY  = REPO_ROOT / "level6" / "data" / "symbol_registry.json"
DEFAULT_RULE_BASE = REPO_ROOT / "level5" / "data" / "rule_base.json"
CANDIDATE_DIR     = REPO_ROOT / "level6" / "data" / "candidate_rules"


# ---------------------------------------------------------------------------
# Antecedent builder
# ---------------------------------------------------------------------------

def _build_antecedent(predicate_profile: dict) -> dict:
    """
    Build a rule_base.json-compatible antecedent from a symbol's predicate_profile.

    Schema mirrors existing L5 rules:
        {"logic": "AND", "operands": [...]}
    where each operand is one of:
        {"predicate": "<name>"}
        {"predicate": "NOT", "operand": {"predicate": "<name>"}}

    Strategy:
        - If present predicates exist: AND all present + NOT all absent.
        - If no present predicates (uncertainty-boundary symbol): use ONLY the
          top-1 uncertain predicate (highest centroid < 0.70) as a single
          positive antecedent, plus NOT for the top-1 absent predicate.
          This keeps the rule conservative in coverage.
    """
    present   = predicate_profile.get("present",   [])
    absent    = predicate_profile.get("absent",    [])
    uncertain = predicate_profile.get("uncertain", [])
    centroid  = predicate_profile.get("centroid",  {})

    operands: list[dict] = []

    if present:
        # Mode 1: strong present predicates
        for pred in present:
            operands.append({"predicate": pred})
        for pred in absent:
            operands.append({"predicate": "NOT", "operand": {"predicate": pred}})
    else:
        # Mode 2: uncertainty-boundary — use top-1 uncertain predicate only
        # Sort uncertain by centroid descending; top-1 is most "almost present"
        sorted_uncertain = sorted(uncertain, key=lambda p: centroid.get(p, 0.0), reverse=True)
        if sorted_uncertain:
            operands.append({"predicate": sorted_uncertain[0]})
        # Add NOT for the single most-absent predicate as a guard
        if absent:
            most_absent = absent[0]   # absent list is already sorted asc by centroid
            operands.append({"predicate": "NOT", "operand": {"predicate": most_absent}})

    if len(operands) == 1:
        # Wrap single operand in an AND for schema consistency
        return {"logic": "AND", "operands": operands}

    return {"logic": "AND", "operands": operands}


# ---------------------------------------------------------------------------
# Duplicate / redundancy detection
# ---------------------------------------------------------------------------

def _positive_predicates(antecedent: dict) -> frozenset[str]:
    """Extract the set of positively asserted predicate names from an antecedent."""
    preds: set[str] = set()
    for op in antecedent.get("operands", []):
        if op.get("predicate") != "NOT":
            preds.add(op["predicate"])
        # recurse one level for OR/AND nesting in existing rules
        for nested in op.get("operands", []):
            if isinstance(nested, dict) and nested.get("predicate") not in (None, "NOT"):
                preds.add(nested["predicate"])
    return frozenset(preds)


def _check_duplicates(
    candidate_antecedent: dict,
    candidate_intent: str,
    existing_rules: list[dict],
) -> tuple[str, str]:
    """
    Returns (status, redundant_with) where status is:
        "ok"         -- no overlap detected
        "redundant"  -- same intent, candidate antecedent is a subset of existing
        "duplicate"  -- exact same positive predicates + same intent
    """
    cand_preds = _positive_predicates(candidate_antecedent)

    for rule in existing_rules:
        if rule.get("consequent_intent") != candidate_intent:
            continue
        existing_preds = _positive_predicates(rule.get("antecedents", {}))
        if cand_preds == existing_preds:
            return "duplicate", rule["name"]
        if cand_preds and cand_preds.issubset(existing_preds):
            return "redundant", rule["name"]

    return "ok", ""


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_candidates(
    registry: SymbolRegistry,
    existing_rules: list[dict],
) -> list[dict]:
    """
    Generate candidate rule dicts for all Proposed symbols.

    Returns a list of candidate dicts (one per symbol), including rejected ones.
    """
    candidates: list[dict] = []

    proposed = registry.by_status(SymbolStatus.PROPOSED)
    if not proposed:
        print("[RuleCandidateGen] No Proposed symbols in registry.")
        return candidates

    for sym in proposed:
        sid     = sym["symbol_id"]
        profile = sym["predicate_profile"]
        intent  = sym["majority_gold_intent"]
        cohesion = sym["cohesion"]

        antecedent = _build_antecedent(profile)
        rule_strength_init = round(cohesion * 0.8, 4)

        status, redundant_with = _check_duplicates(antecedent, intent, existing_rules)

        rule_name = f"R_{sid}_{sym['name']}"

        candidate = {
            # Traceability fields (not in L5 rule_base schema but preserved here)
            "_l6_symbol_id":         sid,
            "_l6_source_cluster_id": None,   # cluster_id set below
            "_l6_dominant_confusion": sym.get("dominant_confusion", ""),
            "_l6_majority_gold_intent": intent,
            "_l6_status":             status,
            "_l6_redundant_with":     redundant_with,

            # rule_base.json schema fields
            "name":              rule_name,
            "description":       (
                f"Candidate rule born from failure cluster {sid}. "
                f"Targets confusion: {sym.get('dominant_confusion', 'n/a')}. "
                f"Coverage: {sym['coverage']} failures."
            ),
            "antecedents":         antecedent,
            "consequent_intent":   intent,
            "rule_strength_init":  rule_strength_init,

            # Extra metadata for Task 11 injection
            "_l6_coverage":           sym["coverage"],
            "_l6_cohesion":           cohesion,
            "_l6_grounding_quality":  sym["grounding_quality"],
        }

        # Back-fill cluster_id from the predicate profile (not stored on symbol,
        # but we can note n/a -- will be linked via symbol_id in all outputs)
        candidate["_l6_source_cluster_id"] = "see symbol_registry.json/" + sid

        candidates.append(candidate)
        print(f"  [{status.upper()}] {sid} -> {rule_name}")
        print(f"      intent={intent}  strength_init={rule_strength_init}")
        print(f"      antecedent operands: {antecedent['operands']}")
        if redundant_with:
            print(f"      note: {status} with existing rule '{redundant_with}'")

    return candidates


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_candidates(candidates: list[dict], dry_run: bool = False) -> Path:
    """
    Write each candidate to CANDIDATE_DIR/R_<symbol_id>.json.
    Write manifest to CANDIDATE_DIR/_manifest.json.
    Returns CANDIDATE_DIR.
    """
    if dry_run:
        print("[RuleCandidateGen] Dry-run: no files written.")
        return CANDIDATE_DIR

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for cand in candidates:
        sid       = cand["_l6_symbol_id"]
        out_path  = CANDIDATE_DIR / f"R_{sid}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(cand, fh, indent=2)
        manifest.append({
            "symbol_id":    sid,
            "rule_name":    cand["name"],
            "status":       cand["_l6_status"],
            "intent":       cand["consequent_intent"],
            "strength_init":cand["rule_strength_init"],
            "file":         str(out_path.name),
        })
        print(f"  [written] {out_path}")

    manifest_path = CANDIDATE_DIR / "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  [manifest] {manifest_path}")

    return CANDIDATE_DIR


# ---------------------------------------------------------------------------
# Update registry with candidate rule names
# ---------------------------------------------------------------------------

def update_registry(registry: SymbolRegistry, candidates: list[dict]):
    """Write candidate_rule_name and rule_strength_init back to each symbol."""
    for cand in candidates:
        if cand["_l6_status"] == "duplicate":
            continue   # rejected duplicates don't get a rule name
        registry.update_validation(
            cand["_l6_symbol_id"],
            candidate_rule_name=cand["name"],
            rule_strength_init=cand["rule_strength_init"],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate candidate rules from Proposed symbols.")
    p.add_argument("--registry",  type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--rule-base", type=Path, default=DEFAULT_RULE_BASE)
    p.add_argument("--dry-run",   action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()

    print(f"[RuleCandidateGen] Registry  : {args.registry}")
    print(f"[RuleCandidateGen] Rule base : {args.rule_base}")

    registry = SymbolRegistry(args.registry)
    summ = registry.summary()
    print(f"[RuleCandidateGen] Proposed symbols: {summ['by_status'].get('proposed', 0)}")

    with open(args.rule_base, encoding="utf-8") as fh:
        rule_base = json.load(fh)
    existing_rules = rule_base.get("rules", [])
    print(f"[RuleCandidateGen] Existing rules  : {len(existing_rules)}")

    print()
    candidates = generate_candidates(registry, existing_rules)

    ok_count  = sum(1 for c in candidates if c["_l6_status"] == "ok")
    dup_count = sum(1 for c in candidates if c["_l6_status"] == "duplicate")
    red_count = sum(1 for c in candidates if c["_l6_status"] == "redundant")
    print()
    print(f"[RuleCandidateGen] {len(candidates)} candidates: "
          f"{ok_count} ok, {red_count} redundant, {dup_count} rejected")

    save_candidates(candidates, dry_run=args.dry_run)

    if not args.dry_run:
        update_registry(registry, candidates)
        registry.save()
        print(f"[RuleCandidateGen] Registry updated with candidate rule names.")

    # Print final candidate summary
    print()
    print("=" * 70)
    print("  Candidate Rule Summary")
    print("=" * 70)
    for cand in candidates:
        status_tag = f"[{cand['_l6_status'].upper()}]"
        print(f"  {status_tag:12s} {cand['_l6_symbol_id']}  {cand['name']}")
        print(f"      confusion    : {cand['_l6_dominant_confusion']}")
        print(f"      gold intent  : {cand['_l6_majority_gold_intent']}")
        print(f"      antecedent   : {cand['antecedents']}")
        print(f"      strength_init: {cand['rule_strength_init']}")
        print()
    print("=" * 70)


if __name__ == "__main__":
    main()
