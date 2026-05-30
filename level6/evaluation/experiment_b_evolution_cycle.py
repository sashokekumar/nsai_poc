# level6/evaluation/experiment_b_evolution_cycle.py
"""
Task 15 — Experiment B: Verify one evolution cycle output.

Reads from SymbolCluster and RuleCandidateGen outputs (clusters.json,
candidate_rules/_manifest.json, individual rule JSONs) and verifies that:

  ✓  Symbols are born with coherent predicate profiles
  ✓  Auto-generated names are interpretable (follow naming convention)
  ✓  Candidate rules are structurally valid (required keys present)
  ✓  No duplicate rules were generated (explicit redundancy flag)
  ✓  Grounding quality thresholds are met

Output: level6/evaluation/experiment_b_evolution_results.json

CLI:
    python -m level6.evaluation.experiment_b_evolution_cycle
    python -m level6.evaluation.experiment_b_evolution_cycle --dry-run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT      = Path(__file__).parent.parent.parent
EVAL_DIR       = REPO_ROOT / "level6" / "evaluation"
CLUSTERS_PATH  = REPO_ROOT / "level6" / "data" / "clusters.json"
CANDIDATE_DIR  = REPO_ROOT / "level6" / "data" / "candidate_rules"
REGISTRY_PATH  = REPO_ROOT / "level6" / "data" / "symbol_registry.json"

# Structural requirements for a valid candidate rule JSON
REQUIRED_RULE_KEYS = {"name", "antecedents", "consequent_intent", "rule_strength_init"}

# Minimum grounding quality for promotion eligibility
MIN_GQ = 0.50


def _check_antecedent(node: dict, depth: int = 0) -> list[str]:
    """
    Recursively validate antecedent tree structure.
    Returns list of validation errors (empty = valid).
    """
    errors = []
    if "predicate" in node:
        pred = node["predicate"]
        if pred == "NOT":
            if "operand" not in node:
                errors.append(f"NOT node missing 'operand' at depth {depth}")
            else:
                errors.extend(_check_antecedent(node["operand"], depth + 1))
    elif "logic" in node:
        if node["logic"] not in ("AND", "OR"):
            errors.append(f"Unknown logic operator '{node['logic']}' at depth {depth}")
        if "operands" not in node or not node["operands"]:
            errors.append(f"Logic node missing 'operands' at depth {depth}")
        else:
            for child in node["operands"]:
                errors.extend(_check_antecedent(child, depth + 1))
    else:
        errors.append(f"Node has neither 'predicate' nor 'logic' key at depth {depth}")
    return errors


def run_experiment_b(
    clusters_path:  Path = CLUSTERS_PATH,
    candidate_dir:  Path = CANDIDATE_DIR,
    registry_path:  Path = REGISTRY_PATH,
) -> dict:
    """
    Verify outputs of one evolution cycle.

    Returns:
        Dict with per-symbol checks and overall pass/fail verdict.
    """
    # ── load clusters ────────────────────────────────────────────────────────
    with open(clusters_path, encoding="utf-8") as fh:
        clusters = json.load(fh)

    non_noise = [c for c in clusters if not c.get("is_noise_cluster", False)]

    # ── load registry ────────────────────────────────────────────────────────
    with open(registry_path, encoding="utf-8") as fh:
        registry = json.load(fh)
    symbols = registry.get("symbols", {})

    # ── load manifest ────────────────────────────────────────────────────────
    manifest_path = candidate_dir / "_manifest.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

    # Also gather all candidate rule files directly (manifest may be partial)
    all_rule_files = sorted(
        f for f in candidate_dir.glob("*.json") if f.name != "_manifest.json"
    )

    # ── per-cluster checks ────────────────────────────────────────────────────
    cluster_checks = []
    for cluster in non_noise:
        name = cluster.get("symbol_name", "")
        size = cluster.get("size", 0)
        coh  = cluster.get("cohesion", 0.0)
        gq   = cluster.get("grounding_quality", 0.0)
        dom  = cluster.get("dominant_confusion", "")

        checks = {
            "cluster_id":       cluster.get("cluster_id"),
            "symbol_name":      name,
            "size":             size,
            "cohesion":         coh,
            "grounding_quality":gq,
            "dominant_confusion": dom,
        }

        errors = []
        if not name.startswith("SYM_"):
            errors.append(f"Symbol name '{name}' does not follow SYM_ convention")
        if size < 10:
            errors.append(f"Cluster too small ({size} < 10)")
        if coh < 0.50:
            errors.append(f"Low cohesion ({coh:.4f} < 0.50)")
        if gq < MIN_GQ:
            errors.append(f"Grounding quality {gq:.4f} < min {MIN_GQ} (low_confidence_grounding)")
        if not dom:
            errors.append("No dominant_confusion recorded")

        checks["errors"]  = errors
        checks["passed"]  = len(errors) == 0
        cluster_checks.append(checks)

    # ── per-rule structural checks ────────────────────────────────────────────
    rule_checks = []
    # Build lookup from manifest for redundancy status
    manifest_by_file = {e.get("file", ""): e for e in manifest}

    for rule_path in all_rule_files:
        manifest_entry = manifest_by_file.get(rule_path.name, {})
        sid       = None
        rule_data = {}
        load_error = None
        try:
            with open(rule_path, encoding="utf-8") as fh:
                rule_data = json.load(fh)
            sid = rule_data.get("_l6_symbol_id", rule_path.stem)
        except Exception as e:
            load_error = str(e)
            sid = rule_path.stem

        rule_name    = rule_data.get("name", rule_path.stem)
        is_redundant = (
            rule_data.get("_l6_status") == "redundant"
            or manifest_entry.get("status") == "redundant"
        )

        errors = []
        if load_error:
            errors.append(f"Failed to load rule JSON: {load_error}")
        else:
            missing = REQUIRED_RULE_KEYS - set(rule_data.keys())
            if missing:
                errors.append(f"Missing required keys: {missing}")
            if "antecedents" in rule_data:
                ant_errors = _check_antecedent(rule_data["antecedents"])
                errors.extend(ant_errors)
            if "consequent_intent" in rule_data:
                valid_intents = {"investigate", "summarization", "execution", "out_of_scope"}
                if rule_data["consequent_intent"] not in valid_intents:
                    errors.append(
                        f"Invalid consequent_intent '{rule_data['consequent_intent']}'"
                    )

        rule_checks.append({
            "symbol_id":   sid,
            "rule_name":   rule_name,
            "rule_file":   rule_path.name,
            "is_redundant":is_redundant,
            "consequent":  rule_data.get("consequent_intent"),
            "strength":    rule_data.get("rule_strength_init"),
            "n_antecedent_operands": (
                len(rule_data.get("antecedents", {}).get("operands", []))
                if "antecedents" in rule_data else None
            ),
            "errors":      errors,
            "passed":      len(errors) == 0,
        })

    # ── no-duplicate check ────────────────────────────────────────────
    unique_names = {r["rule_name"] for r in rule_checks}
    has_duplicates = len(unique_names) < len(rule_checks)

    # ── registered symbols check ─────────────────────────────────────────────
    n_registered = len(symbols)
    n_clusters   = len(non_noise)

    # ── overall verdict ───────────────────────────────────────────────────────
    all_clusters_pass = all(c["passed"] for c in cluster_checks)
    all_rules_pass    = all(r["passed"] for r in rule_checks)
    passed = (
        len(non_noise) >= 1
        and n_registered >= 1
        and all_clusters_pass
        and all_rules_pass
        and not has_duplicates
    )

    results = {
        "experiment":         "B — Evolution Cycle Symbol Birth and Rule Generation",
        "computed_at":        datetime.now(timezone.utc).isoformat(),
        "cluster_summary": {
            "total_clusters":     len(clusters),
            "non_noise_clusters": n_clusters,
            "noise_points":       sum(1 for c in clusters if c.get("is_noise_cluster")),
        },
        "registered_symbols": n_registered,
        "rules_generated":    len(manifest),
        "has_duplicates":     has_duplicates,
        "cluster_checks":     cluster_checks,
        "rule_checks":        rule_checks,
        "verdict":            "PASS" if passed else "FAIL",
    }
    return results


def print_results(results: dict) -> None:
    cs = results["cluster_summary"]
    print(f"\n{'='*62}")
    print(f"  Experiment B — Evolution Cycle Verification")
    print(f"{'='*62}")
    print(f"\n  Clusters: {cs['total_clusters']} total  "
          f"({cs['non_noise_clusters']} non-noise, {cs['noise_points']} noise)")
    print(f"  Registered symbols: {results['registered_symbols']}")
    print(f"  Candidate rules:    {results['rules_generated']}")
    print(f"  Duplicate rules:    {'YES (bug)' if results['has_duplicates'] else 'None'}")

    print(f"\n  Cluster checks:")
    for c in results["cluster_checks"]:
        status = "✓" if c["passed"] else "✗"
        print(f"    {status} Cluster {c['cluster_id']:>2}  {c['symbol_name']}")
        print(f"         size={c['size']}  cohesion={c['cohesion']:.4f}  "
              f"gq={c['grounding_quality']:.4f}  confusion: {c['dominant_confusion']}")
        for err in c["errors"]:
            print(f"         [ERR] {err}")

    print(f"\n  Rule structure checks:")
    for r in results["rule_checks"]:
        status = "✓" if r["passed"] else "✗"
        redundant_tag = "  [REDUNDANT]" if r["is_redundant"] else ""
        print(f"    {status} {r['symbol_id']}  {r['rule_name']}{redundant_tag}")
        if not r["is_redundant"]:
            print(f"         intent={r['consequent']}  "
                  f"strength={r['strength']}  "
                  f"operands={r['n_antecedent_operands']}")
        for err in r["errors"]:
            print(f"         [ERR] {err}")

    print(f"\n  Verdict: {results['verdict']}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 15 — Experiment B: Evolution cycle symbol and rule verification"
    )
    p.add_argument("--clusters",    type=Path, default=CLUSTERS_PATH)
    p.add_argument("--candidates",  type=Path, default=CANDIDATE_DIR)
    p.add_argument("--registry",    type=Path, default=REGISTRY_PATH)
    p.add_argument("--dry-run",     action="store_true")
    args = p.parse_args()

    results = run_experiment_b(
        clusters_path = args.clusters,
        candidate_dir = args.candidates,
        registry_path = args.registry,
    )
    print_results(results)

    out_path = EVAL_DIR / "experiment_b_evolution_results.json"
    if not args.dry_run:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"[ExperimentB] Results -> {out_path}")


if __name__ == "__main__":
    main()
