# level6/evaluation/experiment_a_baseline.py
"""
Task 14 — Experiment A: Level 5 baseline failure analysis on level6_seed.csv.

Reads from existing FailureCollector outputs (failure_summary.json,
failure_set.jsonl) and produces a structured experiment report.

Key questions answered:
  - What is the baseline failure rate on the augmented seed dataset?
  - Which predicate combinations dominate the failure set?
  - Which intent confusion pairs occur most frequently?
  - Is the failure set large enough for meaningful clustering (≥50 failures,
    ≥2 distinguishable predicate patterns)?

Output: level6/evaluation/experiment_a_baseline_results.json

CLI:
    python -m level6.evaluation.experiment_a_baseline
    python -m level6.evaluation.experiment_a_baseline --dry-run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT        = Path(__file__).parent.parent.parent
EVAL_DIR         = REPO_ROOT / "level6" / "evaluation"
FAILURE_SUMMARY  = REPO_ROOT / "level6" / "data" / "failure_summary.json"
FAILURE_SET      = REPO_ROOT / "level6" / "data" / "failure_set.jsonl"
FULL_INFERENCE   = REPO_ROOT / "level6" / "data" / "full_inference.jsonl"

PREDICATE_COLS = [
    "is_infrastructure", "is_service", "is_metric", "is_incident",
    "is_job", "is_pipeline", "is_unknown", "is_sre_domain",
    "has_runbook", "is_known_incident", "is_metric_query",
]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_experiment_a(
    failure_summary_path: Path = FAILURE_SUMMARY,
    failure_set_path:     Path = FAILURE_SET,
    dry_run:              bool = False,
) -> dict:
    """
    Load FailureCollector outputs and compute structured Experiment A results.

    Returns:
        Dict with baseline statistics and pass/fail criteria assessment.
    """
    # ── load failure summary ─────────────────────────────────────────────────
    with open(failure_summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)

    # ── load failure rows for predicate distribution analysis ────────────────
    failure_rows = _load_jsonl(failure_set_path)

    # ── top-K dominant predicates in the failure set ─────────────────────────
    pred_means: dict[str, float] = summary.get("failure_predicate_profile", {})
    top_predicates = sorted(pred_means.items(), key=lambda x: x[1], reverse=True)[:5]

    # ── confusion pair frequencies ───────────────────────────────────────────
    confusion_counts: dict[str, int] = {}
    for row in failure_rows:
        if row.get("is_misclassification"):
            gold = row.get("gold_intent", "?")
            pred = row.get("predicted_intent", "?")
            key  = f"{gold} -> {pred}"
            confusion_counts[key] = confusion_counts.get(key, 0) + 1
    top_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # ── high-FPR predicate co-occurrences in failure rows ────────────────────
    # Find predicates that are "active" (>0.50) in the highest % of failures
    high_activation: dict[str, int] = {p: 0 for p in PREDICATE_COLS}
    for row in failure_rows:
        named = row.get("_predicate_named", {})
        for p in PREDICATE_COLS:
            if named.get(p, 0.0) > 0.50:
                high_activation[p] += 1

    predicate_activation_rate = {
        p: round(c / len(failure_rows), 4)
        for p, c in high_activation.items()
    }
    top_active_predicates = sorted(
        predicate_activation_rate.items(), key=lambda x: x[1], reverse=True
    )[:5]

    # ── clustering viability check ───────────────────────────────────────────
    n_failures    = summary.get("n_failures", 0)
    min_required  = 50
    clustering_viable = n_failures >= min_required

    # ── predicate diversity: predicates with mean activation > 0.10 in failure set ──
    diverse_predicates = [p for p, m in pred_means.items() if m > 0.10]
    predicate_diverse  = len(diverse_predicates) >= 2

    # ── experiment verdict ────────────────────────────────────────────────────
    passed = clustering_viable and predicate_diverse
    verdict_reasons = []
    if clustering_viable:
        verdict_reasons.append(
            f"sufficient failures: {n_failures} >= {min_required}"
        )
    else:
        verdict_reasons.append(
            f"insufficient failures: {n_failures} < {min_required}"
        )
    if predicate_diverse:
        verdict_reasons.append(
            f"predicate diversity: {len(diverse_predicates)} predicates above 0.20 activation"
        )
    else:
        verdict_reasons.append(
            f"low predicate diversity: only {len(diverse_predicates)} predicates above 0.20"
        )

    results = {
        "experiment":       "A — Level 5 Baseline Failure Analysis",
        "computed_at":      datetime.now(timezone.utc).isoformat(),
        "dataset_stats": {
            "n_total":               summary.get("n_total", 0),
            "n_failures":            n_failures,
            "failure_rate":          summary.get("failure_rate", 0.0),
            "n_misclassification":   summary.get("n_misclassification", 0),
            "n_low_confidence":      summary.get("n_low_confidence", 0),
            "intent_accuracy":       summary.get("intent_accuracy", 0.0),
            "low_conf_threshold":    summary.get("low_confidence_threshold", 0.65),
        },
        "top_confused_pairs": [
            {"pair": k, "count": v} for k, v in top_confusions
        ],
        "top_failure_predicates": [
            {"predicate": k, "mean_activation": round(v, 4)} for k, v in top_predicates
        ],
        "predicate_high_activation_rate": predicate_activation_rate,
        "top_active_predicates_in_failures": [
            {"predicate": k, "activation_rate": v} for k, v in top_active_predicates
        ],
        "clustering_viability": {
            "viable":              passed,
            "n_failures":          n_failures,
            "min_required":        min_required,
            "diverse_predicates":  diverse_predicates,
            "reasons":             verdict_reasons,
        },
        "verdict": "PASS" if passed else "FAIL",
    }

    return results


def print_results(results: dict) -> None:
    ds = results["dataset_stats"]
    cv = results["clustering_viability"]

    print(f"\n{'='*62}")
    print(f"  Experiment A — Level 5 Baseline Failure Analysis")
    print(f"{'='*62}")
    print(f"\n  Dataset: {ds['n_total']} samples  "
          f"(failure_rate={ds['failure_rate']*100:.1f}%, "
          f"intent_acc={ds['intent_accuracy']*100:.1f}%)")
    print(f"  Failures: {ds['n_failures']} total  "
          f"({ds['n_misclassification']} misclassification, "
          f"{ds['n_low_confidence']} low-confidence)")

    print(f"\n  Top confusion pairs:")
    for e in results["top_confused_pairs"]:
        print(f"    {e['pair']:<40} {e['count']:>5}")

    print(f"\n  Top predicates in failure set (mean activation):")
    for e in results["top_failure_predicates"]:
        bar = "█" * int(e["mean_activation"] * 20)
        print(f"    {e['predicate']:<22} {e['mean_activation']:.4f}  {bar}")

    print(f"\n  Clustering viability:")
    for r in cv["reasons"]:
        print(f"    ✓  {r}" if "sufficient" in r or "diversity" in r else f"    ✗  {r}")
    print(f"\n  Verdict: {results['verdict']}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 14 — Experiment A: Level 5 baseline failure analysis"
    )
    p.add_argument("--failure-summary", type=Path, default=FAILURE_SUMMARY)
    p.add_argument("--failure-set",     type=Path, default=FAILURE_SET)
    p.add_argument("--dry-run",         action="store_true",
                   help="Print results without saving file")
    args = p.parse_args()

    results = run_experiment_a(
        failure_summary_path = args.failure_summary,
        failure_set_path     = args.failure_set,
        dry_run              = args.dry_run,
    )
    print_results(results)

    out_path = EVAL_DIR / "experiment_a_baseline_results.json"
    if not args.dry_run:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"[ExperimentA] Results -> {out_path}")


if __name__ == "__main__":
    main()
