# level6/evaluation/experiment_c_validate_rules.py
"""
Task 16 — Experiment C: Candidate rule validation and accuracy-delta measurement.

Reads from RuleValidator outputs (validation_reports/) and produces a
structured comparison report showing:

  - Accuracy delta on each symbol's failure cluster (no-retrain injection)
  - False positive rate on the non-failure set
  - Whether each symbol meets the Experimental promotion threshold
  - Summary: how many symbols passed both gates

Output: level6/evaluation/experiment_c_validation_results.json

CLI:
    python -m level6.evaluation.experiment_c_validate_rules
    python -m level6.evaluation.experiment_c_validate_rules --dry-run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT      = Path(__file__).parent.parent.parent
EVAL_DIR       = REPO_ROOT / "level6" / "evaluation"
REPORTS_DIR    = REPO_ROOT / "level6" / "data" / "validation_reports"
REGISTRY_PATH  = REPO_ROOT / "level6" / "data" / "symbol_registry.json"
FAILURE_SUMMARY= REPO_ROOT / "level6" / "data" / "failure_summary.json"

# Task 11 Experimental promotion thresholds (from lifecycle.py)
MIN_ACC_DELTA = 0.03
MAX_FPR       = 0.10


def run_experiment_c(
    reports_dir:   Path = REPORTS_DIR,
    registry_path: Path = REGISTRY_PATH,
    failure_summary_path: Path = FAILURE_SUMMARY,
) -> dict:
    """
    Load validation reports, evaluate eligibility, build comparison table.

    Returns:
        Structured Experiment C results dict.
    """
    # ── load registry for baseline context ───────────────────────────────────
    with open(registry_path, encoding="utf-8") as fh:
        registry = json.load(fh)
    symbols = registry.get("symbols", {})

    # ── load failure summary for baseline accuracy ────────────────────────────
    baseline_intent_acc = None
    if failure_summary_path.exists():
        with open(failure_summary_path, encoding="utf-8") as fh:
            fs = json.load(fh)
        baseline_intent_acc = fs.get("intent_accuracy")

    # ── load per-symbol validation reports ───────────────────────────────────
    report_files = sorted(reports_dir.glob("report_S_*.json"))
    symbol_results = []

    for rp in report_files:
        with open(rp, encoding="utf-8") as fh:
            rpt = json.load(fh)

        sid          = rpt.get("symbol_id", rp.stem.replace("report_", ""))
        sym          = symbols.get(sid, {})
        status       = str(sym.get("status", "unknown"))
        acc_baseline = rpt.get("acc_baseline_cluster", None)
        acc_injected = rpt.get("acc_injected_cluster", None)
        acc_delta    = rpt.get("accuracy_delta_noretrain", None)
        fpr          = rpt.get("false_positive_rate", None)
        eligible     = rpt.get("promotion_eligible", False)
        reasons      = rpt.get("promotion_reasons", [])

        # Determine which gate(s) failed
        gate_results = {}
        if acc_delta is not None:
            gate_results["acc_delta_gate"] = {
                "value":     round(acc_delta, 4),
                "threshold": f">= {MIN_ACC_DELTA}",
                "passed":    acc_delta >= MIN_ACC_DELTA,
            }
        if fpr is not None:
            gate_results["fpr_gate"] = {
                "value":     round(fpr, 4),
                "threshold": f"< {MAX_FPR}",
                "passed":    fpr < MAX_FPR,
            }

        # Retrain delta (Task 11B) if available
        retrain_delta = sym.get("retrain_delta_over_noretrain")
        retrain_eligible = (retrain_delta is not None and retrain_delta >= 0.01)

        symbol_results.append({
            "symbol_id":              sid,
            "symbol_name":            sym.get("name") or sym.get("symbol_name", ""),
            "current_status":         status,
            "dominant_confusion":     rpt.get("dominant_confusion", ""),
            "cluster_size":           rpt.get("cluster_size", 0),
            "non_failure_size":       rpt.get("non_failure_size", 0),
            "acc_baseline_cluster":   round(acc_baseline, 4)  if acc_baseline  is not None else None,
            "acc_injected_cluster":   round(acc_injected, 4)  if acc_injected  is not None else None,
            "accuracy_delta_noretrain":round(acc_delta, 4)    if acc_delta     is not None else None,
            "false_positive_rate":    round(fpr, 4)           if fpr           is not None else None,
            "retrain_delta":          round(retrain_delta, 4) if retrain_delta is not None else None,
            "gate_results":           gate_results,
            "stage1_eligible":        eligible,
            "stage2_eligible":        retrain_eligible,
            "fully_active_eligible":  eligible and retrain_eligible,
            "non_eligible_reasons":   reasons,
        })

    # ── summary stats ─────────────────────────────────────────────────────────
    n_stage1_pass = sum(1 for s in symbol_results if s["stage1_eligible"])
    n_stage2_pass = sum(1 for s in symbol_results if s["stage2_eligible"])
    n_both_pass   = sum(1 for s in symbol_results if s["fully_active_eligible"])

    # Intent accuracy improvement from injecting all eligible rules
    # Best-case estimate: weighted average of acc_delta across eligible symbols
    eligible_deltas = [
        s["accuracy_delta_noretrain"]
        for s in symbol_results
        if s["stage1_eligible"] and s["accuracy_delta_noretrain"] is not None
    ]
    mean_eligible_delta = (
        round(sum(eligible_deltas) / len(eligible_deltas), 4)
        if eligible_deltas else None
    )

    # ── experiment verdict ─────────────────────────────────────────────────────
    # PASS: at least one symbol meets both Stage 1 gates
    passed = n_stage1_pass >= 1

    results = {
        "experiment":       "C — Candidate Rule Validation and Accuracy-Delta Measurement",
        "computed_at":      datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "stage1_min_acc_delta":  MIN_ACC_DELTA,
            "stage1_max_fpr":        MAX_FPR,
            "stage2_min_retrain_delta": 0.01,
        },
        "baseline_intent_accuracy": baseline_intent_acc,
        "symbol_results":           symbol_results,
        "summary": {
            "n_evaluated":     len(symbol_results),
            "n_stage1_pass":   n_stage1_pass,
            "n_stage2_pass":   n_stage2_pass,
            "n_fully_eligible":n_both_pass,
            "mean_eligible_acc_delta": mean_eligible_delta,
        },
        "verdict": "PASS" if passed else "FAIL",
    }
    return results


def print_results(results: dict) -> None:
    sm = results["summary"]
    print(f"\n{'='*70}")
    print(f"  Experiment C — Candidate Rule Validation")
    print(f"{'='*70}")
    if results["baseline_intent_accuracy"] is not None:
        print(f"\n  Baseline intent accuracy (L5): "
              f"{results['baseline_intent_accuracy']*100:.2f}%")
    print(f"\n  Thresholds: acc_delta >= {results['thresholds']['stage1_min_acc_delta']}  |  "
          f"fpr < {results['thresholds']['stage1_max_fpr']}  |  "
          f"retrain_delta >= {results['thresholds']['stage2_min_retrain_delta']}")

    hdr = (f"\n  {'Symbol':<8} {'Status':<12} {'Confusion':<38} "
           f"{'Δacc':>7} {'fpr':>6} {'Δretrain':>9} {'S1':>3} {'S2':>3}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))

    for s in results["symbol_results"]:
        confusion_short = s["dominant_confusion"][:36] if s["dominant_confusion"] else ""
        ad  = f"{s['accuracy_delta_noretrain']:+.4f}" if s["accuracy_delta_noretrain"] is not None else "    n/a"
        fp  = f"{s['false_positive_rate']:.4f}"       if s["false_positive_rate"]       is not None else "   n/a"
        rd  = f"{s['retrain_delta']:+.4f}"            if s["retrain_delta"]             is not None else "      n/a"
        s1  = "✓" if s["stage1_eligible"] else "✗"
        s2  = "✓" if s["stage2_eligible"] else "✗" if s["retrain_delta"] is not None else "-"
        print(f"  {s['symbol_id']:<8} {s['current_status']:<12} "
              f"{confusion_short:<38} {ad:>7} {fp:>6} {rd:>9} {s1:>3} {s2:>3}")

    print(f"\n  Summary:")
    print(f"    Evaluated:         {sm['n_evaluated']}")
    print(f"    Stage 1 pass:      {sm['n_stage1_pass']}  "
          f"(acc_delta >= {results['thresholds']['stage1_min_acc_delta']} "
          f"AND fpr < {results['thresholds']['stage1_max_fpr']})")
    print(f"    Stage 2 pass:      {sm['n_stage2_pass']}  (retrain_delta >= 0.01)")
    print(f"    Fully eligible:    {sm['n_fully_eligible']}")
    if sm["mean_eligible_acc_delta"] is not None:
        print(f"    Mean Δacc (S1 pass): {sm['mean_eligible_acc_delta']:+.4f}")

    print(f"\n  Verdict: {results['verdict']}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 16 — Experiment C: Candidate rule validation results"
    )
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    p.add_argument("--registry",    type=Path, default=REGISTRY_PATH)
    p.add_argument("--failure-summary", type=Path, default=FAILURE_SUMMARY)
    p.add_argument("--dry-run",     action="store_true")
    args = p.parse_args()

    results = run_experiment_c(
        reports_dir          = args.reports_dir,
        registry_path        = args.registry,
        failure_summary_path = args.failure_summary,
    )
    print_results(results)

    out_path = EVAL_DIR / "experiment_c_validation_results.json"
    if not args.dry_run:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"[ExperimentC] Results -> {out_path}")


if __name__ == "__main__":
    main()
