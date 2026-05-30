# level6/evaluation/evolution_metrics.py
"""
Task 13 — Per-cycle evolution metrics for the Level 6 symbol vocabulary.

Computes and saves:
  - symbol counts per lifecycle state
  - accuracy_delta and false_positive_rate distributions (validated symbols)
  - grounding_quality distribution (all symbols)
  - failure coverage: % of failures covered by Active symbols' clusters
  - active symbol details table

Saves:
  level6/evaluation/evolution_report_cycle_{N}.json  — per-cycle report
  level6/evaluation/evolution_summary.json           — rolling multi-cycle summary

CLI:
    python -m level6.evaluation.evolution_metrics
    python -m level6.evaluation.evolution_metrics --cycle 3
    python -m level6.evaluation.evolution_metrics --dry-run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT          = Path(__file__).parent.parent.parent
EVAL_DIR           = REPO_ROOT / "level6" / "evaluation"
REGISTRY_PATH      = REPO_ROOT / "level6" / "data" / "symbol_registry.json"
FAILURE_SUMMARY    = REPO_ROOT / "level6" / "data" / "failure_summary.json"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n":    len(vals),
        "mean": round(sum(vals) / len(vals), 4),
        "min":  round(min(vals), 4),
        "max":  round(max(vals), 4),
    }


def compute_metrics(
    registry_path:      Path = REGISTRY_PATH,
    failure_summary_path: Path = FAILURE_SUMMARY,
    cycle:              int | None = None,
) -> dict[str, Any]:
    """
    Compute per-cycle evolution metrics from the registry and failure summary.

    Args:
        registry_path:       Path to symbol_registry.json.
        failure_summary_path: Path to failure_summary.json (FailureCollector output).
        cycle:               Override cycle number (default: current_cycle from registry).

    Returns:
        Metrics dict ready for JSON serialisation.
    """
    # ── load registry ────────────────────────────────────────────────────────
    with open(registry_path, encoding="utf-8") as fh:
        registry = json.load(fh)

    current_cycle = registry.get("current_cycle", 0)
    if cycle is None:
        cycle = current_cycle
    symbols = registry.get("symbols", {})

    # ── 1. symbol counts per status ──────────────────────────────────────────
    status_counts: dict[str, int] = {}
    for sym in symbols.values():
        s = str(sym.get("status", "unknown"))
        status_counts[s] = status_counts.get(s, 0) + 1

    # ── 2. metric distributions across all validated symbols ─────────────────
    acc_deltas: list[float] = []
    fprs:       list[float] = []
    gqs:        list[float] = []

    for sym in symbols.values():
        ad = sym.get("accuracy_delta_noretrain")
        fp = sym.get("false_positive_rate")
        gq = sym.get("grounding_quality")
        if ad is not None:
            acc_deltas.append(float(ad))
        if fp is not None:
            fprs.append(float(fp))
        if gq is not None:
            gqs.append(float(gq))

    # ── 3. failure coverage by Active symbols ────────────────────────────────
    # A failure is "covered" if it belongs to a cluster whose symbol is Active.
    # We use the `coverage` field (= cluster size in the failure set) stored in
    # each symbol, rather than per-row membership indices.
    total_failures   = 0
    covered_failures = 0

    if failure_summary_path.exists():
        with open(failure_summary_path, encoding="utf-8") as fh:
            fs = json.load(fh)
        total_failures = fs.get("n_failures", 0)

    for sym in symbols.values():
        if str(sym.get("status", "")) == "active":
            covered_failures += int(sym.get("coverage", 0))

    coverage_rate = (
        round(covered_failures / total_failures, 4) if total_failures > 0 else 0.0
    )

    # ── 4. active symbol details ─────────────────────────────────────────────
    active_details = []
    for sid, sym in sorted(symbols.items()):
        if str(sym.get("status", "")) == "active":
            active_details.append({
                "symbol_id":               sid,
                "symbol_name":             sym.get("name") or sym.get("symbol_name"),
                "coverage":                sym.get("coverage"),
                "accuracy_delta_noretrain":sym.get("accuracy_delta_noretrain"),
                "false_positive_rate":     sym.get("false_positive_rate"),
                "retrain_delta":           sym.get("retrain_delta_over_noretrain"),
                "grounding_quality":       sym.get("grounding_quality"),
                "consecutive_weak_cycles": sym.get("consecutive_weak_cycles", 0),
            })

    # ── 5. weakening / deprecated details ────────────────────────────────────
    weakening_details = []
    deprecated_details = []
    for sid, sym in sorted(symbols.items()):
        status = str(sym.get("status", ""))
        entry = {
            "symbol_id":      sid,
            "symbol_name":    sym.get("name") or sym.get("symbol_name"),
            "weakening_count":sym.get("weakening_count", 0),
        }
        if status == "weakening":
            weakening_details.append(entry)
        elif status == "deprecated":
            entry["deprecated_cycle"] = sym.get("deprecated_cycle")
            deprecated_details.append(entry)

    return {
        "cycle":                  cycle,
        "computed_at":            datetime.now(timezone.utc).isoformat(),
        "symbol_counts":          status_counts,
        "total_symbols":          len(symbols),
        "accuracy_delta_dist":    _stats(acc_deltas),
        "fpr_dist":               _stats(fprs),
        "grounding_quality_dist": _stats(gqs),
        "failure_coverage": {
            "total_failures":   total_failures,
            "covered_failures": covered_failures,
            "coverage_rate":    coverage_rate,
        },
        "active_symbol_details":     active_details,
        "weakening_symbol_details":  weakening_details,
        "deprecated_symbol_details": deprecated_details,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_report(
    metrics:  dict[str, Any],
    eval_dir: Path = EVAL_DIR,
    dry_run:  bool = False,
) -> Path:
    """Save per-cycle JSON report and update rolling evolution_summary.json."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    cycle       = metrics["cycle"]
    report_path = eval_dir / f"evolution_report_cycle_{cycle}.json"

    if dry_run:
        print(f"[dry-run] Would write {report_path}")
        return report_path

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[EvolutionMetrics] Report  -> {report_path}")

    # Rolling summary
    summary_path = eval_dir / "evolution_summary.json"
    summary: list[dict] = []
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)

    summary = [e for e in summary if e.get("cycle") != cycle]
    summary.append({
        "cycle":           cycle,
        "computed_at":     metrics["computed_at"],
        "symbol_counts":   metrics["symbol_counts"],
        "total_symbols":   metrics["total_symbols"],
        "failure_coverage":metrics["failure_coverage"]["coverage_rate"],
        "active_count":    metrics["symbol_counts"].get("active",     0),
        "weakening_count": metrics["symbol_counts"].get("weakening",  0),
        "deprecated_count":metrics["symbol_counts"].get("deprecated", 0),
        "proposed_count":  metrics["symbol_counts"].get("proposed",   0),
    })
    summary.sort(key=lambda e: e["cycle"])

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[EvolutionMetrics] Summary -> {summary_path}")

    return report_path


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def print_report(metrics: dict[str, Any]) -> None:
    c = metrics["cycle"]
    print(f"\n{'='*62}")
    print(f"  Evolution Metrics  —  Cycle {c}")
    print(f"{'='*62}")

    print("\n  Symbol counts by status:")
    order = ["proposed", "experimental", "active", "weakening", "deprecated"]
    for s in order:
        if s in metrics["symbol_counts"]:
            print(f"    {s:<14}: {metrics['symbol_counts'][s]}")
    print(f"    {'TOTAL':<14}: {metrics['total_symbols']}")

    ad = metrics["accuracy_delta_dist"]
    fp = metrics["fpr_dist"]
    gq = metrics["grounding_quality_dist"]
    fc = metrics["failure_coverage"]

    print(f"\n  Accuracy delta  (n={ad['n']}): "
          f"mean={ad['mean']}  min={ad['min']}  max={ad['max']}")
    print(f"  False pos rate  (n={fp['n']}): "
          f"mean={fp['mean']}  min={fp['min']}  max={fp['max']}")
    print(f"  Grounding qual  (n={gq['n']}): "
          f"mean={gq['mean']}  min={gq['min']}  max={gq['max']}")

    print(f"\n  Failure coverage by Active symbols:")
    print(f"    {fc['covered_failures']:>5} / {fc['total_failures']} failures covered  "
          f"({fc['coverage_rate']*100:.1f}%)")

    if metrics["active_symbol_details"]:
        print(f"\n  Active symbols:")
        hdr = f"    {'ID':<8} {'acc_delta':>10} {'fpr':>8} {'retrain_d':>10} {'gq':>6} {'cvg':>5}"
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for d in metrics["active_symbol_details"]:
            ad_s = f"{d['accuracy_delta_noretrain']:+.4f}" if d["accuracy_delta_noretrain"] is not None else "     n/a"
            fp_s = f"{d['false_positive_rate']:.4f}"       if d["false_positive_rate"]       is not None else "    n/a"
            rt_s = f"{d['retrain_delta']:+.4f}"            if d["retrain_delta"]              is not None else "     n/a"
            gq_s = f"{d['grounding_quality']:.4f}"         if d["grounding_quality"]          is not None else "   n/a"
            print(f"    {d['symbol_id']:<8} {ad_s:>10} {fp_s:>8} {rt_s:>10} {gq_s:>6} {d.get('coverage', 0):>5}")

    if metrics["weakening_symbol_details"]:
        print(f"\n  Weakening symbols: "
              + ", ".join(d["symbol_id"] for d in metrics["weakening_symbol_details"]))
    if metrics["deprecated_symbol_details"]:
        print(f"  Deprecated symbols: "
              + ", ".join(d["symbol_id"] for d in metrics["deprecated_symbol_details"]))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 13: Compute and save per-cycle evolution metrics"
    )
    p.add_argument("--cycle",          type=int,  default=None,
                   help="Override cycle number (default: current_cycle from registry)")
    p.add_argument("--registry",       type=Path, default=REGISTRY_PATH)
    p.add_argument("--failure-summary",type=Path, default=FAILURE_SUMMARY)
    p.add_argument("--dry-run",        action="store_true",
                   help="Print metrics without writing files")
    args = p.parse_args()

    metrics = compute_metrics(
        registry_path        = args.registry,
        failure_summary_path = args.failure_summary,
        cycle                = args.cycle,
    )
    print_report(metrics)
    save_report(metrics, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
