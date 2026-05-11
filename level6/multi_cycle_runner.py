# level6/multi_cycle_runner.py
"""
Task 17: Multi-Cycle Evolution Runner.

Demonstrates the full symbolic lifecycle:
  Proposed -> Experimental -> Active -> Weakening -> Deprecated -> GC

Runs N evolution cycles sequentially, printing a lifecycle state table after
each one so you can observe every transition.

Two simulation modes:
  --simulate-weakening  : After Active symbols have been stable for
                          ``--stable-cycles`` cycles, inject a low acc_delta
                          (0.005) to trigger the weakening counter.
  (default)             : Pure-data-driven; weakening only fires if the
                          model genuinely stops improving on re-clustered
                          failures (unlikely for a fixed seed dataset).

Usage
-----
    # 3 cycles, skip fine-tuning (fast demo)
    python -m level6.multi_cycle_runner --cycles 3 --skip-retrain

    # 5 cycles with fine-tuning + weakening simulation after 1 stable cycle
    python -m level6.multi_cycle_runner --cycles 5 --simulate-weakening --stable-cycles 1

    # Print current registry state without running any cycles
    python -m level6.multi_cycle_runner --status-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from level6.evolution_engine import run_cycle               # noqa: E402
from level6.symbol_registry  import SymbolRegistry          # noqa: E402

DEFAULT_CHECKPOINT    = REPO_ROOT / "level5" / "saved_models" / "exp_b_l5_main" / "best_model.pt"
DEFAULT_SEED_CSV      = REPO_ROOT / "level6" / "data" / "level6_seed.csv"
DEFAULT_REGISTRY      = REPO_ROOT / "level6" / "data" / "symbol_registry.json"
DEFAULT_RULE_BASE     = REPO_ROOT / "level5" / "data" / "rule_base.json"
DEFAULT_CANDIDATE_DIR = REPO_ROOT / "level6" / "data" / "candidate_rules"
DEFAULT_FAILURE_SET   = REPO_ROOT / "level6" / "data" / "failure_set.jsonl"
DEFAULT_FULL_INF      = REPO_ROOT / "level6" / "data" / "full_inference.jsonl"
DEFAULT_CLUSTERS      = REPO_ROOT / "level6" / "data" / "clusters.json"
DEFAULT_REFINEMENTS   = REPO_ROOT / "level6" / "data" / "antecedent_refinements.json"
DEFAULT_REPORTS_DIR   = REPO_ROOT / "level6" / "data" / "validation_reports"

# Drift injection constants
WEAKENING_ACC_DELTA   = 0.005   # below min_retrain_delta (0.01) and min_acc_delta (0.03)


# ---------------------------------------------------------------------------
# Lifecycle table printer
# ---------------------------------------------------------------------------

STATUS_ORDER = ["proposed", "experimental", "active", "weakening", "deprecated"]
STATUS_COLOUR = {
    "proposed":     "",
    "experimental": "",
    "active":       "",
    "weakening":    "",
    "deprecated":   "",
}


def _print_lifecycle_table(registry: SymbolRegistry, cycle_label: str) -> None:
    """Print a compact per-symbol lifecycle state table."""
    symbols = registry._data["symbols"]
    if not symbols:
        print("  (registry empty)")
        return

    hdr = f"{'Symbol':<10} {'Status':<14} {'acc_delta':>9} {'fpr':>6} {'retrain_d':>10} {'weak_cyc':>8} {'dep_age':>7} {'born':>5}"
    sep = "-" * len(hdr)
    print(f"\n  === Lifecycle State — {cycle_label} ===")
    print(f"  {hdr}")
    print(f"  {sep}")
    current_cycle = registry.current_cycle
    for sid, sym in sorted(symbols.items()):
        raw_status   = sym.get("status", "?")
        # Use .value for StrEnum so Python 3.11+ shows "active" not "SymbolStatus.ACTIVE"
        status       = getattr(raw_status, "value", str(raw_status))
        acc_delta    = sym.get("accuracy_delta_noretrain")
        fpr          = sym.get("false_positive_rate")
        retrain_d    = sym.get("retrain_delta_over_noretrain")
        weak_cyc     = sym.get("consecutive_weak_cycles", 0)
        dep_cycle    = sym.get("deprecated_cycle")
        dep_age      = (current_cycle - dep_cycle) if dep_cycle is not None else 0
        born         = sym.get("born_cycle", "?")

        acc_str      = f"{acc_delta:+.4f}" if acc_delta is not None else "    n/a"
        fpr_str      = f"{fpr:.4f}"        if fpr      is not None else "   n/a"
        ret_str      = f"{retrain_d:+.4f}" if retrain_d is not None else "      n/a"

        print(f"  {sid:<10} {status:<14} {acc_str:>9} {fpr_str:>6} {ret_str:>10} {weak_cyc:>8} {dep_age:>7} {born:>5}")
    print()


# ---------------------------------------------------------------------------
# Drift injection — force Active symbols into weakening territory
# ---------------------------------------------------------------------------

def _inject_drift(registry: SymbolRegistry, stable_cycle: int, current_cycle: int) -> list[str]:
    """
    For every Active symbol that was promoted to Active at or before
    ``stable_cycle``, set accuracy_delta_noretrain to WEAKENING_ACC_DELTA
    so tick_cycle will increment consecutive_weak_cycles and eventually
    transition them to Weakening then Deprecated.

    Returns list of symbol_ids injected.
    """
    injected: list[str] = []
    for sid, sym in registry._data["symbols"].items():
        if sym.get("status") != "active":
            continue
        promoted_at = sym.get("last_evaluated_cycle", current_cycle)
        if promoted_at <= stable_cycle:
            sym["accuracy_delta_noretrain"] = WEAKENING_ACC_DELTA
            injected.append(sid)
            print(f"  [drift-inject] {sid}: acc_delta overridden to {WEAKENING_ACC_DELTA} "
                  f"(simulating concept drift)")
    return injected


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_multi_cycle(
    num_cycles: int,
    checkpoint_path: Path,
    seed_csv: Path,
    registry_path: Path,
    rule_base_path: Path,
    candidate_dir: Path,
    failure_set_path: Path,
    full_inference_path: Path,
    clusters_path: Path,
    refinements_path: Path,
    reports_dir: Path,
    skip_retrain: bool,
    ft_epochs: int,
    ft_lr: float,
    simulate_weakening: bool,
    stable_cycles: int,
) -> list[dict]:
    """
    Run ``num_cycles`` evolution cycles sequentially.
    Returns the list of per-cycle report dicts.
    """
    all_reports: list[dict] = []

    # Print initial state
    registry = SymbolRegistry(registry_path)
    starting_cycle = registry.current_cycle
    print(f"\n{'='*70}")
    print(f"  Multi-Cycle Runner   starting at registry cycle={starting_cycle}")
    print(f"  Planned cycles: {num_cycles}   skip_retrain={skip_retrain}")
    print(f"  Simulate weakening : {simulate_weakening}"
          + (f"  (after {stable_cycles} stable cycle(s))" if simulate_weakening else ""))
    print(f"{'='*70}")
    _print_lifecycle_table(registry, f"BEFORE cycle 1 (registry cycle={starting_cycle})")

    for i in range(num_cycles):
        cycle_number = starting_cycle + i + 1

        # --- optional: inject drift before tick so tick picks it up ---
        if simulate_weakening:
            registry = SymbolRegistry(registry_path)
            # Active symbols stable for >= stable_cycles since promotion
            drift_cutoff = starting_cycle + stable_cycles
            injected = _inject_drift(registry, drift_cutoff, registry.current_cycle)
            if injected:
                registry.save()
                print(f"  [drift] Injected for: {injected}")

        # --- run one full cycle ---
        report = run_cycle(
            cycle_number       = cycle_number,
            checkpoint_path    = checkpoint_path,
            seed_csv           = seed_csv,
            registry_path      = registry_path,
            rule_base_path     = rule_base_path,
            candidate_dir      = candidate_dir,
            failure_set_path   = failure_set_path,
            full_inference_path= full_inference_path,
            clusters_path      = clusters_path,
            refinements_path   = refinements_path,
            reports_dir        = reports_dir,
            skip_retrain       = skip_retrain,
            ft_epochs          = ft_epochs,
            ft_lr              = ft_lr,
        )
        all_reports.append(report)

        # --- print post-cycle table ---
        registry = SymbolRegistry(registry_path)
        _print_lifecycle_table(registry, f"AFTER cycle {cycle_number} (registry cycle={registry.current_cycle})")

        _print_cycle_summary(report)

    # --- final summary ---
    _print_final_summary(all_reports)
    return all_reports


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------

def _print_cycle_summary(report: dict) -> None:
    c = report["cycle_number"]
    print(f"  --- Cycle {c} transitions ---")
    for sid in report.get("promoted_to_exp",    []): print(f"      {sid}: Proposed    -> Experimental")
    for sid in report.get("promoted_to_active", []): print(f"      {sid}: Experimental -> Active")
    for sid in report.get("weakened",           []): print(f"      {sid}: Active       -> Weakening")
    for sid in report.get("deprecated",         []): print(f"      {sid}: Weakening    -> Deprecated")
    for sid in report.get("gc_removed",         []): print(f"      {sid}: Deprecated   -> REMOVED")
    if not any(report.get(k) for k in ("promoted_to_exp", "promoted_to_active",
                                        "weakened", "deprecated", "gc_removed")):
        print("      (no transitions this cycle)")
    print()


def _print_final_summary(reports: list[dict]) -> None:
    print(f"\n{'='*70}")
    print(f"  Multi-Cycle Runner — Final Summary ({len(reports)} cycles)")
    print(f"{'='*70}")

    born_total         = sum(len(r.get("new_symbols",       [])) for r in reports)
    promoted_exp_total = sum(len(r.get("promoted_to_exp",   [])) for r in reports)
    active_total       = sum(len(r.get("promoted_to_active",[])) for r in reports)
    weakened_total     = sum(len(r.get("weakened",          [])) for r in reports)
    deprecated_total   = sum(len(r.get("deprecated",        [])) for r in reports)
    gc_total           = sum(len(r.get("gc_removed",        [])) for r in reports)

    print(f"  Symbols born           : {born_total}")
    print(f"  Proposed -> Experimental: {promoted_exp_total}")
    print(f"  Experimental -> Active  : {active_total}")
    print(f"  Active -> Weakening     : {weakened_total}")
    print(f"  Weakening -> Deprecated : {deprecated_total}")
    print(f"  GC removed              : {gc_total}")
    print()
    full_lifecycle = weakened_total > 0 and deprecated_total > 0
    partial_lifecycle = weakened_total > 0 or deprecated_total > 0 or gc_total > 0
    if full_lifecycle:
        lifecycle_label = "YES — Active -> Weakening -> Deprecated demonstrated"
    elif partial_lifecycle:
        lifecycle_label = "PARTIAL — some transitions observed (run more cycles)"
    else:
        lifecycle_label = "NONE — add --simulate-weakening or run more cycles"
    print(f"  Lifecycle demonstrated  : {lifecycle_label}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Status-only printer (no cycles)
# ---------------------------------------------------------------------------

def print_status(registry_path: Path) -> None:
    registry = SymbolRegistry(registry_path)
    _print_lifecycle_table(registry, f"current (cycle={registry.current_cycle})")
    counts = {}
    for sym in registry._data["symbols"].values():
        s = sym.get("status", "?")
        counts[s] = counts.get(s, 0) + 1
    print("  Symbol counts by status:")
    for s in STATUS_ORDER:
        if s in counts:
            print(f"    {s:<14}: {counts[s]}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 17: Multi-Cycle Evolution Runner — full lifecycle demo"
    )
    p.add_argument("--cycles",            type=int,   default=3,
                   help="Number of evolution cycles to run (default: 3)")
    p.add_argument("--checkpoint",        type=Path,  default=DEFAULT_CHECKPOINT)
    p.add_argument("--data",              type=Path,  default=DEFAULT_SEED_CSV)
    p.add_argument("--registry",          type=Path,  default=DEFAULT_REGISTRY)
    p.add_argument("--rule-base",         type=Path,  default=DEFAULT_RULE_BASE)
    p.add_argument("--candidate-dir",     type=Path,  default=DEFAULT_CANDIDATE_DIR)
    p.add_argument("--failure-set",       type=Path,  default=DEFAULT_FAILURE_SET)
    p.add_argument("--full-inference",    type=Path,  default=DEFAULT_FULL_INF)
    p.add_argument("--clusters",          type=Path,  default=DEFAULT_CLUSTERS)
    p.add_argument("--refinements",       type=Path,  default=DEFAULT_REFINEMENTS)
    p.add_argument("--reports-dir",       type=Path,  default=DEFAULT_REPORTS_DIR)
    p.add_argument("--skip-retrain",      action="store_true",
                   help="Skip Task 11B fine-tuning (faster, Experimental->Active gate disabled)")
    p.add_argument("--ft-epochs",         type=int,   default=5)
    p.add_argument("--ft-lr",             type=float, default=1e-4)
    p.add_argument("--simulate-weakening",action="store_true",
                   help="Inject low acc_delta for Active symbols to trigger Weakening transitions")
    p.add_argument("--stable-cycles",     type=int,   default=1,
                   help="Cycles an Active symbol must exist before drift is injected (default: 1)")
    p.add_argument("--status-only",       action="store_true",
                   help="Print current registry state and exit without running cycles")
    args = p.parse_args()

    if args.status_only:
        print_status(args.registry)
        return

    run_multi_cycle(
        num_cycles         = args.cycles,
        checkpoint_path    = args.checkpoint,
        seed_csv           = args.data,
        registry_path      = args.registry,
        rule_base_path     = args.rule_base,
        candidate_dir      = args.candidate_dir,
        failure_set_path   = args.failure_set,
        full_inference_path= args.full_inference,
        clusters_path      = args.clusters,
        refinements_path   = args.refinements,
        reports_dir        = args.reports_dir,
        skip_retrain       = args.skip_retrain,
        ft_epochs          = args.ft_epochs,
        ft_lr              = args.ft_lr,
        simulate_weakening = args.simulate_weakening,
        stable_cycles      = args.stable_cycles,
    )


if __name__ == "__main__":
    main()
