# level6/evolution_engine.py
"""
EvolutionEngine -- Task 12: orchestrate one full evolution cycle.

One cycle:
  1.  FailureCollector  -- run L5 inference on labeled CSV, collect failures
  2.  SymbolCluster     -- HDBSCAN on predicate_probs, birth new symbols
  3.  SymbolRegistry    -- register new symbols (skip duplicates by name)
  4.  RuleCandidateGen  -- generate candidate rule JSONs for Proposed symbols
  5.  AntecedentRefiner -- tighten antecedents from previous predicate_evolver output
                           (only if antecedent_refinements.json exists)
  6.  RuleValidator     -- no-retrain FPR/acc_delta gate (Task 11)
  7.  PredicateEvolver  -- mine discriminants for FPR-failing symbols (Task 20)
  8.  AntecedentRefiner -- second pass with fresh discriminant output
  9.  RuleValidator     -- second pass validation on refined rules
  10. RetrainValidator  -- fine-tune Experimental symbols (Task 11B)
  11. Promotion         -- promote eligible symbols (Exp->Active, Proposed->Exp)
  12. Tick + GC         -- advance cycle counter, garbage-collect Deprecated

Steps 5-9 are the discriminant refinement loop (introduced by Tasks 20-21).
If a symbol fails FPR after refinement it remains Proposed and carries over
to the next cycle.

Usage
-----
    # Run one evolution cycle
    python -m level6.evolution_engine

    # Run without fine-tuning (skip Task 11B)
    python -m level6.evolution_engine --skip-retrain

    # Use a non-default seed CSV
    python -m level6.evolution_engine --data level6/data/level6_seed.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from level6.failure_collector   import collect_failures, summarize, save_outputs  # noqa: E402
from level6.symbol_cluster      import cluster_failures, load_failures, save_clusters  # noqa: E402
from level6.symbol_registry     import SymbolRegistry                                  # noqa: E402
from level6.rule_candidate_gen  import (                                               # noqa: E402
    generate_candidates, save_candidates,
    update_registry as _apply_candidate_registry_updates,
)
from level6.antecedent_refiner  import run_refinement            # noqa: E402
from level6.rule_validator      import run_validation, save_reports  # noqa: E402
from level6.predicate_evolver   import run_evolution             # noqa: E402
from level6.retrain_validator   import run_retrain_validation    # noqa: E402
from level6.lifecycle           import SymbolStatus, can_promote_to_experimental, can_promote_to_active  # noqa: E402

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


# ---------------------------------------------------------------------------
# Promotion helpers
# ---------------------------------------------------------------------------

def _promote_eligible(registry: SymbolRegistry, cycle: int) -> dict[str, list[str]]:
    """
    Promote all symbols that meet their current lifecycle gate.

    Returns {"proposed_to_experimental": [...], "experimental_to_active": [...]}
    """
    promoted: dict[str, list[str]] = {
        "proposed_to_experimental": [],
        "experimental_to_active":   [],
    }

    for sid, sym in list(registry._data["symbols"].items()):
        status = sym.get("status")

        if status == "proposed":
            eligible, _ = can_promote_to_experimental(sym)
            if eligible:
                ok, msg = registry.promote(sid)
                if ok:
                    print(f"    [promote] {sid}: Proposed -> Experimental  ({msg})")
                    promoted["proposed_to_experimental"].append(sid)

        elif status == "experimental":
            eligible, reasons = can_promote_to_active(sym)
            if eligible:
                ok, msg = registry.promote(sid)
                if ok:
                    print(f"    [promote] {sid}: Experimental -> Active  ({msg})")
                    promoted["experimental_to_active"].append(sid)
            else:
                # Still awaiting Task 11B or retrain_delta insufficient
                pass

    return promoted


# ---------------------------------------------------------------------------
# One cycle
# ---------------------------------------------------------------------------

def run_cycle(
    cycle_number: int,
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
) -> dict:
    """
    Execute one full evolution cycle.  Returns a cycle report dict.
    """
    registry = SymbolRegistry(registry_path)

    print(f"\n{'='*70}")
    print(f"  Evolution Cycle {cycle_number}  (registry cycle={registry.current_cycle})")
    print(f"{'='*70}")

    report: dict = {
        "cycle_number":          cycle_number,
        "registry_cycle_before": registry.current_cycle,
        "new_symbols":           [],
        "promoted_to_exp":       [],
        "promoted_to_active":    [],
        "weakened":              [],
        "deprecated":            [],
        "gc_removed":            [],
    }

    device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Step 1: Failure collection
    # ------------------------------------------------------------------
    print(f"\n[Cycle {cycle_number}] Step 1: Failure collection")
    failures, all_states = collect_failures(
        checkpoint_path = checkpoint_path,
        data_csv        = seed_csv,
        device          = device,
    )
    summary = summarize(failures, all_states)
    save_outputs(failures, all_states, summary, failure_set_path.parent)
    print(f"  Failures collected: {len(failures)} / {len(all_states)} total")

    # Load as raw dicts for downstream steps
    failure_rows     = load_failures(failure_set_path)
    all_rows         = load_failures(full_inference_path)
    non_failure_rows = [r for r in all_rows if not r.get("is_failure", True)]

    # ------------------------------------------------------------------
    # Step 2: Cluster in predicate space
    # ------------------------------------------------------------------
    print(f"\n[Cycle {cycle_number}] Step 2: Symbol clustering")
    new_clusters = cluster_failures(failure_rows)
    save_clusters(new_clusters, clusters_path)
    n_clusters = sum(1 for c in new_clusters if not c.get("is_noise_cluster"))
    print(f"  Non-noise clusters: {n_clusters}")

    # ------------------------------------------------------------------
    # Step 3: Register new symbols
    # ------------------------------------------------------------------
    print(f"\n[Cycle {cycle_number}] Step 3: Symbol registration")
    registry = SymbolRegistry(registry_path)
    n_before  = len(registry._data["symbols"])
    new_sids  = registry.register_from_clusters_file(clusters_path)
    n_after   = len(registry._data["symbols"])
    print(f"  New symbols registered: {n_after - n_before}  -> {new_sids}")
    report["new_symbols"] = new_sids

    # ------------------------------------------------------------------
    # Step 4: Candidate rule generation for all Proposed symbols
    # ------------------------------------------------------------------
    print(f"\n[Cycle {cycle_number}] Step 4: Rule candidate generation")
    with open(rule_base_path, encoding="utf-8") as _f:
        existing_rule_base = json.load(_f)
    existing_rules = existing_rule_base.get("rules", [])
    candidates = generate_candidates(registry, existing_rules)
    save_candidates(candidates)
    _apply_candidate_registry_updates(registry, candidates)
    registry.save()

    # ------------------------------------------------------------------
    # Step 5: First-pass validation (Task 11)
    # ------------------------------------------------------------------
    print(f"\n[Cycle {cycle_number}] Step 5: Rule validation (pass 1)")
    registry = SymbolRegistry(registry_path)
    with open(rule_base_path, encoding="utf-8") as _f:
        existing_rule_base = json.load(_f)
    with open(clusters_path, encoding="utf-8") as _f:
        clusters_loaded = json.load(_f)
    reports_pass1 = run_validation(
        symbol_ids         = None,
        registry           = registry,
        existing_rule_base = existing_rule_base,
        candidate_dir      = candidate_dir,
        failure_rows       = failure_rows,
        non_failure_rows   = non_failure_rows,
        clusters           = clusters_loaded,
        checkpoint_path    = checkpoint_path,
        device             = device,
        dry_run            = False,
    )
    registry.save()
    save_reports(reports_pass1)

    # Reload registry to check which symbols are now FPR-failing
    registry = SymbolRegistry(registry_path)
    fpr_failing = [
        sid for sid, sym in registry._data["symbols"].items()
        if sym.get("status") == "proposed"
        and (sym.get("accuracy_delta_noretrain") or 0) >= 0.03
        and (sym.get("false_positive_rate") or 1.0) >= 0.10
    ]

    if fpr_failing:
        # ------------------------------------------------------------------
        # Step 6: Discriminant mining (Task 20) for FPR-failing symbols
        # ------------------------------------------------------------------
        print(f"\n[Cycle {cycle_number}] Step 6: Discriminant mining for {fpr_failing}")
        run_evolution(
            symbol_ids         = fpr_failing,
            checkpoint_path    = checkpoint_path,
            registry_path      = registry_path,
            rule_base_path     = rule_base_path,
            candidate_dir      = candidate_dir,
            failure_set_path   = failure_set_path,
            full_inference_path= full_inference_path,
            clusters_path      = clusters_path,
            report_dir         = reports_dir,
            top_k              = 4,
        )

        # ------------------------------------------------------------------
        # Step 7: Antecedent refinement (Task 21)
        # ------------------------------------------------------------------
        print(f"\n[Cycle {cycle_number}] Step 7: Antecedent refinement")
        if refinements_path.exists():
            run_refinement(
                refinements_path = refinements_path,
                candidate_dir    = candidate_dir,
                registry_path    = registry_path,
                symbol_ids       = fpr_failing,
                dry_run          = False,
            )

            # ------------------------------------------------------------------
            # Step 8: Second-pass validation with refined antecedents
            # ------------------------------------------------------------------
            print(f"\n[Cycle {cycle_number}] Step 8: Rule validation (pass 2 -- refined rules)")
            registry = SymbolRegistry(registry_path)
            with open(rule_base_path, encoding="utf-8") as _f:
                existing_rule_base = json.load(_f)
            reports_pass2 = run_validation(
                symbol_ids         = fpr_failing,
                registry           = registry,
                existing_rule_base = existing_rule_base,
                candidate_dir      = candidate_dir,
                failure_rows       = failure_rows,
                non_failure_rows   = non_failure_rows,
                clusters           = clusters_loaded,
                checkpoint_path    = checkpoint_path,
                device             = device,
                dry_run            = False,
            )
            registry.save()
            save_reports(reports_pass2)

    # ------------------------------------------------------------------
    # Step 9: Task 11B -- fine-tune Experimental symbols
    # ------------------------------------------------------------------
    if not skip_retrain:
        print(f"\n[Cycle {cycle_number}] Step 9: Retrain validation (Task 11B)")
        run_retrain_validation(
            symbol_ids     = None,
            checkpoint_path= checkpoint_path,
            registry_path  = registry_path,
            rule_base_path = rule_base_path,
            candidate_dir  = candidate_dir,
            seed_csv       = seed_csv,
            epochs         = ft_epochs,
            lr             = ft_lr,
            batch_size     = 32,
            dry_run        = False,
        )
    else:
        print(f"\n[Cycle {cycle_number}] Step 9: Retrain validation -- SKIPPED (--skip-retrain)")

    # ------------------------------------------------------------------
    # Step 10: Promote eligible symbols
    # ------------------------------------------------------------------
    print(f"\n[Cycle {cycle_number}] Step 10: Lifecycle promotion")
    registry = SymbolRegistry(registry_path)
    promoted = _promote_eligible(registry, cycle_number)
    report["promoted_to_exp"]    = promoted["proposed_to_experimental"]
    report["promoted_to_active"] = promoted["experimental_to_active"]
    registry.save()

    # ------------------------------------------------------------------
    # Step 11: Tick cycle counter + auto-transitions (Active->Weakening->Deprecated)
    # ------------------------------------------------------------------
    print(f"\n[Cycle {cycle_number}] Step 11: Tick cycle + auto-transitions")
    registry = SymbolRegistry(registry_path)
    registry.tick_cycle()

    # Collect what happened — detect transitions that just occurred
    new_cycle = registry.current_cycle
    for sid, sym in registry._data["symbols"].items():
        status = sym.get("status")
        # Newly weakened: status changed to Weakening this tick (weakening_count==1)
        if str(status) in ("weakening", "SymbolStatus.WEAKENING") and sym.get("weakening_count", 0) == 1:
            report["weakened"].append(sid)
            print(f"    [weaken]    {sid}: Active -> Weakening")
        # Newly deprecated: deprecated_cycle set to this cycle
        if str(status) in ("deprecated", "SymbolStatus.DEPRECATED"):
            dep_cycle = sym.get("deprecated_cycle", -1)
            if dep_cycle == new_cycle:
                report["deprecated"].append(sid)
                print(f"    [deprecate] {sid}: Weakening -> Deprecated")

    # ------------------------------------------------------------------
    # Step 12: Garbage collection
    # ------------------------------------------------------------------
    print(f"\n[Cycle {cycle_number}] Step 12: Garbage collection")
    gc_result = registry.garbage_collect()
    report["gc_removed"] = gc_result if isinstance(gc_result, list) else gc_result.get("removed", [])
    if report["gc_removed"]:
        print(f"    [GC] Removed {len(report['gc_removed'])} deprecated symbol(s): {report['gc_removed']}")
    else:
        print(f"    [GC] No symbols removed")

    registry.save()   # persist tick + GC in one final write
    report["registry_cycle_after"] = registry.current_cycle
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EvolutionEngine -- Task 12: one full evolution cycle"
    )
    parser.add_argument("--checkpoint",    type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data",          type=Path, default=DEFAULT_SEED_CSV)
    parser.add_argument("--registry",      type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--rule-base",     type=Path, default=DEFAULT_RULE_BASE)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--failure-set",   type=Path, default=DEFAULT_FAILURE_SET)
    parser.add_argument("--full-inference",type=Path, default=DEFAULT_FULL_INF)
    parser.add_argument("--clusters",      type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--refinements",   type=Path, default=DEFAULT_REFINEMENTS)
    parser.add_argument("--reports-dir",   type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--skip-retrain",  action="store_true",
                        help="Skip Task 11B fine-tuning (faster cycle, no Exp->Active)")
    parser.add_argument("--ft-epochs",     type=int,   default=5)
    parser.add_argument("--ft-lr",         type=float, default=1e-4)
    args = parser.parse_args()

    registry = SymbolRegistry(args.registry)
    cycle_number = registry.current_cycle + 1

    report = run_cycle(
        cycle_number       = cycle_number,
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
    )

    print(f"\n{'='*70}")
    print(f"  Evolution Cycle {cycle_number} Complete")
    print(f"{'='*70}")
    print(f"  New symbols born       : {len(report['new_symbols'])}  {report['new_symbols']}")
    print(f"  Promoted -> Experimental: {report['promoted_to_exp']}")
    print(f"  Promoted -> Active      : {report['promoted_to_active']}")
    print(f"  Weakened               : {report['weakened']}")
    print(f"  Deprecated             : {report['deprecated']}")
    print(f"  GC removed             : {report['gc_removed']}")
    print(f"  Registry cycle         : {report['registry_cycle_before']} -> {report['registry_cycle_after']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
