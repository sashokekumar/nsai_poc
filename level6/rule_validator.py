# level6/rule_validator.py
"""
RuleValidator -- Task 11: no-retrain candidate rule injection and evaluation.

For each candidate rule in level6/data/candidate_rules/, this module:

    1. Loads the Level 5 checkpoint (weights frozen -- no retraining).
    2. Injects the candidate rule into a temporary rule_base.json that
       extends the existing 4 rules with the candidate as rule N+1.
    3. Rebuilds the RuleCompiler with the extended rule base.
    4. Runs the model forward pass (predicate_head unchanged, rule_layer
       re-initialised with the new rule at rule_strength_init) on:
         a. The symbol's failure cluster (from failure_set.jsonl, filtered
            by symbol membership via cluster assignment stored in clusters.json)
         b. The non-failure set (all_states from full_inference.jsonl where
            is_failure=False) -- used to measure the false positive rate.
    5. Computes:
         accuracy_delta_noretrain  = acc_with_rule - acc_without_rule
                                     (on the failure cluster only)
         false_positive_rate       = fraction of non-failure examples where
                                     the new rule fires AND changes the
                                     previously-correct prediction to wrong
    6. Updates the SymbolRegistry with both metrics via update_validation().
    7. Checks lifecycle promotion criteria (can_promote_to_experimental).
       Does NOT auto-promote -- prints eligibility and leaves promotion to
       the EvolutionEngine (Task 12) or CLI --promote flag.
    8. Saves a validation report to level6/data/validation_reports/report_<sid>.json

Key constraint
--------------
No model retraining.  The neural trunk weights (encoder, shared, predicate_head)
are frozen at checkpoint values.  Only the rule_layer is rebuilt with the
injected rule, and the new rule's rule_strength is fixed at rule_strength_init
(not fine-tuned).  This isolates the symbolic contribution of the new rule.

S_003 note
----------
S_003 (is_unknown AND NOT is_incident -> summarization) has the highest risk
of false positives -- is_unknown fires on generic/non-SRE utterances and
redirecting them to summarization could pull genuine out_of_scope examples
into the wrong bucket.  The FPR check (< 0.10) is the critical gate.

Usage
-----
    # Validate all Proposed symbols with candidate rules
    python -m level6.rule_validator

    # Validate a specific symbol
    python -m level6.rule_validator --symbol S_001

    # Dry-run: print report but do not update registry
    python -m level6.rule_validator --dry-run

    # Explicit paths
    python -m level6.rule_validator \\
        --checkpoint level5/saved_models/exp_b_l5_main/best_model.pt \\
        --failure-set level6/data/failure_set.jsonl \\
        --full-inference level6/data/full_inference.jsonl \\
        --clusters level6/data/clusters.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from level5.model.level5_model import Level5IntentModel       # noqa: E402
from level5.model.dataset import INTENT_LABELS, PREDICATE_COLS  # noqa: E402
from level6.symbol_registry import SymbolRegistry              # noqa: E402
from level6.lifecycle import (                                  # noqa: E402
    SymbolStatus, can_promote_to_experimental,
    PROPOSED_TO_EXPERIMENTAL,
)

DEFAULT_CHECKPOINT     = REPO_ROOT / "level5" / "saved_models" / "exp_b_l5_main" / "best_model.pt"
DEFAULT_REGISTRY       = REPO_ROOT / "level6" / "data" / "symbol_registry.json"
DEFAULT_RULE_BASE      = REPO_ROOT / "level5" / "data" / "rule_base.json"
DEFAULT_CANDIDATE_DIR  = REPO_ROOT / "level6" / "data" / "candidate_rules"
DEFAULT_FAILURE_SET    = REPO_ROOT / "level6" / "data" / "failure_set.jsonl"
DEFAULT_FULL_INFERENCE = REPO_ROOT / "level6" / "data" / "full_inference.jsonl"
DEFAULT_CLUSTERS       = REPO_ROOT / "level6" / "data" / "clusters.json"
REPORT_DIR             = REPO_ROOT / "level6" / "data" / "validation_reports"


# ---------------------------------------------------------------------------
# Model loading (same strict=False pattern as failure_collector)
# ---------------------------------------------------------------------------

def _load_model_with_rule_base(
    checkpoint_path: Path,
    rule_base_path: str,
    device: torch.device,
) -> Level5IntentModel:
    """
    Load L5 checkpoint with a custom rule_base_path (for injected rules).

    When the rule_base has N+1 rules but the checkpoint has N-rule tensors,
    we cannot call load_state_dict directly.  Instead we:
      1. Load the baseline model (original N rules) with strict=False.
      2. Construct the target model with the new rule_base_path (N+1 rules).
      3. Copy all weights EXCEPT rule_layer (encoder + trunk + predicate_head).
      4. Leave rule_layer at its randomly-initialised default, then set the
         first N rule_strength_logits to the checkpoint values and the N+1th
         to logit(rule_strength_init) from the candidate rule JSON.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build baseline (N rules) just to extract trunk weights
    baseline = Level5IntentModel(
        hard_rules=ckpt.get("hard_rules", False),
    )
    baseline.load_state_dict(ckpt["state_dict"], strict=False)

    # Build target model with (potentially different-sized) rule_base
    target = Level5IntentModel(
        rule_base_path=rule_base_path,
        hard_rules=ckpt.get("hard_rules", False),
    )

    # Copy encoder + trunk + predicate_head from baseline
    trunk_keys = [k for k in ckpt["state_dict"] if not k.startswith("rule_layer.")]
    target_sd  = target.state_dict()
    for k in trunk_keys:
        if k in target_sd:
            target_sd[k] = ckpt["state_dict"][k]
    target.load_state_dict(target_sd, strict=False)

    # Copy rule_layer weights for the original N rules; new rules keep init value
    n_orig = baseline.rule_layer.rule_strength_logits.shape[0]
    n_new  = target.rule_layer.rule_strength_logits.shape[0]
    if n_new >= n_orig:
        with torch.no_grad():
            target.rule_layer.rule_strength_logits[:n_orig] = (
                baseline.rule_layer.rule_strength_logits
            )
            target.rule_layer.rule_to_intent[:n_orig] = (
                baseline.rule_layer.rule_to_intent
            )

    target.eval()
    target.to(device)
    return target


# ---------------------------------------------------------------------------
# Rule base injection
# ---------------------------------------------------------------------------

def _inject_rule(existing_rule_base: dict, candidate_rule: dict) -> dict:
    """
    Return a NEW rule_base dict with the candidate appended as an extra rule.

    Only the rule_base.json schema fields are copied (not _l6_* metadata).
    """
    schema_keys = {"name", "description", "antecedents", "consequent_intent",
                   "rule_strength_init"}
    clean_rule = {k: v for k, v in candidate_rule.items() if k in schema_keys}

    new_rb = {
        "_description": existing_rule_base.get("_description", ""),
        "_predicate_columns": existing_rule_base.get("_predicate_columns", []),
        "_intent_labels": existing_rule_base.get("_intent_labels", []),
        "rules": existing_rule_base["rules"] + [clean_rule],
    }
    return new_rb


# ---------------------------------------------------------------------------
# Forward pass on a pre-computed predicate_probs matrix
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_from_predicates(
    model: Level5IntentModel,
    predicate_probs: np.ndarray,   # [N, 11]
    device: torch.device,
) -> np.ndarray:
    """
    Run the rule layer only, using pre-computed predicate_probs.

    The neural trunk is bypassed -- we feed stored predicate_probs through the
    RuleCompiler directly.  This is valid because:
      - The trunk weights are unchanged between baseline and injected runs.
      - We want to isolate the effect of the new rule, not re-encode utterances.
      - predicate_probs are already stored in failure_set.jsonl / full_inference.jsonl.
    """
    preds_tensor = torch.tensor(predicate_probs, dtype=torch.float32, device=device)
    rule_acts    = model.rule_layer(preds_tensor)           # [N, n_rules]
    intent_scores = model.rule_layer.scatter_to_intents(rule_acts)  # [N, 4]
    intent_logits = torch.logit(intent_scores.clamp(1e-6, 1 - 1e-6))
    intent_preds  = intent_logits.argmax(dim=1).cpu().numpy()
    return intent_preds


# ---------------------------------------------------------------------------
# Cluster membership matching
# ---------------------------------------------------------------------------

def _match_cluster_members(
    failure_rows: list[dict],
    cluster: dict,
) -> list[dict]:
    """
    Return the subset of failure_rows that belong to the given cluster.
    Matching is done by comparing predicate_probs cosine distance to centroid.
    Members within distance 0.15 (cosine) of the centroid are considered
    cluster members.  This avoids storing explicit cluster labels in JSONL.
    """
    centroid = np.array(
        [cluster["centroid"][col] for col in PREDICATE_COLS], dtype=np.float32
    )
    c_norm = np.linalg.norm(centroid)

    members: list[dict] = []
    for row in failure_rows:
        pp = np.array(row["predicate_probs"], dtype=np.float32)
        pp_norm = np.linalg.norm(pp)
        if c_norm < 1e-9 or pp_norm < 1e-9:
            continue
        cos_sim = float(np.dot(pp, centroid) / (pp_norm * c_norm))
        if cos_sim >= 0.85:   # tight cosine threshold for cluster membership
            members.append(row)
    return members


# ---------------------------------------------------------------------------
# Accuracy measurement helpers
# ---------------------------------------------------------------------------

def _intent_to_idx(intent_str: str) -> int:
    try:
        return INTENT_LABELS.index(intent_str)
    except ValueError:
        return -1


def _accuracy_on_rows(
    rows: list[dict],
    model: Level5IntentModel,
    device: torch.device,
) -> float:
    """Accuracy of model (via rule layer only) on a list of inference rows."""
    if not rows:
        return 0.0
    predicate_matrix = np.array([r["predicate_probs"] for r in rows], dtype=np.float32)
    gold_labels      = np.array([_intent_to_idx(r["gold_intent"]) for r in rows])
    pred_labels      = _predict_from_predicates(model, predicate_matrix, device)
    valid = gold_labels >= 0
    return float(np.mean(pred_labels[valid] == gold_labels[valid])) if valid.any() else 0.0


def _false_positive_rate(
    non_failure_rows: list[dict],
    baseline_model: Level5IntentModel,
    injected_model: Level5IntentModel,
    device: torch.device,
) -> float:
    """
    FPR = fraction of previously-correct non-failure examples that the
    injected rule flips to incorrect.

    Only previously-correct examples are included in the denominator so that
    pre-existing errors don't inflate FPR artificially.
    """
    if not non_failure_rows:
        return 0.0

    pmatrix   = np.array([r["predicate_probs"] for r in non_failure_rows], dtype=np.float32)
    gold      = np.array([_intent_to_idx(r["gold_intent"]) for r in non_failure_rows])

    base_pred = _predict_from_predicates(baseline_model, pmatrix, device)
    inj_pred  = _predict_from_predicates(injected_model, pmatrix, device)

    valid = gold >= 0
    # Only examples that baseline got RIGHT
    previously_correct = valid & (base_pred == gold)
    if not previously_correct.any():
        return 0.0

    # Among those, how many did the injected model flip to wrong?
    flipped = previously_correct & (inj_pred != gold)
    return float(np.sum(flipped) / np.sum(previously_correct))


# ---------------------------------------------------------------------------
# Single-symbol validation
# ---------------------------------------------------------------------------

def validate_symbol(
    symbol_id: str,
    registry: SymbolRegistry,
    existing_rule_base: dict,
    candidate_dir: Path,
    failure_rows: list[dict],
    non_failure_rows: list[dict],
    clusters: list[dict],
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    """
    Validate one Proposed symbol.  Returns a report dict.
    """
    sym = registry.get(symbol_id)

    # Resolve the candidate rule file.
    # Registry candidate_rule_name stores the file-stem (e.g. "R_S_001_refined").
    # Fall back to the original "R_{symbol_id}.json" if the stem file is missing.
    reg_rule_name  = sym.get("candidate_rule_name", "")
    candidate_path = candidate_dir / f"{reg_rule_name}.json" if reg_rule_name else None
    if candidate_path is None or not candidate_path.exists():
        # Prefer _refined if it exists, else original
        matches  = sorted(candidate_dir.glob(f"R_{symbol_id}*.json"))
        refined  = [m for m in matches if "_refined" in m.name]
        original = [m for m in matches if "_refined" not in m.name]
        candidate_path = (refined or original or [candidate_dir / f"R_{symbol_id}.json"])[0]

    if not candidate_path.exists():
        return {"symbol_id": symbol_id, "error": f"Candidate file not found: {candidate_path}"}

    with open(candidate_path, encoding="utf-8") as fh:
        candidate = json.load(fh)

    # Skip duplicate-rejected candidates
    if candidate.get("_l6_status") == "duplicate":
        return {
            "symbol_id": symbol_id,
            "skipped":   True,
            "reason":    f"duplicate of {candidate.get('_l6_redundant_with')}",
        }

    # Find the matching cluster by symbol name
    cluster = next(
        (c for c in clusters if c["symbol_name"] == sym["name"]),
        None,
    )
    if cluster is None:
        return {"symbol_id": symbol_id, "error": "No matching cluster found in clusters.json"}

    # Cluster members (failure set)
    cluster_members = _match_cluster_members(failure_rows, cluster)
    if not cluster_members:
        return {"symbol_id": symbol_id, "error": "No cluster members found in failure set"}

    print(f"\n  [{symbol_id}] {sym['name']}")
    print(f"      cluster members (failures) : {len(cluster_members)}")
    print(f"      non-failure rows for FPR   : {len(non_failure_rows)}")

    # Build injected rule base in a temp file
    injected_rb = _inject_rule(existing_rule_base, candidate)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(injected_rb, tmp)
        tmp_path = tmp.name

    try:
        # Load baseline model (original 4 rules)
        baseline_model = _load_model_with_rule_base(
            checkpoint_path, str(DEFAULT_RULE_BASE), device
        )
        # Load injected model (4 + 1 candidate rule)
        injected_model = _load_model_with_rule_base(
            checkpoint_path, tmp_path, device
        )

        # Accuracy on cluster members
        acc_base     = _accuracy_on_rows(cluster_members, baseline_model, device)
        acc_injected = _accuracy_on_rows(cluster_members, injected_model, device)
        acc_delta    = round(acc_injected - acc_base, 4)

        # False positive rate on non-failure set
        fpr = round(_false_positive_rate(
            non_failure_rows, baseline_model, injected_model, device
        ), 4)

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    print(f"      acc_baseline (cluster)     : {acc_base:.4f}")
    print(f"      acc_injected (cluster)     : {acc_injected:.4f}")
    print(f"      accuracy_delta_noretrain   : {acc_delta:+.4f}  "
          f"(threshold >= {PROPOSED_TO_EXPERIMENTAL['min_accuracy_delta_noretrain']})")
    print(f"      false_positive_rate        : {fpr:.4f}  "
          f"(threshold < {PROPOSED_TO_EXPERIMENTAL['max_false_positive_rate']})")

    report = {
        "symbol_id":              symbol_id,
        "symbol_name":            sym["name"],
        "candidate_rule_name":    candidate["name"],
        "dominant_confusion":     candidate.get("_l6_dominant_confusion", ""),
        "majority_gold_intent":   candidate.get("_l6_majority_gold_intent", ""),
        "cluster_size":           len(cluster_members),
        "non_failure_size":       len(non_failure_rows),
        "acc_baseline_cluster":   round(acc_base, 4),
        "acc_injected_cluster":   round(acc_injected, 4),
        "accuracy_delta_noretrain": acc_delta,
        "false_positive_rate":    fpr,
        "promotion_eligible":     None,
        "promotion_reasons":      [],
    }

    # Check promotion eligibility (updates sym first)
    sym_copy = {**sym, "accuracy_delta_noretrain": acc_delta, "false_positive_rate": fpr}
    eligible, reasons = can_promote_to_experimental(sym_copy)
    report["promotion_eligible"] = eligible
    report["promotion_reasons"]  = reasons

    if eligible:
        print(f"      ELIGIBLE for Proposed -> Experimental")
    else:
        print(f"      NOT eligible: {'; '.join(reasons)}")

    return report


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def run_validation(
    symbol_ids: list[str] | None,
    registry: SymbolRegistry,
    existing_rule_base: dict,
    candidate_dir: Path,
    failure_rows: list[dict],
    non_failure_rows: list[dict],
    clusters: list[dict],
    checkpoint_path: Path,
    device: torch.device,
    dry_run: bool = False,
) -> list[dict]:
    """Validate all (or selected) Proposed symbols.  Returns list of reports."""
    proposed = registry.by_status(SymbolStatus.PROPOSED)
    if symbol_ids:
        proposed = [s for s in proposed if s["symbol_id"] in symbol_ids]

    if not proposed:
        print("[RuleValidator] No Proposed symbols to validate.")
        return []

    reports: list[dict] = []
    for sym in proposed:
        sid    = sym["symbol_id"]
        report = validate_symbol(
            sid, registry, existing_rule_base, candidate_dir,
            failure_rows, non_failure_rows, clusters, checkpoint_path, device,
        )
        reports.append(report)

        if dry_run or report.get("error") or report.get("skipped"):
            continue

        # Write metrics back to registry (but do NOT promote)
        registry.update_validation(
            sid,
            accuracy_delta_noretrain=report["accuracy_delta_noretrain"],
            false_positive_rate=report["false_positive_rate"],
        )

    return reports


# ---------------------------------------------------------------------------
# Save reports
# ---------------------------------------------------------------------------

def save_reports(reports: list[dict], dry_run: bool = False):
    if dry_run:
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for report in reports:
        sid = report.get("symbol_id", "unknown")
        out = REPORT_DIR / f"report_{sid}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  [report] {out}")


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_validation_summary(reports: list[dict]):
    print()
    print("=" * 70)
    print("  RuleValidator Summary")
    print("=" * 70)
    criteria = PROPOSED_TO_EXPERIMENTAL

    for r in reports:
        sid = r.get("symbol_id", "?")
        if r.get("error"):
            print(f"  {sid}  ERROR: {r['error']}")
            continue
        if r.get("skipped"):
            print(f"  {sid}  SKIPPED: {r['reason']}")
            continue

        eligible = r.get("promotion_eligible", False)
        tag      = "ELIGIBLE" if eligible else "NOT ELIGIBLE"
        delta    = r.get("accuracy_delta_noretrain", 0.0)
        fpr      = r.get("false_positive_rate", 1.0)
        delta_ok = delta >= criteria["min_accuracy_delta_noretrain"]
        fpr_ok   = fpr   <  criteria["max_false_positive_rate"]

        print(f"  {sid}  [{tag}]")
        print(f"      {r.get('symbol_name', '')}")
        print(f"      acc_delta = {delta:+.4f}  "
              f"{'OK' if delta_ok else 'FAIL'}  "
              f"(threshold >= {criteria['min_accuracy_delta_noretrain']})")
        print(f"      fpr       = {fpr:.4f}   "
              f"{'OK' if fpr_ok else 'FAIL'}  "
              f"(threshold < {criteria['max_false_positive_rate']})")
        if not eligible:
            for reason in r.get("promotion_reasons", []):
                print(f"      ! {reason}")
        print()
    print("=" * 70)
    print()
    print("  Symbols remain Proposed. To promote an eligible symbol:")
    print("    python -m level6.symbol_registry --promote <SYMBOL_ID>")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate candidate rules via no-retrain injection.")
    p.add_argument("--symbol",         nargs="*", metavar="SYMBOL_ID",
                   help="Validate specific symbols (default: all Proposed)")
    p.add_argument("--checkpoint",     type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--registry",       type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--rule-base",      type=Path, default=DEFAULT_RULE_BASE)
    p.add_argument("--candidate-dir",  type=Path, default=DEFAULT_CANDIDATE_DIR)
    p.add_argument("--failure-set",    type=Path, default=DEFAULT_FAILURE_SET)
    p.add_argument("--full-inference", type=Path, default=DEFAULT_FULL_INFERENCE)
    p.add_argument("--clusters",       type=Path, default=DEFAULT_CLUSTERS)
    p.add_argument("--dry-run",        action="store_true",
                   help="Print reports but do not update registry or write files.")
    return p.parse_args()


def main():
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[RuleValidator] Checkpoint     : {args.checkpoint}")
    print(f"[RuleValidator] Device         : {device}")

    # Load data
    print(f"[RuleValidator] Loading failure set     : {args.failure_set}")
    all_rows      = _load_jsonl(args.failure_set)
    failure_rows  = [r for r in all_rows if r.get("is_failure")]

    print(f"[RuleValidator] Loading full inference  : {args.full_inference}")
    full_rows     = _load_jsonl(args.full_inference)
    non_failure_rows = [r for r in full_rows if not r.get("is_failure")]

    print(f"[RuleValidator] Failure rows     : {len(failure_rows)}")
    print(f"[RuleValidator] Non-failure rows : {len(non_failure_rows)}")

    with open(args.clusters, encoding="utf-8") as fh:
        clusters = json.load(fh)

    with open(args.rule_base, encoding="utf-8") as fh:
        existing_rule_base = json.load(fh)

    registry = SymbolRegistry(args.registry)

    reports = run_validation(
        symbol_ids=args.symbol,
        registry=registry,
        existing_rule_base=existing_rule_base,
        candidate_dir=args.candidate_dir,
        failure_rows=failure_rows,
        non_failure_rows=non_failure_rows,
        clusters=clusters,
        checkpoint_path=args.checkpoint,
        device=device,
        dry_run=args.dry_run,
    )

    print_validation_summary(reports)

    save_reports(reports, dry_run=args.dry_run)

    if not args.dry_run and reports:
        registry.save()
        print(f"[RuleValidator] Registry updated -> {args.registry}")


if __name__ == "__main__":
    main()
