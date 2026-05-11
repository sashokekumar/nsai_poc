# level6/failure_collector.py
"""
FailureCollector — extracts failure ReasoningStates from a Level 5 checkpoint.

Runs the Level 5 model over a labeled CSV and identifies two categories of
failure, both of which signal a symbolic gap in the current rule vocabulary:

    Misclassification : predicted_intent ≠ gold_intent
    Low-confidence    : max(intent_dist) < LOW_CONFIDENCE_THRESHOLD (0.65)
                        even when the prediction is correct — the symbolic
                        rule layer is uncertain, which reveals coverage gaps

The trunk_repr [256] is captured via a PyTorch forward hook on model.shared
so Level 5 code is not modified.

Output
------
Writes two JSONL files to level6/data/:

    failure_set.jsonl     — only ReasoningStates where is_failure=True
    full_inference.jsonl  — all ReasoningStates (useful for non-failure FPR checks)

Each line is the JSON output of ReasoningState.to_dict().

Summary statistics are printed and saved to level6/data/failure_summary.json.

Usage (from repo root)
----------------------
    # Use default checkpoint (exp_b_l5_main) and default seed CSV
    python -m level6.failure_collector

    # Explicit paths
    python -m level6.failure_collector \\
        --checkpoint level5/saved_models/exp_b_l5_main/best_model.pt \\
        --data level6/data/level6_seed.csv \\
        --batch-size 64

    # Dry-run: print summary only, do not write files
    python -m level6.failure_collector --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from level5.model.level5_model import Level5IntentModel
from level5.model.dataset import INTENT_LABELS, PREDICATE_COLS
from level6.reasoning_state import ReasoningState, LOW_CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Checkpoint loading (Level 6 own loader — strict=False)
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str | Path, device: torch.device):
    """
    Load a Level 5 checkpoint into the current Level5IntentModel.

    Uses strict=False so that legacy checkpoints saved with the blended
    architecture (which had rule_weight_logit, intent_head, rule_score_projection)
    load cleanly into the current pure-rule architecture. The extra keys are
    silently discarded; the essential components (encoder, shared trunk,
    predicate_head, rule_layer) are loaded correctly.
    """
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Level5IntentModel(
        hard_rules=ckpt.get("hard_rules", False),
    )
    missing, unexpected = model.load_state_dict(
        ckpt["state_dict"], strict=False
    )
    if missing:
        raise RuntimeError(
            f"[FailureCollector] Missing keys in checkpoint: {missing}\n"
            "The checkpoint is missing components required by Level5IntentModel."
        )
    # unexpected keys are from the legacy blended architecture — silently accepted
    model.to(device)
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "level5" / "saved_models" / "exp_b_l5_main" / "best_model.pt"
)
DEFAULT_DATA = REPO_ROOT / "level6" / "data" / "level6_seed.csv"
OUT_DIR      = REPO_ROOT / "level6" / "data"


# ---------------------------------------------------------------------------
# Trunk hook
# ---------------------------------------------------------------------------

class _TrunkHook:
    """
    Captures the output of model.shared (the 256-dim trunk activations)
    via a PyTorch forward hook.

    Register with:
        hook = _TrunkHook()
        handle = model.shared.register_forward_hook(hook)
        # ... run forward pass ...
        trunk = hook.output   # [B, 256]
        handle.remove()
    """
    def __init__(self):
        self.output: torch.Tensor = None

    def __call__(self, module, input, output):
        self.output = output.detach().cpu()


# ---------------------------------------------------------------------------
# Core inference + failure extraction
# ---------------------------------------------------------------------------

def collect_failures(
    checkpoint_path: str | Path,
    data_csv: str | Path,
    batch_size: int = 64,
    device: torch.device = None,
) -> tuple[list[ReasoningState], list[ReasoningState]]:
    """
    Run Level 5 inference over a labeled CSV and extract ReasoningStates.

    Args:
        checkpoint_path : path to best_model.pt
        data_csv        : labeled CSV with 'utterance' and 'intent' columns
        batch_size      : inference batch size
        device          : torch device (auto-detected if None)

    Returns:
        (failures, all_states)
            failures   — ReasoningStates where is_failure=True
            all_states — all ReasoningStates (failures + correct predictions)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, meta = _load_model(str(checkpoint_path), device)
    model.eval()
    print(
        f"[FailureCollector] Checkpoint   : {Path(checkpoint_path).name}\n"
        f"[FailureCollector] Run name     : {meta.get('run_name', '?')}\n"
        f"[FailureCollector] Val acc      : {meta.get('val_intent_acc', 0):.4f}\n"
        f"[FailureCollector] Rule strengths: {meta.get('rule_strengths', {})}\n"
        f"[FailureCollector] Device       : {device}"
    )

    # ------------------------------------------------------------------
    # Load data — require utterance + intent columns
    # ------------------------------------------------------------------
    df = pd.read_csv(data_csv)
    required = {"utterance", "intent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[FailureCollector] CSV missing required columns: {missing}"
        )
    bad_intents = set(df["intent"].unique()) - set(INTENT_LABELS)
    if bad_intents:
        raise ValueError(
            f"[FailureCollector] Unknown intents in CSV: {bad_intents}"
        )

    utterances   = df["utterance"].astype(str).tolist()
    gold_intents = df["intent"].astype(str).tolist()
    n_total      = len(utterances)
    print(f"[FailureCollector] Dataset rows : {n_total}")

    # ------------------------------------------------------------------
    # Register trunk hook (captures model.shared output → [B, 256])
    # ------------------------------------------------------------------
    trunk_hook   = _TrunkHook()
    hook_handle  = model.shared.register_forward_hook(trunk_hook)

    all_states: list[ReasoningState] = []

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------
    try:
        with torch.no_grad():
            for start in range(0, n_total, batch_size):
                batch_utts  = utterances[start: start + batch_size]
                batch_golds = gold_intents[start: start + batch_size]
                B = len(batch_utts)

                # Forward pass — hook fires, capturing trunk [B, 256]
                out = model.forward(batch_utts, device)

                # Tensors for this batch
                predicate_probs  = out["predicate_probs"].detach().cpu()   # [B, 11]
                rule_activations = out["rule_activations"].detach().cpu()  # [B, n_rules]
                intent_dist      = torch.softmax(
                    out["intent_logits"].detach().cpu(), dim=-1
                )                                                           # [B, 4]
                trunk_batch      = trunk_hook.output                        # [B, 256]

                for i in range(B):
                    rs = ReasoningState.from_model_output(
                        utterance       = batch_utts[i],
                        trunk_repr      = trunk_batch[i],
                        predicate_probs = predicate_probs[i],
                        rule_activations= rule_activations[i],
                        intent_dist     = intent_dist[i],
                        gold_intent     = batch_golds[i],
                    )
                    all_states.append(rs)

                if (start // batch_size) % 5 == 0:
                    done = min(start + batch_size, n_total)
                    print(f"[FailureCollector]   {done}/{n_total} processed...")
    finally:
        hook_handle.remove()

    failures = [rs for rs in all_states if rs.is_failure]
    return failures, all_states


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def summarize(
    failures: list[ReasoningState],
    all_states: list[ReasoningState],
) -> dict:
    n_total          = len(all_states)
    n_failures       = len(failures)
    n_misclass       = sum(1 for rs in failures if rs.is_misclassification)
    n_low_conf       = sum(1 for rs in failures if rs.is_low_confidence)
    n_both           = sum(
        1 for rs in failures if rs.is_misclassification and rs.is_low_confidence
    )
    failure_rate     = n_failures / n_total if n_total else 0.0
    misclass_rate    = n_misclass / n_total if n_total else 0.0

    # Intent confusion among misclassified (predicted → gold)
    confusion: dict[str, dict[str, int]] = {}
    for rs in failures:
        if not rs.is_misclassification:
            continue
        pred, gold = rs.predicted_intent, rs.gold_intent
        confusion.setdefault(pred, {})
        confusion[pred][gold] = confusion[pred].get(gold, 0) + 1

    # Predicate profile of failure set (mean predicate activations)
    if failures:
        import torch as _torch
        pred_matrix = _torch.stack([rs.predicate_probs for rs in failures])  # [F, 11]
        pred_means  = pred_matrix.mean(dim=0).tolist()
        predicate_profile = dict(zip(PREDICATE_COLS, [round(v, 4) for v in pred_means]))
    else:
        predicate_profile = {col: 0.0 for col in PREDICATE_COLS}

    # Failure breakdown by source (level5_base vs level6_seed) if available
    source_breakdown: dict[str, int] = {}
    for rs in failures:
        # We can't read source from ReasoningState directly — use predicate as proxy
        source_breakdown["total"] = n_failures

    summary = {
        "n_total":           n_total,
        "n_failures":        n_failures,
        "n_misclassification": n_misclass,
        "n_low_confidence":  n_low_conf,
        "n_both":            n_both,
        "failure_rate":      round(failure_rate, 4),
        "misclassification_rate": round(misclass_rate, 4),
        "intent_accuracy":   round(1.0 - misclass_rate, 4),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "failure_predicate_profile": predicate_profile,
        "intent_confusion_matrix":  confusion,
    }
    return summary


def print_summary(summary: dict):
    print()
    print("=" * 60)
    print("  FailureCollector Summary")
    print("=" * 60)
    print(f"  Total rows         : {summary['n_total']}")
    print(f"  Total failures     : {summary['n_failures']}  ({summary['failure_rate']:.1%})")
    print(f"  Misclassifications : {summary['n_misclassification']}  "
          f"(L6 stress dataset accuracy = {summary['intent_accuracy']:.4f})")
    print(f"  Low-confidence     : {summary['n_low_confidence']}  "
          f"(threshold = {summary['low_confidence_threshold']})")
    print(f"  Both               : {summary['n_both']}")
    print()
    print("  Failure predicate profile (mean activations):")
    for pred, val in summary["failure_predicate_profile"].items():
        bar = "#" * int(val * 20)
        print(f"    {pred:<22}  {val:.3f}  {bar}")
    print()
    if summary["intent_confusion_matrix"]:
        print("  Misclassification confusion (predicted → gold):")
        for pred_intent, golds in summary["intent_confusion_matrix"].items():
            for gold_intent, count in sorted(golds.items(), key=lambda x: -x[1]):
                print(f"    {pred_intent:>15} → {gold_intent:<15}  {count}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    failures: list[ReasoningState],
    all_states: list[ReasoningState],
    summary: dict,
    out_dir: Path,
    dry_run: bool = False,
):
    if dry_run:
        print("[FailureCollector] Dry-run mode — no files written.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    failure_path  = out_dir / "failure_set.jsonl"
    full_path     = out_dir / "full_inference.jsonl"
    summary_path  = out_dir / "failure_summary.json"

    with open(failure_path, "w", encoding="utf-8") as f:
        for rs in failures:
            f.write(rs.to_json() + "\n")
    print(f"[FailureCollector] failure_set.jsonl   → {failure_path}  ({len(failures)} rows)")

    with open(full_path, "w", encoding="utf-8") as f:
        for rs in all_states:
            f.write(rs.to_json() + "\n")
    print(f"[FailureCollector] full_inference.jsonl → {full_path}  ({len(all_states)} rows)")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[FailureCollector] failure_summary.json → {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Level 6 FailureCollector — extract failure ReasoningStates from L5"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
        help="Path to Level 5 best_model.pt"
    )
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA),
        help="Labeled CSV with utterance + intent columns"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--out-dir", type=str, default=str(OUT_DIR),
        help="Directory to write failure_set.jsonl, full_inference.jsonl, failure_summary.json"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print summary only; do not write output files"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    failures, all_states = collect_failures(
        checkpoint_path = args.checkpoint,
        data_csv        = args.data,
        batch_size      = args.batch_size,
        device          = device,
    )

    summary = summarize(failures, all_states)
    print_summary(summary)
    save_outputs(
        failures, all_states, summary,
        out_dir = Path(args.out_dir),
        dry_run = args.dry_run,
    )


if __name__ == "__main__":
    main()
