# level6/retrain_validator.py
"""
RetrainValidator -- Task 11B: fine-tune validation for Experimental symbols.

For each Experimental symbol with a refined candidate rule:
  1. Build an augmented rule_base that injects the candidate rule.
  2. Fine-tune the full Level 5 model for a small number of epochs
     (default 5, lr=1e-4) using the level6 seed dataset.
  3. Evaluate the fine-tuned model vs the frozen no-retrain baseline:
         retrain_delta_over_noretrain =
             acc(retrained, full_set) - acc(noretrain_injected, full_set)
  4. If retrain_delta >= 0.01 --> neuro-symbolic co-evolution confirmed.
  5. Update registry retrain_delta_over_noretrain field.
  6. Saves fine-tuned checkpoint to
     level5/saved_models/l6_ft_{symbol_id}/best_model.pt.

Why fine-tune rather than full retrain
---------------------------------------
Full retraining from scratch would take 20+ epochs.  Fine-tuning for 5 epochs
starting from the checkpoint is sufficient to measure whether the neural trunk
adapts to the new symbolic structure.  The encoder (SentenceTransformer) is
always frozen; only the shared trunk, predicate_head, and rule_layer are
updated.  The injected rule's rule_strength_logit is also learnable so the
model can calibrate how much it trusts the new rule.

Neuro-symbolic co-evolution signal
------------------------------------
retrain_delta = 0.00 --> rule is injected but neural trunk ignores it.
                         Symbolic patching only.  Promote to Active anyway
                         if the user wishes (still useful as a runtime guard).
retrain_delta > 0.01 --> neural trunk adapted.  Genuine co-evolution.
                         Required for automatic Experimental --> Active.

Usage
-----
    python -m level6.retrain_validator

    python -m level6.retrain_validator --symbol S_001 --epochs 5
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from level5.model.dataset import Level5Dataset, INTENT_LABELS        # noqa: E402
from level5.model.level5_model import Level5IntentModel               # noqa: E402
from level5.model.losses import Level5Loss                             # noqa: E402
from level5.train import (                                             # noqa: E402
    collate_fn, train_one_epoch, evaluate, intent_accuracy,
)
from level6.rule_validator import (                                    # noqa: E402
    DEFAULT_CHECKPOINT, DEFAULT_REGISTRY, DEFAULT_RULE_BASE,
    DEFAULT_CANDIDATE_DIR,
    _inject_rule, _load_model_with_rule_base,
)
from level6.symbol_registry import SymbolRegistry                     # noqa: E402
from level6.lifecycle import can_promote_to_active                    # noqa: E402

DEFAULT_SEED_CSV   = REPO_ROOT / "level6" / "data" / "level6_seed.csv"
DEFAULT_FT_EPOCHS  = 5
DEFAULT_FT_LR      = 1e-4
DEFAULT_FT_BATCH   = 32
FT_MODEL_DIR       = REPO_ROOT / "level5" / "saved_models"


# ---------------------------------------------------------------------------
# Baseline accuracy (no-retrain injected model)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _full_set_accuracy(
    model: Level5IntentModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Intent accuracy over a full DataLoader."""
    model.eval()
    correct = total = 0
    for batch in loader:
        utterances = batch["utterances"]
        intent_tgt = batch["intent_idx"].to(device)
        out = model.forward(utterances, device)
        preds = out["intent_logits"].argmax(dim=-1)
        correct += (preds == intent_tgt).sum().item()
        total   += len(intent_tgt)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Fine-tune one symbol
# ---------------------------------------------------------------------------

def finetune_symbol(
    symbol_id: str,
    registry: SymbolRegistry,
    existing_rule_base: dict,
    candidate_dir: Path,
    seed_csv: Path,
    checkpoint_path: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    dry_run: bool,
) -> dict:
    """
    Fine-tune Level 5 with the candidate rule injected.
    Returns a report dict with retrain_delta_over_noretrain.
    """
    sym = registry.get(symbol_id)
    sym_name        = sym.get("name", symbol_id)
    reg_rule_name   = sym.get("candidate_rule_name", "")

    # Locate candidate rule file
    candidate_path  = candidate_dir / f"{reg_rule_name}.json" if reg_rule_name else None
    if candidate_path is None or not candidate_path.exists():
        refined = sorted(candidate_dir.glob(f"R_{symbol_id}*_refined.json"))
        original = sorted(candidate_dir.glob(f"R_{symbol_id}*.json"))
        original = [f for f in original if "_refined" not in f.name]
        candidate_path = (refined or original or [None])[0]

    if candidate_path is None or not candidate_path.exists():
        return {"symbol_id": symbol_id, "error": "No candidate rule file found"}

    with open(candidate_path) as f:
        candidate_rule = json.load(f)

    print(f"\n  [{symbol_id}] {sym_name}")
    print(f"      candidate rule  : {candidate_rule.get('name')}")
    print(f"      fine-tune epochs: {epochs}  lr={lr}  batch={batch_size}")

    # Build dataset from seed CSV
    dataset  = Level5Dataset(str(seed_csv))
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    ft_loader= DataLoader(dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    print(f"      dataset rows    : {len(dataset)}")

    # Inject rule into a temp rule_base file
    injected_rb = _inject_rule(existing_rule_base, candidate_rule)
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(injected_rb, tf)
    tf.close()
    tmp_path = tf.name

    try:
        # No-retrain baseline (frozen weights, injected rule)
        noretrain_model = _load_model_with_rule_base(checkpoint_path, tmp_path, device)
        acc_noretrain   = _full_set_accuracy(noretrain_model, loader, device)
        print(f"      acc_noretrain   : {acc_noretrain:.4f}")

        if dry_run:
            Path(tmp_path).unlink(missing_ok=True)
            return {
                "symbol_id":               symbol_id,
                "acc_noretrain":           round(acc_noretrain, 4),
                "acc_retrained":           None,
                "retrain_delta_over_noretrain": None,
                "dry_run":                 True,
            }

        # Fine-tune starting from the no-retrain model (it already has the right weights)
        ft_model = _load_model_with_rule_base(checkpoint_path, tmp_path, device)
        ft_model.train()

        # Only tune trunk + predicate_head + rule_layer (encoder stays frozen)
        trainable = [p for p in ft_model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        loss_fn   = Level5Loss(pred_weight=0.5)

        best_acc  = acc_noretrain
        best_sd   = copy.deepcopy(ft_model.state_dict())

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            metrics = train_one_epoch(ft_model, ft_loader, optimizer, loss_fn, device)
            acc_val = _full_set_accuracy(ft_model, loader, device)
            elapsed = time.time() - t0
            print(
                f"        epoch {epoch}/{epochs}  "
                f"loss={metrics['loss']:.4f}  "
                f"intent_acc={acc_val:.4f}  "
                f"({elapsed:.1f}s)"
            )
            if acc_val > best_acc:
                best_acc = acc_val
                best_sd  = copy.deepcopy(ft_model.state_dict())

        acc_retrained = best_acc
        retrain_delta = round(acc_retrained - acc_noretrain, 4)
        print(f"      acc_retrained   : {acc_retrained:.4f}")
        print(f"      retrain_delta   : {retrain_delta:+.4f}  "
              f"(threshold >= 0.01)")

        # Save fine-tuned checkpoint
        ft_dir = FT_MODEL_DIR / f"l6_ft_{symbol_id}"
        ft_dir.mkdir(parents=True, exist_ok=True)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        torch.save({
            "epoch":               epochs,
            "run_name":            f"l6_ft_{symbol_id}",
            "state_dict":          best_sd,
            "val_intent_acc":      acc_retrained,
            "hard_rules":          ckpt.get("hard_rules", False),
            "l6_symbol_id":        symbol_id,
            "l6_candidate_rule":   candidate_rule.get("name"),
            "l6_rule_base":        injected_rb,
        }, ft_dir / "best_model.pt")
        print(f"      [saved] {ft_dir / 'best_model.pt'}")

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Promotion check
    eligible, reasons = can_promote_to_active({**sym, "retrain_delta_over_noretrain": retrain_delta})
    print(f"      eligible (Exp->Active): {eligible}")
    if not eligible:
        for r in reasons:
            print(f"        ! {r}")

    return {
        "symbol_id":                    symbol_id,
        "symbol_name":                  sym_name,
        "candidate_rule_name":          candidate_rule.get("name"),
        "acc_noretrain":                round(acc_noretrain, 4),
        "acc_retrained":                round(acc_retrained, 4),
        "retrain_delta_over_noretrain": retrain_delta,
        "promotion_eligible":           eligible,
        "promotion_reasons":            reasons,
        "ft_checkpoint":                str(FT_MODEL_DIR / f"l6_ft_{symbol_id}" / "best_model.pt"),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_retrain_validation(
    symbol_ids: list[str] | None,
    checkpoint_path: Path,
    registry_path: Path,
    rule_base_path: Path,
    candidate_dir: Path,
    seed_csv: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    dry_run: bool,
) -> list[dict]:
    device = torch.device("cpu")

    print("[RetrainValidator] Checkpoint :", checkpoint_path)
    print("[RetrainValidator] Seed CSV   :", seed_csv)
    print(f"[RetrainValidator] Epochs={epochs}  lr={lr}  batch={batch_size}")

    with open(rule_base_path) as f:
        existing_rule_base = json.load(f)

    registry = SymbolRegistry(registry_path)

    # Experimental symbols only
    experimental = [
        sid for sid, sym in registry._data["symbols"].items()
        if sym.get("status") == "experimental"
    ]
    if symbol_ids:
        experimental = [s for s in experimental if s in symbol_ids]

    print(f"[RetrainValidator] Experimental symbols: {experimental}")

    reports: list[dict] = []
    for sid in experimental:
        report = finetune_symbol(
            symbol_id      = sid,
            registry       = registry,
            existing_rule_base = existing_rule_base,
            candidate_dir  = candidate_dir,
            seed_csv       = seed_csv,
            checkpoint_path= checkpoint_path,
            device         = device,
            epochs         = epochs,
            lr             = lr,
            batch_size     = batch_size,
            dry_run        = dry_run,
        )
        reports.append(report)

        if not dry_run and "error" not in report:
            registry.update_validation(
                sid,
                retrain_delta_over_noretrain=report["retrain_delta_over_noretrain"],
            )

    if not dry_run:
        registry.save()
        print(f"\n[RetrainValidator] Registry updated -> {registry_path}")

    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RetrainValidator -- Task 11B: fine-tune validation"
    )
    parser.add_argument("--symbol",       nargs="*",  default=None)
    parser.add_argument("--checkpoint",   type=Path,  default=DEFAULT_CHECKPOINT)
    parser.add_argument("--registry",     type=Path,  default=DEFAULT_REGISTRY)
    parser.add_argument("--rule-base",    type=Path,  default=DEFAULT_RULE_BASE)
    parser.add_argument("--candidate-dir",type=Path,  default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--seed-csv",     type=Path,  default=DEFAULT_SEED_CSV)
    parser.add_argument("--epochs",       type=int,   default=DEFAULT_FT_EPOCHS)
    parser.add_argument("--lr",           type=float, default=DEFAULT_FT_LR)
    parser.add_argument("--batch",        type=int,   default=DEFAULT_FT_BATCH)
    parser.add_argument("--dry-run",      action="store_true")
    args = parser.parse_args()

    reports = run_retrain_validation(
        symbol_ids     = args.symbol,
        checkpoint_path= args.checkpoint,
        registry_path  = args.registry,
        rule_base_path = args.rule_base,
        candidate_dir  = args.candidate_dir,
        seed_csv       = args.seed_csv,
        epochs         = args.epochs,
        lr             = args.lr,
        batch_size     = args.batch,
        dry_run        = args.dry_run,
    )

    print(f"\n{'='*70}")
    print(f"  RetrainValidator Summary (Task 11B)")
    print(f"{'='*70}")
    for r in reports:
        if "error" in r:
            print(f"  [{r['symbol_id']}]  ERROR: {r['error']}")
            continue
        eligible = r.get("promotion_eligible")
        tag = "ELIGIBLE (Exp->Active)" if eligible else "NOT ELIGIBLE"
        print(f"  [{r['symbol_id']}]  [{tag}]")
        print(f"      acc_noretrain  : {r.get('acc_noretrain'):.4f}")
        if r.get("acc_retrained") is not None:
            print(f"      acc_retrained  : {r.get('acc_retrained'):.4f}")
            delta = r.get("retrain_delta_over_noretrain", 0)
            print(f"      retrain_delta  : {delta:+.4f}  (threshold >= 0.01)")
        for reason in r.get("promotion_reasons", []):
            print(f"      ! {reason}")
    print(f"{'='*70}")
    print(f"\n  To promote eligible symbols to Active:")
    print(f"    python -m level6.symbol_registry --promote <SYMBOL_ID>")


if __name__ == "__main__":
    main()
