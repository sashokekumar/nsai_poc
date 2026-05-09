# level4/train.py
"""
Training script for Level 4 symbolically supervised neural model.

Usage:
    # Experiment A — baseline (no symbolic loss)
    python -m level4.train --lam 0.0 --run-name baseline

    # Experiment C — symbolically supervised
    python -m level4.train --lam 0.5 --run-name level4_lam0.5

    # λ ablation sweep (all values)
    for lam in 0.0 0.1 0.25 0.5 1.0 2.0:
        python -m level4.train --lam <lam> --run-name ablation_lam<lam>

Saves per-run checkpoint and training log to level4/saved_models/<run-name>/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from level4.model.dataset import IntentDataset, INTENT_LABELS, ENTITY_TYPE_LABELS
from level4.model.neural_intent_model import Level4IntentModel
from level4.model.losses import Level4Loss


# -------------------------------------------------------
# Collate: DataLoader batches utterances as list[str]
# -------------------------------------------------------
def collate_fn(batch: list[dict]) -> dict:
    return {
        "utterances": [item["utterance"] for item in batch],
        "intent_idx":   torch.stack([item["intent_idx"]   for item in batch]),
        "entity_idx":   torch.stack([item["entity_idx"]   for item in batch]),
        "domain_valid": torch.stack([item["domain_valid"] for item in batch]),
    }


# -------------------------------------------------------
# Accuracy helpers
# -------------------------------------------------------
def intent_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def entity_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def domain_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == targets).float().mean().item()


# -------------------------------------------------------
# One epoch of training
# -------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device) -> dict:
    model.train()
    totals = dict(loss=0.0, intent_loss=0.0, entity_loss=0.0,
                  domain_loss=0.0, constraint_loss=0.0,
                  intent_acc=0.0, entity_acc=0.0, domain_acc=0.0)
    n_batches = 0

    for batch in loader:
        utterances   = batch["utterances"]
        intent_tgt   = batch["intent_idx"].to(device)
        entity_tgt   = batch["entity_idx"].to(device)
        domain_tgt   = batch["domain_valid"].to(device)

        optimizer.zero_grad()

        out = model.forward(utterances, device)
        loss_dict = loss_fn(
            out["intent_logits"], out["entity_logits"], out["domain_logits"],
            intent_tgt, entity_tgt, domain_tgt,
        )

        loss_dict["loss"].backward()
        optimizer.step()

        totals["loss"]            += loss_dict["loss"].item()
        totals["intent_loss"]     += loss_dict["intent_loss"]
        totals["entity_loss"]     += loss_dict["entity_loss"]
        totals["domain_loss"]     += loss_dict["domain_loss"]
        totals["constraint_loss"] += loss_dict["constraint_loss"]
        totals["intent_acc"]      += intent_accuracy(out["intent_logits"].detach(), intent_tgt)
        totals["entity_acc"]      += entity_accuracy(out["entity_logits"].detach(), entity_tgt)
        totals["domain_acc"]      += domain_accuracy(out["domain_logits"].detach(), domain_tgt)
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# -------------------------------------------------------
# Evaluation pass (no gradient)
# -------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    totals = dict(loss=0.0, intent_loss=0.0, entity_loss=0.0,
                  domain_loss=0.0, constraint_loss=0.0,
                  intent_acc=0.0, entity_acc=0.0, domain_acc=0.0)
    n_batches = 0

    for batch in loader:
        utterances   = batch["utterances"]
        intent_tgt   = batch["intent_idx"].to(device)
        entity_tgt   = batch["entity_idx"].to(device)
        domain_tgt   = batch["domain_valid"].to(device)

        out = model.forward(utterances, device)
        loss_dict = loss_fn(
            out["intent_logits"], out["entity_logits"], out["domain_logits"],
            intent_tgt, entity_tgt, domain_tgt,
        )

        totals["loss"]            += loss_dict["loss"].item()
        totals["intent_loss"]     += loss_dict["intent_loss"]
        totals["entity_loss"]     += loss_dict["entity_loss"]
        totals["domain_loss"]     += loss_dict["domain_loss"]
        totals["constraint_loss"] += loss_dict["constraint_loss"]
        totals["intent_acc"]      += intent_accuracy(out["intent_logits"], intent_tgt)
        totals["entity_acc"]      += entity_accuracy(out["entity_logits"], entity_tgt)
        totals["domain_acc"]      += domain_accuracy(out["domain_logits"], domain_tgt)
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# -------------------------------------------------------
# Main training loop
# -------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Run: {args.run_name} | λ={args.lam} | epochs={args.epochs} | batch={args.batch_size} | lr={args.lr}")

    # Output dir
    out_dir = Path(__file__).parent / "saved_models" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data_dir = Path(__file__).parent / "data"
    train_ds = IntentDataset(str(data_dir / "train.csv"))
    test_ds  = IntentDataset(str(data_dir / "test.csv"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_ds)} rows | Test: {len(test_ds)} rows")

    # Model + loss + optimiser
    model    = Level4IntentModel(dropout=args.dropout).to(device)
    loss_fn  = Level4Loss(lam=args.lam, alpha=args.alpha, beta=args.beta)
    # Only train the heads and shared trunk — encoder is frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    history = []
    best_val_intent_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics   = evaluate(model, test_loader, loss_fn, device)

        elapsed = time.time() - t0

        row = {
            "epoch":               epoch,
            "lam":                 args.lam,
            "train_loss":          round(train_metrics["loss"], 4),
            "train_intent_loss":   round(train_metrics["intent_loss"], 4),
            "train_entity_loss":   round(train_metrics["entity_loss"], 4),
            "train_domain_loss":   round(train_metrics["domain_loss"], 4),
            "train_constraint":    round(train_metrics["constraint_loss"], 4),
            "train_intent_acc":    round(train_metrics["intent_acc"], 4),
            "train_entity_acc":    round(train_metrics["entity_acc"], 4),
            "train_domain_acc":    round(train_metrics["domain_acc"], 4),
            "val_loss":            round(val_metrics["loss"], 4),
            "val_intent_loss":     round(val_metrics["intent_loss"], 4),
            "val_entity_loss":     round(val_metrics["entity_loss"], 4),
            "val_domain_loss":     round(val_metrics["domain_loss"], 4),
            "val_constraint":      round(val_metrics["constraint_loss"], 4),
            "val_intent_acc":      round(val_metrics["intent_acc"], 4),
            "val_entity_acc":      round(val_metrics["entity_acc"], 4),
            "val_domain_acc":      round(val_metrics["domain_acc"], 4),
            "elapsed_sec":         round(elapsed, 1),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_intent_acc={row['train_intent_acc']:.3f} val_intent_acc={row['val_intent_acc']:.3f} | "
            f"train_loss={row['train_loss']:.4f} (constraint={row['train_constraint']:.4f}) | "
            f"val_loss={row['val_loss']:.4f} (constraint={row['val_constraint']:.4f}) | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint by val intent accuracy
        if val_metrics["intent_acc"] > best_val_intent_acc:
            best_val_intent_acc = val_metrics["intent_acc"]
            best_epoch = epoch
            torch.save({
                "epoch":    epoch,
                "lam":      args.lam,
                "state_dict": model.state_dict(),
                "val_intent_acc": best_val_intent_acc,
            }, out_dir / "best_model.pt")

    print(f"\nBest val_intent_acc={best_val_intent_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to {out_dir / 'best_model.pt'}")

    # Save training history
    history_path = out_dir / "training_log.json"
    with open(history_path, "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"Training log saved to {history_path}")


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 4 symbolic-loss training")
    parser.add_argument("--lam",        type=float, default=0.5,  help="Constraint loss weight λ (0.0 = baseline)")
    parser.add_argument("--alpha",      type=float, default=1.0,  help="Entity loss weight α")
    parser.add_argument("--beta",       type=float, default=1.0,  help="Domain loss weight β")
    parser.add_argument("--epochs",     type=int,   default=20,   help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--dropout",    type=float, default=0.2,  help="Dropout on shared layer")
    parser.add_argument("--run-name",   type=str,   default="run", help="Output subdirectory name under saved_models/")
    args = parser.parse_args()

    train(args)
