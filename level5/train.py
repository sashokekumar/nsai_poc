# level5/train.py
"""
Training script for Level 5 rule-compiled neural model.

Architecture: neural predicate estimator → compiled differentiable symbolic rules → intent.
No residual neural intent classifier. No blend weight.

Usage:
    # L5A — soft compiled symbolic logic (rule_strength learnable)
    python -m level5.train --run-name exp_l5a_soft

    # L5B — hard compiled symbolic logic (rule_strength fixed at 1.0)
    python -m level5.train --run-name exp_l5b_hard --hard-rules

    # L5A with predicate supervision ablation (no pred loss)
    python -m level5.train --run-name exp_l5a_no_pred --pred-weight 0.0

Saves per-run checkpoint and training log to level5/saved_models/<run-name>/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from level5.model.dataset import Level5Dataset, INTENT_LABELS, PREDICATE_COLS
from level5.model.level5_model import Level5IntentModel
from level5.model.losses import Level5Loss

REPO_ROOT = Path(__file__).parent.parent
DATA_CSV  = Path(__file__).parent / "data" / "level5_labeled.csv"


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    return {
        "utterances":       [item["utterance"] for item in batch],
        "intent_idx":       torch.stack([item["intent_idx"]       for item in batch]),
        "predicate_labels": torch.stack([item["predicate_labels"] for item in batch]),
    }


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def intent_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def predicate_accuracy(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean per-predicate binary accuracy across all 11 heads."""
    preds = (probs >= 0.5).float()
    return (preds == targets).float().mean().item()


def predicate_per_head_acc(probs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Per-predicate accuracy dict for logging."""
    preds = (probs >= 0.5).float()
    return {
        PREDICATE_COLS[j]: round(float((preds[:, j] == targets[:, j]).float().mean()), 4)
        for j in range(len(PREDICATE_COLS))
    }


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device) -> dict:
    model.train()
    totals = dict(loss=0.0, intent_loss=0.0, predicate_loss=0.0,
                  intent_acc=0.0, pred_acc=0.0)
    n = 0

    for batch in loader:
        utterances   = batch["utterances"]
        intent_tgt   = batch["intent_idx"].to(device)
        pred_tgt     = batch["predicate_labels"].to(device)

        optimizer.zero_grad()
        out = model.forward(utterances, device)

        loss_dict = loss_fn(
            out["intent_logits"],
            out["predicate_probs"],
            intent_tgt,
            pred_tgt,
        )
        loss_dict["loss"].backward()
        optimizer.step()

        totals["loss"]           += loss_dict["loss"].item()
        totals["intent_loss"]    += loss_dict["intent_loss"]
        totals["predicate_loss"] += loss_dict["predicate_loss"]
        totals["intent_acc"]     += intent_accuracy(out["intent_logits"].detach(), intent_tgt)
        totals["pred_acc"]       += predicate_accuracy(out["predicate_probs"].detach(), pred_tgt)
        n += 1

    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    totals = dict(loss=0.0, intent_loss=0.0, predicate_loss=0.0,
                  intent_acc=0.0, pred_acc=0.0)
    all_pred_probs    = []
    all_pred_tgts     = []
    all_rule_acts     = []
    all_intent_scores = []
    n = 0

    for batch in loader:
        utterances = batch["utterances"]
        intent_tgt = batch["intent_idx"].to(device)
        pred_tgt   = batch["predicate_labels"].to(device)

        out = model.forward(utterances, device)
        loss_dict = loss_fn(
            out["intent_logits"],
            out["predicate_probs"],
            intent_tgt,
            pred_tgt,
        )

        totals["loss"]           += loss_dict["loss"].item()
        totals["intent_loss"]    += loss_dict["intent_loss"]
        totals["predicate_loss"] += loss_dict["predicate_loss"]
        totals["intent_acc"]     += intent_accuracy(out["intent_logits"], intent_tgt)
        totals["pred_acc"]       += predicate_accuracy(out["predicate_probs"], pred_tgt)

        all_pred_probs.append(out["predicate_probs"].cpu())
        all_pred_tgts.append(pred_tgt.cpu())
        all_rule_acts.append(out["rule_activations"].cpu())
        all_intent_scores.append(out["intent_rule_scores"].cpu())
        n += 1

    metrics = {k: v / n for k, v in totals.items()}

    # Per-head predicate accuracy
    all_pred_probs = torch.cat(all_pred_probs, dim=0)
    all_pred_tgts  = torch.cat(all_pred_tgts,  dim=0)
    metrics["pred_per_head"] = predicate_per_head_acc(all_pred_probs, all_pred_tgts)

    # Mean rule activation per rule
    all_rule_acts = torch.cat(all_rule_acts, dim=0)   # [N, n_rules]
    rule_names = [r["name"] for r in model.rule_layer.rules]
    metrics["mean_rule_activations"] = {
        rule_names[r]: round(float(all_rule_acts[:, r].mean()), 4)
        for r in range(all_rule_acts.shape[1])
    }

    # No-rule-fired: inputs where no intent rule scores above threshold
    all_intent_scores = torch.cat(all_intent_scores, dim=0)  # [N, n_intents]
    no_rule_fired = (all_intent_scores.max(dim=1).values < 0.1).sum().item()
    metrics["no_rule_fired_count"] = int(no_rule_fired)
    metrics["no_rule_fired_frac"]  = round(no_rule_fired / len(all_intent_scores), 4)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Run: {args.run_name} | epochs={args.epochs} | "
        f"batch={args.batch_size} | lr={args.lr} | "
        f"pred_weight={args.pred_weight} | hard_rules={args.hard_rules}"
    )

    out_dir = Path(__file__).parent / "saved_models" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data — stratified 80/20 split
    full_ds = Level5Dataset(str(DATA_CSV))
    indices = list(range(len(full_ds)))
    intent_labels_all = [full_ds.intent_idxs[i] for i in indices]
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=intent_labels_all
    )
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Train: {len(train_ds)} rows | Val: {len(val_ds)} rows")

    # Model
    model = Level5IntentModel(
        dropout=args.dropout,
        hard_rules=args.hard_rules,
    )
    if args.hard_rules:
        print("Hard rules mode (L5B): rule_strength fixed at 1.0, no grad")
    else:
        print("Soft rules mode (L5A): rule_strength_logits learnable")

    model = model.to(device)
    loss_fn = Level5Loss(pred_weight=args.pred_weight)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    history = []
    best_val_intent_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_m   = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        # Rule diagnostics
        rule_strengths = model.rule_strength_dict()

        row = {
            "epoch":              epoch,
            "train_loss":         round(train_m["loss"], 4),
            "train_intent_loss":  round(train_m["intent_loss"], 4),
            "train_pred_loss":    round(train_m["predicate_loss"], 4),
            "train_intent_acc":   round(train_m["intent_acc"], 4),
            "train_pred_acc":     round(train_m["pred_acc"], 4),
            "val_loss":           round(val_m["loss"], 4),
            "val_intent_loss":    round(val_m["intent_loss"], 4),
            "val_pred_loss":      round(val_m["predicate_loss"], 4),
            "val_intent_acc":     round(val_m["intent_acc"], 4),
            "val_pred_acc":       round(val_m["pred_acc"], 4),
            "val_pred_per_head":  val_m["pred_per_head"],
            "mean_rule_activations": val_m["mean_rule_activations"],
            "rule_strengths":     rule_strengths,
            "no_rule_fired":      val_m["no_rule_fired_count"],
            "no_rule_fired_frac": val_m["no_rule_fired_frac"],
            "elapsed_sec":        round(elapsed, 1),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"tr_int_acc={row['train_intent_acc']:.3f}  "
            f"val_int_acc={row['val_intent_acc']:.3f}  "
            f"tr_loss={row['train_loss']:.4f}  "
            f"val_loss={row['val_loss']:.4f}  "
            f"no_rule_fired={row['no_rule_fired']}  "
            f"rule_str={list(rule_strengths.values())}  "
            f"{elapsed:.1f}s"
        )

        if val_m["intent_acc"] > best_val_intent_acc:
            best_val_intent_acc = val_m["intent_acc"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch":            epoch,
                    "run_name":         args.run_name,
                    "state_dict":       model.state_dict(),
                    "val_intent_acc":   best_val_intent_acc,
                    "rule_strengths":   rule_strengths,
                    "hard_rules":       args.hard_rules,
                    "args":             vars(args),
                },
                out_dir / "best_model.pt",
            )

    print(f"\nBest val_intent_acc={best_val_intent_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {out_dir / 'best_model.pt'}")

    log_path = out_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"Training log: {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 5 rule-compiled neural training")
    parser.add_argument("--run-name",          type=str,   default="run",
                        help="Output subdirectory name under saved_models/")
    parser.add_argument("--epochs",            type=int,   default=20)
    parser.add_argument("--batch-size",        type=int,   default=32)
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--dropout",           type=float, default=0.2)
    parser.add_argument("--pred-weight",       type=float, default=0.5,
                        help="Weight on predicate BCE loss (0.0 = no predicate supervision)")
    parser.add_argument("--hard-rules",        action="store_true",
                        help="L5B mode: fix rule_strength=1.0, no grad (hard compiled symbolic)")
    parser.add_argument("--rule-strength-init", type=float, default=None,
                        help="Override all rule_strength_init values before training")
    args = parser.parse_args()

    # Optionally override rule_strength_init (used for hard-symbolic ablation)
    if args.rule_strength_init is not None:
        import json as _json, math as _math
        rb_path = Path(__file__).parent / "data" / "rule_base.json"
        rb = _json.loads(rb_path.read_text())
        for rule in rb["rules"]:
            rule["rule_strength_init"] = args.rule_strength_init
        # Write to a temp path so the main rule_base is unchanged
        tmp_path = Path(__file__).parent / "data" / "_rule_base_override.json"
        tmp_path.write_text(_json.dumps(rb, indent=2))
        # Monkey-patch default in Level5IntentModel
        import level5.model.level5_model as _m5
        _m5._DEFAULT_RULE_BASE = tmp_path
        print(f"Rule strength overridden to {args.rule_strength_init} for all rules")

    train(args)
