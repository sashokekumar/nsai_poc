# level4/evaluation/violation_metrics.py
"""
Offline evaluation of constraint violations for Level 4.

Loads a saved checkpoint, runs inference on the test set, and computes:
  - Intent / entity / domain accuracy
  - Constraint violation rate (all disallowed pairs)
  - TYPE_A violation rate: false rejection (out_of_scope predicted, SRE entity present)
  - TYPE_B violation rate: false execution (execution predicted, entity=metric or incident)
  - TYPE_C violation rate: ungrounded SRE (SRE intent predicted, entity=unknown)
  - Per-intent accuracy breakdown

This script is OFFLINE ONLY — it never corrects predictions.
The key Level 4 claim is that training-time symbolic loss reduces violations
without any runtime correction.

Usage:
    python -m level4.evaluation.violation_metrics \
        --checkpoint saved_models/baseline/best_model.pt \
        --run-name baseline

    python -m level4.evaluation.violation_metrics \
        --checkpoint saved_models/level4_lam0_5/best_model.pt \
        --run-name level4_lam0_5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from level4.model.neural_intent_model import Level4IntentModel


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Level4IntentModel()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def run_inference(model, utterances: list[str], device, batch_size=64) -> list[dict]:
    all_preds = []
    for start in range(0, len(utterances), batch_size):
        batch = utterances[start: start + batch_size]
        all_preds.extend(model.predict(batch, device))
    return all_preds


def load_constraint_rules() -> list[dict]:
    rules_path = Path(__file__).parent.parent / "ontology" / "constraint_rules.json"
    with open(rules_path) as f:
        cfg = json.load(f)
    return cfg["disallowed_intent_entity_pairs"]


# -------------------------------------------------------
# Metric computation
# -------------------------------------------------------
def compute_metrics(df_true: pd.DataFrame, preds: list[dict]) -> dict:
    df_pred = pd.DataFrame(preds)

    n = len(df_true)
    assert len(df_pred) == n, "Prediction count mismatch"

    # Core accuracy
    intent_acc = (df_pred["intent"] == df_true["intent"].values).mean()
    entity_acc = (df_pred["entity_type"] == df_true["entity_type"].values).mean()
    domain_acc = (df_pred["domain_valid"].astype(bool) == df_true["domain_valid"].astype(bool).values).mean()

    # Per-intent accuracy
    intent_acc_per = {}
    for intent in df_true["intent"].unique():
        mask = df_true["intent"] == intent
        if mask.sum() == 0:
            continue
        intent_acc_per[intent] = float((df_pred.loc[mask.values, "intent"] == intent).mean())

    # Constraint violations (from rules file)
    rules = load_constraint_rules()
    violation_counts = {}
    for rule in rules:
        intent    = rule["intent"]
        entity    = rule["entity_type"]
        key       = f"{intent}+{entity}"
        mask      = (df_pred["intent"] == intent) & (df_pred["entity_type"] == entity)
        violation_counts[key] = int(mask.sum())

    total_violations = sum(violation_counts.values())
    overall_violation_rate = total_violations / n

    # TYPE_A — false rejection: out_of_scope predicted, known SRE entity in pred
    sre_entities = ["infrastructure", "service", "metric", "incident", "job", "pipeline"]
    type_a_mask = (df_pred["intent"] == "out_of_scope") & (df_pred["entity_type"].isin(sre_entities))
    type_a_rate = float(type_a_mask.mean())

    # TYPE_B — false execution: execution predicted against observe/report targets
    type_b_mask = (df_pred["intent"] == "execution") & (df_pred["entity_type"].isin(["metric", "incident"]))
    type_b_rate = float(type_b_mask.mean())

    # TYPE_C — ungrounded SRE: SRE intent predicted, entity=unknown
    sre_intents = ["investigate", "summarization", "execution"]
    type_c_mask = (df_pred["intent"].isin(sre_intents)) & (df_pred["entity_type"] == "unknown")
    type_c_rate = float(type_c_mask.mean())

    return {
        "n": n,
        "intent_acc":             round(float(intent_acc), 4),
        "entity_acc":             round(float(entity_acc), 4),
        "domain_acc":             round(float(domain_acc), 4),
        "intent_acc_per_class":   {k: round(v, 4) for k, v in intent_acc_per.items()},
        "total_violations":       total_violations,
        "overall_violation_rate": round(overall_violation_rate, 4),
        "type_a_false_rejection": round(type_a_rate, 4),
        "type_b_false_execution": round(type_b_rate, 4),
        "type_c_ungrounded_sre":  round(type_c_rate, 4),
        "per_pair_violations":    violation_counts,
    }


# -------------------------------------------------------
# Pretty-print
# -------------------------------------------------------
def print_metrics(metrics: dict, run_name: str, lam):
    w = 50
    print("=" * w)
    print(f"  Run: {run_name}   λ={lam}")
    print("=" * w)
    print(f"  Test rows      : {metrics['n']}")
    print(f"  Intent acc     : {metrics['intent_acc']:.4f}")
    print(f"  Entity acc     : {metrics['entity_acc']:.4f}")
    print(f"  Domain acc     : {metrics['domain_acc']:.4f}")
    print()
    print("  Per-class intent accuracy:")
    for intent, acc in sorted(metrics["intent_acc_per_class"].items()):
        print(f"    {intent:20s}: {acc:.4f}")
    print()
    print(f"  Constraint violation rate: {metrics['overall_violation_rate']:.4f}  ({metrics['total_violations']} / {metrics['n']})")
    print(f"  TYPE_A false rejection   : {metrics['type_a_false_rejection']:.4f}")
    print(f"  TYPE_B false execution   : {metrics['type_b_false_execution']:.4f}")
    print(f"  TYPE_C ungrounded SRE    : {metrics['type_c_ungrounded_sre']:.4f}")
    print()
    if any(v > 0 for v in metrics["per_pair_violations"].values()):
        print("  Per-pair violations:")
        for pair, count in sorted(metrics["per_pair_violations"].items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"    {pair:40s}: {count}")
    print("=" * w)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Level 4 offline constraint violation evaluation")
    parser.add_argument("--checkpoint",    required=True, help="Path to best_model.pt checkpoint")
    parser.add_argument("--run-name",      type=str, default="run", help="Label for this evaluation run")
    parser.add_argument("--test-csv",      type=str, default=None, help="Override default test.csv path")
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--output-dir",    type=str, default=None, help="Save metrics JSON here")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint path
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).parent.parent / args.checkpoint

    # Load model
    model, ckpt_meta = load_model(str(ckpt_path), device)
    lam = ckpt_meta.get("lam", "?")

    # Load test data
    test_csv = args.test_csv or str(Path(__file__).parent.parent / "data" / "test.csv")
    df_true = pd.read_csv(test_csv)
    print(f"Test set: {test_csv}  ({len(df_true)} rows)")

    # Run inference
    preds = run_inference(model, df_true["utterance"].tolist(), device, args.batch_size)

    # Compute + display
    metrics = compute_metrics(df_true, preds)
    print_metrics(metrics, args.run_name, lam)

    # Save
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_path = out_dir / "evaluation_metrics.json"
    full_results = {"run_name": args.run_name, "lam": lam, "checkpoint": str(ckpt_path), **metrics}
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
