# level5/evaluation/violation_metrics.py
"""
Offline evaluation for Level 5 rule-compiled neural model.

Loads a saved checkpoint, runs inference on the val split of level5_labeled.csv,
and computes:
  - Intent accuracy (overall and per-class)
  - Predicate accuracy (overall and per-head)
  - Rule fidelity: % of predictions consistent with the highest-activated rule
  - Ontology violation rate: predictions that break constraint_rules.json pairs
    (same ontology used in Level 4 for cross-level comparison)
  - TYPE_A / B / C violation breakdown (same taxonomy as Level 4)
  - Mean rule activation and learned rule_strength per rule

This script is OFFLINE ONLY — it never corrects predictions.
The Level 5 claim: violations are reduced because rules are structural
(compiled into the forward pass), not post-hoc.

Usage:
    python -m level5.evaluation.violation_metrics \\
        --checkpoint saved_models/exp_b_l5_main/best_model.pt \\
        --run-name exp_b_l5_main

    # Compare against Level 4 results
    python -m level5.evaluation.violation_metrics \\
        --checkpoint saved_models/exp_b_l5_main/best_model.pt \\
        --run-name exp_b_l5_main --compare-l4 level4/saved_models/lam2_0/evaluation_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from level5.model.level5_model import Level5IntentModel
from level5.model.dataset import (
    Level5Dataset, PREDICATE_COLS, INTENT_LABELS,
)

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_CSV  = Path(__file__).parent.parent / "data" / "level5_labeled.csv"
L4_RULES  = REPO_ROOT / "level4" / "ontology" / "constraint_rules.json"


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Level5IntentModel()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Inference over a list of utterances
# ---------------------------------------------------------------------------

def run_inference(model, utterances, device, batch_size=64):
    all_preds = []
    for start in range(0, len(utterances), batch_size):
        batch = utterances[start: start + batch_size]
        all_preds.extend(model.predict(batch, device=device))
    return all_preds


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def load_l4_constraint_rules() -> list:
    with open(L4_RULES) as f:
        return json.load(f)["disallowed_intent_entity_pairs"]


def compute_metrics(
    df_true: pd.DataFrame,
    preds: list,
    model: Level5IntentModel,
) -> dict:
    n = len(df_true)
    assert len(preds) == n

    pred_intents  = [p["intent"] for p in preds]
    pred_intents_s = pd.Series(pred_intents)
    true_intents  = df_true["intent"].values

    # --- Intent accuracy ---
    intent_acc = (pred_intents_s.values == true_intents).mean()
    intent_acc_per = {}
    for intent in INTENT_LABELS:
        mask = true_intents == intent
        if mask.sum() == 0:
            continue
        intent_acc_per[intent] = float(
            (pred_intents_s.values[mask] == intent).mean()
        )

    # --- Predicate accuracy (vs ground-truth predicate cols in CSV) ---
    pred_cols_available = [c for c in PREDICATE_COLS if c in df_true.columns]
    pred_acc_per = {}
    for col in pred_cols_available:
        pred_vals = [p["predicate_activations"].get(col, 0.0) for p in preds]
        pred_binary = [1 if v >= 0.5 else 0 for v in pred_vals]
        true_binary = df_true[col].astype(int).values
        pred_acc_per[col] = round(
            float((pd.Series(pred_binary).values == true_binary).mean()), 4
        )
    pred_acc_overall = round(
        float(sum(pred_acc_per.values()) / len(pred_acc_per)) if pred_acc_per else 0.0, 4
    )

    # --- Rule fidelity ---
    # For each prediction, identify the highest-activated rule and check
    # whether the predicted intent matches that rule's consequent.
    rule_names = [r["name"] for r in model.rule_layer.rules]
    rule_to_consequent = {
        r["name"]: r["consequent_intent"] for r in model.rule_layer.rules
    }
    fidelity_count = 0
    for pred in preds:
        acts = pred.get("rule_activations", {})
        if not acts or max(acts.values()) < 1e-4:
            # No rule fired meaningfully — skip from fidelity check
            continue
        top_rule = max(acts, key=acts.get)
        if pred["intent"] == rule_to_consequent.get(top_rule, ""):
            fidelity_count += 1
    fidelity_eligible = sum(
        1 for p in preds
        if p.get("rule_activations") and max(p["rule_activations"].values()) >= 1e-4
    )
    rule_fidelity = (
        round(fidelity_count / fidelity_eligible, 4) if fidelity_eligible > 0 else 0.0
    )

    # --- Mean rule activations ---
    mean_rule_activations = {}
    for rname in rule_names:
        vals = [p["rule_activations"].get(rname, 0.0) for p in preds]
        mean_rule_activations[rname] = round(float(sum(vals) / len(vals)), 4)

    # --- Ontology violation rate (Level 4 constraint_rules.json) ---
    # entity_type for Level 5 preds: derive from predicate_activations
    # (the predicate with highest activation among entity-type predicates)
    ENTITY_PREDS = [
        "is_infrastructure", "is_service", "is_metric",
        "is_incident", "is_job", "is_pipeline", "is_unknown",
    ]
    PRED_TO_ENTITY = {
        "is_infrastructure": "infrastructure",
        "is_service":        "service",
        "is_metric":         "metric",
        "is_incident":       "incident",
        "is_job":            "job",
        "is_pipeline":       "pipeline",
        "is_unknown":        "unknown",
    }

    def inferred_entity(pred_dict):
        acts = pred_dict.get("predicate_activations", {})
        best = max(ENTITY_PREDS, key=lambda k: acts.get(k, 0.0))
        return PRED_TO_ENTITY[best]

    pred_entity_types = [inferred_entity(p) for p in preds]

    rules = load_l4_constraint_rules()
    violation_counts = {}
    for rule in rules:
        intent = rule["intent"]
        entity = rule["entity_type"]
        key    = f"{intent}+{entity}"
        count  = sum(
            1 for pi, pe in zip(pred_intents, pred_entity_types)
            if pi == intent and pe == entity
        )
        violation_counts[key] = count

    total_violations = sum(violation_counts.values())
    overall_violation_rate = total_violations / n

    sre_entities = ["infrastructure", "service", "metric", "incident", "job", "pipeline"]
    type_a_rate = sum(
        1 for pi, pe in zip(pred_intents, pred_entity_types)
        if pi == "out_of_scope" and pe in sre_entities
    ) / n
    type_b_rate = sum(
        1 for pi, pe in zip(pred_intents, pred_entity_types)
        if pi == "execution" and pe in ["metric", "incident"]
    ) / n
    type_c_rate = sum(
        1 for pi, pe in zip(pred_intents, pred_entity_types)
        if pi in ["investigate", "summarization", "execution"] and pe == "unknown"
    ) / n

    return {
        "n": n,
        "intent_acc":              round(float(intent_acc), 4),
        "intent_acc_per_class":    {k: round(v, 4) for k, v in intent_acc_per.items()},
        "predicate_acc_overall":   pred_acc_overall,
        "predicate_acc_per_head":  pred_acc_per,
        "rule_fidelity":           rule_fidelity,
        "fidelity_eligible_rows":  fidelity_eligible,
        "mean_rule_activations":   mean_rule_activations,
        "rule_strengths":          model.rule_strength_dict(),
        "blend_weight":            model.blend_weight(),
        "total_violations":        total_violations,
        "overall_violation_rate":  round(overall_violation_rate, 4),
        "type_a_false_rejection":  round(type_a_rate, 4),
        "type_b_false_execution":  round(type_b_rate, 4),
        "type_c_ungrounded_sre":   round(type_c_rate, 4),
        "per_pair_violations":     violation_counts,
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict, run_name: str, compare_l4: dict = None):
    w = 60
    print("=" * w)
    print(f"  Level 5 Evaluation: {run_name}")
    print("=" * w)
    print(f"  Val rows           : {metrics['n']}")
    print(f"  Intent accuracy    : {metrics['intent_acc']:.4f}")
    print(f"  Predicate acc (avg): {metrics['predicate_acc_overall']:.4f}")
    print()
    print("  Per-class intent accuracy:")
    for k, v in metrics["intent_acc_per_class"].items():
        print(f"    {k:<18} {v:.4f}")
    print()
    print("  Rule fidelity      : {:.4f}  ({} eligible rows)".format(
        metrics["rule_fidelity"], metrics["fidelity_eligible_rows"]))
    print("  Rule strengths     :", metrics["rule_strengths"])
    print("  Blend weight α     : {:.4f}".format(metrics["blend_weight"]))
    print("  Mean rule activations:")
    for k, v in metrics["mean_rule_activations"].items():
        print(f"    {k:<35} {v:.4f}")
    print()
    print(f"  Total violations   : {metrics['total_violations']}")
    print(f"  Violation rate     : {metrics['overall_violation_rate']:.4f}")
    print(f"  TYPE_A (false rej) : {metrics['type_a_false_rejection']:.4f}")
    print(f"  TYPE_B (false exec): {metrics['type_b_false_execution']:.4f}")
    print(f"  TYPE_C (ungrounded): {metrics['type_c_ungrounded_sre']:.4f}")

    if compare_l4:
        print()
        print("  Comparison vs Level 4 (λ=2.0):")
        l4_acc  = compare_l4.get("intent_acc", "?")
        l4_viol = compare_l4.get("overall_violation_rate", "?")
        l5_acc  = metrics["intent_acc"]
        l5_viol = metrics["overall_violation_rate"]
        print(f"    {'Metric':<25} {'L4 λ=2.0':>10} {'L5':>10}")
        print(f"    {'Intent accuracy':<25} {l4_acc:>10} {l5_acc:>10}")
        print(f"    {'Violation rate':<25} {l4_viol:>10} {l5_viol:>10}")

    print("=" * w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Level 5 offline evaluation — violation metrics"
    )
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to best_model.pt (absolute or relative to level5/)")
    parser.add_argument("--run-name",    type=str, default="eval",
                        help="Label for this evaluation run")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--compare-l4", type=str, default=None,
                        help="Path to Level 4 evaluation_metrics.json for side-by-side comparison")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).parent.parent / args.checkpoint

    model, ckpt_meta = load_model(str(ckpt_path), device)
    print(f"Loaded: {ckpt_path}  val_intent_acc={ckpt_meta.get('val_intent_acc', 0):.4f}\n")

    # Reproducible val split — same seed/ratio as train.py
    full_ds = Level5Dataset(str(DATA_CSV))
    indices = list(range(len(full_ds)))
    intent_labels_all = [full_ds.intent_idxs[i] for i in indices]
    _, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=intent_labels_all
    )

    df_full = pd.read_csv(DATA_CSV)
    df_val  = df_full.iloc[val_idx].reset_index(drop=True)
    utterances = df_val["utterance"].tolist()

    print(f"Val set: {len(utterances)} rows")
    print("Running inference...")
    preds = run_inference(model, utterances, device, batch_size=args.batch_size)

    metrics = compute_metrics(df_val, preds, model)

    compare_l4 = None
    if args.compare_l4:
        with open(args.compare_l4) as f:
            compare_l4 = json.load(f)

    print_metrics(metrics, args.run_name, compare_l4)

    # Save metrics alongside the checkpoint
    out_path = ckpt_path.parent / "evaluation_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"run_name": args.run_name, **metrics}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
