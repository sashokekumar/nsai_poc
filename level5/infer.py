# level5/infer.py
"""
Clean inference for Level 5 rule-compiled neural model.

Loads a saved checkpoint and runs forward pass through the full neuro-symbolic
architecture (encoder → trunk → predicate heads → rule layer → intent blend).
No symbolic post-processing beyond what the rule layer contributes structurally.

Output includes predicate activations and rule activation scores per utterance,
making the model's symbolic reasoning interpretable.

Usage:
    # Single utterance
    python -m level5.infer --checkpoint saved_models/exp_b_l5_main/best_model.pt \\
        --utterance "why is the payment service latency spiking"

    # File of utterances (CSV)
    python -m level5.infer --checkpoint saved_models/exp_b_l5_main/best_model.pt \\
        --input-file data/level5_labeled.csv --utterance-col utterance

    # Interactive REPL
    python -m level5.infer --checkpoint saved_models/exp_b_l5_main/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from level5.model.level5_model import Level5IntentModel


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Level5IntentModel()
    incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        import warnings
        warnings.warn(
            f"load_model: checkpoint contains keys not present in the current model "
            f"(skipped): {incompatible.unexpected_keys}",
            UserWarning,
        )
    if incompatible.missing_keys:
        raise RuntimeError(
            f"load_model: model weights missing from checkpoint: {incompatible.missing_keys}"
        )
    model.to(device)
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_result(pred: dict, json_out: bool) -> str:
    if json_out:
        return json.dumps(pred, indent=2)

    rule_str = "  ".join(
        f"{name}={v:.3f}"
        for name, v in pred.get("rule_activations", {}).items()
    )
    top_preds = sorted(
        pred.get("predicate_activations", {}).items(),
        key=lambda x: x[1], reverse=True
    )[:4]
    pred_str = "  ".join(f"{k}={v:.3f}" for k, v in top_preds)

    return (
        f"utterance   : {pred['utterance']}\n"
        f"intent      : {pred['intent']}  (conf={pred['intent_prob']:.3f})\n"
        f"rules       : {rule_str}\n"
        f"top preds   : {pred_str}"
    )


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def run_inference(
    model: Level5IntentModel,
    utterances: list,
    device: torch.device,
    batch_size: int = 64,
) -> list:
    all_preds = []
    for start in range(0, len(utterances), batch_size):
        batch = utterances[start: start + batch_size]
        all_preds.extend(model.predict(batch, device=device))
    return all_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Level 5 inference — rules compiled into architecture, no post-hoc correction"
    )
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to best_model.pt (absolute or relative to level5/)")
    parser.add_argument("--utterance",     type=str, default=None,
                        help="Single utterance to classify")
    parser.add_argument("--input-file",    type=str, default=None,
                        help="CSV or TXT file of utterances")
    parser.add_argument("--utterance-col", type=str, default="utterance",
                        help="Column name when --input-file is a CSV")
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--json",          action="store_true", dest="json_out",
                        help="Output results as JSON")
    parser.add_argument("--output-file",   type=str, default=None,
                        help="Write results to this JSONL file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).parent / args.checkpoint

    model, meta = load_model(str(ckpt_path), device)
    print(
        f"Loaded: {ckpt_path.name}  "
        f"run={meta.get('run_name', '?')}  "
        f"epoch={meta.get('epoch', '?')}  "
        f"val_intent_acc={meta.get('val_intent_acc', 0):.4f}"
    )
    print(f"Rule strengths: {meta.get('rule_strengths', {})}")
    print(f"Hard rules    : {meta.get('hard_rules', '?')}")
    print(f"Device        : {device}\n")

    output_lines = []

    # Single utterance
    if args.utterance:
        preds = run_inference(model, [args.utterance], device)
        print(format_result(preds[0], args.json_out))
        output_lines.append(json.dumps(preds[0]))

    # File mode
    elif args.input_file:
        input_path = Path(args.input_file)
        if input_path.suffix == ".csv":
            import csv
            with open(input_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                utterances = [row[args.utterance_col] for row in reader]
        else:
            utterances = [
                line.strip()
                for line in input_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        print(f"Running inference on {len(utterances)} utterances...")
        all_preds = run_inference(model, utterances, device, batch_size=args.batch_size)

        for pred in all_preds:
            line = json.dumps(pred)
            output_lines.append(line)
            if args.json_out:
                print(line)
            else:
                print(format_result(pred, False))
                print("---")

        print(f"\nTotal predictions: {len(all_preds)}")

    # Interactive REPL
    else:
        print("Interactive mode — type an utterance and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                utt = input("utterance> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not utt or utt.lower() in ("quit", "exit", "q"):
                break
            preds = run_inference(model, [utt], device)
            print(format_result(preds[0], args.json_out))
            print()

    # Write output file
    if args.output_file and output_lines:
        out_path = Path(args.output_file)
        out_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
