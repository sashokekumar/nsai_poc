# level4/infer.py
"""
Clean inference for Level 4.

Loads a saved checkpoint and runs the neural model on one or more utterances.
No symbolic guards, reasoners, planners, or post-processing — model output only.

Usage:
    # Single utterance
    python -m level4.infer --checkpoint saved_models/baseline/best_model.pt \
        --utterance "restart the payment-service deployment"

    # File of utterances (one per line)
    python -m level4.infer --checkpoint saved_models/level4_lam0_5/best_model.pt \
        --input-file data/test.csv --utterance-col utterance

    # Interactive REPL
    python -m level4.infer --checkpoint saved_models/baseline/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from level4.model.neural_intent_model import Level4IntentModel


def load_model(checkpoint_path: str, device: torch.device) -> Level4IntentModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Level4IntentModel()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta


def format_result(pred: dict, json_out: bool) -> str:
    if json_out:
        return json.dumps(pred, indent=2)
    return (
        f"utterance   : {pred['utterance']}\n"
        f"intent      : {pred['intent']}  (conf={pred['intent_conf']:.3f})\n"
        f"entity_type : {pred['entity_type']}  (conf={pred['entity_conf']:.3f})\n"
        f"domain_valid: {pred['domain_valid']}  (prob={pred['domain_prob']:.3f})"
    )


def main():
    parser = argparse.ArgumentParser(description="Level 4 inference — neural model, no symbolic post-processing")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt checkpoint")
    parser.add_argument("--utterance",  type=str, default=None, help="Single utterance to classify")
    parser.add_argument("--input-file", type=str, default=None, help="CSV/TXT file of utterances")
    parser.add_argument("--utterance-col", type=str, default="utterance", help="Column name when using --input-file with a CSV")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for file input")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Output results as JSON")
    parser.add_argument("--output-file", type=str, default=None, help="Write results to this file (JSONL)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(__file__).parent / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)

    model, meta = load_model(str(ckpt_path), device)
    print(f"Loaded checkpoint: {ckpt_path.name}  (lam={meta.get('lam')}, epoch={meta.get('epoch')}, val_intent_acc={meta.get('val_intent_acc', '?'):.4f})")
    print(f"Device: {device}\n")

    output_lines = []

    # ---- Single utterance mode ----
    if args.utterance:
        preds = model.predict([args.utterance], device)
        print(format_result(preds[0], args.json_out))
        output_lines.append(json.dumps(preds[0]))

    # ---- File mode ----
    elif args.input_file:
        input_path = Path(args.input_file)
        if input_path.suffix == ".csv":
            import csv
            with open(input_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                utterances = [row[args.utterance_col] for row in reader]
        else:
            utterances = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        print(f"Running inference on {len(utterances)} utterances...")
        all_preds = []
        for start in range(0, len(utterances), args.batch_size):
            batch = utterances[start: start + args.batch_size]
            all_preds.extend(model.predict(batch, device))

        for pred in all_preds:
            line = json.dumps(pred)
            output_lines.append(line)
            if args.json_out:
                print(line)
            else:
                print(format_result(pred, False))
                print("---")

        print(f"\nTotal predictions: {len(all_preds)}")

    # ---- Interactive REPL mode ----
    else:
        print("Interactive mode — type an utterance and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                utt = input("utterance> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not utt or utt.lower() in ("quit", "exit", "q"):
                break
            preds = model.predict([utt], device)
            print(format_result(preds[0], args.json_out))
            print()
            output_lines.append(json.dumps(preds[0]))

    # ---- Write output file ----
    if args.output_file and output_lines:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
