# level4/data_audit.py
"""
Pre-training dataset sanity check for Level 4.

Prints:
  - Row counts by intent, entity_type, domain_valid
  - Cross-tab: intent x entity_type
  - Cross-tab: intent x domain_valid
  - Constraint violations present in labels (per constraint_rules.json)

Usage:
    python -m level4.data_audit                     # audits labeled_clean.csv
    python -m level4.data_audit --split train       # audits train.csv
    python -m level4.data_audit --split test        # audits test.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR    = Path(__file__).parent / "data"
ONTOLOGY_DIR = Path(__file__).parent / "ontology"


def load_df(split: str) -> pd.DataFrame:
    file_map = {
        "clean": DATA_DIR / "labeled_clean.csv",
        "train": DATA_DIR / "train.csv",
        "test":  DATA_DIR / "test.csv",
    }
    path = file_map[split]
    df = pd.read_csv(path)
    print(f"Loaded: {path.name}  ({len(df)} rows)\n")
    return df


def section(title: str):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def audit_distributions(df: pd.DataFrame):
    section("Row counts by intent")
    print(df["intent"].value_counts().to_string())

    section("Row counts by entity_type")
    print(df["entity_type"].value_counts().to_string())

    section("Row counts by domain_valid")
    print(df["domain_valid"].value_counts().to_string())


def audit_crosstabs(df: pd.DataFrame):
    section("Cross-tab: intent x entity_type")
    ct = pd.crosstab(df["intent"], df["entity_type"], margins=True)
    print(ct.to_string())

    section("Cross-tab: intent x domain_valid")
    ct2 = pd.crosstab(df["intent"], df["domain_valid"], margins=True)
    print(ct2.to_string())


def audit_constraint_violations(df: pd.DataFrame):
    rules_path = ONTOLOGY_DIR / "constraint_rules.json"
    if not rules_path.exists():
        print(f"[WARN] constraint_rules.json not found at {rules_path}, skipping violation check")
        return

    with open(rules_path) as f:
        rules_cfg = json.load(f)

    disallowed = rules_cfg.get("disallowed_pairs", [])

    section("Constraint violations in labels")
    total_violations = 0
    for rule in disallowed:
        intent      = rule["intent"]
        entity_type = rule["entity_type"]
        weight      = rule.get("weight", 1.0)
        tier        = rule.get("tier", "?")
        count = ((df["intent"] == intent) & (df["entity_type"] == entity_type)).sum()
        if count > 0:
            print(f"  [{tier}] intent={intent:20s} entity_type={entity_type:15s}  w={weight}  violations={count}")
            total_violations += count

    if total_violations == 0:
        print("  No violations found. Labels are constraint-clean.")
    else:
        pct = 100.0 * total_violations / len(df)
        print(f"\n  Total violations: {total_violations} / {len(df)} rows ({pct:.1f}%)")
        print()
        print("  NOTE: violations in labels = noise / annotation ambiguity.")
        print("  The constraint loss penalises the MODEL for predicting these pairs,")
        print("  but ground-truth labels are not changed based on violations alone.")


def main():
    parser = argparse.ArgumentParser(description="Level 4 dataset audit")
    parser.add_argument(
        "--split",
        choices=["clean", "train", "test"],
        default="clean",
        help="Which dataset file to audit (default: labeled_clean.csv)",
    )
    args = parser.parse_args()

    df = load_df(args.split)
    audit_distributions(df)
    print()
    audit_crosstabs(df)
    print()
    audit_constraint_violations(df)


if __name__ == "__main__":
    main()
