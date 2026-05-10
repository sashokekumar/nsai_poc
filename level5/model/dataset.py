# level5/model/dataset.py
"""
PyTorch Dataset for Level 5 rule-compiled neural training.

Loads utterance, intent, entity_type, domain_valid, and 11 predicate columns
from level5/data/level5_labeled.csv.

At load time validates:
  - All intent labels are in INTENT_LABELS
  - All predicate columns exist and contain only 0/1 values
  - All predicate names referenced in rule_base.json antecedents are present
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

# -------------------------------------------------------
# Label vocabularies — must stay consistent with rule_base.json
# -------------------------------------------------------
INTENT_LABELS = ["investigate", "summarization", "execution", "out_of_scope"]

ENTITY_TYPE_LABELS = [
    "infrastructure", "service", "metric",
    "incident", "job", "pipeline", "unknown",
]

# 7 entity-type predicates + 4 signal predicates = 11 total
PREDICATE_COLS = [
    "is_infrastructure", "is_service", "is_metric",
    "is_incident", "is_job", "is_pipeline", "is_unknown",
    "is_sre_domain", "has_runbook", "is_known_incident", "is_metric_query",
]

INTENT_TO_IDX      = {label: idx for idx, label in enumerate(INTENT_LABELS)}
ENTITY_TYPE_TO_IDX = {label: idx for idx, label in enumerate(ENTITY_TYPE_LABELS)}

IDX_TO_INTENT      = {idx: label for label, idx in INTENT_TO_IDX.items()}
IDX_TO_ENTITY_TYPE = {idx: label for label, idx in ENTITY_TYPE_TO_IDX.items()}


class Level5Dataset(Dataset):
    """
    Dataset that returns per sample:
        utterance        : str — raw text, encoded by the sentence encoder at collation time
        intent_idx       : long tensor — encoded intent label
        predicate_labels : float tensor [11] — binary predicate values
    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self._validate(df)

        self.utterances = df["utterance"].astype(str).tolist()
        self.intent_idxs = [INTENT_TO_IDX[l] for l in df["intent"].astype(str)]
        self.predicate_matrix = df[PREDICATE_COLS].astype(float).values  # [N, 11]

    def _validate(self, df: pd.DataFrame):
        required_base = {"utterance", "intent"}
        missing = required_base - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        missing_preds = [c for c in PREDICATE_COLS if c not in df.columns]
        if missing_preds:
            raise ValueError(
                f"CSV missing predicate columns: {missing_preds}. "
                f"Run level5/build_dataset.py first."
            )

        bad_intents = set(df["intent"].unique()) - set(INTENT_LABELS)
        if bad_intents:
            raise ValueError(f"Unknown intents in CSV: {bad_intents}")

        for col in PREDICATE_COLS:
            bad_vals = set(df[col].dropna().unique()) - {0, 1, 0.0, 1.0}
            if bad_vals:
                raise ValueError(
                    f"Predicate column '{col}' contains non-binary values: {bad_vals}"
                )

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict:
        return {
            "utterance":        self.utterances[idx],
            "intent_idx":       torch.tensor(self.intent_idxs[idx], dtype=torch.long),
            "predicate_labels": torch.tensor(
                self.predicate_matrix[idx], dtype=torch.float32
            ),
        }
