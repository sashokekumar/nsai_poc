# level4/model/dataset.py
"""
PyTorch Dataset for Level 4 symbolic-loss training.
Loads utterance, intent, entity_type, domain_valid from a CSV file.
Label encoding is consistent with constraint_rules.json vocab order.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

# -------------------------------------------------------
# Label vocabularies — must match constraint_rules.json
# -------------------------------------------------------
INTENT_LABELS = ["investigate", "summarization", "execution", "out_of_scope"]
ENTITY_TYPE_LABELS = ["infrastructure", "service", "metric", "incident", "job", "pipeline", "unknown"]

INTENT_TO_IDX = {label: idx for idx, label in enumerate(INTENT_LABELS)}
ENTITY_TYPE_TO_IDX = {label: idx for idx, label in enumerate(ENTITY_TYPE_LABELS)}

IDX_TO_INTENT = {idx: label for label, idx in INTENT_TO_IDX.items()}
IDX_TO_ENTITY_TYPE = {idx: label for label, idx in ENTITY_TYPE_TO_IDX.items()}


class IntentDataset(Dataset):
    """
    Dataset that returns:
        utterance  : str — raw text, encoded at collation time by the sentence encoder
        intent_idx : int tensor — encoded intent label
        entity_idx : int tensor — encoded entity_type label
        domain_valid: float tensor — 1.0 if in-domain, 0.0 if out-of-domain
    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self._validate(df)

        self.utterances = df["utterance"].astype(str).tolist()
        self.intent_labels = df["intent"].astype(str).tolist()
        self.entity_labels = df["entity_type"].astype(str).tolist()
        self.domain_valid = df["domain_valid"].astype(bool).tolist()

        # Encode to ints once at load time
        self.intent_idxs = [INTENT_TO_IDX[l] for l in self.intent_labels]
        self.entity_idxs = [ENTITY_TYPE_TO_IDX[l] for l in self.entity_labels]

    def _validate(self, df: pd.DataFrame):
        required = {"utterance", "intent", "entity_type", "domain_valid"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        bad_intents = set(df["intent"].unique()) - set(INTENT_LABELS)
        if bad_intents:
            raise ValueError(f"Unknown intents in CSV: {bad_intents}")

        bad_entities = set(df["entity_type"].unique()) - set(ENTITY_TYPE_LABELS)
        if bad_entities:
            raise ValueError(f"Unknown entity_types in CSV: {bad_entities}")

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict:
        return {
            "utterance": self.utterances[idx],
            "intent_idx": torch.tensor(self.intent_idxs[idx], dtype=torch.long),
            "entity_idx": torch.tensor(self.entity_idxs[idx], dtype=torch.long),
            "domain_valid": torch.tensor(float(self.domain_valid[idx]), dtype=torch.float),
        }
