# level4/model/neural_intent_model.py
"""
Level 4 neural model: all-MiniLM-L6-v2 encoder (frozen) + shared dense layer
+ three classification heads (intent, entity_type, domain_valid).

Architecture:
    utterance
        ↓
    SentenceTransformer encoder  [384-dim, frozen]
        ↓
    Shared dense layer           [384 → 256, ReLU, Dropout]
        ↓
    ┌───────────────┬────────────────┬──────────────────┐
    intent head     entity_type head  domain_valid head
    [256 → 4]       [256 → 7]         [256 → 1]
    (CrossEntropy)  (CrossEntropy)    (BCEWithLogits)

At inference: utterance → model.predict() → intent, entity_type, domain_valid
No symbolic post-processing.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from level4.model.dataset import (
    INTENT_LABELS, ENTITY_TYPE_LABELS,
    IDX_TO_INTENT, IDX_TO_ENTITY_TYPE,
)

NUM_INTENTS = len(INTENT_LABELS)        # 4
NUM_ENTITY_TYPES = len(ENTITY_TYPE_LABELS)  # 7
ENCODER_DIM = 384   # all-MiniLM-L6-v2 output dimension
HIDDEN_DIM = 256


class Level4IntentModel(nn.Module):
    """
    Frozen sentence encoder + trainable MLP classification heads.
    """

    def __init__(self, encoder_name: str = "all-MiniLM-L6-v2", dropout: float = 0.2):
        super().__init__()

        # Frozen sentence encoder — weights not updated during training
        self._encoder = SentenceTransformer(encoder_name)
        for param in self._encoder.parameters():
            param.requires_grad = False

        # Shared dense trunk
        self.shared = nn.Sequential(
            nn.Linear(ENCODER_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification heads
        self.intent_head = nn.Linear(HIDDEN_DIM, NUM_INTENTS)
        self.entity_head = nn.Linear(HIDDEN_DIM, NUM_ENTITY_TYPES)
        self.domain_head = nn.Linear(HIDDEN_DIM, 1)   # binary → BCEWithLogits

    def encode(self, utterances: list[str], device: torch.device) -> torch.Tensor:
        """Encode a batch of utterances → [B, 384] tensor."""
        embeddings = self._encoder.encode(
            utterances,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device,
        )
        # .clone() exits inference mode so the tensor can flow through
        # autograd-tracked layers (shared dense + heads) during training
        return embeddings.to(device).clone()

    def forward(self, utterances: list[str], device: torch.device) -> dict:
        """
        Returns logits for all three heads.
        Input:  list of raw utterance strings
        Output: dict with keys intent_logits, entity_logits, domain_logits
        """
        embeddings = self.encode(utterances, device)          # [B, 384]
        hidden = self.shared(embeddings)                       # [B, 256]

        return {
            "intent_logits": self.intent_head(hidden),         # [B, 4]
            "entity_logits": self.entity_head(hidden),         # [B, 7]
            "domain_logits": self.domain_head(hidden).squeeze(-1),  # [B]
        }

    def predict(self, utterances: list[str], device: torch.device = None) -> list[dict]:
        """
        Clean inference — no symbolic post-processing.
        Returns a list of prediction dicts per utterance.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        with torch.no_grad():
            out = self.forward(utterances, device)

            intent_probs = torch.softmax(out["intent_logits"], dim=-1)
            entity_probs = torch.softmax(out["entity_logits"], dim=-1)
            domain_probs = torch.sigmoid(out["domain_logits"])

            intent_idxs = intent_probs.argmax(dim=-1).tolist()
            entity_idxs = entity_probs.argmax(dim=-1).tolist()

        results = []
        for i, utt in enumerate(utterances):
            results.append({
                "utterance": utt,
                "intent": IDX_TO_INTENT[intent_idxs[i]],
                "entity_type": IDX_TO_ENTITY_TYPE[entity_idxs[i]],
                "domain_valid": bool(domain_probs[i].item() >= 0.5),
                "intent_conf": float(intent_probs[i].max().item()),
                "entity_conf": float(entity_probs[i].max().item()),
                "domain_prob": float(domain_probs[i].item()),
            })
        return results
