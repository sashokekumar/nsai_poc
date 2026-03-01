# level3/neural_intent_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralIntentModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        """
        input_ids: (batch_size, seq_len)
        """
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)  # simple average pooling
        logits = self.fc(pooled)
        return logits

    def predict(self, input_ids):
        """
        Returns discrete class index.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            predicted = torch.argmax(logits, dim=-1)
        return predicted