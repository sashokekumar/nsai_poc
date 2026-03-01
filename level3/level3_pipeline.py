# level3/level3_pipeline.py

import torch

from level3.neural_intent_model import NeuralIntentModel
from level3.symbol_emitter import SymbolEmitter
from level3.rule_engine import RuleEngine


class Level3Pipeline:
    def __init__(self, vocab_size, embed_dim, num_classes):
        self.model = NeuralIntentModel(vocab_size, embed_dim, num_classes)
        self.symbol_emitter = SymbolEmitter()
        self.rule_engine = RuleEngine()

    def load_model_state(self, state_dict_path: str):
        self.model.load_state_dict(torch.load(state_dict_path))
        self.model.eval()

    def process(self, input_ids: torch.Tensor):
        """
        input_ids: Tensor shape (1, seq_len)
        """

        # 1️⃣ Neural perception
        class_index = self.model.predict(input_ids).item()

        # 2️⃣ Symbol emission
        symbol = self.symbol_emitter.emit(class_index)

        # 3️⃣ Symbolic reasoning
        decision = self.rule_engine.route(symbol)

        return {
            "symbol": symbol.intent,
            "decision": decision,
        }