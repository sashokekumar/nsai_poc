# level5/model/rule_compiler.py
"""
RuleCompiler — reads rule_base.json and builds a differentiable
forward pass expression for each rule using product t-norm operators.

Each rule has:
  - antecedents: a nested logic tree (AND / OR / NOT over predicate names)
  - consequent_intent: which intent this rule fires for
  - rule_strength_init: starting value for the learnable rule_strength param

The rule_strength parameter (one per rule, sigmoid-bounded to [0,1])
controls how much each rule contributes to the final intent logit blend.
It is a trainable nn.Parameter — the network learns how much to trust
each symbolic rule on this dataset.

Usage:
    compiler = RuleCompiler(rule_base_path, predicate_columns, intent_labels)
    rule_activations = compiler.forward(predicate_probs)
    # predicate_probs: [B, n_predicates] in [0, 1]
    # rule_activations: [B, n_rules] in [0, 1]

    intent_rule_scores = compiler.scatter_to_intents(rule_activations)
    # intent_rule_scores: [B, n_intents] — max rule activation per intent
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from level5.model.differentiable_logic import (
    logic_and, logic_or, logic_not,
    logic_and_multi, logic_or_multi,
)


class RuleCompiler(nn.Module):
    """
    Compiles symbolic rules from rule_base.json into a differentiable
    PyTorch module with one learnable rule_strength parameter per rule.

    Args:
        rule_base_path  : path to rule_base.json
        predicate_cols  : ordered list of predicate column names
                          (must match level5_labeled.csv order)
        intent_labels   : ordered list of intent label strings
                          (must match dataset.INTENT_LABELS)
    """

    def __init__(
        self,
        rule_base_path: str,
        predicate_cols: list[str],
        intent_labels: list[str],
    ):
        super().__init__()

        with open(rule_base_path) as f:
            rb = json.load(f)

        self.rules = rb["rules"]
        self.predicate_cols = predicate_cols
        self.predicate_to_idx = {p: i for i, p in enumerate(predicate_cols)}
        self.intent_labels = intent_labels
        self.intent_to_idx = {intent: i for i, intent in enumerate(intent_labels)}
        self.n_rules = len(self.rules)
        self.n_intents = len(intent_labels)

        # Validate all predicates and consequents referenced in rules
        for rule in self.rules:
            self._validate_node(rule["antecedents"], rule["name"])
            consequent = rule["consequent_intent"]
            if consequent not in self.intent_to_idx:
                raise ValueError(
                    f"Rule '{rule['name']}': unknown consequent intent '{consequent}'. "
                    f"Valid: {intent_labels}"
                )

        # Learnable rule_strength per rule — sigmoid-bounded to [0, 1]
        # Initialised from rule_strength_init values in rule_base.json
        init_values = [
            _inverse_sigmoid(rule.get("rule_strength_init", 0.7))
            for rule in self.rules
        ]
        self.rule_strength_logits = nn.Parameter(
            torch.tensor(init_values, dtype=torch.float32)
        )

        # Static mapping: rule index → consequent intent index (not a parameter)
        self.register_buffer(
            "rule_to_intent",
            torch.tensor(
                [self.intent_to_idx[r["consequent_intent"]] for r in self.rules],
                dtype=torch.long,
            ),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_node(self, node: Any, rule_name: str):
        """Recursively validate that all predicate references exist."""
        if "predicate" in node:
            p = node["predicate"]
            if p not in self.predicate_to_idx and p != "NOT":
                raise ValueError(
                    f"Rule '{rule_name}': unknown predicate '{p}'. "
                    f"Valid: {self.predicate_cols}"
                )
            # Handle NOT node with nested operand
            if p == "NOT":
                self._validate_node(node["operand"], rule_name)
        elif "logic" in node:
            for operand in node.get("operands", []):
                self._validate_node(operand, rule_name)
        else:
            raise ValueError(f"Rule '{rule_name}': malformed node: {node}")

    # ------------------------------------------------------------------
    # Logic tree evaluation
    # ------------------------------------------------------------------

    def _eval_node(self, node: Any, predicate_probs: torch.Tensor) -> torch.Tensor:
        """
        Recursively evaluate a logic node against predicate_probs.

        Args:
            node            : dict from rule_base.json antecedents tree
            predicate_probs : [B, n_predicates] — sigmoid activations from
                              the predicate heads
        Returns:
            [B] tensor of activation scores in [0, 1]
        """
        if "logic" in node:
            logic_op = node["logic"].upper()
            operand_tensors = [
                self._eval_node(op, predicate_probs)
                for op in node["operands"]
            ]

            if logic_op == "AND":
                return logic_and_multi(*operand_tensors)
            elif logic_op == "OR":
                return logic_or_multi(*operand_tensors)
            elif logic_op == "NOT":
                # Unary NOT inside a logic node
                assert len(operand_tensors) == 1
                return logic_not(operand_tensors[0])
            else:
                raise ValueError(f"Unknown logic op: {logic_op}")

        elif "predicate" in node:
            p = node["predicate"]
            if p == "NOT":
                # NOT as predicate key with nested operand (alternate schema)
                inner = self._eval_node(node["operand"], predicate_probs)
                return logic_not(inner)
            idx = self.predicate_to_idx[p]
            return predicate_probs[:, idx]          # [B]

        else:
            raise ValueError(f"Malformed logic node: {node}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, predicate_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute rule activation scores for a batch.

        Args:
            predicate_probs : [B, n_predicates] — sigmoid outputs from
                              the predicate heads, values in [0, 1]
        Returns:
            rule_activations : [B, n_rules] — each rule's activation score,
                               weighted by its learnable rule_strength
        """
        B = predicate_probs.shape[0]
        strengths = torch.sigmoid(self.rule_strength_logits)  # [n_rules], in [0,1]

        activations = torch.stack(
            [
                self._eval_node(rule["antecedents"], predicate_probs) * strengths[i]
                for i, rule in enumerate(self.rules)
            ],
            dim=1,
        )  # [B, n_rules]

        return activations

    def scatter_to_intents(self, rule_activations: torch.Tensor) -> torch.Tensor:
        """
        Aggregate rule activations into per-intent scores.
        Multiple rules for the same intent take the max activation.

        Fully differentiable — avoids in-place ops by using a mask matmul
        followed by a max over the rules dimension.

        Args:
            rule_activations : [B, n_rules]
        Returns:
            intent_rule_scores : [B, n_intents]
        """
        # Build rule→intent one-hot mask: [n_rules, n_intents]
        # mask[r, i] = 1 if rule r fires for intent i, else 0
        mask = torch.zeros(
            self.n_rules, self.n_intents,
            device=rule_activations.device,
            dtype=rule_activations.dtype,
        )
        for r, i in enumerate(self.rule_to_intent.tolist()):
            mask[r, i] = 1.0

        # [B, n_rules, 1] * [1, n_rules, n_intents] → [B, n_rules, n_intents]
        # For positions where mask=0, contribution is 0 (rule doesn't apply).
        per_rule_intent = rule_activations.unsqueeze(-1) * mask.unsqueeze(0)

        # Max over rules dimension → [B, n_intents]
        intent_scores, _ = per_rule_intent.max(dim=1)
        return intent_scores

    def rule_strength_dict(self) -> dict:
        """Return current learned rule strengths as a plain dict (for logging)."""
        strengths = torch.sigmoid(self.rule_strength_logits).detach().cpu()
        return {
            rule["name"]: round(float(strengths[i]), 4)
            for i, rule in enumerate(self.rules)
        }


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _inverse_sigmoid(p: float) -> float:
    """Compute logit(p) = log(p / (1-p)) for initialising rule_strength_logits."""
    import math
    p = max(min(p, 0.9999), 0.0001)
    return math.log(p / (1.0 - p))
