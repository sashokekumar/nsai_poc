# level5/model/losses.py
"""
Loss functions for Level 5 rule-compiled neural training.

Total loss:
    L = CrossEntropy(intent_logits, intent_targets)
      + pred_weight * BCELoss(predicate_probs, predicate_targets)

No explicit constraint penalty term — symbolic constraints are structural.
They are compiled into the DifferentiableRuleLayer (rule_compiler.py) and
influence the intent_logits via the blend:
    intent_logits = α * rule_scores + (1-α) * trunk_logits

Gradients flow back through the rule layer into the predicate heads and
shared trunk, so the constraint signal is implicit in the intent loss itself.

The predicate BCE loss ensures predicate heads learn accurate binary
classifiers — a necessary condition for the rule activations to be
semantically meaningful.
"""

import torch
import torch.nn as nn


class Level5Loss(nn.Module):
    """
    Combined intent + predicate loss for Level 5 training.

    Args:
        pred_weight : weight on predicate BCE loss (default 0.5).
                      Set to 0.0 to disable predicate supervision
                      (equivalent to Experiment A: rules disabled).
    """

    def __init__(self, pred_weight: float = 0.5):
        super().__init__()
        self.pred_weight = pred_weight
        self.intent_ce  = nn.CrossEntropyLoss()
        self.pred_bce   = nn.BCELoss()

    def forward(
        self,
        intent_logits:    torch.Tensor,   # [B, 4]
        predicate_probs:  torch.Tensor,   # [B, 11]  — sigmoid outputs
        intent_targets:   torch.Tensor,   # [B]       — long
        predicate_targets: torch.Tensor,  # [B, 11]  — float 0/1
    ) -> dict:
        """
        Returns a dict with:
            loss            : total scalar loss (backprop target)
            intent_loss     : float
            predicate_loss  : float
        """
        intent_loss    = self.intent_ce(intent_logits, intent_targets)
        predicate_loss = self.pred_bce(predicate_probs, predicate_targets)

        total = intent_loss + self.pred_weight * predicate_loss

        return {
            "loss":           total,
            "intent_loss":    intent_loss.item(),
            "predicate_loss": predicate_loss.item(),
        }
