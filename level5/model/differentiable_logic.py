# level5/model/differentiable_logic.py
"""
Product t-norm differentiable logic operators.

All operators work on tensors of predicate confidence scores in [0, 1].
They are fully differentiable and can be composed to build arbitrarily
complex rule antecedents that pass gradients back through the network.

Product t-norm semantics:
    AND(a, b)  = a * b               (both must be active)
    OR(a, b)   = a + b - a * b       (at least one active)
    NOT(a)     = 1 - a               (negation)

These satisfy the standard fuzzy logic boundary conditions:
    AND(1, 1) = 1   AND(0, x) = 0
    OR(0, 0)  = 0   OR(1, x)  = 1
    NOT(1) = 0      NOT(0) = 1
"""

import torch


def logic_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Product t-norm AND.
    Both operands must be active for the result to be high.

    Args:
        a, b: tensors of shape [B] with values in [0, 1]
    Returns:
        tensor of shape [B] in [0, 1]
    """
    return a * b


def logic_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Product t-norm OR (probabilistic sum).
    At least one operand being active drives the result high.

    Args:
        a, b: tensors of shape [B] with values in [0, 1]
    Returns:
        tensor of shape [B] in [0, 1]
    """
    return a + b - a * b


def logic_not(a: torch.Tensor) -> torch.Tensor:
    """
    Standard fuzzy NOT.

    Args:
        a: tensor of shape [B] with values in [0, 1]
    Returns:
        tensor of shape [B] in [0, 1]
    """
    return 1.0 - a


def logic_and_multi(*tensors: torch.Tensor) -> torch.Tensor:
    """
    N-ary product t-norm AND. Reduces a sequence of tensors with AND.
    logic_and_multi(a, b, c) == AND(AND(a, b), c)
    """
    result = tensors[0]
    for t in tensors[1:]:
        result = logic_and(result, t)
    return result


def logic_or_multi(*tensors: torch.Tensor) -> torch.Tensor:
    """
    N-ary product t-norm OR. Reduces a sequence of tensors with OR.
    """
    result = tensors[0]
    for t in tensors[1:]:
        result = logic_or(result, t)
    return result
