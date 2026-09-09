"""
Optional per-entry maps on live edge weights.
"""

import torch
from torch import nn

from ._errors import Kpnn2Error


def as_constraint(constraint: object) -> nn.Module | None:
    """
    Return ``constraint`` if it is an ``nn.Module`` or ``None``.

    Parameters
    ----------
    constraint : object
        The constructor value. Must be an ``nn.Module`` or
        ``None``.

    Returns
    -------
    torch.nn.Module or None
        ``constraint`` unchanged when valid.

    Raises
    ------
    Kpnn2Error
        If ``constraint`` is neither an ``nn.Module`` nor
        ``None``.
    """
    if constraint is None:
        return None
    if not isinstance(constraint, nn.Module):
        raise Kpnn2Error("'constraint' must be a torch.nn.Module or None.")
    return constraint


def check_constraint_shape(
    constraint: nn.Module,
    weight: torch.Tensor,
) -> None:
    """
    Raise if ``constraint`` does not preserve ``weight``'s shape.

    Parameters
    ----------
    constraint : torch.nn.Module
        The per-entry map to probe with a zeros tensor of
        ``weight``'s shape.
    weight : torch.Tensor
        The unconstrained weight whose shape must be preserved.

    Raises
    ------
    Kpnn2Error
        If ``constraint`` does not return a tensor of the same
        shape as ``weight``.
    """
    with torch.no_grad():
        sample = constraint(
            torch.zeros(
                weight.shape,
                dtype=weight.dtype,
                device=weight.device,
            )
        )
    if not isinstance(sample, torch.Tensor):
        raise Kpnn2Error(
            "'constraint' must return a tensor of the same shape as the weight."
        )
    if sample.shape != weight.shape:
        raise Kpnn2Error(
            "'constraint' must return a tensor of the same shape as the weight."
        )
