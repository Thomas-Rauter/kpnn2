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


def stored_from_effective(
    constraint: nn.Module | None,
    sample: torch.Tensor,
    live: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Return the tensor to store so ``constraint`` yields ``sample``.

    ``sample`` is the degree-aware draw for the effective weight.
    When ``constraint`` defines ``right_inverse`` (the
    ``torch.nn.utils.parametrize`` convention), the result is
    ``constraint.right_inverse(sample)``; otherwise it is
    ``sample`` itself, unchanged. No inverse is ever guessed.

    Parameters
    ----------
    constraint : torch.nn.Module or None
        The layer's ``constraint``.
    sample : torch.Tensor
        Degree-aware draw, shape of the stored weight. Not
        modified.
    live : torch.Tensor or None, default=None
        Boolean tensor of ``sample``'s shape marking entries that
        can reach the output. Only those must come back finite.
        ``None`` means every entry is live.

    Returns
    -------
    torch.Tensor
        ``sample`` itself when there is no ``right_inverse``, else
        a new tensor of the same shape.

    Raises
    ------
    Kpnn2Error
        If ``right_inverse`` does not return a tensor of
        ``sample``'s shape, or returns a non-finite value on a
        live entry.
    """
    if constraint is None:
        return sample
    right_inverse = getattr(
        constraint,
        "right_inverse",
        None,
    )
    if not callable(right_inverse):
        return sample
    stored = right_inverse(sample.clone())
    if not isinstance(stored, torch.Tensor) or stored.shape != sample.shape:
        raise Kpnn2Error(
            "'constraint.right_inverse' must return a tensor of the "
            "same shape as the weight."
        )
    checked = stored if live is None else stored[live]
    if not bool(torch.isfinite(checked).all()):
        raise Kpnn2Error(
            "'constraint.right_inverse' returned a non-finite value "
            "for a live edge. It receives the degree-aware draw for "
            "the effective weight, uniform in [-bound, bound] and "
            "possibly exactly 0, and must map every such value to a "
            "finite stored value."
        )
    return stored
