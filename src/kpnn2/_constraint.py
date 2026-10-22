"""
Optional per-entry maps on live edge weights.
"""

import torch
from torch import nn

from ._errors import caller_error
from ._validate import describe


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
        raise caller_error(
            "'constraint' must be a torch.nn.Module or None.",
            got=describe(constraint),
            fix="Pass an instance such as nn.Softplus().",
        )
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
    _check_weight_shape(
        sample,
        weight.shape,
        "constraint",
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
    _check_weight_shape(
        stored,
        sample.shape,
        "constraint.right_inverse",
    )
    checked = stored if live is None else stored[live]
    finite = torch.isfinite(checked)
    if not bool(finite.all()):
        count = int((~finite).sum())
        raise caller_error(
            "'constraint.right_inverse' must return a finite value for "
            "every live edge.",
            got=f"{count} non-finite value(s) on live edges",
            fix=(
                "Make it map every value in [-b, b], 0 included, to a "
                "finite value, where b is the layer's init_bound() for "
                "that entry."
            ),
        )
    return stored


def _check_weight_shape(
    result: object,
    shape: torch.Size,
    name: str,
) -> None:
    """
    Raise unless ``result`` is a tensor of the weight's ``shape``.

    ``name`` is the callable that returned ``result``.
    """
    if not isinstance(result, torch.Tensor) or result.shape != shape:
        raise caller_error(
            f"'{name}' must return a tensor of the same shape as the weight.",
            got=describe(result),
            expected=f"shape {tuple(shape)}",
        )
