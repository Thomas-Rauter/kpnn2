"""Optional ``torch.Generator`` for isolated parameter init."""

from collections.abc import Callable
from typing import TypeVar

import torch

from ._errors import Kpnn2Error

T = TypeVar("T")


def as_generator(generator: object) -> torch.Generator | None:
    """
    Return ``None`` or a ``torch.Generator``; reject other types.

    Parameters
    ----------
    generator : torch.Generator or None
        Constructor or ``reset_parameters`` value.

    Returns
    -------
    torch.Generator or None
        ``generator`` unchanged when it is a
        ``torch.Generator`` or ``None``.

    Raises
    ------
    Kpnn2Error
        If ``generator`` is neither a ``torch.Generator``
        nor ``None``.
    """
    if generator is None:
        return None
    if not isinstance(generator, torch.Generator):
        raise Kpnn2Error("'generator' must be a torch.Generator or None.")
    return generator


def run_preserving_default_rng(fn: Callable[[], T]) -> T:
    """
    Run ``fn`` and restore the default CPU generator.

    ``nn.Linear`` constructors kaiming-init on the global
    stream. When a caller-supplied ``generator`` will
    overwrite those weights, wrap the constructors so they
    do not shift later global draws.

    Parameters
    ----------
    fn : callable
        Zero-argument callable.

    Returns
    -------
    object
        ``fn()``.
    """
    state = torch.get_rng_state()
    try:
        return fn()
    finally:
        torch.set_rng_state(state)
