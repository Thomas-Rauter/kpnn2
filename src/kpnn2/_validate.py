"""
Argument checks shared by more than one public entry point.

At module level this imports nothing from kpnn2 but ``_errors``,
so the spec and layout modules can import it without a cycle.
"""

from numbers import Integral
from typing import TYPE_CHECKING, TypeGuard

import torch

from ._errors import Kpnn2Error

if TYPE_CHECKING:
    from ._adjacency_spec import AdjacencySpec
    from ._spec import LayeredSpec

_DESCRIBE_LIMIT = 60


def is_integer(value: object) -> TypeGuard[Integral]:
    """
    Return whether ``value`` counts as an integer argument.

    A Python ``int`` or a numpy integer scalar (any
    ``numbers.Integral``) that is not a ``bool``. ``numpy.bool_``
    and floats such as ``2.0`` are not ``Integral`` and fail. A
    caller converts an accepted value with ``int(...)`` before
    keeping, returning, or comparing it, so no numpy integer
    reaches kpnn2's data. Every integer test on a caller's
    argument goes through here.
    """
    return isinstance(value, Integral) and not isinstance(value, bool)


def as_positive_int(
    value: object,
    name: str,
) -> int:
    """
    Return ``value`` as an ``int`` if it is an integer ``>= 1``.

    Raises
    ------
    Kpnn2Error
        If ``value`` is not an integer or is below 1. The message
        names ``name`` and describes ``value``.
    """
    if not is_integer(value) or value < 1:
        raise Kpnn2Error(
            f"'{name}' must be a positive int. Got {describe(value)}."
        )
    return int(value)


def as_bool(
    value: object,
    name: str,
) -> bool:
    """
    Return ``value`` if it is a ``bool``.

    A flag is read by its value, never by truthiness, so a
    stand-in such as ``"False"``, ``0``, ``None``, or a
    ``numpy.bool_`` is rejected rather than guessed at.

    Raises
    ------
    Kpnn2Error
        If ``value`` is not a ``bool``. The message names ``name``
        and describes ``value``.
    """
    if not isinstance(value, bool):
        raise Kpnn2Error(
            f"'{name}' must be True or False. Got {describe(value)}."
        )
    return value


def describe(value: object) -> str:
    """
    Describe a received value in one short line for a message.

    A value with a tuple ``shape`` (tensor, array, frame, series)
    is described by type name, shape, and ``dtype`` when it has
    one, never by its values. Anything else is its ``repr``, cut
    at the first line break and at about 60 characters, followed
    by its type name: ``'no' (str)``, ``1.5 (float)``. ``None``
    is ``None``.
    """
    if value is None:
        return "None"
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple):
        text = f"{type(value).__name__} of shape {tuple(shape)}"
        dtype = getattr(value, "dtype", None)
        if dtype is not None:
            text = f"{text}, dtype {dtype}"
        return text
    full = repr(value)
    text = full.partition("\n")[0]
    if text != full or len(text) > _DESCRIBE_LIMIT:
        text = f"{text[: _DESCRIBE_LIMIT - 3]}..."
    return f"{text} ({type(value).__name__})"


def require_spec(spec: "LayeredSpec | AdjacencySpec") -> None:
    """
    Raise unless ``spec`` is a ``LayeredSpec`` or an ``AdjacencySpec``.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is any other type.
    """
    from ._adjacency_spec import AdjacencySpec
    from ._spec import LayeredSpec

    if not isinstance(
        spec,
        (LayeredSpec, AdjacencySpec),
    ):
        raise Kpnn2Error("'spec' must be a LayeredSpec or an AdjacencySpec.")


def check_layer_input(
    x: object,
    in_features: int,
    owner: str,
) -> None:
    """
    Raise unless ``x`` is a tensor of shape ``(..., in_features)``.

    Shared by the forward of both linear layers; ``owner`` is the
    class name the message starts with. Without this check a wider
    tensor would be read silently by the ``PackedLinear`` gather,
    and a raw torch error would leak from ``MaskedLinear``.

    Raises
    ------
    Kpnn2Error
        If ``x`` is not a ``torch.Tensor``, is 0-dimensional, or
        its last dimension is not ``in_features``.
    """
    if not isinstance(x, torch.Tensor):
        raise Kpnn2Error(f"{owner} input must be a torch.Tensor.")
    if x.ndim < 1:
        raise Kpnn2Error(
            f"{owner} input must have shape (..., in_features) "
            f"with in_features={in_features}. Got a 0-dimensional "
            "tensor."
        )
    if x.shape[-1] != in_features:
        raise Kpnn2Error(
            f"{owner} input must have shape (..., in_features) "
            f"with in_features={in_features}. Got last dimension "
            f"{x.shape[-1]}."
        )
