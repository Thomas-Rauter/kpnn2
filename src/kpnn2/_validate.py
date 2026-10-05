"""
Argument checks shared by more than one public entry point.

At module level this imports nothing from kpnn2 but ``_errors``,
so the spec and layout modules can import it without a cycle.
"""

from typing import TYPE_CHECKING, TypeGuard

from ._errors import Kpnn2Error

if TYPE_CHECKING:
    from ._adjacency_spec import AdjacencySpec
    from ._spec import LayeredSpec

_DESCRIBE_LIMIT = 60


def is_integer(value: object) -> TypeGuard[int]:
    """
    Return whether ``value`` counts as an integer argument.

    An ``int`` that is not a ``bool``. Every integer test on a
    caller's argument goes through here.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def as_positive_int(
    value: object,
    name: str,
) -> int:
    """
    Return ``value`` if it is an integer ``>= 1``.

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
