"""
Argument checks shared by more than one public entry point.

At module level this imports nothing from kpnn2 but ``_errors``,
so the spec and layout modules can import it without a cycle.
"""

from collections.abc import Iterable, Mapping, Set
from numbers import Integral
from typing import TYPE_CHECKING, TypeGuard

import numpy as np
import torch

from ._errors import caller_error

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
        raise caller_error(
            f"'{name}' must be a positive int.",
            got=describe(value),
        )
    return int(value)


def as_layer_index(
    value: object,
    n_layers: int,
) -> int:
    """
    Return ``value`` as an ``int`` if it is a layer of the spec.

    A layer is a 0-based index into ``spec.layer_nodes``.
    Negative indices are rejected rather than counted from the
    end, so a layer number always means the same depth.

    Raises
    ------
    Kpnn2Error
        If ``value`` is not an integer, or is outside
        ``[0, n_layers)``. The message names ``'layer'`` and
        reports ``value``.
    """
    if not is_integer(value):
        raise caller_error(
            "'layer' must be an int.",
            got=describe(value),
        )
    layer = int(value)
    if layer < 0 or layer >= n_layers:
        if layer < 0:
            fix = (
                f"Count from 0: the last layer is {n_layers - 1}. "
                "Negative indices do not count from the end."
            )
        else:
            fix = None
        raise caller_error(
            f"'layer' must be in range [0, {n_layers}).",
            got=describe(value),
            fix=fix,
        )
    return layer


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
        raise caller_error(
            f"'{name}' must be True or False.",
            got=describe(value),
        )
    return value


def describe(value: object) -> str:
    """
    Describe a received value in one short line for a message.

    A value with a tuple ``shape`` (tensor, array, frame, series)
    is described by type name, shape, and ``dtype`` when it has
    one, never by its values. A list, tuple, dict, set, or
    frozenset that holds such a value is described by type name
    and length, so its ``repr`` cannot print those values
    either: ``list of length 2``. Anything else is its ``repr``,
    cut at the first line break and at about 60 characters,
    followed by its type name: ``'no' (str)``, ``1.5 (float)``.
    A numpy scalar has shape ``()`` but holds one value, not an
    array, so it is described by that value in the same way, with
    its type name after ``numpy.``: ``1.5 (numpy.float64)``,
    ``True (numpy.bool)``. Without the module, a numpy ``bool``
    would read like a Python ``bool`` in a message that rejects
    it. A 0-dimensional array or tensor still gets the shape form.
    ``None`` is ``None``.
    """
    if value is None:
        return "None"
    type_name = type(value).__name__
    if isinstance(value, np.generic):
        full = _numpy_scalar_text(value)
        type_name = f"numpy.{type_name}"
    else:
        shape = getattr(value, "shape", None)
        if isinstance(shape, tuple):
            text = f"{type(value).__name__} of shape {tuple(shape)}"
            dtype = getattr(value, "dtype", None)
            if dtype is not None:
                text = f"{text}, dtype {dtype}"
            return text
        if isinstance(value, (list, tuple, dict, set, frozenset)):
            items = value.values() if isinstance(value, dict) else value
            if any(
                isinstance(getattr(item, "shape", None), tuple)
                and not isinstance(item, np.generic)
                for item in items
            ):
                return f"{type(value).__name__} of length {len(value)}"
        full = repr(value)
    text = full.partition("\n")[0]
    if text != full or len(text) > _DESCRIBE_LIMIT:
        text = f"{text[: _DESCRIBE_LIMIT - 3]}..."
    return f"{text} ({type_name})"


def _numpy_scalar_text(value: np.generic) -> str:
    """
    Return the text of one numpy scalar as a message shows it.

    A number or ``bool`` uses numpy's own shortest form, so
    ``np.float32(0.1)`` reads ``0.1``. Anything else, such as
    ``np.str_``, shows the ``repr`` of the Python value, quotes
    included: ``'no'``.
    """
    if isinstance(value, (np.number, np.bool_)):
        return str(value)
    return repr(value.item())


def reject_unordered(
    value: object,
    name: str,
    expected: str,
) -> None:
    """
    Raise if ``value`` is a set or a mapping where order matters.

    The items of ``value`` are matched to positions: an index
    pair, a tensor axis, a feature column. A set has no order
    and a mapping iterates its keys, so either would be read in
    an arbitrary or unintended order instead of failing.

    Raises
    ------
    Kpnn2Error
        If ``value`` is a ``collections.abc.Set`` or
        ``collections.abc.Mapping``. The message says ``name``
        must be ``expected``, names the kind, and describes
        ``value``.
    """
    if isinstance(value, Mapping):
        kind = "a mapping"
    elif isinstance(value, Set):
        kind = "a set"
    else:
        return
    raise caller_error(
        f"'{name}' must be {expected}, not {kind}: its items are "
        "matched by position.",
        got=describe(value),
        fix="Pass a list or tuple in the intended order.",
    )


def as_name_list(
    names: object,
    argument: str,
    noun: str,
    *,
    container: str = "sequence",
    ndim_fix: str | None = None,
) -> list[str]:
    """
    Return ``names`` as a list of ``str`` labels, or raise.

    Each item is a label matched by position, so a value that
    would iterate into the wrong labels is rejected rather than
    read: a string into characters, a matrix into rows, a
    mapping into its keys. Labels go through ``str(...)``, the
    conversion edgelist node names go through, so an integer
    label ``1`` becomes ``"1"``.

    Parameters
    ----------
    names : object
        Labels in the caller's order: a list, a tuple, a
        one-dimensional numpy array, a ``pandas.Index``, or any
        other iterable that is not rejected below. It is not
        modified.
    argument : str
        Name of the caller's argument that the messages quote,
        such as ``"names"``.
    noun : str
        What the labels are, in the plural, such as
        ``"feature names"``.
    container : str, default "sequence"
        What the headlines call ``names``: ``"sequence"`` where
        order matters, ``"collection"`` where it does not.
    ndim_fix : str, optional
        The ``Fix`` for an ``ndim`` other than 1, for a caller
        who holds a matrix that the result indexes. Without it,
        that message has no ``Fix``.

    Returns
    -------
    list of str
        ``str(item)`` for each item of ``names``, in order.

    Raises
    ------
    Kpnn2Error
        If ``names`` is a ``torch.Tensor``, a ``str``, ``bytes``,
        a set or a mapping, has an ``ndim`` other than 1, or is
        not iterable, checked in that order. The message quotes
        ``argument``, names ``noun``, and describes ``names``.
    """
    rule = f"'{argument}' must be a {container} of {noun}"
    if isinstance(names, torch.Tensor):
        raise caller_error(
            f"{rule}, not a tensor.",
            got=describe(names),
        )
    if isinstance(names, str):
        raise caller_error(
            f"{rule}, not a single string.",
            got=describe(names),
            fix="Wrap it in a list.",
        )
    if isinstance(names, bytes):
        raise caller_error(
            f"{rule}, not bytes.",
            got=describe(names),
        )
    reject_unordered(
        names,
        argument,
        f"a {container} of {noun}",
    )
    ndim = getattr(names, "ndim", None)
    if ndim is not None and ndim != 1:
        raise caller_error(
            f"'{argument}' must be a 1-dimensional {container} of {noun}.",
            got=describe(names),
            fix=ndim_fix,
        )
    if not isinstance(names, Iterable):
        raise caller_error(
            f"{rule}.",
            got=describe(names),
        )
    return [str(item) for item in names]


def require_node_mapping(
    value: object,
    name: str,
) -> None:
    """
    Raise unless ``value`` is a mapping of node name to int.

    Only the container is checked here: ``widths`` and ``ranks``
    each check their own keys and values.

    Raises
    ------
    Kpnn2Error
        If ``value`` is not a ``collections.abc.Mapping``. The
        message names ``name`` and describes ``value``.
    """
    if not isinstance(value, Mapping):
        raise caller_error(
            f"'{name}' must be a mapping of node name to int.",
            got=describe(value),
        )


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
        raise caller_error(
            "'spec' must be a LayeredSpec or an AdjacencySpec.",
            got=describe(spec),
            fix="Build one with parse_layered() or parse_adjacency(), or "
            "load a saved one with LayeredSpec.from_dict() or "
            "AdjacencySpec.from_dict().",
        )


def require_floating(
    tensor: torch.Tensor,
    what: str,
) -> None:
    """
    Raise unless ``tensor`` has a floating-point dtype.

    Layer forwards cast their input to the parameter dtype, so
    without this check an integer, ``bool``, or complex tensor
    would be converted silently, and a complex one would lose
    its imaginary part. ``what`` names the input at the start of
    the message.

    Raises
    ------
    Kpnn2Error
        If ``tensor`` is not floating point. The message gives its
        shape and dtype, never its values.
    """
    if not tensor.is_floating_point():
        raise caller_error(
            f"{what} must be a floating-point tensor.",
            got=describe(tensor),
            fix="Convert it first, for example with .float().",
        )


def check_layer_input(
    x: object,
    in_features: int,
    owner: str,
) -> None:
    """
    Raise unless ``x`` is a floating tensor of shape ``(..., in_features)``.

    Shared by the forward of both linear layers; ``owner`` is the
    class name the message starts with. Without this check a wider
    tensor would be read silently by the ``PackedLinear`` gather,
    and a raw torch error would leak from ``MaskedLinear``.

    Raises
    ------
    Kpnn2Error
        If ``x`` is not a ``torch.Tensor``, is not floating point,
        is 0-dimensional, or its last dimension is not
        ``in_features``.
    """
    if not isinstance(x, torch.Tensor):
        raise caller_error(
            f"{owner} input must be a torch.Tensor.",
            got=describe(x),
        )
    require_floating(
        x,
        f"{owner} input",
    )
    if x.ndim < 1 or x.shape[-1] != in_features:
        raise caller_error(
            f"{owner} input must have shape (..., in_features) "
            f"with in_features={in_features}.",
            got=describe(x),
        )
