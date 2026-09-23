"""
Align feature names to ``spec.input_nodes``.
"""

from collections.abc import Iterable, Mapping, Set

import numpy as np
import pandas as pd
import torch

from ._adjacency_spec import AdjacencySpec
from ._errors import Kpnn2Error
from ._layout import DEFAULT_NODE_WIDTH, build_layout
from ._spec import LayeredSpec

_TENSOR_NOT_ACCEPTED_MSG = (
    "'names' is a tensor; pass the feature names that "
    "label that axis. A tensor whose columns already follow "
    "spec.input_nodes goes straight to the model."
)
_DATAFRAME_NOT_ACCEPTED_MSG = (
    "'names' is a DataFrame; pass the feature names "
    "(for example data.columns). Apply the returned index "
    "on the matrix yourself."
)
_STRING_NOT_ACCEPTED_MSG = (
    "'names' is a string; pass a sequence of feature names."
)
_BYTES_NOT_ACCEPTED_MSG = "'names' is bytes; pass a sequence of feature names."
_MAPPING_NOT_ACCEPTED_MSG = (
    "'names' must be a sequence of feature names, not a mapping."
)
_SET_NOT_ACCEPTED_MSG = (
    "'names' must be a sequence of feature names, not a set."
)
_ANNDATA_NOT_ACCEPTED_MSG = (
    "'names' looks like AnnData; pass the .var_names "
    "and apply the index to .X yourself."
)
_NOT_1D_MSG = (
    "'names' must be one-dimensional; a matrix is not a "
    "name list. Pass the feature names and index the "
    "matrix yourself."
)
_UNSUPPORTED_TYPE_MSG = (
    "Unsupported names type. Expected a sequence of feature names."
)


def align_inputs(
    names: object,
    spec: LayeredSpec | AdjacencySpec,
) -> np.ndarray:
    """
    Return an integer index that orders features to spec inputs.

    Feature-table labels rarely match the input-node order the
    parsed edgelist fixes. This returns a 1-D ``int64`` index
    into the caller's feature axis so that axis can be gathered
    into that order. It does not take the matrix, copy sample
    rows, or densify. Apply the index on whatever holds X:
    ``X[:, col]`` for numpy, scipy CSR/CSC, AnnData ``.X`` in
    those formats, or a tensor; ``df.to_numpy()[:, col]`` for
    a DataFrame. COO-style sparse layouts do not support
    integer column indexing; convert first. For a
    ``LayeredSpec``, a node with width greater than 1 repeats
    its column index across those units, so the length is
    ``spec.layer_dims[0]``. On CSR/CSC that repeat copies
    those columns and stays sparse; it is not a view. For an
    ``AdjacencySpec`` the same repeat uses ``node_widths``, so
    the length is ``len(spec.input_index)``.
    Call it after parsing, once, instead of hand-ordering
    columns. DataFrames, tensors, and matrices are rejected.

    Parameters
    ----------
    names : sequence of labels
        Feature-axis labels, in the order they currently sit on
        the matrix: ``df.columns``, ``adata.var_names``, a
        one-dimensional numpy array, or a list. The annotated
        type is ``object`` so a DataFrame or tensor is not
        treated as valid names; those are rejected at runtime.
        Labels are matched after ``str(...)``, the conversion
        edgelist node names go through, so an integer label
        ``1`` matches node ``"1"``. Extra names are ignored.
        Neither ``names`` nor the matrix is modified.
    spec : LayeredSpec or AdjacencySpec
        Parsed edgelist whose ``input_nodes`` — the in-degree-0
        nodes, alphabetically sorted — fix both the required
        names and their order. A ``LayeredSpec`` also uses
        ``layer_widths[0]``, an ``AdjacencySpec`` its
        ``node_widths``, to repeat columns.

    Returns
    -------
    numpy.ndarray of dtype int64, shape (width,)
        Positions into ``names``. Shape is
        ``(spec.layer_dims[0],)`` for a ``LayeredSpec`` and
        ``(len(spec.input_index),)`` for an ``AdjacencySpec``.
        Index the caller's feature axis with it. An empty
        ``names`` sequence is allowed when ``input_nodes`` is
        empty; otherwise missing names raise.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``; ``names`` is a DataFrame, tensor,
        string, bytes, mapping, set, AnnData-like object, or a
        matrix (ndim ≠ 1); ``names`` is not a sequence of
        labels; required names are missing or duplicated
        (including after ``str`` conversion; the message names
        the unique duplicated labels, sorted, comma-separated).

    See Also
    --------
    parse_layered : Builds the ``LayeredSpec`` whose
        ``input_nodes`` set this column order.
    parse_adjacency : Builds the ``AdjacencySpec`` for the packed
        layout, where the result needs scattering first.
    gather_hop_inputs : Assembles a later hop's input; layer 0
        comes from indexing with this result instead.

    Notes
    -----
    PyTorch never sees feature names. Passing a hand-stacked
    array into the model can silently wire the wrong features if
    the column order differs, which is what this guards against.
    A tensor whose columns already follow ``spec.input_nodes``
    needs no alignment and goes straight to the model. AnnData,
    numpy matrices, and scipy sparse matrices are not accepted
    as ``names``; pass the labels that sit on that axis and
    apply the index on the matrix. scipy CSR/CSC and AnnData
    ``.X`` in those formats stay sparse under ``X[:, col]``
    until the caller densifies one row block. COO and similar
    layouts cannot be indexed that way. Width greater than 1
    repeats indices; on CSR/CSC that copies the duplicated
    columns.

    The returned length for a ``LayeredSpec`` is
    ``spec.layer_dims[0]``, which equals
    ``len(spec.input_nodes)`` only when every input node has
    width 1. ``hops[0]`` reads layer 0 alone, so a row block
    indexed with this result feeds the first hop directly and
    needs no gathering. For an ``AdjacencySpec`` it is **not**
    the state width: ``to_mask()`` is
    ``(state_dim, state_dim)`` over every node's units, while
    the index covers the input units only
    (``len(spec.input_index)``). Scatter the gathered columns
    into the ``spec.state_dim``-wide state vector with
    ``spec.input_index`` before calling
    ``MaskedLinear(spec.to_mask())``.

    Examples
    --------
    Extra names are ignored and remaining columns are reordered:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H"],
    ...         "target": ["H", "C"],
    ...     }
    ... )
    >>> spec = kpnn2.parse_layered(edgelist)
    >>> spec.input_nodes
    ('A',)
    >>> names = ["unused", "A"]
    >>> col = kpnn2.align_inputs(names, spec)
    >>> col.dtype
    dtype('int64')
    >>> col.tolist()
    [1]
    >>> values = np.array([[9.0, 0.5], [8.0, 1.5]])
    >>> values[:, col].tolist()
    [[0.5], [1.5]]

    An ``AdjacencySpec`` works the same way, but the index is
    ``len(spec.input_index)`` long and the gathered columns must be
    scattered into the state vector before they reach
    ``MaskedLinear(spec.to_mask())``:

    >>> import torch
    >>> cyclic = pd.DataFrame(
    ...     {
    ...         "source": ["x", "a", "b", "a"],
    ...         "target": ["a", "b", "a", "y"],
    ...     }
    ... )
    >>> state_spec = kpnn2.parse_adjacency(cyclic)
    >>> col = kpnn2.align_inputs(["x"], state_spec)
    >>> col.tolist(), tuple(state_spec.to_mask().shape)
    ([0], (4, 4))
    >>> x = torch.tensor([[0.5], [1.5]])[:, col]
    >>> state = torch.zeros(
    ...     2,
    ...     state_spec.state_dim,
    ... )
    >>> state[:, state_spec.input_index] = x
    >>> state.tolist()
    [[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 1.5, 0.0]]

    A tensor is not accepted; pass the names that label it:

    >>> t = torch.tensor([[0.5], [1.5]])
    >>> kpnn2.align_inputs(t, spec)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    Kpnn2Error: 'names' is a tensor; pass the feature names that ...
    """
    if not isinstance(
        spec,
        (LayeredSpec, AdjacencySpec),
    ):
        raise Kpnn2Error("'spec' must be a LayeredSpec or an AdjacencySpec.")
    labels = _labels_from_names(names)
    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    duplicated_labels = sorted(
        label for label, count in label_counts.items() if count > 1
    )
    if duplicated_labels:
        labels_str = ", ".join(duplicated_labels)
        raise Kpnn2Error(
            "Feature names must not contain duplicate labels "
            "(including after converting labels to strings): "
            f"{labels_str}."
        )

    missing = [name for name in spec.input_nodes if name not in label_counts]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise Kpnn2Error(
            f"Feature names are missing required name(s): {missing_str}."
        )

    position = {label: index for index, label in enumerate(labels)}
    node_index = np.array(
        [position[name] for name in spec.input_nodes],
        dtype=np.int64,
    )
    if isinstance(spec, LayeredSpec):
        widths = build_layout(
            spec.input_nodes,
            spec.layer_widths[0],
        ).widths()
    else:
        width_of = dict(
            zip(
                spec.nodes,
                spec.node_widths,
                strict=True,
            )
        )
        widths = tuple(width_of[name] for name in spec.input_nodes)
    if any(width != DEFAULT_NODE_WIDTH for width in widths):
        node_index = np.repeat(
            node_index,
            widths,
        )
    return node_index


def _labels_from_names(names: object) -> list[str]:
    """
    Convert ``names`` to ``str`` labels, or raise ``Kpnn2Error``.
    """
    if isinstance(names, torch.Tensor):
        raise Kpnn2Error(_TENSOR_NOT_ACCEPTED_MSG)
    if isinstance(names, pd.DataFrame):
        raise Kpnn2Error(_DATAFRAME_NOT_ACCEPTED_MSG)
    if isinstance(names, str):
        raise Kpnn2Error(_STRING_NOT_ACCEPTED_MSG)
    if isinstance(names, bytes):
        raise Kpnn2Error(_BYTES_NOT_ACCEPTED_MSG)
    if isinstance(names, Mapping):
        raise Kpnn2Error(_MAPPING_NOT_ACCEPTED_MSG)
    if isinstance(names, Set):
        raise Kpnn2Error(_SET_NOT_ACCEPTED_MSG)
    if hasattr(names, "var_names") and hasattr(names, "X"):
        raise Kpnn2Error(_ANNDATA_NOT_ACCEPTED_MSG)
    ndim = getattr(names, "ndim", None)
    if ndim is not None and ndim != 1:
        raise Kpnn2Error(_NOT_1D_MSG)
    if not isinstance(names, Iterable):
        raise Kpnn2Error(_UNSUPPORTED_TYPE_MSG)
    return [str(name) for name in names]
