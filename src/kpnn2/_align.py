"""
Align named DataFrame columns to ``spec.input_nodes``.
"""

import pandas as pd
import torch

from ._adjacency_spec import AdjacencySpec
from ._errors import Kpnn2Error
from ._layout import build_layout, expand_columns
from ._spec import LayeredSpec

_TENSOR_NOT_ACCEPTED_MSG = (
    "'data' is a tensor; a pandas DataFrame is required. "
    "Pass a DataFrame so columns can be matched to "
    "spec.input_nodes. Pre-ordered tensors go straight to "
    "the model."
)


def align_inputs(
    data: pd.DataFrame,
    spec: LayeredSpec | AdjacencySpec,
) -> torch.Tensor:
    """
    Return a float32 tensor whose columns follow ``spec.input_nodes``.

    Feature-table column labels rarely match the input-node order
    the parsed edgelist fixes; the returned tensor puts them in
    that order, so column ``i`` is ``spec.input_nodes[i]``. Call
    it after parsing, before the model's first layer, instead of
    hand-ordering columns. Every row is materialized as one dense
    CPU tensor; it is not a minibatch or device helper, and
    tensors are rejected.

    Parameters
    ----------
    data : pandas.DataFrame of shape (n_samples, n_columns)
        Feature table, one row per sample. It must carry a
        numeric column for every name in ``spec.input_nodes``, in
        any order; extra columns are ignored. Labels are matched
        after ``str(...)``, the conversion edgelist node names go
        through, so an integer column ``1`` matches node ``"1"``.
        Values keep whatever units the table holds: nothing is
        scaled or imputed, and ``NaN`` survives into the result.
        Neither the frame nor its values are modified, and the
        returned tensor shares no memory with it.
    spec : LayeredSpec or AdjacencySpec
        Parsed edgelist whose ``input_nodes`` — the in-degree-0
        nodes, alphabetically sorted — fix both the required
        column set and the output column order. Only that field
        is read, so the two layouts behave identically here.

    Returns
    -------
    torch.Tensor
        Dense ``float32`` CPU tensor of shape
        ``(n_samples, len(spec.input_nodes))``, column ``i``
        holding the values of node ``spec.input_nodes[i]``. A
        DataFrame with no rows gives a ``(0, len(input_nodes))``
        tensor rather than an error.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``; ``data`` is a tensor; ``data`` is not a
        DataFrame; required DataFrame columns are missing or
        duplicated (including after ``str`` conversion; the
        message names the unique duplicated labels, sorted,
        comma-separated); or required columns are non-numeric.

    See Also
    --------
    parse_layered : Builds the ``LayeredSpec`` whose
        ``input_nodes`` set this column order.
    parse_adjacency : Builds the ``AdjacencySpec`` for the packed
        layout, where the result needs scattering first.
    gather_hop_inputs : Assembles a later hop's input; layer 0
        comes from here instead.

    Notes
    -----
    PyTorch never sees feature names. Passing
    ``DataFrame.to_numpy()`` (or any hand-stacked array) into the
    model can silently wire the wrong features if the column order
    differs, which is what this guards against. A tensor whose
    columns already follow ``spec.input_nodes`` needs no
    alignment and goes straight to the model. AnnData, numpy
    arrays, and scipy sparse matrices are not accepted; a sparse
    host matrix is column-aligned by the caller and densified one
    row block at a time, never through here.

    The returned width is always ``len(spec.input_nodes)``. For a
    ``LayeredSpec`` that is the width of ``hops[0].mask``, whose
    only source layer is layer 0, so the tensor feeds the first
    hop directly and needs no gathering. For an ``AdjacencySpec``
    it is **not** the state width: scatter the tensor into the
    ``len(spec.nodes)``-wide state vector with ``spec.input_index``
    before calling ``MaskedLinear(spec.to_mask())``.

    Examples
    --------
    Extra columns are dropped and remaining columns are reordered:

    >>> import pandas as pd
    >>> import torch
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
    >>> df = pd.DataFrame(
    ...     {
    ...         "unused": [9.0, 8.0],
    ...         "A": [0.5, 1.5],
    ...     }
    ... )
    >>> x = kpnn2.align_inputs(df, spec)
    >>> x.dtype
    torch.float32
    >>> tuple(x.shape)
    (2, 1)
    >>> x.tolist()
    [[0.5], [1.5]]

    An ``AdjacencySpec`` works the same way, but the result is
    ``len(input_nodes)`` wide and must be scattered into the state
    vector before it reaches ``MaskedLinear(spec.to_mask())``:

    >>> cyclic = pd.DataFrame(
    ...     {
    ...         "source": ["x", "a", "b", "a"],
    ...         "target": ["a", "b", "a", "y"],
    ...     }
    ... )
    >>> state_spec = kpnn2.parse_adjacency(cyclic)
    >>> inputs = pd.DataFrame({"x": [0.5, 1.5]})
    >>> x = kpnn2.align_inputs(inputs, state_spec)
    >>> tuple(x.shape), tuple(state_spec.to_mask().shape)
    ((2, 1), (4, 4))
    >>> state = torch.zeros(
    ...     2,
    ...     len(state_spec.nodes),
    ... )
    >>> state[:, state_spec.input_index] = x
    >>> state.tolist()
    [[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 1.5, 0.0]]

    A tensor is not accepted; pass a DataFrame instead:

    >>> t = torch.tensor([[0.5], [1.5]])
    >>> kpnn2.align_inputs(t, spec)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    Kpnn2Error: 'data' is a tensor; a pandas DataFrame is required. ...
    """
    if not isinstance(
        spec,
        (LayeredSpec, AdjacencySpec),
    ):
        raise Kpnn2Error("'spec' must be a LayeredSpec or an AdjacencySpec.")
    if isinstance(data, torch.Tensor):
        raise Kpnn2Error(_TENSOR_NOT_ACCEPTED_MSG)
    if isinstance(data, pd.DataFrame):
        return _align_dataframe(
            data,
            spec,
        )
    raise Kpnn2Error(
        "Unsupported input data type. Expected a pandas DataFrame."
    )


def _align_dataframe(
    data: pd.DataFrame,
    spec: LayeredSpec | AdjacencySpec,
) -> torch.Tensor:
    """
    Reorder numeric DataFrame columns to ``spec.input_nodes``.
    """
    str_columns = [str(name) for name in data.columns]
    label_counts: dict[str, int] = {}
    for label in str_columns:
        label_counts[label] = label_counts.get(label, 0) + 1
    duplicated_labels = sorted(
        label for label, count in label_counts.items() if count > 1
    )
    if duplicated_labels:
        labels_str = ", ".join(duplicated_labels)
        raise Kpnn2Error(
            "Input DataFrame must not contain duplicate column names "
            "(including after converting labels to strings): "
            f"{labels_str}."
        )

    renamed = data.copy(deep=False)
    renamed.columns = str_columns

    missing = [name for name in spec.input_nodes if name not in renamed.columns]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise Kpnn2Error(
            f"Input data is missing required feature name(s): {missing_str}."
        )

    ordered = renamed[list(spec.input_nodes)]
    non_numeric = [
        name
        for name in spec.input_nodes
        if not pd.api.types.is_numeric_dtype(ordered[name])
    ]
    if non_numeric:
        non_numeric_str = ", ".join(sorted(non_numeric))
        raise Kpnn2Error(
            "Input data contains non-numeric feature column(s): "
            f"{non_numeric_str}."
        )

    layout = build_layout(spec.input_nodes)
    values = expand_columns(
        ordered.to_numpy(copy=True),
        layout,
    )
    return torch.tensor(
        values,
        dtype=torch.float32,
    )
