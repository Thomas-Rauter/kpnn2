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

    **DataFrame.** Required columns are ``spec.input_nodes``. Labels
    are matched after ``str(...)``, the same conversion used for
    edgelist node names, so an integer column ``1`` matches node
    ``"1"``. Extra columns are ignored. Columns are reordered to
    ``spec.input_nodes``. Missing, duplicate, or non-numeric required
    columns raise an error.

    **Tensor.** Not accepted. Raise ``Kpnn2Error``. Pre-ordered
    tensors go straight to the model. Users who need alignment pass
    a DataFrame.

    AnnData is not supported.

    Parameters
    ----------
    data : DataFrame
        Feature table with named columns.
    spec : LayeredSpec or AdjacencySpec
        Graph structure whose ``input_nodes`` define column order.
        Both spec types work the same way here; only
        ``input_nodes`` is read.

    Returns
    -------
    Tensor
        Float32 tensor of shape
        ``(n_samples, len(spec.input_nodes))``.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``; ``data`` is a tensor; ``data`` is not a
        DataFrame; required DataFrame columns are missing or
        duplicated (including after ``str`` conversion; the message
        names the unique duplicated labels, sorted,
        comma-separated); or required columns are non-numeric.

    Notes
    -----
    PyTorch never sees feature names. Column ``i`` of the returned
    tensor is ``spec.input_nodes[i]``. Use this function when the
    table has named columns. A tensor whose columns already follow
    ``spec.input_nodes`` goes straight to the model. Passing
    ``DataFrame.to_numpy()`` (or any hand-stacked array) into the
    model can silently wire the wrong features if the column order
    differs.

    The returned width is always ``len(spec.input_nodes)``. For a
    ``LayeredSpec`` that is the width of ``masks[0]``, so the
    tensor feeds the first hop directly. For an ``AdjacencySpec``
    it is **not** the mask width: scatter the tensor into the
    ``len(spec.nodes)``-wide state vector with ``spec.input_index``
    before calling ``MaskedLinear(spec.mask)``.

    Examples
    --------
    Extra columns are dropped and remaining columns are reordered:

    >>> import pandas as pd
    >>> import torch
    >>> import kpnn2 as k2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H"],
    ...         "target": ["H", "C"],
    ...     }
    ... )
    >>> spec = k2.parse_layered(edgelist)
    >>> spec.input_nodes
    ('A',)
    >>> df = pd.DataFrame(
    ...     {
    ...         "unused": [9.0, 8.0],
    ...         "A": [0.5, 1.5],
    ...     }
    ... )
    >>> x = k2.align_inputs(df, spec)
    >>> x.dtype
    torch.float32
    >>> tuple(x.shape)
    (2, 1)
    >>> x.tolist()
    [[0.5], [1.5]]

    An ``AdjacencySpec`` works the same way, but the result is
    ``len(input_nodes)`` wide and must be scattered into the state
    vector before it reaches ``MaskedLinear(spec.mask)``:

    >>> cyclic = pd.DataFrame(
    ...     {
    ...         "source": ["x", "a", "b", "a"],
    ...         "target": ["a", "b", "a", "y"],
    ...     }
    ... )
    >>> state_spec = k2.parse_adjacency(cyclic)
    >>> inputs = pd.DataFrame({"x": [0.5, 1.5]})
    >>> x = k2.align_inputs(inputs, state_spec)
    >>> tuple(x.shape), state_spec.mask.shape
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
    >>> k2.align_inputs(t, spec)  # doctest: +IGNORE_EXCEPTION_DETAIL
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
