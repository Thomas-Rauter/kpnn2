"""
Align named or tensor inputs to ``GraphSpec.input_nodes``.
"""

import pandas as pd
import torch

from .errors import Kpnn2Error
from .graph_spec import GraphSpec

_DF_DUPLICATE_COLUMNS_MSG = (
    "Input DataFrame must not contain duplicate column names "
    "(including after converting labels to strings)."
)


def align_inputs(
    data: pd.DataFrame | torch.Tensor,
    spec: GraphSpec,
) -> torch.Tensor:
    """
    Return a float32 tensor whose columns follow ``spec.input_nodes``.

    **DataFrame.** Required columns are ``spec.input_nodes``. Labels
    are matched after ``str(...)``, the same conversion used for
    edgelist node names, so an integer column ``1`` matches node
    ``"1"``. Extra columns are ignored. Columns are reordered to
    ``spec.input_nodes``. Missing, duplicate, or non-numeric required
    columns raise an error.

    **Tensor.** Must be 2-D with
    ``shape[1] == len(spec.input_nodes)``. Columns are assumed to
    already follow ``spec.input_nodes``; they are not reordered.

    AnnData is not supported.

    Parameters
    ----------
    data : DataFrame | Tensor
        Feature table with named columns, or a pre-ordered 2-D
        tensor.
    spec : GraphSpec
        Graph structure whose ``input_nodes`` define column order.

    Returns
    -------
    Tensor
        Float32 tensor of shape
        ``(n_samples, len(spec.input_nodes))``.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is not a ``GraphSpec``; ``data`` is neither a
        DataFrame nor a tensor; required DataFrame columns are
        missing or duplicated (including after ``str`` conversion);
        required columns are non-numeric; or a tensor has the wrong
        number of dimensions or the wrong width.

    Notes
    -----
    PyTorch never sees feature names. Column ``i`` of the returned
    tensor is ``spec.input_nodes[i]``. Always build train, validation,
    and test tensors with this function and the same ``spec``. Passing
    ``DataFrame.to_numpy()`` (or any hand-stacked array) into the
    model can silently wire the wrong features if the column order
    differs.

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
    >>> spec = k2.parse_edgelist(edgelist)
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

    A 2-D tensor is checked and cast, not reordered:

    >>> t = torch.tensor([[0.5], [1.5]])
    >>> y = k2.align_inputs(t, spec)
    >>> y.dtype
    torch.float32
    >>> y.tolist()
    [[0.5], [1.5]]
    """
    if not isinstance(spec, GraphSpec):
        raise Kpnn2Error("'spec' must be a GraphSpec.")
    if isinstance(data, pd.DataFrame):
        return _align_dataframe(
            data,
            spec,
        )
    if isinstance(data, torch.Tensor):
        return _align_tensor(
            data,
            spec,
        )
    raise Kpnn2Error(
        "Unsupported input data type. Expected a pandas DataFrame "
        "or a torch.Tensor."
    )


def _align_dataframe(
    data: pd.DataFrame,
    spec: GraphSpec,
) -> torch.Tensor:
    """
    Reorder numeric DataFrame columns to ``spec.input_nodes``.
    """
    if data.columns.duplicated().any():
        raise Kpnn2Error(_DF_DUPLICATE_COLUMNS_MSG)

    str_columns = [str(name) for name in data.columns]
    if len(set(str_columns)) != len(str_columns):
        raise Kpnn2Error(_DF_DUPLICATE_COLUMNS_MSG)

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

    return torch.tensor(
        ordered.to_numpy(copy=True),
        dtype=torch.float32,
    )


def _align_tensor(
    data: torch.Tensor,
    spec: GraphSpec,
) -> torch.Tensor:
    """
    Validate tensor shape and cast to float32 without reordering.
    """
    if data.ndim != 2:
        raise Kpnn2Error("Input tensor must be 2-dimensional.")
    n_features = len(spec.input_nodes)
    if data.shape[1] != n_features:
        raise Kpnn2Error(
            "Input tensor has the wrong number of features. "
            f"Expected {n_features}, got {data.shape[1]}."
        )
    return data.to(dtype=torch.float32)
