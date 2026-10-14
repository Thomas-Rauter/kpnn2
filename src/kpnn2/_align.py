"""
Align feature names to ``spec.input_nodes``.
"""

import numpy as np
import pandas as pd
import torch

from ._adjacency_spec import AdjacencySpec
from ._errors import Kpnn2Error
from ._spec import LayeredSpec
from ._validate import as_name_list, require_spec

_TENSOR_NOT_ACCEPTED_MSG = (
    "'names' is a tensor; pass the feature names that "
    "label that axis, even when its columns already follow "
    "spec.input_nodes."
)
_INPUT_ROLE_RULE = (
    "Input nodes are the edgelist nodes with no incoming edge, "
    "so the edgelist starts at the measured features, and each "
    "input node is named like its feature column."
)
_MISSING_FIX = (
    "Add the missing feature columns, or give a computed node "
    "with no incoming edge its parents in the edgelist. Do not "
    "fit the width with a dense layer in front of the model: it "
    "discards the prior wiring from the features."
)
_COMPUTED_FIX = (
    "Give each measured feature its own input node, named "
    "differently from every computed node (for example "
    "STAT3_gene for the gene and STAT3 for the transcription "
    "factor). A column you do not feed, such as a label column "
    "named like an output node, belongs out of the table before "
    "you take its names."
)
_DATAFRAME_NOT_ACCEPTED_MSG = (
    "'names' is a DataFrame; pass the feature names "
    "(for example data.columns). Apply the returned index "
    "on the matrix yourself."
)
_ANNDATA_NOT_ACCEPTED_MSG = (
    "'names' looks like AnnData; pass the .var_names "
    "and apply the index to .X yourself."
)


def align_inputs(
    names: object,
    spec: LayeredSpec | AdjacencySpec,
) -> np.ndarray:
    """
    Return an integer index that orders features to spec inputs.

    The columns of a feature table rarely come in the order of
    ``spec.input_nodes``, which the parsed edgelist fixes. The
    model reads columns by position, not by name, so a column
    in the wrong place feeds the wrong input node without any
    error. This returns a 1-D ``int64`` index into the caller's
    feature axis; indexing that axis with it puts the columns in
    input-node order. It does not take the matrix, copy sample
    rows, or densify. Apply the index on whatever holds X:
    ``X[:, col]`` for numpy, scipy CSR/CSC, AnnData ``.X`` in
    those formats, or a tensor; ``df.to_numpy()[:, col]`` for
    a DataFrame. COO-style sparse layouts do not support
    integer column indexing; convert first. Input nodes always
    have width 1, so the index has one position per input node.
    Call it after parsing, once, instead of hand-ordering
    columns. Pass the names, not the data: DataFrames, tensors,
    and matrices are rejected.

    Parameters
    ----------
    names : sequence of labels
        The feature names, one per column, in the order the
        columns sit in your data: ``df.columns``,
        ``adata.var_names``, a 1-D numpy array, or a list. The
        annotated type is ``object`` because a DataFrame or
        tensor passed here by mistake is rejected at runtime.
        Names are compared to node names as text, after the
        same ``str(...)`` conversion edgelist nodes go through,
        so an integer ``1`` matches node ``"1"``. Names that
        are not graph nodes are ignored, so extra columns are
        fine. A name of a hidden or output node raises, because
        the model computes that node and would never read the
        column. Neither ``names`` nor the matrix is modified.
    spec : LayeredSpec or AdjacencySpec
        Parsed edgelist whose ``input_nodes`` — the in-degree-0
        nodes, alphabetically sorted — fix both the required
        names and their order.

    Returns
    -------
    numpy.ndarray of dtype int64, shape (len(spec.input_nodes),)
        Positions into ``names``, one per input node in
        ``spec.input_nodes`` order. That length is
        ``spec.layer_dims[0]`` for a ``LayeredSpec`` and
        ``len(spec.input_index)`` for an ``AdjacencySpec``.
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
        labels; labels are duplicated (including after ``str``
        conversion; the message names the unique duplicated
        labels, sorted, comma-separated); required names are
        missing (the message names them, sorted, and counts the
        input nodes found); or a name matches a hidden or output
        node (the message names each with its role, sorted).
        Missing names and computed-node names are reported in
        one message.

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
    Call it for every feature table, including one whose columns
    already follow ``spec.input_nodes``: it is the one check
    that the data match the graph's input nodes by name.

    Input nodes are not declared; they are the nodes with no
    incoming edge. The edgelist therefore starts at the measured
    features, one input node per feature column, named like that
    column. Name measured features so no computed node shares a
    name, for example ``STAT3_gene`` for the gene and ``STAT3``
    for the transcription factor node. Then an edgelist that
    starts at the first computed layer, which makes those
    computed nodes inputs, raises here as missing names. One
    name is one node, so a feature named like a hidden or output
    node has no input node of its own, and that raises too. With
    shared names, such as transcription factor nodes named by
    gene symbol beside a table of every gene, an edgelist that
    starts at a computed layer can match feature columns and
    pass.

    AnnData,
    numpy matrices, and scipy sparse matrices are not accepted
    as ``names``; pass the labels that sit on that axis and
    apply the index on the matrix. scipy CSR/CSC and AnnData
    ``.X`` in those formats stay sparse under ``X[:, col]``
    until the caller densifies one row block. COO and similar
    layouts cannot be indexed that way.

    ``hops[0]`` of a ``LayeredSpec`` reads layer 0 alone, so a
    row block indexed with this result feeds the first hop
    directly and needs no gathering. For an ``AdjacencySpec``
    the returned length is **not** the state width: ``to_mask()`` is
    ``(state_dim, state_dim)`` over every node's units, while
    the index covers the input units only
    (``len(spec.input_index)``). Scatter the gathered columns
    into the ``spec.state_dim``-wide state vector with
    ``state.index_copy(-1, input_index, x)`` before calling
    ``MaskedLinear(spec.to_mask())``. ``input_index`` is a
    buffer, ``torch.as_tensor(spec.input_index)``.

    Examples
    --------
    Names that are not graph nodes are ignored, and the rest are
    reordered:

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
    >>> index = torch.as_tensor(state_spec.input_index)
    >>> state = state.index_copy(
    ...     -1,
    ...     index,
    ...     x,
    ... )
    >>> state.tolist()
    [[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 1.5, 0.0]]

    A feature named like a node the model computes raises,
    because its column would never be read:

    >>> computed = ["A", "H"]
    >>> kpnn2.align_inputs(computed, spec)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    Kpnn2Error: Feature names match node(s) the model computes, ...

    A tensor is not accepted; pass the names that label it:

    >>> t = torch.tensor([[0.5], [1.5]])
    >>> kpnn2.align_inputs(t, spec)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    Kpnn2Error: 'names' is a tensor; pass the feature names that ...
    """
    require_spec(spec)
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

    _require_input_roles(
        set(label_counts),
        spec,
    )

    position = {label: index for index, label in enumerate(labels)}
    return np.array(
        [position[name] for name in spec.input_nodes],
        dtype=np.int64,
    )


def _require_input_roles(
    labels: set[str],
    spec: LayeredSpec | AdjacencySpec,
) -> None:
    """
    Raise ``Kpnn2Error`` unless ``labels`` fit the node roles.

    Every input node must be a label, and no label may name a
    hidden or output node, whose column the model would never
    read. Both findings go into one message.
    """
    missing = [name for name in spec.input_nodes if name not in labels]
    roles = {name: "hidden" for name in spec.hidden_nodes}
    roles.update({name: "output" for name in spec.output_nodes})
    computed = sorted(labels & roles.keys())
    if not missing and not computed:
        return

    parts = []
    if missing:
        found = len(spec.input_nodes) - len(missing)
        parts.append(
            "Feature names are missing required name(s): "
            f"{', '.join(sorted(missing))}. Found {found} of "
            f"{len(spec.input_nodes)} input nodes."
        )
    if computed:
        matched = ", ".join(f"{name} ({roles[name]})" for name in computed)
        parts.append(
            "Feature names match node(s) the model computes, so "
            f"their columns would never be read: {matched}."
        )
    parts.append(_INPUT_ROLE_RULE)
    if missing:
        parts.append(_MISSING_FIX)
    if computed:
        parts.append(_COMPUTED_FIX)
    raise Kpnn2Error(" ".join(parts))


def _labels_from_names(names: object) -> list[str]:
    """
    Convert ``names`` to ``str`` labels, or raise ``Kpnn2Error``.
    """
    if isinstance(names, torch.Tensor):
        raise Kpnn2Error(_TENSOR_NOT_ACCEPTED_MSG)
    if isinstance(names, pd.DataFrame):
        raise Kpnn2Error(_DATAFRAME_NOT_ACCEPTED_MSG)
    if hasattr(names, "var_names") and hasattr(names, "X"):
        raise Kpnn2Error(_ANNDATA_NOT_ACCEPTED_MSG)
    return as_name_list(
        names,
        "names",
        "feature names",
    )
