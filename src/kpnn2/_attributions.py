"""
Map attribution tensors onto spec node names.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch
import xarray as xr

from ._adjacency_spec import AdjacencySpec
from ._errors import caller_error, close_matches, listed
from ._layout import Layout, build_layout, concat_layouts
from ._spec import Hop, LayeredSpec, require_spec_hop
from ._validate import (
    as_layer_index,
    describe,
    reject_unordered,
    require_spec,
)

_NODE_DIM = "node"
_LAYER_COORD = "layer"
_OBS_DIM = "observation"
_STEP_DIM = "step"
# Which argument names which axis of a LayeredSpec.
_AXIS_CHOICE_FIX = (
    "Pass layer= with a 0-based index into spec.layer_nodes, or a Hop "
    "from spec.hops as hop_output= (what its module returns) or "
    "hop_input= (what its module reads)."
)


def map_node_attributions(  # numpydoc ignore=PR06
    attributions: torch.Tensor | Sequence[torch.Tensor],
    spec: LayeredSpec | AdjacencySpec,
    layer: int | None = None,
    *,
    hop_input: Hop | None = None,
    hop_output: Hop | None = None,
    axis: str | None = None,
    dims: Sequence[str] | None = None,
    coords: Mapping[str, Sequence] | None = None,
) -> xr.DataArray:
    """
    Label an attribution tensor's node axis with names from a spec.

    Attribution methods return unlabeled tensors whose node axis is
    bare positions; the spec knows the name at each one. Reach for
    it after Captum or your own gradients rather than zipping names
    to columns yourself. Which spec you pass decides the contract: a
    ``LayeredSpec`` needs exactly one of ``layer``, ``hop_input``,
    ``hop_output``, or ``axis="inputs"``; an ``AdjacencySpec``
    forbids the first three. Omit ``axis`` there to name the
    whole state vector, or pass ``axis="inputs"`` to name the
    input units. The width is never inferred.
    Values are detached onto CPU and never aggregated.

    Parameters
    ----------
    attributions : torch.Tensor or sequence of torch.Tensor
        Scores exactly as the attribution method produced them, in
        whatever units it works in; nothing is scaled, summed, or
        made absolute. The node axis must be as long as the named
        units — ``spec.layer_dims[layer]`` when ``layer`` is given,
        ``hop.in_features`` for ``hop_input=hop``,
        ``hop.out_features`` for ``hop_output=hop``,
        ``len(spec.input_index)`` for ``axis="inputs"`` on an
        ``AdjacencySpec``, ``spec.layer_dims[0]`` for
        ``axis="inputs"`` on a ``LayeredSpec`` (the same axis
        as ``layer=0``), and ``spec.state_dim`` for an
        ``AdjacencySpec`` when ``axis`` is omitted — and the
        remaining axes are yours. A non-empty tuple or list of
        equal-shaped tensors is stacked on a new leading ``step``
        axis, one entry per unrolled step or module call. The
        tensors are read, never modified, and the result holds a
        detached CPU copy that shares no memory with them.
    spec : LayeredSpec or AdjacencySpec
        Parsed edgelist supplying the ``node`` coordinate. A
        ``LayeredSpec`` names one depth (``layer`` or
        ``hop_output``) or the concatenated source units of one hop
        (``hop_input``); an ``AdjacencySpec`` names the whole state
        vector at once and has no depth to report, unless
        ``axis="inputs"``, which names ``spec.input_nodes``,
        one unit each.
    layer : int or numpy integer, optional
        0-based depth into ``spec.layer_nodes``, index 0 being the
        input layer, whose names label the node axis; the
        ``layer`` coordinate repeats the index on every unit, the
        same for a numpy integer as for an ``int``.
        ``LayeredSpec`` only, and mutually exclusive with
        ``hop_input`` and ``hop_output``.
    hop_input : Hop, optional
        A ``Hop`` that equals one entry of ``spec.hops``. Labels
        what that hop's module reads: the concatenated source axis
        ``gather_hop_inputs`` returns, length ``hop.in_features``,
        with unit names (a wide node's name repeats). Captum layer
        methods score this side only with
        ``attribute_to_layer_input=True``. The ``layer``
        coordinate gives each unit the depth of its source layer,
        so a skip hop's axis holds several depths.
        ``LayeredSpec`` only, and mutually exclusive with
        ``layer`` and ``hop_output``.
    hop_output : Hop, optional
        A ``Hop`` that equals one entry of ``spec.hops``. Labels
        what that hop's module returns: layer ``hop.target_layer``,
        length ``hop.out_features``. This is the side Captum layer
        methods score by default. The result is the same as
        ``layer=hop.target_layer``, ``layer`` coordinate
        included. ``LayeredSpec`` only, and mutually exclusive with
        ``layer`` and ``hop_input``.
    axis : {"inputs"} or None, optional
        ``"inputs"`` labels the model-input axis and nothing
        else. On a ``LayeredSpec`` that is ``layer=0``,
        including the ``layer`` coordinate, and it is
        mutually exclusive with ``layer``, ``hop_input``, and
        ``hop_output``. On an ``AdjacencySpec`` the names are
        the input units in ``spec.input_index`` order, length
        ``len(spec.input_index)``, with no ``layer``
        coordinate. Any other string is rejected. The length
        of the tensor is not used to choose this axis.
    dims : sequence of str, optional
        One name per axis of the tensor after any stacking,
        containing ``node`` exactly once and never ``layer``. A
        single ``str`` is rejected, not split into characters,
        and so are a set and a mapping, which carry no axis order.
        Required at 3 or more axes, unless the tensor is a stacked
        sequence of 1-D or 2-D pieces. The defaults are
        ``("node",)`` for 1-D and ``("observation", "node")`` for
        2-D, with ``step`` prepended when a sequence was stacked.
    coords : mapping of str to sequence, optional
        Labels for axes other than ``node`` and ``layer``, keyed by
        dim name; each sequence must be as long as its axis. A
        single ``str`` is rejected, not split into characters, and
        so are a set and a mapping (a dict would give its keys).
        Axes left out are labelled with their integer positions.

    Returns
    -------
    xarray.DataArray
        The values and shape of the (stacked) tensor, carrying the
        spec's node names as the ``node`` coordinate in spec order
        (a name repeats once per unit when that node is wider
        than 1). On a ``LayeredSpec`` it also carries ``layer``,
        an ``int64`` coordinate along ``node`` that holds each
        unit's depth in ``spec.layer_nodes``: one depth repeated
        for ``layer``, ``hop_output``, and ``axis="inputs"``, and
        each unit's source layer for ``hop_input``. It stays
        with its units when you concatenate results along
        ``node``. Read one depth with ``int(da["layer"][0])``. An
        ``AdjacencySpec`` result has no ``layer`` coordinate.
        ``bfloat16`` scores are stored as ``float32``: NumPy has
        no bfloat16 dtype, and every bfloat16 value fits in
        float32, so the scores are unchanged. Other dtypes are
        kept. Use ``.to_dataframe(name="score").reset_index()``
        for a long table, or ``.to_pandas()`` for a 2-D wide
        table.

        Units of a wide node share its edges, so the spec gives
        them no names of their own. On an axis with a wide node
        the ``node`` index is not unique. ``da.sel(node=name)``
        returns all units of that node, and a fold you choose,
        such as ``da.groupby("node").mean()``, gives one score
        per node, sorted by name. Selecting a list of names,
        ``reindex``, and aligning with an array whose ``node``
        index differs raise in xarray, even for width-1 names.
        Select several names with
        ``da.sel(node=da["node"].isin(names))``. Two arrays with
        the same ``node`` index combine unit by unit, by
        position. With every node at width 1 the index is unique.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``; a ``LayeredSpec`` gets none or more
        than one of ``layer``, ``hop_input``, ``hop_output``,
        and ``axis``; ``axis`` is neither ``"inputs"`` nor
        ``None``; any of ``layer``, ``hop_input``, or
        ``hop_output`` is given for an ``AdjacencySpec``;
        ``layer`` is not an integer in range; ``hop_input`` or
        ``hop_output`` is not a ``Hop`` that matches an entry
        of ``spec.hops``;
        ``attributions`` is neither a tensor nor a non-empty
        sequence of equal-shaped tensors; ``dims`` is missing, not
        a sequence of strings (a ``str``, set, or mapping
        included), the wrong length, non-unique, or does not name
        ``node`` exactly once; the node axis is not as long as the
        named units; or ``coords`` is not a mapping, names an
        unknown axis, or holds a value that is not a sequence of
        labels (a ``str``, set, or mapping included) or has the
        wrong length.

    See Also
    --------
    aggregate_node_attributions : Folds named scores to one
        value per node; call this after mapping.
    align_inputs : Shared ``input_nodes`` order on the way
        in; a column index into the caller's feature names,
        not the full state axis of an ``AdjacencySpec``.
    gather_hop_inputs : Builds the tensor whose columns
        ``hop_input`` names.
    Hop : One hop; pass ``hop_output=spec.hops[i]`` or
        ``hop_input=spec.hops[i]`` to name either side of it.
    LayeredSpec : Holds ``layer_nodes`` and ``hops``, the names
        used when ``layer``, ``hop_input``, or ``hop_output`` is
        given.
    AdjacencySpec : Holds ``nodes``, the state-vector names used
        when all three are omitted.

    Notes
    -----
    Captum is not imported anywhere in this package; mapping is
    name alignment only, and any attribution method will do.

    Name a hop module's scores with the same index ``i`` you use
    for the module. For what ``PackedLinear`` (or ``MaskedLinear``)
    on ``spec.hops[i]`` returns, which is what Captum layer methods
    such as ``LayerConductance`` score by default, pass
    ``hop_output=spec.hops[i]``. For what it reads (the
    ``gather_hop_inputs`` tensor, or Captum with
    ``attribute_to_layer_input=True``), pass
    ``hop_input=spec.hops[i]``. The side is always stated, never
    inferred: a hop's input and output often have the same width,
    so a width check alone cannot catch the wrong side. Only map
    units that are spec nodes; BatchNorm and other unnamed modules
    have no node axis to name.

    DeepLift, DeepLiftShap, and LRP rescale ``nn.Module``
    nonlinearities. The activation after a hop is an
    ``nn.ReLU`` held in an ``nn.ModuleList``, one entry per
    hop, and called from ``forward``. The hop module returns
    the pre-activation tensor: ``hop_output`` names that
    linear map. Post-activation node states are the output
    of the ``nn.ReLU`` that follows the hop.
    ``LayerActivation`` and ``LayerGradientXActivation``
    hook that module.

    A recurrent net on an ``AdjacencySpec`` has no layer to index,
    and the natural extra axis there is ``step``: pass one tensor
    per unrolled step as a sequence and they are stacked for you.

    Examples
    --------
    Name a two-observation tensor at the output layer of a
    ``LayeredSpec``:

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
    >>> spec.layer_nodes
    (('A',), ('H',), ('C',))
    >>> scores = torch.tensor([[0.5], [1.0]])
    >>> da = kpnn2.map_node_attributions(
    ...     attributions=scores,
    ...     spec=spec,
    ...     layer=2,
    ... )
    >>> da["node"].values.tolist()
    ['C']
    >>> da.sel(node="C").values.tolist()
    [0.5, 1.0]
    >>> da["layer"].values.tolist()
    [2]

    Both sides of a hop can have the same width. Here ``hops[0]``
    reads ``A``, ``B`` and writes ``H1``, ``H2``, so one
    ``(1, 2)`` tensor fits either side; the keyword decides the
    names:

    >>> square = pd.DataFrame(
    ...     {
    ...         "source": ["A", "B", "A", "B", "H1", "H2"],
    ...         "target": ["H1", "H1", "H2", "H2", "C", "C"],
    ...     }
    ... )
    >>> square_spec = kpnn2.parse_layered(square)
    >>> hop = square_spec.hops[0]
    >>> hop.in_features, hop.out_features
    (2, 2)
    >>> hop_scores = torch.tensor([[0.1, 0.2]])
    >>> out_da = kpnn2.map_node_attributions(
    ...     attributions=hop_scores,
    ...     spec=square_spec,
    ...     hop_output=hop,
    ... )
    >>> out_da["node"].values.tolist()
    ['H1', 'H2']
    >>> out_da["layer"].values.tolist()
    [1, 1]
    >>> in_da = kpnn2.map_node_attributions(
    ...     attributions=hop_scores,
    ...     spec=square_spec,
    ...     hop_input=hop,
    ... )
    >>> in_da["node"].values.tolist()
    ['A', 'B']
    >>> in_da["layer"].values.tolist()
    [0, 0]

    A skip hop reads several layers, so its input axis is those
    layers concatenated (width ``hop.in_features``), and
    ``layer`` gives each unit its own depth:

    >>> skip_edges = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> skip_spec = kpnn2.parse_layered(skip_edges)
    >>> skip_hop = skip_spec.hops[1]
    >>> skip_hop.source_layers
    (0, 1)
    >>> skip_da = kpnn2.map_node_attributions(
    ...     attributions=torch.tensor([[0.1, 0.2]]),
    ...     spec=skip_spec,
    ...     hop_input=skip_hop,
    ... )
    >>> skip_da["node"].values.tolist()
    ['A', 'H']
    >>> skip_da["layer"].values.tolist()
    [0, 1]

    Concatenated along ``node``, each unit keeps its depth:

    >>> import xarray as xr
    >>> top_da = kpnn2.map_node_attributions(
    ...     attributions=torch.tensor([[0.3]]),
    ...     spec=skip_spec,
    ...     layer=2,
    ... )
    >>> both = xr.concat(
    ...     [skip_da, top_da],
    ...     dim="node",
    ... )
    >>> both["node"].values.tolist()
    ['A', 'H', 'C']
    >>> both["layer"].values.tolist()
    [0, 1, 2]

    A wide node's name repeats once per unit. ``sel`` takes one
    name; pick several with ``isin``, and fold units yourself:

    >>> fan = pd.DataFrame(
    ...     {
    ...         "source": ["A", "A", "A", "H1", "H2", "H3"],
    ...         "target": ["H1", "H2", "H3", "C", "C", "C"],
    ...     }
    ... )
    >>> fan_spec = kpnn2.parse_layered(
    ...     fan,
    ...     widths={"H1": 2},
    ... )
    >>> fan_da = kpnn2.map_node_attributions(
    ...     attributions=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    ...     spec=fan_spec,
    ...     layer=1,
    ... )
    >>> fan_da["node"].values.tolist()
    ['H1', 'H1', 'H2', 'H3']
    >>> fan_da.sel(node="H1").values.tolist()
    [[1.0, 2.0]]
    >>> picked = fan_da.sel(node=fan_da["node"].isin(["H1", "H3"]))
    >>> picked["node"].values.tolist()
    ['H1', 'H1', 'H3']
    >>> fan_da.groupby("node").sum().values.tolist()
    [[3.0, 3.0, 4.0]]

    On an ``AdjacencySpec`` there are no layers: omit ``layer``,
    ``hop_input``, and ``hop_output`` and the whole state vector
    is named. One tensor per unrolled step stacks onto a ``step``
    axis:

    >>> cyclic = pd.DataFrame(
    ...     {
    ...         "source": ["x", "a", "b", "a"],
    ...         "target": ["a", "b", "a", "y"],
    ...     }
    ... )
    >>> state_spec = kpnn2.parse_adjacency(cyclic)
    >>> per_step = kpnn2.map_node_attributions(
    ...     attributions=[
    ...         torch.zeros(2, 4),
    ...         torch.ones(2, 4),
    ...     ],
    ...     spec=state_spec,
    ... )
    >>> per_step.dims
    ('step', 'observation', 'node')
    >>> per_step["node"].values.tolist()
    ['a', 'b', 'x', 'y']
    >>> "layer" in per_step.coords
    False
    >>> input_scores = kpnn2.map_node_attributions(
    ...     attributions=torch.tensor([[0.5], [1.0]]),
    ...     spec=state_spec,
    ...     axis="inputs",
    ... )
    >>> input_scores["node"].values.tolist()
    ['x']
    >>> "layer" in input_scores.coords
    False
    """
    layout, unit_layers = _resolve_node_layout(
        spec,
        layer,
        hop_input,
        hop_output,
        axis,
    )
    names = layout.unit_names()
    n_units = layout.n_units
    tensor, used_default_step = _as_tensor(attributions)
    dim_names = _resolve_dims(
        tensor=tensor,
        dims=dims,
        used_default_step=used_default_step,
    )
    node_axis = dim_names.index(_NODE_DIM)
    if tensor.shape[node_axis] != n_units:
        units, size_source = _node_axis_units(
            spec,
            layer,
            hop_input,
            hop_output,
            axis,
        )
        stacked = "stacked " if used_default_step else ""
        raise caller_error(
            "The 'node' axis of 'attributions' must have one entry per "
            f"{units}.",
            got=(
                f"{tensor.shape[node_axis]} (axis {node_axis} of "
                f"{stacked}shape {tuple(tensor.shape)})"
            ),
            expected=f"{n_units} ({size_source})",
        )

    coord_map = _build_coords(
        tensor=tensor,
        dim_names=dim_names,
        names=names,
        unit_layers=unit_layers,
        coords=coords,
    )
    values = tensor.detach().cpu()
    # NumPy has no bfloat16; every value fits in float32.
    if values.dtype is torch.bfloat16:
        values = values.float()
    values = values.numpy().copy()
    return xr.DataArray(
        data=values,
        dims=dim_names,
        coords=coord_map,
    )


def _resolve_node_layout(
    spec: LayeredSpec | AdjacencySpec,
    layer: int | None,
    hop_input: Hop | None,
    hop_output: Hop | None,
    axis: str | None,
) -> tuple[Layout, tuple[int, ...] | None]:
    """
    Return the ``node`` axis layout and each unit's depth.

    The layout supplies both the expected axis length and the
    per-unit names, so a node owning several units labels each
    of them. On a ``LayeredSpec`` the depths repeat one layer
    (``layer``, ``hop_output.target_layer``, or 0 for
    ``axis="inputs"``), or follow ``hop_input.source_layers``
    unit by unit. An ``AdjacencySpec`` has no depths and gives
    ``None``; its ``axis="inputs"`` is the input units only.
    """
    if axis is not None and (not isinstance(axis, str) or axis != "inputs"):
        raise caller_error(
            "'axis' must be 'inputs' or None.",
            got=describe(axis),
            candidates=(
                close_matches(
                    [axis],
                    ["inputs"],
                )
                if isinstance(axis, str)
                else None
            ),
        )
    require_spec(spec)
    if isinstance(spec, LayeredSpec):
        given = [
            name
            for name, value in (
                ("layer", layer),
                ("hop_input", hop_input),
                ("hop_output", hop_output),
                ("axis", axis),
            )
            if value is not None
        ]
        if not given:
            raise caller_error(
                "A LayeredSpec needs one of 'layer', 'hop_input', "
                "'hop_output', or axis='inputs'.",
                fix=_AXIS_CHOICE_FIX,
            )
        if len(given) > 1:
            raise caller_error(
                "A LayeredSpec takes only one of 'layer', 'hop_input', "
                "'hop_output', and 'axis'.",
                got=listed(repr(name) for name in given),
                fix=_AXIS_CHOICE_FIX,
            )
        if hop_input is not None:
            require_spec_hop(
                hop_input,
                spec.hops,
                "hop_input",
            )
            return _layered_axis(
                spec,
                hop_input.source_layers,
            )
        if hop_output is not None:
            require_spec_hop(
                hop_output,
                spec.hops,
                "hop_output",
            )
            layer = hop_output.target_layer
        if axis == "inputs":
            layer = 0
        layer = as_layer_index(
            layer,
            len(spec.layer_nodes),
        )
        return _layered_axis(
            spec,
            (layer,),
        )
    if layer is not None or hop_input is not None or hop_output is not None:
        given = [
            name
            for name, value in (
                ("layer", layer),
                ("hop_input", hop_input),
                ("hop_output", hop_output),
            )
            if value is not None
        ]
        raise caller_error(
            "'layer', 'hop_input', and 'hop_output' do not apply to an "
            "AdjacencySpec: it has no layers, only one state vector.",
            got=listed(repr(name) for name in given),
            fix=(
                "Omit them to label the node axis with spec.nodes, or "
                "pass axis='inputs' to label the input units."
            ),
        )
    if axis == "inputs":
        return build_layout(spec.input_nodes), None
    return build_layout(
        spec.nodes,
        spec.node_widths,
    ), None


def _node_axis_units(
    spec: LayeredSpec | AdjacencySpec,
    layer: int | None,
    hop_input: Hop | None,
    hop_output: Hop | None,
    axis: str | None,
) -> tuple[str, str]:
    """
    Name the units the ``node`` axis must match, and their count.

    For the message on a wrong axis length, after
    ``_resolve_node_layout`` has accepted the arguments: what the
    caller's choice names, and the field that holds its length.
    """
    if hop_input is not None:
        return "input unit of 'hop_input'", "hop_input.in_features"
    if hop_output is not None:
        return "output unit of 'hop_output'", "hop_output.out_features"
    if axis == "inputs":
        if isinstance(spec, LayeredSpec):
            return "input unit", "spec.layer_dims[0]"
        return "input unit", "len(spec.input_index)"
    if isinstance(spec, LayeredSpec):
        return f"unit of layer {layer}", f"spec.layer_dims[{layer}]"
    return "unit of the state vector", "spec.state_dim"


def _layered_axis(
    spec: LayeredSpec,
    layers: tuple[int, ...],
) -> tuple[Layout, tuple[int, ...]]:
    """
    Lay ``layers`` of ``spec`` end to end, with each unit's depth.

    One layer is a ``layer`` axis. Several, in order, are the
    source axis of a hop that reads more than one layer.
    """
    layouts = [
        build_layout(
            spec.layer_nodes[i],
            spec.layer_widths[i],
        )
        for i in layers
    ]
    depths = tuple(
        i
        for i, layout in zip(
            layers,
            layouts,
        )
        for _ in range(layout.n_units)
    )
    return concat_layouts(layouts), depths


def _as_tensor(
    attributions: torch.Tensor | Sequence[torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    """
    Return one tensor and whether a default ``step`` axis was added.
    """
    if isinstance(attributions, torch.Tensor):
        return attributions, False
    if not isinstance(attributions, (tuple, list)):
        raise caller_error(
            "'attributions' must be a torch.Tensor or a list or tuple of "
            "tensors.",
            got=describe(attributions),
        )
    if len(attributions) == 0:
        raise caller_error("'attributions' must hold at least one tensor.")
    pieces = list(attributions)
    for position, piece in enumerate(pieces):
        if not isinstance(piece, torch.Tensor):
            raise caller_error(
                "Each item of 'attributions' must be a torch.Tensor.",
                got=f"{describe(piece)} at position {position}",
            )
    first_shape = tuple(pieces[0].shape)
    for position, piece in enumerate(
        pieces[1:],
        start=1,
    ):
        if tuple(piece.shape) != first_shape:
            raise caller_error(
                "Tensors in 'attributions' must all have the same shape.",
                got=(
                    f"shape {first_shape} at position 0 and shape "
                    f"{tuple(piece.shape)} at position {position}"
                ),
            )
    stacked = torch.stack(
        pieces,
        dim=0,
    )
    return stacked, True


def _resolve_dims(
    tensor: torch.Tensor,
    dims: Sequence[str] | None,
    used_default_step: bool,
) -> tuple[str, ...]:
    """
    Choose axis names, requiring ``node`` exactly once.
    """
    if dims is None:
        dim_names = _default_dims(
            tensor=tensor,
            used_default_step=used_default_step,
        )
    else:
        dim_names = tuple(
            _sequence_items(
                dims,
                "dims",
                "a sequence of strings, one per tensor axis",
            )
        )
        for position, name in enumerate(dim_names):
            if not isinstance(name, str):
                raise caller_error(
                    "'dims' must be a sequence of strings.",
                    got=f"{describe(name)} at position {position}",
                )
        if len(dim_names) != tensor.ndim:
            stacked = "the stacked " if used_default_step else ""
            raise caller_error(
                "'dims' must have one name per axis of 'attributions'.",
                got=f"length {len(dim_names)}",
                expected=(
                    f"length {tensor.ndim}, for {stacked}shape "
                    f"{tuple(tensor.shape)}"
                ),
            )
    if _LAYER_COORD in dim_names:
        raise caller_error(
            f"'dims' must not include '{_LAYER_COORD}': kpnn2 reserves it "
            f"for the depth coordinate along '{_NODE_DIM}'."
        )
    repeated = sorted(
        name for name, count in Counter(dim_names).items() if count > 1
    )
    if repeated:
        raise caller_error(
            "'dims' names must be unique.",
            got=f"repeated {listed(repr(name) for name in repeated)}",
        )
    node_count = dim_names.count(_NODE_DIM)
    if node_count != 1:
        raise caller_error(
            f"'dims' must contain '{_NODE_DIM}' exactly once.",
            got=listed(repr(name) for name in dim_names),
            candidates=close_matches(
                dim_names,
                [_NODE_DIM],
            ),
        )
    return dim_names


def _default_dims(
    tensor: torch.Tensor,
    used_default_step: bool,
) -> tuple[str, ...]:
    """
    Default axis names for 1-D and 2-D scores, plus stacked steps.
    """
    ndim = tensor.ndim
    if used_default_step:
        piece_ndim = ndim - 1
        piece_shape = tuple(tensor.shape[1:])
        if piece_ndim == 1:
            return (_STEP_DIM, _NODE_DIM)
        if piece_ndim == 2:
            return (_STEP_DIM, _OBS_DIM, _NODE_DIM)
        if piece_ndim == 0:
            raise caller_error(
                f"Tensors in 'attributions' need a '{_NODE_DIM}' axis.",
                got=f"tensors of shape {piece_shape}",
            )
        raise caller_error(
            "'dims' is required when 'attributions' stacks tensors with "
            "3 or more axes.",
            got=f"tensors of shape {piece_shape}",
            fix=(
                "Pass dims= with one name per axis of the stacked "
                "tensor, the new leading axis included."
            ),
        )
    if ndim == 1:
        return (_NODE_DIM,)
    if ndim == 2:
        return (_OBS_DIM, _NODE_DIM)
    if ndim == 0:
        raise caller_error(
            f"'attributions' needs a '{_NODE_DIM}' axis.",
            got=describe(tensor),
        )
    raise caller_error(
        "'dims' is required when 'attributions' has 3 or more axes.",
        got=describe(tensor),
        fix=f"Pass dims= with one name per axis, '{_NODE_DIM}' among them.",
    )


def _build_coords(
    tensor: torch.Tensor,
    dim_names: tuple[str, ...],
    names: list[str],
    unit_layers: tuple[int, ...] | None,
    coords: Mapping[str, Sequence] | None,
) -> dict[str, object]:
    """
    Build xarray coordinates, with ``node`` fixed from the spec.

    ``layer`` runs along ``node``, one depth per unit, so it stays
    with its units when arrays are concatenated along ``node``.
    It is added only when ``unit_layers`` is not ``None``; an
    ``AdjacencySpec`` result carries no such coordinate.
    """
    extra: dict[str, Sequence] = {}
    if coords is not None:
        if not isinstance(coords, Mapping):
            raise caller_error(
                "'coords' must be a mapping of dim name to labels.",
                got=describe(coords),
            )
        extra = dict(coords)
        if _NODE_DIM in extra or _LAYER_COORD in extra:
            reserved = [
                name for name in (_NODE_DIM, _LAYER_COORD) if name in extra
            ]
            raise caller_error(
                f"'coords' must not include '{_NODE_DIM}' or "
                f"'{_LAYER_COORD}': kpnn2 reserves them for the node names "
                "and depths.",
                got=listed(repr(name) for name in reserved),
            )
        unknown = set(extra) - set(dim_names)
        if unknown:
            # In the caller's order: keys of mixed types do not sort.
            unknown_keys = [key for key in extra if key in unknown]
            others = [name for name in dim_names if name != _NODE_DIM]
            raise caller_error(
                "Unknown dim name(s) in 'coords'.",
                got=listed(repr(key) for key in unknown_keys),
                expected=(
                    f"dim names {listed(repr(name) for name in others)}"
                    if others
                    else None
                ),
                candidates=close_matches(
                    [key for key in unknown_keys if isinstance(key, str)],
                    others,
                ),
            )
    coord_map: dict[str, object] = {}
    for axis, dim in enumerate(dim_names):
        size = int(tensor.shape[axis])
        if dim == _NODE_DIM:
            coord_map[dim] = names
            continue
        if dim in extra:
            labels = _sequence_items(
                extra[dim],
                f"coords[{dim!r}]",
                "a sequence of labels, one per position on that axis",
            )
            if len(labels) != size:
                raise caller_error(
                    f"'coords[{dim!r}]' must have one label per position "
                    f"on the {dim!r} axis.",
                    got=f"length {len(labels)}",
                    expected=f"length {size}",
                )
            coord_map[dim] = labels
        else:
            coord_map[dim] = np.arange(size)
    if unit_layers is not None:
        coord_map[_LAYER_COORD] = (
            _NODE_DIM,
            np.asarray(
                unit_layers,
                dtype=np.int64,
            ),
        )
    return coord_map


def _sequence_items(
    value: object,
    name: str,
    expected: str,
) -> list:
    """
    Return the items of ``value``, an argument that holds one per axis.

    A ``str`` or ``bytes`` is one value, not a sequence of them, so
    it is rejected rather than split into characters. A set or a
    mapping is rejected because its order is not the caller's. So
    is anything ``list`` cannot iterate, a 0-dimensional array
    included.

    Raises
    ------
    Kpnn2Error
        If ``value`` is a ``str``, ``bytes``, set, mapping, or not
        iterable. The message says ``name`` must be ``expected``
        and describes ``value``.
    """
    if isinstance(value, (str, bytes)):
        raise caller_error(
            f"'{name}' must be {expected}, not a single string.",
            got=describe(value),
            fix="Wrap it in a list.",
        )
    reject_unordered(
        value,
        name,
        expected,
    )
    if not isinstance(value, Iterable):
        raise caller_error(
            f"'{name}' must be {expected}.",
            got=describe(value),
        )
    try:
        return list(value)
    except TypeError as exc:
        raise caller_error(
            f"'{name}' must be {expected}.",
            got=describe(value),
        ) from exc
