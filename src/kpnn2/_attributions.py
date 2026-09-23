"""
Map attribution tensors onto spec node names.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import xarray as xr

from ._adjacency_spec import AdjacencySpec
from ._errors import Kpnn2Error
from ._layout import Layout, build_layout, concat_layouts
from ._spec import Hop, LayeredSpec

_NODE_DIM = "node"
_LAYER_COORD = "layer"
_OBS_DIM = "observation"
_STEP_DIM = "step"


def map_node_attributions(
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
        ``axis="inputs"``, which names ``spec.input_nodes``
        (a wide node's name repeats once per unit).
    layer : int, optional
        0-based depth into ``spec.layer_nodes``, index 0 being the
        input layer, whose names label the node axis; the index
        itself is attached as a scalar ``layer`` coordinate.
        ``LayeredSpec`` only, and mutually exclusive with
        ``hop_input`` and ``hop_output``.
    hop_input : Hop, optional
        A ``Hop`` that equals one entry of ``spec.hops``. Labels
        what that hop's module reads: the concatenated source axis
        ``gather_hop_inputs`` returns, length ``hop.in_features``,
        with unit names (a wide node's name repeats). Captum layer
        methods score this side only with
        ``attribute_to_layer_input=True``. No ``layer`` coordinate
        is attached; that axis is not one depth. ``LayeredSpec``
        only, and mutually exclusive with ``layer`` and
        ``hop_output``.
    hop_output : Hop, optional
        A ``Hop`` that equals one entry of ``spec.hops``. Labels
        what that hop's module returns: layer ``hop.target_layer``,
        length ``hop.out_features``. This is the side Captum layer
        methods score by default. The result is the same as
        ``layer=hop.target_layer``, scalar ``layer`` coordinate
        included. ``LayeredSpec`` only, and mutually exclusive with
        ``layer`` and ``hop_input``.
    axis : {"inputs"} or None, optional
        ``"inputs"`` labels the model-input axis and nothing
        else. On a ``LayeredSpec`` that is ``layer=0``,
        including the scalar ``layer`` coordinate, and it is
        mutually exclusive with ``layer``, ``hop_input``, and
        ``hop_output``. On an ``AdjacencySpec`` the names are
        the input units in ``spec.input_index`` order, length
        ``len(spec.input_index)``, with no ``layer``
        coordinate. Any other string is rejected. The length
        of the tensor is not used to choose this axis.
    dims : sequence of str, optional
        One name per axis of the tensor after any stacking,
        containing ``node`` exactly once and never ``layer``.
        Required at 3 or more axes, unless the tensor is a stacked
        sequence of 1-D or 2-D pieces. The defaults are
        ``("node",)`` for 1-D and ``("observation", "node")`` for
        2-D, with ``step`` prepended when a sequence was stacked.
    coords : mapping of str to sequence, optional
        Labels for axes other than ``node`` and ``layer``, keyed by
        dim name; each sequence must be as long as its axis. Axes
        left out are labelled with their integer positions.

    Returns
    -------
    xarray.DataArray
        The values and shape of the (stacked) tensor, carrying the
        spec's node names as the ``node`` coordinate in spec order
        (a name repeats once per unit when that node is wider
        than 1), and a scalar ``layer`` coordinate when ``layer``
        or ``hop_output`` was passed, and when ``axis="inputs"``
        is passed on a ``LayeredSpec``. A ``hop_input`` mapping,
        and ``axis="inputs"`` on an ``AdjacencySpec``, have no
        ``layer`` coordinate. ``bfloat16`` scores are stored as
        ``float32``: NumPy has no bfloat16 dtype, and every
        bfloat16 value fits in float32, so the scores are
        unchanged. Other dtypes are kept. Use
        ``.to_dataframe(name="score").reset_index()`` for a long
        table, or ``.to_pandas()`` for a 2-D wide table.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``; a ``LayeredSpec`` gets none or more
        than one of ``layer``, ``hop_input``, ``hop_output``,
        and ``axis``; ``axis`` is neither ``"inputs"`` nor
        ``None``; any of ``layer``, ``hop_input``, or
        ``hop_output`` is given for an ``AdjacencySpec``;
        ``layer`` is not an int in range; ``hop_input`` or
        ``hop_output`` is not a ``Hop`` that matches an entry
        of ``spec.hops``;
        ``attributions`` is neither a tensor nor a non-empty
        sequence of equal-shaped tensors; ``dims`` is missing, the
        wrong length, non-unique, or does not name ``node`` exactly
        once; the node axis is not as long as the named units; or
        ``coords`` names an unknown axis or a wrong length.

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
    >>> int(da.coords["layer"])
    2

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
    >>> int(out_da.coords["layer"])
    1
    >>> in_da = kpnn2.map_node_attributions(
    ...     attributions=hop_scores,
    ...     spec=square_spec,
    ...     hop_input=hop,
    ... )
    >>> in_da["node"].values.tolist()
    ['A', 'B']
    >>> "layer" in in_da.coords
    False

    A skip hop reads several layers, so its input axis is those
    layers concatenated (width ``hop.in_features``):

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
    layout, layer_coord = _resolve_node_layout(
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
        raise Kpnn2Error(
            "Attribution tensor has the wrong number of units. "
            f"Expected {n_units}, got {tensor.shape[node_axis]}."
        )

    coord_map = _build_coords(
        tensor=tensor,
        dim_names=dim_names,
        names=names,
        layer=layer_coord,
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
) -> tuple[Layout, int | None]:
    """
    Return the ``node`` axis layout and the ``layer`` coordinate.

    The layout supplies both the expected axis length and the
    per-unit names, so a node owning several units would label
    each of them. The coordinate is the depth when ``layer`` or
    ``hop_output`` is given, and ``None`` for an
    ``AdjacencySpec`` or a hop input axis, neither of which is
    one depth. ``axis="inputs"`` on a ``LayeredSpec`` is layer
    0. On an ``AdjacencySpec`` it is the input units only.
    """
    if axis is not None and axis != "inputs":
        raise Kpnn2Error("'axis' must be 'inputs' or None.")
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
            raise Kpnn2Error(
                "A LayeredSpec needs one of 'layer', 'hop_input', "
                "or 'hop_output', or axis='inputs'. Pass the "
                "0-based index into spec.layer_nodes, or a Hop "
                "from spec.hops as hop_output= (what its module "
                "returns) or hop_input= (what its module reads)."
            )
        if len(given) > 1:
            raise Kpnn2Error(
                "Pass exactly one of 'layer', 'hop_input', "
                "'hop_output', or 'axis'. "
                f"Got {', '.join(given)}."
            )
        if hop_input is not None:
            return _layout_for_hop_input(
                spec,
                hop_input,
            ), None
        if hop_output is not None:
            _check_hop(
                spec,
                hop_output,
                "hop_output",
            )
            layer = hop_output.target_layer
        if axis == "inputs":
            return build_layout(
                spec.layer_nodes[0],
                spec.layer_widths[0],
            ), 0
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise Kpnn2Error("'layer' must be an int.")
        n_layers = len(spec.layer_nodes)
        if layer < 0 or layer >= n_layers:
            raise Kpnn2Error(
                f"'layer' must be in range [0, {n_layers}). Got {layer}."
            )
        return build_layout(
            spec.layer_nodes[layer],
            spec.layer_widths[layer],
        ), layer
    if isinstance(spec, AdjacencySpec):
        if layer is not None or hop_input is not None or hop_output is not None:
            raise Kpnn2Error(
                "'layer', 'hop_input', and 'hop_output' do not apply "
                "to an AdjacencySpec: every node is one unit of a "
                "single state vector. Omit them to label the node "
                "axis with spec.nodes, or pass axis='inputs' to "
                "label the input units."
            )
        if axis == "inputs":
            width_of = {
                name: width
                for name, width in zip(
                    spec.nodes,
                    spec.node_widths,
                )
            }
            return build_layout(
                spec.input_nodes,
                [width_of[name] for name in spec.input_nodes],
            ), None
        return build_layout(
            spec.nodes,
            spec.node_widths,
        ), None
    raise Kpnn2Error("'spec' must be a LayeredSpec or an AdjacencySpec.")


def _check_hop(
    spec: LayeredSpec,
    hop: object,
    name: str,
) -> None:
    """
    Raise unless ``hop`` is a ``Hop`` equal to one of ``spec.hops``.
    """
    if not isinstance(hop, Hop):
        raise Kpnn2Error(f"'{name}' must be a Hop from spec.hops.")
    if hop not in spec.hops:
        raise Kpnn2Error(f"'{name}' must match an entry of spec.hops.")


def _layout_for_hop_input(
    spec: LayeredSpec,
    hop: Hop,
) -> Layout:
    """
    Concatenate source-layer layouts of one hop on the spec.
    """
    _check_hop(
        spec,
        hop,
        "hop_input",
    )
    return concat_layouts(
        [
            build_layout(
                spec.layer_nodes[i],
                spec.layer_widths[i],
            )
            for i in hop.source_layers
        ]
    )


def _as_tensor(
    attributions: torch.Tensor | Sequence[torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    """
    Return one tensor and whether a default ``step`` axis was added.
    """
    if isinstance(attributions, torch.Tensor):
        return attributions, False
    if not isinstance(attributions, (tuple, list)):
        raise Kpnn2Error(
            "'attributions' must be a torch.Tensor or a sequence of tensors."
        )
    if len(attributions) == 0:
        raise Kpnn2Error("'attributions' sequence must not be empty.")
    pieces = list(attributions)
    for piece in pieces:
        if not isinstance(piece, torch.Tensor):
            raise Kpnn2Error(
                "Each item in 'attributions' must be a torch.Tensor."
            )
    first_shape = tuple(pieces[0].shape)
    for piece in pieces[1:]:
        if tuple(piece.shape) != first_shape:
            raise Kpnn2Error(
                "Tensors in 'attributions' must all have the same shape."
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
            ndim=tensor.ndim,
            used_default_step=used_default_step,
        )
    else:
        dim_names = tuple(dims)
        if any(not isinstance(name, str) for name in dim_names):
            raise Kpnn2Error("'dims' must be a sequence of strings.")
        if len(dim_names) != tensor.ndim:
            raise Kpnn2Error(
                "'dims' length must match the number of tensor "
                f"axes. Expected {tensor.ndim}, got "
                f"{len(dim_names)}."
            )
    if _LAYER_COORD in dim_names:
        raise Kpnn2Error(f"'dims' must not include '{_LAYER_COORD}'.")
    if len(set(dim_names)) != len(dim_names):
        raise Kpnn2Error("'dims' names must be unique.")
    node_count = dim_names.count(_NODE_DIM)
    if node_count != 1:
        raise Kpnn2Error(f"'dims' must contain '{_NODE_DIM}' exactly once.")
    return dim_names


def _default_dims(
    ndim: int,
    used_default_step: bool,
) -> tuple[str, ...]:
    """
    Default axis names for 1-D and 2-D scores, plus stacked steps.
    """
    if used_default_step:
        piece_ndim = ndim - 1
        if piece_ndim == 1:
            return (_STEP_DIM, _NODE_DIM)
        if piece_ndim == 2:
            return (_STEP_DIM, _OBS_DIM, _NODE_DIM)
        raise Kpnn2Error(
            "Pass dims= when stacking tensors with 3 or more axes."
        )
    if ndim == 1:
        return (_NODE_DIM,)
    if ndim == 2:
        return (_OBS_DIM, _NODE_DIM)
    raise Kpnn2Error("Pass dims= for attribution tensors with 3 or more axes.")


def _build_coords(
    tensor: torch.Tensor,
    dim_names: tuple[str, ...],
    names: list[str],
    layer: int | None,
    coords: Mapping[str, Sequence] | None,
) -> dict[str, object]:
    """
    Build xarray coordinates, with ``node`` fixed from the spec.

    The scalar ``layer`` coordinate is added only when ``layer`` is
    not ``None``; an ``AdjacencySpec`` or hop-source result
    carries no such coordinate.
    """
    extra: dict[str, Sequence] = {}
    if coords is not None:
        if not isinstance(coords, Mapping):
            raise Kpnn2Error("'coords' must be a mapping.")
        extra = dict(coords)
        if _NODE_DIM in extra or _LAYER_COORD in extra:
            raise Kpnn2Error(
                f"'coords' must not include '{_NODE_DIM}' or '{_LAYER_COORD}'."
            )
        unknown = set(extra) - set(dim_names)
        if unknown:
            keys = ", ".join(sorted(unknown))
            raise Kpnn2Error(f"'coords' has unknown dim name(s): {keys}.")
    coord_map: dict[str, object] = {}
    for axis, dim in enumerate(dim_names):
        size = int(tensor.shape[axis])
        if dim == _NODE_DIM:
            coord_map[dim] = names
            continue
        if dim in extra:
            labels = list(extra[dim])
            if len(labels) != size:
                raise Kpnn2Error(
                    f"'coords[{dim!r}]' length must be {size}, "
                    f"got {len(labels)}."
                )
            coord_map[dim] = labels
        else:
            coord_map[dim] = np.arange(size)
    if layer is not None:
        coord_map[_LAYER_COORD] = layer
    return coord_map
