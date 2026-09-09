"""
Map attribution tensors onto spec node names.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import xarray as xr

from ._adjacency_spec import AdjacencySpec
from ._errors import Kpnn2Error
from ._layout import Layout, build_layout
from ._spec import LayeredSpec

_NODE_DIM = "node"
_LAYER_COORD = "layer"
_OBS_DIM = "observation"
_STEP_DIM = "step"


def map_node_attributions(
    attributions: torch.Tensor | Sequence[torch.Tensor],
    spec: LayeredSpec | AdjacencySpec,
    layer: int | None = None,
    *,
    dims: Sequence[str] | None = None,
    coords: Mapping[str, Sequence] | None = None,
) -> xr.DataArray:
    """
    Label an attribution tensor's node axis with names from a spec.

    Attribution methods return unlabeled tensors whose node axis is
    bare positions; the spec knows the name at each one. Reach for
    it after Captum or your own gradients rather than zipping names
    to columns yourself. Which spec you pass decides the contract: a
    ``LayeredSpec`` requires ``layer``, an ``AdjacencySpec`` forbids
    it. Values are detached onto CPU and never aggregated.

    Parameters
    ----------
    attributions : torch.Tensor or sequence of torch.Tensor
        Scores exactly as the attribution method produced them, in
        whatever units it works in; nothing is scaled, summed, or
        made absolute. The node axis must be as long as the named
        units — ``spec.layer_dims[layer]`` for a ``LayeredSpec``,
        ``len(spec.nodes)`` for an ``AdjacencySpec`` — and the
        remaining axes are yours. A
        non-empty tuple or list of equal-shaped tensors is stacked
        on a new leading ``step`` axis, one entry per unrolled step
        or module call. The tensors are read, never modified, and
        the result holds a detached CPU copy that shares no memory
        with them.
    spec : LayeredSpec or AdjacencySpec
        Parsed edgelist supplying the ``node`` coordinate. A
        ``LayeredSpec`` names one depth at a time, so it needs
        ``layer``; an ``AdjacencySpec`` names the whole state
        vector at once and has no depth to report.
    layer : int, optional
        0-based depth into ``spec.layer_nodes``, index 0 being the
        input layer, whose names label the node axis; the index
        itself is attached as a scalar ``layer`` coordinate.
        Required for a ``LayeredSpec``, rejected for an
        ``AdjacencySpec``. Omitting it is not a default but the
        other half of a spec-dependent contract.
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
        than 1), and a scalar ``layer`` coordinate when a
        ``LayeredSpec`` was passed. Use
        ``.to_dataframe(name="score").reset_index()`` for a long
        table, or ``.to_pandas()`` for a 2-D wide table.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``; ``layer`` is missing for a
        ``LayeredSpec``, given for an ``AdjacencySpec``, or is not
        an int in range; ``attributions`` is neither a tensor nor a
        non-empty sequence of equal-shaped tensors; ``dims`` is
        missing, the wrong length, non-unique, or does not name
        ``node`` exactly once; the node axis is not as long as the
        named units; or ``coords`` names an unknown axis or a
        wrong length.

    See Also
    --------
    align_inputs : Applies the same node order on the way in,
        mapping a named DataFrame onto ``spec.input_nodes``.
    LayeredSpec : Holds ``layer_nodes``, the per-depth names used
        when ``layer`` is given.
    AdjacencySpec : Holds ``nodes``, the state-vector names used
        when ``layer`` is omitted.

    Notes
    -----
    Captum is not imported anywhere in this package; mapping is
    name alignment only, and any attribution method will do. For
    the output of ``PackedLinear`` (or ``MaskedLinear``) on
    ``spec.hops[i]``, pass ``layer=spec.hops[i].target_layer``,
    that is ``i + 1``: the hop output, not its input. Only map
    units that are spec nodes; BatchNorm and other unnamed
    modules have no node axis to name.

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

    On an ``AdjacencySpec`` there are no layers: omit ``layer`` and
    the whole state vector is named. One tensor per unrolled step
    stacks onto a ``step`` axis:

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
    """
    layout, layer_coord = _resolve_node_layout(
        spec,
        layer,
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
    values = tensor.detach().cpu().numpy()
    return xr.DataArray(
        data=values,
        dims=dim_names,
        coords=coord_map,
    )


def _resolve_node_layout(
    spec: LayeredSpec | AdjacencySpec,
    layer: int | None,
) -> tuple[Layout, int | None]:
    """
    Return the ``node`` axis layout and the ``layer`` coordinate.

    The layout supplies both the expected axis length and the
    per-unit names, so a node owning several units would label
    each of them. The coordinate is ``None`` for an
    ``AdjacencySpec``, which has no depths and therefore nothing
    to report as a layer.
    """
    if isinstance(spec, LayeredSpec):
        if layer is None:
            raise Kpnn2Error(
                "'layer' is required for a LayeredSpec. Pass the "
                "0-based index into spec.layer_nodes."
            )
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
        if layer is not None:
            raise Kpnn2Error(
                "'layer' does not apply to an AdjacencySpec: every "
                "node is one unit of a single state vector. Omit "
                "'layer' to label the node axis with spec.nodes."
            )
        return build_layout(spec.nodes), None
    raise Kpnn2Error("'spec' must be a LayeredSpec or an AdjacencySpec.")


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
    not ``None``; an ``AdjacencySpec`` result carries no such
    coordinate.
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
