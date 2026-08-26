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
    Label an attribution tensor with node names from a spec.

    Values are copied with ``detach()`` onto CPU. This function does
    not run an attribution method and does not import Captum. Pass
    the tensor (or per-call tensors) you already computed. Nothing
    is summed or averaged.

    A 2-D ``(batch, n_units)`` tensor becomes dims
    ``(observation, node)``. Extra Captum axes need ``dims`` so
    one axis is named ``node``. A tuple of equal-shaped tensors is
    stacked on a new ``step`` axis (one entry per module call).

    Where the names come from depends on the spec:

    - ``LayeredSpec``: ``layer`` is required and names come from
      ``spec.layer_nodes[layer]``. The result carries a scalar
      ``layer`` coordinate.
    - ``AdjacencySpec``: ``layer`` must be omitted and names come
      from ``spec.nodes``, the whole state vector. The result
      carries no ``layer`` coordinate, because there is no depth.

    Parameters
    ----------
    attributions : Tensor or sequence of Tensor
        Scores whose ``node`` axis length equals the number of
        named units: ``len(spec.layer_nodes[layer])`` for a
        ``LayeredSpec``, ``len(spec.nodes)`` for an
        ``AdjacencySpec``. A sequence is stacked along ``step``.
    spec : LayeredSpec or AdjacencySpec
        Graph structure supplying the ``node`` coordinate.
    layer : int, optional
        0-based index into ``spec.layer_nodes``. Layer 0 is the
        input layer. Stored as a scalar coordinate ``layer``.
        Required for a ``LayeredSpec``; must be omitted for an
        ``AdjacencySpec``.
    dims : sequence of str, optional
        Name of each axis of the (stacked) tensor. Must contain
        ``node`` exactly once. Required when the tensor has 3 or
        more axes (except a default-stacked 2-D sequence).
    coords : mapping, optional
        Labels for axes other than ``node`` and ``layer``. Length
        of each entry must match that axis. ``node`` is set from
        ``spec``, and ``layer`` from the ``layer`` argument.

    Returns
    -------
    DataArray
        Raw scores with a ``node`` coordinate from the spec, plus a
        scalar ``layer`` coordinate for a ``LayeredSpec``. Use
        ``.to_dataframe(name="score").reset_index()`` for a long
        table, or ``.to_pandas()`` for a 2-D wide table.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is neither a ``LayeredSpec`` nor an
        ``AdjacencySpec``; ``layer`` is missing for a
        ``LayeredSpec``, given for an ``AdjacencySpec``, or not a
        valid int; ``attributions`` is not a tensor or a non-empty
        sequence of equal-shaped tensors; ``dims`` is missing or
        inconsistent; a ``node`` axis length does not match the
        named units; or ``coords`` is invalid.

    Notes
    -----
    For a ``MaskedLinear`` built from ``spec.hops[i].mask``, the
    layer index to pass here is ``spec.hops[i].target_layer``,
    that is ``i + 1``: the hop output, not its input.
    Do not name-map tensors from BatchNorm or other unnamed
    modules; only map units that are spec nodes.

    A recurrent net built on an ``AdjacencySpec`` has no layers to
    index. The natural extra axis there is ``step``: pass one
    tensor per time step as a sequence and they are stacked for
    you.

    Examples
    --------
    Name a two-row tensor at the output layer:

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
    >>> spec.layer_nodes
    (('A',), ('H',), ('C',))
    >>> scores = torch.tensor([[0.5], [1.0]])
    >>> da = k2.map_node_attributions(
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

    The ``layer`` argument is an index into ``spec.layer_nodes``:

    >>> hidden = k2.map_node_attributions(
    ...     attributions=torch.zeros(2, 1),
    ...     spec=spec,
    ...     layer=1,
    ... )
    >>> hidden["node"].values.tolist()
    ['H']

    On an ``AdjacencySpec`` there are no layers: omit ``layer``
    and the whole state vector is named. One tensor per time step
    stacks onto a ``step`` axis:

    >>> cyclic = pd.DataFrame(
    ...     {
    ...         "source": ["x", "a", "b", "a"],
    ...         "target": ["a", "b", "a", "y"],
    ...     }
    ... )
    >>> state_spec = k2.parse_adjacency(cyclic)
    >>> per_step = k2.map_node_attributions(
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
        return build_layout(spec.layer_nodes[layer]), layer
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
