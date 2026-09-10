"""
Structural blueprint for an edgelist-defined DAG.
"""

from dataclasses import dataclass
from itertools import accumulate

import pandas as pd
from torch import Tensor

from ._errors import Kpnn2Error
from ._layout import (
    build_layout,
    hop_axis_layouts,
    packed_indices_for_named_edge,
    resolve_edge_names,
)


def _layered_node_layer(
    layer_nodes: tuple[tuple[str, ...], ...],
    name: object,
) -> tuple[int, str]:
    """
    Return the depth and string name of one layered node.
    """
    node_name = str(name)
    if node_name == "":
        raise Kpnn2Error("Node name is empty.")
    for layer, names in enumerate(layer_nodes):
        if node_name in names:
            return layer, node_name
    raise Kpnn2Error(f"Unknown node name: {node_name}.")


@dataclass(frozen=True)
class Hop:
    """
    Every edge entering one layer, as packed indices.

    One entry of ``LayeredSpec.hops``, never built by hand: a hop
    is exactly what a single ``PackedLinear`` or
    ``MaskedLinear`` computes. Packed ``source_index`` /
    ``target_index`` hold **all** parents of ``target_layer``, so
    a skip edge is an ordinary pair rather than a term added
    later, and no edge can be dropped. Columns are the source
    layers concatenated, the axis ``gather_hop_inputs``
    assembles. There is no stored mask; ``to_mask()`` allocates
    one.

    Parameters
    ----------
    target_layer : int
        Depth of the layer this hop produces. Always at least 1;
        layer 0 has no parents. ``LayeredSpec.hops[i]`` has
        ``target_layer == i + 1``.
    source_layers : tuple[int, ...]
        Depths this hop reads, ascending, each one below
        ``target_layer``. Only layers that really feed the
        target appear. Under longest-path ranking,
        ``target_layer - 1`` is always one of them; with
        ``parse_layered(..., ranks=)`` a hop may omit the
        previous layer when every parent is a skip. A single
        entry is a plain adjacent hop when that entry is
        ``target_layer - 1``; ``hops[0]`` is always ``(0,)``,
        so an ``align_inputs`` tensor feeds it with no
        gathering.
    source_dims : tuple[int, ...]
        Units contributed by each entry of ``source_layers``,
        same order. Their sum is ``in_features``.
    source_nodes : tuple[str, ...]
        Node names of the concatenated source axis, source
        layers in ``source_layers`` order. One name per node,
        not per unit. With width greater than 1,
        ``len(source_nodes)`` is smaller than ``in_features``.
    target_dim : int
        Units in the target layer, equal to
        ``layer_dims[target_layer]`` and to ``out_features``.
    source_index : tuple[int, ...]
        Concat-column of each live unit pair. A named edge
        ``A -> B`` expands into every pair of the
        ``(k_B, k_A)`` block, target-unit outer, source-unit
        inner. Same length as ``target_index``. Named edges
        are canonical: lexicographic by
        ``(source name, target name)``.
    target_index : tuple[int, ...]
        Target-layer row of each live unit pair. A dense
        rectangle would have ``1.0`` at
        ``[target_index[i], source_index[i]]``.

    See Also
    --------
    LayeredSpec : Holds ``hops``, one per layer after the first.
    gather_hop_inputs : Builds the tensor whose columns these
        indices address.
    scatter_hop_outputs : Splits that concatenated axis back
        onto source layers.
    PackedLinear : Applies one hop from the packed indices.
    PackedLinear.transpose : Tied decode of this hop.
    MaskedLinear : Applies one hop after ``to_mask()``.
    Skip : Metadata for the edges in this hop that span layers.
    LayeredSpec.hop_units : Slice of one named node on this
        hop's concatenated source axis.
    LayeredSpec.node_units : Slice of one named node on its
        layer tensor.

    Notes
    -----
    Every named edge is a block of packed unit pairs in exactly
    one hop, the one of its target layer. At width 1 that block
    is a single pair and the pairs summed over all hops give
    the named-edge count. Applying a hop applies every parent
    of its layer at once.

    To locate one source layer's block on the concatenated axis,
    add the widths in front of it:

    ``offset = sum(source_dims[:source_layers.index(layer)])``

    ``column_offsets`` does that for you.
    ``LayeredSpec.hop_units`` locates one named node on that
    axis, widths included.

    Examples
    --------
    A chain ``A -> H -> C`` plus the skip ``A -> C``:

    >>> import pandas as pd
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> spec = kpnn2.parse_layered(edgelist)
    >>> hop = spec.hops[1]
    >>> hop.target_layer, hop.source_layers
    (2, (0, 1))
    >>> hop.source_nodes
    ('A', 'H')
    >>> hop.source_index, hop.target_index
    ((0, 1), (0, 0))
    >>> hop.to_mask().tolist()
    [[1.0, 1.0]]
    """

    target_layer: int
    source_layers: tuple[int, ...]
    source_dims: tuple[int, ...]
    source_nodes: tuple[str, ...]
    target_dim: int
    source_index: tuple[int, ...]
    target_index: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_layers",
            tuple(self.source_layers),
        )
        object.__setattr__(
            self,
            "source_dims",
            tuple(self.source_dims),
        )
        object.__setattr__(
            self,
            "source_nodes",
            tuple(self.source_nodes),
        )
        object.__setattr__(
            self,
            "source_index",
            tuple(self.source_index),
        )
        object.__setattr__(
            self,
            "target_index",
            tuple(self.target_index),
        )

    @property
    def in_features(self) -> int:
        """
        Width of the concatenated source axis.

        Equal to ``sum(source_dims)``. This is
        ``PackedLinear`` ``in_features`` and the last axis
        ``gather_hop_inputs`` returns. Unused skip columns stay
        in this width.
        """
        return sum(self.source_dims)

    @property
    def out_features(self) -> int:
        """
        Width of the target layer.

        Equal to ``target_dim``. This is ``PackedLinear``
        ``out_features``.
        """
        return self.target_dim

    @property
    def column_offsets(self) -> tuple[int, ...]:
        """
        First source column of each entry of ``source_layers``.

        Same length and order as ``source_layers``. Add a node's
        block start inside its own layer to get its concatenated
        first unit. At width 1 the block start equals the node's
        ordinal in ``layer_nodes``.
        """
        return tuple(
            accumulate(
                self.source_dims[:-1],
                initial=0,
            )
        )

    def to_mask(self) -> Tensor:
        """
        Allocate a dense float32 rectangle from the packed edges.

        Shape is ``(out_features, in_features)``. The result
        starts at zeros; each live edge sets ``1.0`` at
        ``[target_index[i], source_index[i]]``. Every call
        returns a fresh tensor. Mutating it does not change this
        hop or the next ``to_mask()`` call.

        Returns
        -------
        torch.Tensor
            New dense connectivity rectangle. This allocates.

        Examples
        --------
        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["A", "H", "A"],
        ...         "target": ["H", "C", "C"],
        ...     }
        ... )
        >>> hop = kpnn2.parse_layered(edgelist).hops[1]
        >>> tuple(hop.to_mask().shape)
        (1, 2)
        >>> hop.to_mask().tolist()
        [[1.0, 1.0]]
        """
        from ._layout import dense_mask_from_indices

        return dense_mask_from_indices(
            self.source_index,
            self.target_index,
            self.out_features,
            self.in_features,
        )


@dataclass(frozen=True)
class Skip:
    """
    One original edge whose endpoints are more than one layer apart.

    Metadata, not a second computation: the named edge is already
    a block of packed unit pairs in
    ``LayeredSpec.hops[target_layer - 1]``, exactly like an
    adjacent edge, so nothing has to add it back later and
    nothing can forget to. Read ``LayeredSpec.skips`` to inspect
    which prior-knowledge edges span layers; a forward pass
    never reads it. ``parse_layered`` builds these, never the
    caller.

    Parameters
    ----------
    source : str
        Name of the node the edge leaves, as it appears in
        ``LayeredSpec.layer_nodes[source_layer]``.
    target : str
        Name of the node the edge enters, as it appears in
        ``LayeredSpec.layer_nodes[target_layer]``.
    source_layer : int
        Depth of ``source``: its index into
        ``LayeredSpec.layer_nodes``.
    target_layer : int
        Depth of ``target``. Always at least two above
        ``source_layer``; that gap is what makes the edge a skip,
        and it names the hop carrying it,
        ``hops[target_layer - 1]``.
    source_in_layer : int
        First unit of ``source`` inside its layer (the block
        start). At width 1 this equals the node's index in
        ``layer_nodes[source_layer]``. It is not a column of
        the hop's concatenated source axis.
        ``LayeredSpec.edge_location`` returns the packed slots.
    target_in_layer : int
        First unit of ``target`` inside its layer (the block
        start). At width 1 this equals the node's index in
        ``layer_nodes[target_layer]``. Locating the skip among
        packed indices means every unit pair in that block is
        live, not a single ``(column, row)`` pair. Use
        ``LayeredSpec.edge_location``.

    See Also
    --------
    Hop : The packed edges this skip is already one of, alongside
        every other parent of ``target_layer``.
    LayeredSpec : Holds ``skips``, empty when no edge spans layers.
    parse_layered : Builds the spec these records come from.

    Notes
    -----
    Every original edge with a depth gap greater than 1 is recorded
    once; adjacent edges never are. Membership changes nothing about
    how the edge is computed: its weight, the unit bias, and the
    fan-in the degree-aware initialization uses all stay on the
    target layer's ``PackedLinear`` or ``MaskedLinear``. Expanding
    a skip into dummy neurons is not the intended use. At width
    greater than 1 the stored indices are block starts; every
    unit pair of that block is a packed index of the hop.

    Examples
    --------
    Locate a skip among packed indices. This graph is
    width 1, so the skip is one packed pair:

    >>> import pandas as pd
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "B", "H", "A"],
    ...         "target": ["H", "H", "C", "C"],
    ...     }
    ... )
    >>> spec = kpnn2.parse_layered(edgelist)
    >>> skip = spec.skips[0]
    >>> skip.source, skip.target
    ('A', 'C')
    >>> skip.source_layer, skip.target_layer
    (0, 2)
    >>> hop_index, packed = spec.edge_location(
    ...     skip.source,
    ...     skip.target,
    ... )
    >>> hop_index
    1
    >>> packed
    (0,)
    """

    source: str
    target: str
    source_layer: int
    target_layer: int
    source_in_layer: int
    target_in_layer: int


@dataclass(frozen=True)
class LayeredSpec:
    """
    Frozen blueprint from ``parse_layered``.

    Depth-ranked wiring for one knowledge-primed network: every
    node sits at a layer, and one ``Hop`` per layer after the
    first holds every edge entering it, skips included, as packed
    indices. Build one ``PackedLinear`` per hop, and use the name
    tuples to label tensors. It is structure only — no
    ``nn.Module``, no parameters — and there is no stored mask.
    ``AdjacencySpec`` is the shared-state packed alternative.

    Parameters
    ----------
    input_nodes : tuple[str, ...]
        In-degree 0 names, alphabetical. This is the DataFrame
        column order ``align_inputs`` reads. The returned tensor
        width is ``layer_dims[0]``.
    output_nodes : tuple[str, ...]
        Out-degree 0 names, alphabetical. A terminal node below
        maximum depth belongs here too, so this is not the same
        tuple as ``layer_nodes[-1]``.
    hidden_nodes : tuple[str, ...]
        Names that are neither input nor output, alphabetical.
    layer_nodes : tuple[tuple[str, ...], ...]
        ``layer_nodes[i]`` is the names at depth ``i``, alphabetical.
        Index 0 is the input layer. Depth is longest path from the
        inputs unless ``parse_layered(..., ranks=)`` assigned
        compact ranks, and there are always at least two layers.
        One name per node, not per unit.
    layer_dims : tuple[int, ...]
        Unit count of each layer: ``layer_dims[i]`` is
        ``sum(layer_widths[i])``. Equal to
        ``len(layer_nodes[i])`` only when every node at that
        depth has width 1.
    layer_widths : tuple[tuple[int, ...], ...]
        ``layer_widths[i][j]`` is the width of
        ``layer_nodes[i][j]``. Default parse yields 1 for every
        node.
    hops : tuple[Hop, ...]
        One hop per layer after the first:
        ``len(hops) == len(layer_nodes) - 1`` and
        ``hops[i].target_layer == i + 1``. ``hops[i]`` holds
        **every** edge entering layer ``i + 1``, adjacent and
        skip alike, as packed indices over the concatenated
        source layers. ``hops[0]`` always reads layer 0 only.
    skips : tuple[Skip, ...]
        Original edges with depth gap greater than 1, as metadata.
        Each one is already a block of packed unit pairs in
        ``hops[target_layer - 1]``; this list only says which
        edges span layers, and is empty when none do. Use
        ``edge_location`` to find the packed slots of a named
        edge, skip or adjacent.

    See Also
    --------
    parse_layered : Builds this spec from a ``source`` / ``target``
        edgelist.
    AdjacencySpec : Packed sibling layout, for cycles, self-loops,
        or one shared state vector instead of depths.
    gather_hop_inputs : Assembles one hop's input from the layer
        tensors produced so far.
    scatter_hop_outputs : Splits a transposed hop's output back
        onto those source layers.
    PackedLinear : Consumes ``hops[i].source_index`` /
        ``target_index`` as one layer.
    PackedLinear.transpose : Tied decode of that hop.
    MaskedLinear : Consumes ``hops[i].to_mask()`` as one layer.

    Notes
    -----
    Fields cannot be reassigned and sequences are tuples, so the
    structure itself is fixed. There is no stored mask tensor and
    no densifying ``mask`` property. ``to_mask()`` on each hop
    allocates a fresh dense rectangle; mutating that tensor does
    not change this spec. ``MaskedLinear(hop.to_mask())`` clones
    the rectangle into a non-persistent buffer, so a layer built
    earlier keeps its own connectivity.

    Because a hop carries every parent of its target, the
    per-row degree ``PackedLinear`` and ``MaskedLinear``
    initialize from is the real fan-in of that unit, skips
    included.

    ``to_edgelist()``, ``to_dict()`` with ``from_dict()``, and
    ``fingerprint`` are the supported interchange. Widths and
    ranks live on ``to_dict()``, not on the edgelist: reparse of
    ``to_edgelist()`` without ``widths=`` / ``ranks=`` is
    width 1 and longest-path. ``to_dict()`` omits ``"ranks"``
    when compacted layers equal longest-path on the same edges.
    Pickle and ``torch.save`` of the dataclass are not.
    ``edge_location`` finds packed slots of a named edge; it is
    not a constraint. ``node_units`` and ``hop_units`` map a
    named node to its contiguous unit slice on a layer tensor
    or a hop source axis.

    Examples
    --------
    Inspect layers, a hop, and a skip after parsing:

    >>> import pandas as pd
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> spec = kpnn2.parse_layered(edgelist)
    >>> spec.layer_nodes
    (('A',), ('H',), ('C',))
    >>> spec.hops[0].source_layers, spec.hops[0].to_mask().tolist()
    ((0,), [[1.0]])
    >>> spec.hops[1].source_layers, spec.hops[1].to_mask().tolist()
    ((0, 1), [[1.0, 1.0]])
    >>> spec.skips[0].source, spec.skips[0].target
    ('A', 'C')
    >>> spec.skips[0].source_layer, spec.skips[0].target_layer
    (0, 2)
    >>> spec.edge_location("A", "C")
    (1, (0,))
    """

    input_nodes: tuple[str, ...]
    output_nodes: tuple[str, ...]
    hidden_nodes: tuple[str, ...]
    layer_nodes: tuple[tuple[str, ...], ...]
    layer_dims: tuple[int, ...]
    layer_widths: tuple[tuple[int, ...], ...]
    hops: tuple[Hop, ...]
    skips: tuple[Skip, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_nodes",
            tuple(self.input_nodes),
        )
        object.__setattr__(
            self,
            "output_nodes",
            tuple(self.output_nodes),
        )
        object.__setattr__(
            self,
            "hidden_nodes",
            tuple(self.hidden_nodes),
        )
        object.__setattr__(
            self,
            "layer_nodes",
            tuple(tuple(layer) for layer in self.layer_nodes),
        )
        object.__setattr__(
            self,
            "layer_dims",
            tuple(self.layer_dims),
        )
        object.__setattr__(
            self,
            "layer_widths",
            tuple(tuple(layer) for layer in self.layer_widths),
        )
        object.__setattr__(
            self,
            "hops",
            tuple(self.hops),
        )
        object.__setattr__(
            self,
            "skips",
            tuple(self.skips),
        )

    def to_edgelist(self) -> pd.DataFrame:
        """
        Return this spec's edges as a two-column table.

        Columns are exactly ``source`` then ``target``. Rows
        follow the packed hop edges in canonical order: sorted
        lexicographically by ``(source, target)``, one row per
        original edge, names as strings. Extra columns from the
        DataFrame that was parsed are not reproduced.

        ``parse_layered`` on this table reconstructs the same
        node lists and named edges when every node has width 1
        and ranking is longest-path. Packed hop indices and
        ``layer_dims`` match in that case. Otherwise pass
        ``widths=`` and/or ``ranks=``, or use ``from_dict()``,
        which reads those keys from the tagged dict. Skip tuple
        order follows these sorted rows rather than the original
        parse input order; the skip *set* matches.

        Returns
        -------
        pandas.DataFrame
            One row per original edge.

        Examples
        --------
        Unsorted input comes back sorted:

        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["H", "A", "A"],
        ...         "target": ["C", "C", "H"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_layered(edgelist)
        >>> table = spec.to_edgelist()
        >>> list(table.columns)
        ['source', 'target']
        >>> table["source"].tolist()
        ['A', 'A', 'H']
        >>> table["target"].tolist()
        ['C', 'H', 'C']
        """
        from ._serialize import spec_to_edgelist

        return spec_to_edgelist(self)

    def edge_location(
        self,
        source: object,
        target: object,
    ) -> tuple[int, tuple[int, ...]]:
        """
        Return packed weight slots of one named edge.

        ``source`` and ``target`` are matched after ``str(...)``,
        same as parse. This is identity into the hop that
        carries the edge, not a constraint.

        Parameters
        ----------
        source : str
            Source node name. Non-strings are converted with
            ``str(...)``.
        target : str
            Target node name. Non-strings are converted with
            ``str(...)``.

        Returns
        -------
        hop_index : int
            Index ``i`` such that the named edge is in
            ``hops[i]``.
        packed_indices : tuple of int
            Indices into ``hops[i].source_index`` /
            ``target_index`` and the corresponding
            ``PackedLinear.weight``. Length is
            ``k_source * k_target`` (1 at default width).
            Order is the stored order: named edges canonical
            lexicographic by ``(source name, target name)``;
            within one named edge, target-unit outer,
            source-unit inner.

        Raises
        ------
        Kpnn2Error
            If the pair is missing, a name is empty, or a name
            is not a node. The message names the pair as
            ``{source} -> {target}``.

        Notes
        -----
        Width greater than 1 does not change the named edge. It
        returns several packed indices, one per unit pair of
        the block.

        Examples
        --------
        A chain plus a skip. The skip is on the later hop:

        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["A", "H", "A"],
        ...         "target": ["H", "C", "C"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_layered(edgelist)
        >>> spec.edge_location("A", "H")
        (0, (0,))
        >>> spec.edge_location("A", "C")
        (1, (0,))
        """
        known_names: set[str] = set()
        for layer in self.layer_nodes:
            known_names.update(layer)
        source_name, target_name = resolve_edge_names(
            source,
            target,
            known_names,
        )
        for hop_index, hop in enumerate(self.hops):
            source_layout, target_layout = hop_axis_layouts(
                self.layer_nodes,
                self.layer_widths,
                hop.source_layers,
                hop.target_layer,
            )
            packed = packed_indices_for_named_edge(
                hop.source_index,
                hop.target_index,
                source_layout,
                target_layout,
                source_name,
                target_name,
            )
            if packed:
                return hop_index, packed
        raise Kpnn2Error(f"No edge {source_name} -> {target_name}.")

    def node_units(
        self,
        name: object,
    ) -> tuple[int, slice]:
        """
        Return the layer and unit slice of one named node.

        ``name`` is matched after ``str(...)``, same as parse.
        The slice indexes the last axis of that layer's tensor
        (``saved[layer]``). It is always a slice, including at
        width 1.

        Parameters
        ----------
        name : str
            Node name. Non-strings are converted with
            ``str(...)``.

        Returns
        -------
        layer : int
            Depth of the node: index into ``layer_nodes`` and
            the key of a saved-layer dict.
        units : slice
            Contiguous columns on that layer's last axis.
            Length is the node's width.

        Raises
        ------
        Kpnn2Error
            If ``name`` is empty or is not a node.

        Notes
        -----
        This is identity into the unit axis, not a dropout
        module and not a head helper. Index as
        ``saved[layer][..., units]``.

        Examples
        --------
        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["A", "H"],
        ...         "target": ["H", "C"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_layered(
        ...     edgelist,
        ...     widths={"H": 3},
        ... )
        >>> layer, units = spec.node_units("H")
        >>> layer, units.start, units.stop
        (1, 0, 3)
        """
        layer, node_name = _layered_node_layer(
            self.layer_nodes,
            name,
        )
        layout = build_layout(
            self.layer_nodes[layer],
            self.layer_widths[layer],
        )
        return layer, layout.slot(node_name).units

    def hop_units(
        self,
        hop: Hop,
        name: object,
    ) -> slice:
        """
        Return the unit slice of one named node on a hop axis.

        The axis is the concatenated source of ``hop``, the
        same tensor ``gather_hop_inputs`` returns.
        ``name`` is matched after ``str(...)``, same as parse.
        ``hop`` must compare equal to one entry of ``hops``.

        Parameters
        ----------
        hop : Hop
            A hop from ``spec.hops``.
        name : str
            Node name. Non-strings are converted with
            ``str(...)``.

        Returns
        -------
        slice
            Contiguous columns on ``hop.in_features``. Length
            is the node's width. Index as
            ``sources[..., units]``.

        Raises
        ------
        Kpnn2Error
            If ``hop`` is not a ``Hop``, does not match an
            entry of ``hops``, ``name`` is empty, ``name`` is
            not a node, or the node is not a source of this
            hop.

        Notes
        -----
        ``column_offsets`` locates a whole source layer on this
        axis. This method locates one named node, widths
        included. A target-layer name is not on the source
        axis.

        Examples
        --------
        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["A", "H", "A"],
        ...         "target": ["H", "C", "C"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_layered(edgelist)
        >>> hop = spec.hops[1]
        >>> spec.hop_units(hop, "A").start
        0
        >>> spec.hop_units(hop, "H").start
        1
        """
        if not isinstance(hop, Hop):
            raise Kpnn2Error("'hop' must be a Hop from spec.hops.")
        if hop not in self.hops:
            raise Kpnn2Error("'hop' must match an entry of spec.hops.")
        _, node_name = _layered_node_layer(
            self.layer_nodes,
            name,
        )
        source_layout, _ = hop_axis_layouts(
            self.layer_nodes,
            self.layer_widths,
            hop.source_layers,
            hop.target_layer,
        )
        if node_name not in source_layout.names:
            raise Kpnn2Error(
                f"Node {node_name} is not on this hop's source axis."
            )
        return source_layout.slot(node_name).units

    def to_dict(self) -> dict:
        """
        Return this spec as a JSON-safe tagged dict.

        Keys are ``kpnn2_spec`` (integer ``1``), ``layout``
        (``"layered"``), and ``edges`` (list of
        ``[source, target]`` lists in the same order as
        ``to_edgelist()`` rows). When any node has width other
        than 1, a ``"widths"`` object of those names is included.
        All-1 graphs omit ``"widths"``. When compacted layers
        differ from longest-path on the same edges, a ``"ranks"``
        object maps every node to its compacted 0-based layer
        index. A ``ranks=`` parse that matches longest-path
        omits ``"ranks"``. The returned dict is new on every
        call.

        Returns
        -------
        dict
            Tagged edge list plus layout.

        Examples
        --------
        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["A", "H"],
        ...         "target": ["H", "C"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_layered(edgelist)
        >>> payload = spec.to_dict()
        >>> payload["kpnn2_spec"]
        1
        >>> payload["layout"]
        'layered'
        >>> payload["edges"]
        [['A', 'H'], ['H', 'C']]
        """
        from ._serialize import spec_to_dict

        return spec_to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "LayeredSpec":
        """
        Rebuild a ``LayeredSpec`` from ``to_dict()`` output.

        Calls ``parse_layered`` on a DataFrame built from
        ``payload["edges"]``, passing ``payload["widths"]`` and
        ``payload["ranks"]`` when present. Hops and packed
        indices are not assembled by hand. Extra unknown keys
        are ignored. An absent or empty ``"widths"`` object is
        width 1. Absent ``"ranks"`` is longest-path.

        Parameters
        ----------
        payload : dict
            A dict with ``kpnn2_spec``, ``layout``, and
            ``edges``. ``layout`` must be ``"layered"``.
            Optional ``"widths"`` is a node-name-to-int object.
            Optional ``"ranks"`` is a node-name-to-int object of
            compacted (or user) depths.

        Returns
        -------
        LayeredSpec
            The parsed spec.

        Raises
        ------
        Kpnn2Error
            If ``payload`` is not a dict; ``kpnn2_spec`` is
            missing or not ``1``; ``layout`` is missing, not a
            known layout, or is ``"adjacency"``; ``edges``
            is missing or not a sequence of two nonempty names;
            ``"widths"`` is present and not a mapping of
            positive ints; or ``"ranks"`` is present and not a
            mapping of non-negative ints covering every node.

        Examples
        --------
        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["A", "H"],
        ...         "target": ["H", "C"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_layered(edgelist)
        >>> roundtrip = kpnn2.LayeredSpec.from_dict(spec.to_dict())
        >>> roundtrip.layer_nodes == spec.layer_nodes
        True
        """
        from ._serialize import layered_spec_from_dict

        return layered_spec_from_dict(payload)

    @property
    def fingerprint(self) -> str:
        """
        SHA-256 hex digest of the canonical ``to_dict()`` JSON.

        The payload is ``json.dumps(self.to_dict(),
        sort_keys=True, separators=(",", ":"),
        ensure_ascii=False)`` encoded as UTF-8. The result is
        64 lowercase hex characters. It is not Python
        ``hash()``.

        Returns
        -------
        str
            Hex digest of the tagged spec dict.

        Examples
        --------
        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["A", "H"],
        ...         "target": ["H", "C"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_layered(edgelist)
        >>> len(spec.fingerprint)
        64
        >>> (
        ...     spec.fingerprint
        ...     == kpnn2.parse_layered(spec.to_edgelist()).fingerprint
        ... )
        True
        """
        from ._serialize import spec_fingerprint

        return spec_fingerprint(self)
