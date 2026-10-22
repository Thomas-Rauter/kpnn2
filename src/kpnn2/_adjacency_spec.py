"""
Structural blueprint for an edgelist-defined node network.
"""

from dataclasses import dataclass

import pandas as pd
from torch import Tensor

from ._errors import caller_error
from ._layout import (
    Layout,
    build_layout,
    dense_mask_from_indices,
    edges_by_slot,
    no_edge_error,
    resolve_edge_names,
    resolve_edge_pairs,
    resolve_node_name,
    resolve_node_names,
)
from ._validate import as_name_list, describe


@dataclass(frozen=True)
class AdjacencySpec:
    """
    Frozen blueprint from ``parse_adjacency``.

    Packed wiring for one knowledge-primed network: every node
    owns a contiguous block of units (one unit unless
    ``parse_adjacency(widths=)`` says otherwise) of one
    alphabetical state vector, and every edge is packed unit
    pairs, so cycles and self-loops are ordinary. Structure
    only — no ``nn.Module``, no parameters, no stored square — and
    the update is yours. Input nodes have no incoming edges, so
    writing the inputs into the state each step is required.
    ``LayeredSpec`` is the depth-ranked alternative.

    Parameters
    ----------
    nodes : tuple[str, ...]
        Every node name, alphabetical. This is the node order of
        the state vector and of the rows and columns of
        ``to_mask()``; ``node_units(name)`` is each node's unit
        block.
    node_widths : tuple[int, ...]
        Units per entry of ``nodes``, same order. All 1 unless
        ``parse_adjacency`` was given ``widths``. Their sum is
        ``state_dim``.
    input_nodes : tuple[str, ...]
        In-degree 0 names, alphabetical. This is the column order
        ``align_inputs`` indexes into, which is
        narrower than ``nodes`` unless every node is an input.
    output_nodes : tuple[str, ...]
        Out-degree 0 names, alphabetical.
    hidden_nodes : tuple[str, ...]
        Names that are neither input nor output, alphabetical.
        Empty when every node is an input or an output.
    source_index : tuple[int, ...]
        Source unit of each live unit pair. A named edge
        ``A -> B`` expands into every pair of its ``(k_B, k_A)``
        block, target-unit outer, source-unit inner; at width 1
        that is one pair per named edge, the source's position in
        ``nodes``. Same length as ``target_index``. Named edges
        are canonical: lexicographic by
        ``(source name, target name)``, identical to
        ``to_edgelist()`` row order. Cycles and self-loops are
        included. ``edge_location`` returns the packed indices of
        one named edge.
    target_index : tuple[int, ...]
        Target unit of each live unit pair. A dense square would
        have ``1.0`` at ``[target_index[i], source_index[i]]``.
    input_index : tuple[int, ...]
        The unit of each ``input_nodes`` name, in that order;
        input nodes always have width 1. Same length as
        ``align_inputs(names, spec)``: scatter the
        gathered columns into the state vector along these
        units.
    output_index : tuple[int, ...]
        Every unit of each ``output_nodes`` name, in that order.
        Read the network's outputs from the state vector along
        these units.

    See Also
    --------
    parse_adjacency : Builds this spec from a ``source`` /
        ``target`` edgelist; the only supported constructor.
    LayeredSpec : Depth-ranked sibling layout, one incoming packed
        hop per layer, rejecting cycles and self-loops.
    PackedLinear.from_adjacency : One weight per edge of this
        spec, with no ``(n, n)``.
    PackedMultiheadAttention.from_adjacency : Scores only those
        same packed pairs, for an attention update instead of a
        linear one.
    align_inputs : Column index that puts named features onto
        ``input_nodes``.

    Notes
    -----
    Fields cannot be reassigned and sequences are tuples, so the
    structure itself is fixed. There is no ``mask`` field and no
    densifying ``mask`` property. ``to_mask()`` allocates a
    fresh dense square on every call; mutating that tensor does
    not change this spec. ``MaskedLinear(spec.to_mask())``
    clones the square into a non-persistent buffer, so a layer
    built earlier keeps its own connectivity.

    ``align_inputs`` returns ``len(input_index)`` positions (one
    per input node), which is not the state width. Scatter the
    gathered columns into the ``state_dim``-wide state vector
    with ``input_index``. Input rows of
    ``to_mask()`` are all zeros, so under the degree-aware init
    of ``MaskedLinear`` and ``PackedLinear`` those units stay
    zero: writing the inputs in is required, not cosmetic.

    Depth does not exist in this layout, so there is no
    ``layer_nodes``, ``layer_dims``, ``hops``, or ``skips``, and
    ``gather_hop_inputs`` does not accept this spec. An edge
    that would span layers is already an ordinary index pair.
    This is not a one-layer ``LayeredSpec``; the layout is your
    choice, and a DAG is valid input to either parser.

    ``to_dict()`` with ``from_dict()`` and ``fingerprint`` are
    the supported interchange; they round-trip through
    ``parse_adjacency``, cycle edges, self-loops, and widths
    included. ``to_edgelist()`` round-trips at width 1. Pickle
    and ``torch.save`` of the
    dataclass are not. ``edge_location`` finds packed slots of
    a named edge, and ``edge_index`` those of several edges, in
    order; neither is a constraint. A hard freeze of those
    slots is ``torch.where`` inside ``constraint=``.
    ``slot_table`` goes the other way and names the edge of
    every slot.
    ``node_table`` counts each node's edges and the nodes
    that reach it or that it reaches.

    Examples
    --------
    An input feeding a two-node cycle plus one output:

    >>> import pandas as pd
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["x", "a", "b", "a"],
    ...         "target": ["a", "b", "a", "y"],
    ...     }
    ... )
    >>> spec = kpnn2.parse_adjacency(edgelist)
    >>> spec.nodes
    ('a', 'b', 'x', 'y')
    >>> spec.input_nodes, spec.output_nodes
    (('x',), ('y',))
    >>> spec.hidden_nodes
    ('a', 'b')
    >>> spec.input_index, spec.output_index
    ((2,), (3,))
    >>> spec.source_index
    (0, 0, 1, 2)
    >>> spec.target_index
    (1, 3, 0, 0)
    >>> spec.edge_location("a", "b")
    (0,)
    >>> spec.state_dim
    4
    >>> tuple(spec.to_mask().shape)
    (4, 4)
    >>> spec.to_mask()[0].tolist()
    [0.0, 1.0, 1.0, 0.0]
    """

    nodes: tuple[str, ...]
    node_widths: tuple[int, ...]
    input_nodes: tuple[str, ...]
    output_nodes: tuple[str, ...]
    hidden_nodes: tuple[str, ...]
    source_index: tuple[int, ...]
    target_index: tuple[int, ...]
    input_index: tuple[int, ...]
    output_index: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "nodes",
            tuple(self.nodes),
        )
        object.__setattr__(
            self,
            "node_widths",
            tuple(self.node_widths),
        )
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
            "source_index",
            tuple(self.source_index),
        )
        object.__setattr__(
            self,
            "target_index",
            tuple(self.target_index),
        )
        object.__setattr__(
            self,
            "input_index",
            tuple(self.input_index),
        )
        object.__setattr__(
            self,
            "output_index",
            tuple(self.output_index),
        )

    @property
    def state_dim(self) -> int:
        """
        Number of units in the state vector, ``sum(node_widths)``.

        This is the ``in_features`` / ``out_features`` of a
        ``PackedLinear`` built on this spec and the side of
        ``to_mask()``. It equals ``len(nodes)`` when every node
        has width 1.

        Returns
        -------
        int
            Number of units in the state vector.
        """
        return sum(self.node_widths)

    def node_units(
        self,
        name: object,
    ) -> slice:
        """
        Return the unit slice of one named node in the state vector.

        ``name`` is matched after ``str(...)``, same as parse.
        Always a slice, including at width 1. Index as
        ``state[..., units]``. ``LayeredSpec.node_units`` also
        returns the depth; this layout has none.

        Parameters
        ----------
        name : str
            Node name. Non-strings are converted with
            ``str(...)``.

        Returns
        -------
        slice
            Contiguous units of that node. Length is its width.

        Raises
        ------
        Kpnn2Error
            If ``name`` is empty or is not a node.

        Examples
        --------
        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["x", "a", "b", "a"],
        ...         "target": ["a", "b", "a", "y"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_adjacency(
        ...     edgelist,
        ...     widths={"b": 3},
        ... )
        >>> spec.node_units("b")
        slice(1, 4, None)
        >>> spec.node_units("x")
        slice(4, 5, None)
        """
        node_name = resolve_node_name(
            name,
            self.nodes,
            "spec.nodes",
        )
        return self._layout().slot(node_name).units

    def unit_index(
        self,
        names: object,
    ) -> tuple[int, ...]:
        """
        Return every state unit of the named nodes, in order.

        Each named node contributes all of its units, ascending,
        in ``names`` order. This is the rule that builds
        ``input_index`` and ``output_index``, so
        ``spec.unit_index(spec.input_nodes) == spec.input_index``
        (same for outputs). Use it for nodes you name yourself,
        such as readouts that are not in ``output_nodes`` or data
        written onto hidden nodes. At width 1, position ``i``
        belongs to ``names[i]``; a node of width ``k`` contributes
        ``k`` entries. ``node_units`` is the slice of one node.
        ``LayeredSpec.unit_index`` is the same rule on one layer
        and also takes ``layer=``; this layout has none.

        Parameters
        ----------
        names : sequence of str
            Node names in the order to index: a list, a tuple, a
            one-dimensional numpy array, a ``pandas.Index``, or
            another ordered iterable. Each name is matched after
            ``str(...)``, same as parse. A repeated name repeats
            its units. ``names`` is not modified.

        Returns
        -------
        tuple of int
            State units of the named nodes, in order. ``()`` when
            ``names`` is empty.

        Raises
        ------
        Kpnn2Error
            If ``names`` is a tensor, a ``str``, ``bytes``, a set, a
            mapping, not one-dimensional, or not iterable; if a
            name is empty; or if a name is not a node. One message
            lists the unknown names.

        Notes
        -----
        This is identity into the state axis: it takes no tensor
        and no data. Keep it as a buffer, as ``input_index`` is,
        with ``torch.as_tensor(index, dtype=torch.int64)``. The
        ``dtype`` is needed because ``torch.as_tensor(())`` is
        float32 and fails as an index.

        Reading through a repeated unit is safe. Writing through
        one is your concern: ``index_copy`` through repeated units
        is ambiguous, while ``index_add`` sums.

        Examples
        --------
        At width 1, each name gives its position in ``nodes``:

        >>> import pandas as pd
        >>> import torch
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["x", "a", "b", "a"],
        ...         "target": ["a", "b", "a", "y"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_adjacency(edgelist)
        >>> spec.nodes
        ('a', 'b', 'x', 'y')
        >>> spec.unit_index(["b", "a"])
        (1, 0)

        A node of width 3 gives its whole block:

        >>> wide = kpnn2.parse_adjacency(
        ...     edgelist,
        ...     widths={"b": 3},
        ... )
        >>> wide.unit_index(["b", "a"])
        (1, 2, 3, 0)
        >>> wide.unit_index(wide.input_nodes) == wide.input_index
        True

        As a buffer, it reads those units from a state tensor:

        >>> index = torch.as_tensor(
        ...     wide.unit_index(["b"]),
        ...     dtype=torch.int64,
        ... )
        >>> state = torch.arange(6.0).unsqueeze(0)
        >>> state[..., index].tolist()
        [[1.0, 2.0, 3.0]]
        """
        node_names = resolve_node_names(
            as_name_list(
                names,
                "names",
                "node names",
            ),
            self.nodes,
            "spec.nodes",
        )
        return self._layout().units_of(node_names)

    def _layout(self) -> Layout:
        """
        Return the unit placement of every node on the state vector.
        """
        return build_layout(
            self.nodes,
            self.node_widths,
        )

    def to_mask(self) -> Tensor:
        """
        Allocate a dense float32 square from the packed edges.

        Shape is ``(state_dim, state_dim)``. The result starts at
        zeros;
        each live edge sets ``1.0`` at
        ``[target_index[i], source_index[i]]``. Every call
        returns a fresh tensor. Mutating it does not change this
        spec or the next ``to_mask()`` call.

        Returns
        -------
        torch.Tensor
            New dense connectivity square. This allocates.

        Examples
        --------
        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["x", "a", "b", "a"],
        ...         "target": ["a", "b", "a", "y"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_adjacency(edgelist)
        >>> mask = spec.to_mask()
        >>> tuple(mask.shape)
        (4, 4)
        >>> mask[0, 1].item(), mask[1, 0].item()
        (1.0, 1.0)
        """
        n_units = self._layout().n_units
        return dense_mask_from_indices(
            self.source_index,
            self.target_index,
            n_units,
            n_units,
        )

    def to_edgelist(self) -> pd.DataFrame:
        """
        Return this spec's edges as a two-column table.

        Columns are exactly ``source`` then ``target``. Rows
        follow the packed indices in canonical order: sorted
        lexicographically by ``(source, target)``, one row per
        original edge, names as strings, including cycle edges
        and self-loops. Extra columns from the DataFrame that
        was parsed are not reproduced.

        ``parse_adjacency`` on this table reconstructs the same
        node lists, packed indices, and input/output indices
        when every node has width 1. Widths live on the spec
        dict, not the edgelist: round-trip a wide spec with
        ``from_dict`` or ``parse_adjacency(..., widths=)``.

        Returns
        -------
        pandas.DataFrame
            One row per original edge.

        Examples
        --------
        A cycle comes back as sorted pairs:

        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["x", "a", "b", "a"],
        ...         "target": ["a", "b", "a", "y"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_adjacency(edgelist)
        >>> table = spec.to_edgelist()
        >>> list(table.columns)
        ['source', 'target']
        >>> table["source"].tolist()
        ['a', 'a', 'b', 'x']
        >>> table["target"].tolist()
        ['b', 'y', 'a', 'a']
        """
        from ._serialize import spec_to_edgelist

        return spec_to_edgelist(self)

    def edge_location(
        self,
        source: object,
        target: object,
    ) -> tuple[int, ...]:
        """
        Return packed weight slots of one named edge.

        ``source`` and ``target`` are matched after ``str(...)``,
        same as parse. Indices address ``source_index`` /
        ``target_index`` and ``PackedLinear.weight`` built from
        those arrays. This is identity into the packed arrays,
        not a constraint. A named edge ``A -> B`` is the full
        ``(k_B, k_A)`` block of unit pairs; at width 1 one pair.

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
        tuple of int
            Packed indices of the named edge, in stored order
            (canonical lexicographic by
            ``(source name, target name)``, same as
            ``to_edgelist()`` rows; within the edge, target-unit
            outer, source-unit inner). Length is
            ``k_source * k_target``, 1 at default width.

        Raises
        ------
        Kpnn2Error
            If the pair is missing, a name is empty, or a name
            is not a node. The message names the pair as
            ``{source} -> {target}``.

        Notes
        -----
        There is no hop index. Width greater than 1 does not
        change the named edge; it returns one packed index per
        unit pair of the block. Mixed signs and frozen values are
        caller PyTorch on these slots. A hard freeze is
        ``torch.where`` inside ``constraint=``, not a
        gradient hook.

        Examples
        --------
        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["x", "a", "b", "a"],
        ...         "target": ["a", "b", "a", "y"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_adjacency(edgelist)
        >>> spec.edge_location("a", "b")
        (0,)
        >>> spec.edge_location("x", "a")
        (3,)
        """
        locations, known_names = self._edge_locations()
        source_name, target_name = resolve_edge_names(
            source,
            target,
            known_names,
            "spec.nodes",
        )
        packed = locations.get(
            (
                source_name,
                target_name,
            )
        )
        if not packed:
            raise no_edge_error(
                [(source_name, target_name)],
                locations,
            )
        return packed

    def edge_index(
        self,
        sources: object,
        targets: object,
    ) -> tuple[int, ...]:
        """
        Return every packed slot of the named edges, in order.

        Edge ``i`` is ``sources[i] -> targets[i]``. Each edge
        contributes all of its slots, in the order
        ``edge_location`` gives them, in edge order. The slots
        index ``source_index`` / ``target_index`` and the
        ``weight`` of a ``PackedLinear`` built from this spec, so
        a value you keep per edge on your own table, such as a
        sign, a prior flag, or an edge type, lands on its slots
        without a loop, and a per-slot tensor such as a trained
        weight reads back in your table's order. At width 1,
        position ``i`` belongs to edge ``i``; an edge between
        nodes of widths ``k_source`` and ``k_target`` contributes
        ``k_source * k_target`` entries. ``edge_location`` gives
        the slots of one edge. ``LayeredSpec.edge_index`` is the
        same rule on one hop and also takes ``hop=``.

        Parameters
        ----------
        sources : sequence of str
            Source node name of each edge, in the order to index:
            a list, a tuple, a one-dimensional numpy array, a
            ``pandas.Series`` or ``pandas.Index``, or another
            ordered iterable. Each name is matched after
            ``str(...)``, same as parse. ``sources`` is not
            modified.
        targets : sequence of str
            Target node name of each edge, same length and order
            as ``sources``, and accepted in the same forms.
            ``targets`` is not modified.

        Returns
        -------
        tuple of int
            Packed slots of the named edges, in order. ``()`` when
            there are no edges.

        Raises
        ------
        Kpnn2Error
            If ``sources`` or ``targets`` is a tensor, a ``str``,
            ``bytes``, a set, a mapping, not one-dimensional, or
            not iterable; if their lengths differ; if a name is
            empty; if a name is not a node; or if a pair is not an
            edge. One message lists the unknown names, and one
            lists the missing edges.

        Notes
        -----
        This is identity into the packed slots: it takes no
        tensor and no values. Index with
        ``torch.as_tensor(index, dtype=torch.int64)``. The
        ``dtype`` is needed because ``torch.as_tensor(())`` is
        float32 and fails as an index.

        With a wide node, one value per edge has to be repeated
        across that edge's slots, ``len(spec.edge_location(source,
        target))`` of them. A repeated edge repeats its slots.
        Reading through a repeated slot is safe. Writing through
        one is your concern: ``index_copy`` through repeated slots
        is ambiguous, while ``index_add`` sums.

        Examples
        --------
        At width 1, each edge gives its row in ``to_edgelist()``:

        >>> import pandas as pd
        >>> import torch
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["x", "a", "b", "a"],
        ...         "target": ["a", "b", "a", "y"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_adjacency(edgelist)
        >>> spec.to_edgelist().values.tolist()
        [['a', 'b'], ['a', 'y'], ['b', 'a'], ['x', 'a']]
        >>> spec.edge_index(["x", "b"], ["a", "a"])
        (3, 2)

        A sign kept on your own table lands on its slots:

        >>> signs = pd.DataFrame(
        ...     {
        ...         "source": ["b", "a"],
        ...         "target": ["a", "b"],
        ...         "sign": [-1.0, 1.0],
        ...     }
        ... )
        >>> index = torch.as_tensor(
        ...     spec.edge_index(signs["source"], signs["target"]),
        ...     dtype=torch.int64,
        ... )
        >>> sign = torch.zeros(len(spec.source_index))
        >>> sign[index] = torch.tensor(signs["sign"].tolist())
        >>> sign.tolist()
        [1.0, 0.0, -1.0, 0.0]

        An edge into a node of width 3 gives its whole block:

        >>> wide = kpnn2.parse_adjacency(
        ...     edgelist,
        ...     widths={"b": 3},
        ... )
        >>> wide.edge_index(["x", "a"], ["a", "b"])
        (7, 0, 1, 2)
        """
        source_names = as_name_list(
            sources,
            "sources",
            "source node names",
        )
        target_names = as_name_list(
            targets,
            "targets",
            "target node names",
        )
        locations, known_names = self._edge_locations()
        pairs = resolve_edge_pairs(
            source_names,
            target_names,
            known_names,
            locations,
            "spec.nodes",
        )
        slots: list[int] = []
        for pair in pairs:
            slots.extend(locations[pair])
        return tuple(slots)

    def slot_table(self) -> pd.DataFrame:
        """
        Return the named edge of every packed slot, in slot order.

        One row per packed slot, in the order of ``source_index``
        and of the ``weight`` of ``PackedLinear.from_adjacency``
        on this spec, so a per-slot tensor such as a trained
        weight gets its edge names by assignment as a numpy
        array: ``table["weight"] = weight.detach().numpy()``.
        At width 1 a slot is a named edge, and the rows are the
        rows of ``to_edgelist()``. An edge between nodes of
        widths ``k_source`` and ``k_target`` fills
        ``k_source * k_target`` rows, one per unit pair, all with
        the same names. ``edge_index`` goes the other way, from
        named edges to their slots.

        Returns
        -------
        pandas.DataFrame
            Index ``slot``: ``0`` to ``len(source_index) - 1``,
            ``int64``. Columns ``source`` and ``target``: the
            names of the edge that owns the slot.

        See Also
        --------
        edge_index : Slots of named edges, in order.
        to_edgelist : One row per named edge.

        Notes
        -----
        The table is computed on each call and is a new DataFrame
        each time, so changing it does not change this spec. The
        units of slot ``i`` are ``source_index[i]`` and
        ``target_index[i]``. To get one value per named edge on a
        wide spec, group the rows by ``["source", "target"]``.

        Examples
        --------
        Four edges, one slot each:

        >>> import pandas as pd
        >>> import torch
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["x", "a", "b", "a"],
        ...         "target": ["a", "b", "a", "y"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_adjacency(edgelist)
        >>> table = spec.slot_table()
        >>> table
             source target
        slot
        0         a      b
        1         a      y
        2         b      a
        3         x      a

        A per-slot tensor, here standing in for
        ``layer.effective_weight().detach()``, names its values:

        >>> weight = torch.tensor([0.5, -1.0, 2.0, 0.25])
        >>> table["weight"] = weight.numpy()
        >>> table.loc[table["target"] == "a"]
             source target  weight
        slot
        2         b      a    2.00
        3         x      a    0.25

        With ``b`` three units wide, ``a -> b`` and ``b -> a``
        fill three rows each:

        >>> wide = kpnn2.parse_adjacency(
        ...     edgelist,
        ...     widths={"b": 3},
        ... )
        >>> wide.slot_table().groupby(["source", "target"]).size()
        source  target
        a       b         3
                y         1
        b       a         3
        x       a         1
        dtype: int64
        """
        locations, _ = self._edge_locations()
        edges = edges_by_slot(
            locations.items(),
            len(self.source_index),
        )
        return pd.DataFrame(
            edges,
            columns=["source", "target"],
            index=pd.RangeIndex(
                len(edges),
                name="slot",
            ),
        )

    def _edge_locations(
        self,
    ) -> tuple[
        dict[tuple[str, str], tuple[int, ...]],
        frozenset[str],
    ]:
        """
        Map each named edge to its packed slots.

        The first call walks the packed indices once and
        remembers the node names. Later calls reuse both.
        Neither is a dataclass field. A unit belongs to the node
        whose block contains it (``Layout.slot_containing``).
        """
        cached = getattr(
            self,
            "_edge_location_cache",
            None,
        )
        if cached is not None:
            return cached
        layout = self._layout()
        grouped: dict[tuple[str, str], list[int]] = {}
        for slot, (source, target) in enumerate(
            zip(
                self.source_index,
                self.target_index,
                strict=True,
            )
        ):
            key = (
                layout.slot_containing(source).name,
                layout.slot_containing(target).name,
            )
            grouped.setdefault(
                key,
                [],
            ).append(slot)
        index = {key: tuple(slots) for key, slots in grouped.items()}
        cached = (
            index,
            frozenset(self.nodes),
        )
        object.__setattr__(
            self,
            "_edge_location_cache",
            cached,
        )
        return cached

    def node_table(self) -> pd.DataFrame:
        """
        Return one row of graph counts per node.

        Each row says where a node sits in the graph: how many
        units it owns, how many edges enter and leave it, and how
        many nodes reach it or are reached from it. A node with
        many connections tends to get large sums and large scores
        only because it is large, so these counts are what such a
        node is scaled by: a ``widths`` value set from its number
        of parents, a division of its input by its in-degree in
        ``forward()``, or a check of whether node scores follow
        connectivity. The table holds the raw counts; the formula
        stays yours.

        Counts are over named edges in the direction of
        computation, so widths never change them. An input that
        feeds a node, directly or through other nodes, is one of
        its ancestors. A self-loop adds 1 to both degrees of its
        node. A node is never counted among its own ancestors or
        descendants, even on a cycle; the other nodes of its
        cycle are both. The table is computed on each call and is
        a new DataFrame each time, so changing it does not change
        this spec.

        Returns
        -------
        pandas.DataFrame
            Index ``node``: every node name in ``nodes`` order,
            the order of the state vector. Columns, all
            ``int64``:

            - ``width``: the number of units the node owns
            - ``in_degree``: edges into the node, one per parent
            - ``out_degree``: edges out of the node, one per
              child
            - ``n_ancestors``: other nodes with a path to the
              node
            - ``n_descendants``: other nodes the node has a path
              to
            - ``n_input_ancestors``: how many of its ancestors
              are in ``input_nodes``

        See Also
        --------
        to_edgelist : The named edges the counts are taken over.
        node_units : One node's unit slice on the state vector.

        Notes
        -----
        ``in_degree == 0`` selects exactly ``input_nodes``, and
        ``out_degree == 0`` exactly ``output_nodes``, self-loops
        included. Rows follow the state vector, so
        ``numpy.repeat(table[column], table["width"])`` gives one
        value per state unit, ``state_dim`` long. When counts
        should set ``widths``, read them from a first parse
        without ``widths``: they count named edges, so the second
        parse has the same counts.

        Examples
        --------
        The cycle ``a -> b -> a`` and a self-loop on ``b``:

        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["x", "a", "b", "a", "b"],
        ...         "target": ["a", "b", "a", "y", "b"],
        ...     }
        ... )
        >>> table = kpnn2.parse_adjacency(edgelist).node_table()
        >>> table[["width", "in_degree", "out_degree"]]
              width  in_degree  out_degree
        node
        a         1          2           2
        b         1          2           2
        x         1          0           1
        y         1          1           0
        >>> table[["n_ancestors", "n_descendants", "n_input_ancestors"]]
              n_ancestors  n_descendants  n_input_ancestors
        node
        a               2              2                  1
        b               2              2                  1
        x               0              3                  0
        y               3              0                  1

        A wider node keeps its counts, and repeating a column by
        ``width`` gives one value per state unit:

        >>> import numpy as np
        >>> wide = kpnn2.parse_adjacency(
        ...     edgelist,
        ...     widths={"a": 2},
        ... ).node_table()
        >>> np.repeat(wide["in_degree"], wide["width"]).tolist()
        [2, 2, 2, 0, 1]
        """
        from ._node_table import spec_node_table

        return spec_node_table(self)

    def to_dict(self) -> dict:
        """
        Return this spec as a JSON-safe tagged dict.

        Keys are ``kpnn2_spec`` (integer ``1``), ``layout``
        (``"adjacency"``), and ``edges`` (list of
        ``[source, target]`` lists in the same order as
        ``to_edgelist()`` rows, including cycle edges and
        self-loops). When any node has width other than 1, a
        ``"widths"`` object of those names is included; all-1
        specs omit it. The returned dict is new on every call.

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
        ...         "source": ["x", "a", "b", "a"],
        ...         "target": ["a", "b", "a", "y"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_adjacency(edgelist)
        >>> payload = spec.to_dict()
        >>> payload["kpnn2_spec"]
        1
        >>> payload["layout"]
        'adjacency'
        >>> payload["edges"]
        [['a', 'b'], ['a', 'y'], ['b', 'a'], ['x', 'a']]
        """
        from ._serialize import spec_to_dict

        return spec_to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "AdjacencySpec":
        """
        Rebuild an ``AdjacencySpec`` from ``to_dict()`` output.

        Calls ``parse_adjacency`` on a DataFrame built from
        ``payload["edges"]``, passing ``payload["widths"]`` when
        present. Packed indices are not assembled by hand. Extra
        unknown keys are ignored, including a stray ``"ranks"``.

        Parameters
        ----------
        payload : dict
            A dict with ``kpnn2_spec``, ``layout``, and
            ``edges``. ``layout`` must be ``"adjacency"``.

        Returns
        -------
        AdjacencySpec
            The parsed spec.

        Raises
        ------
        Kpnn2Error
            If ``payload`` is not a dict; ``kpnn2_spec`` is
            missing or not ``1``; ``layout`` is missing, not a
            known layout, or is ``"layered"``; ``edges`` is
            missing or not a sequence of two nonempty names, or a
            name is not a ``str``, integer, float, or ``bool``; or
            ``"widths"`` is present and not a mapping of positive
            integers over known nodes.

        Warns
        -----
        UserWarning
            If an edge name is a float or a ``bool``, as in
            ``parse_adjacency``.

        Examples
        --------
        >>> import pandas as pd
        >>> import kpnn2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["x", "a", "b", "a"],
        ...         "target": ["a", "b", "a", "y"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_adjacency(edgelist)
        >>> roundtrip = kpnn2.AdjacencySpec.from_dict(spec.to_dict())
        >>> roundtrip.nodes == spec.nodes
        True
        """
        from ._serialize import adjacency_spec_from_dict

        return adjacency_spec_from_dict(payload)

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
        ...         "source": ["x", "a", "b", "a"],
        ...         "target": ["a", "b", "a", "y"],
        ...     }
        ... )
        >>> spec = kpnn2.parse_adjacency(edgelist)
        >>> len(spec.fingerprint)
        64
        >>> (
        ...     spec.fingerprint
        ...     == kpnn2.parse_adjacency(spec.to_edgelist()).fingerprint
        ... )
        True
        """
        from ._serialize import spec_fingerprint

        return spec_fingerprint(self)


def require_adjacency_spec(
    spec: object,
    fix: str,
) -> None:
    """
    Raise unless ``spec`` is an ``AdjacencySpec``.

    Parameters
    ----------
    spec : object
        The ``spec`` argument of the caller.
    fix : str
        Sentence that ends the message and says what to pass.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is any other type, a ``LayeredSpec`` or a
        ``Hop`` included.
    """
    if not isinstance(spec, AdjacencySpec):
        raise caller_error(
            "'spec' must be an AdjacencySpec.",
            got=describe(spec),
            fix=fix,
        )
