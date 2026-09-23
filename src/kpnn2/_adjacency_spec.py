"""
Structural blueprint for an edgelist-defined node network.
"""

from dataclasses import dataclass

import pandas as pd
from torch import Tensor

from ._errors import Kpnn2Error
from ._layout import (
    Layout,
    build_layout,
    dense_mask_from_indices,
    resolve_edge_names,
)


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
        Every unit of each ``input_nodes`` name, in that order,
        so a wide input contributes each of its units. Same
        length as ``align_inputs(names, spec)``: scatter the
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
    PackedLinear : Consumes ``source_index`` and ``target_index``
        as they are, with one weight per edge and no ``(n, n)``.
    PackedMultiheadAttention : Scores only those same packed
        pairs, for an attention update instead of a linear one.
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
    per input unit), which is not the state width. Scatter the
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
    a named edge; it is not a constraint. A hard freeze of
    those slots is ``torch.where`` inside ``constraint=``.

    Examples
    --------
    An input feeding a two-node feedback core plus one output:

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
        node_name = str(name)
        if node_name == "":
            raise Kpnn2Error("Node name is empty.")
        if node_name not in self.nodes:
            raise Kpnn2Error(f"Unknown node name: {node_name}.")
        return self._layout().slot(node_name).units

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
        )
        packed = locations.get(
            (
                source_name,
                target_name,
            )
        )
        if not packed:
            raise Kpnn2Error(f"No edge {source_name} -> {target_name}.")
        return packed

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
            missing or not a sequence of two nonempty names; or
            ``"widths"`` is present and not a mapping of positive
            ints over known nodes.

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
