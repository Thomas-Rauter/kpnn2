"""
Unit placement for graph nodes on a tensor axis.

Every graph node owns a contiguous slice of units on the last
axis of a tensor. Default width is ``DEFAULT_NODE_WIDTH == 1``.
``parse_layered(..., widths=)`` may give a node several units;
a named edge then occupies every unit pair of that block.

Routing index arithmetic through this module is what keeps node
width additive: give ``build_layout`` real widths and mask
construction, input alignment, hop concatenation, and attribution
naming follow without changes at their call sites.

Everything this module is handed comes from kpnn2 itself: spec
fields and values the parsers computed. Public entry points
validate caller-supplied names before they ask a ``Layout``, so
an inconsistent layout or a failed lookup here is a kpnn2 bug and
raises through ``internal_error``. ``resolve_edge_names``,
``resolve_edge_pairs``, ``resolve_node_name``, and
``resolve_node_names`` are the exception: they check
caller-supplied names and raise ``Kpnn2Error``.
``unknown_name_hints`` and ``no_edge_error`` build the parts of
those messages that other lookups share.
"""

from collections.abc import (
    Collection,
    Container,
    Iterable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass, field

import numpy as np
import torch

from ._errors import (
    Kpnn2Error,
    caller_error,
    close_matches,
    internal_error,
    listed,
)

DEFAULT_NODE_WIDTH = 1

_EMPTY_NODE_NAME = "Node name is empty."


@dataclass(frozen=True)
class NodeSlot:
    """
    The units one graph node owns on a tensor axis.

    Parameters
    ----------
    name : str
        Node name.
    start : int
        First unit index of the node.
    width : int
        Number of units the node owns. ``DEFAULT_NODE_WIDTH``
        unless ``parse_layered`` was given ``widths``.
    """

    name: str
    start: int
    width: int

    @property
    def stop(self) -> int:
        """
        One past the last unit index of the node.
        """
        return self.start + self.width

    @property
    def units(self) -> slice:
        """
        The node's units as a slice, for indexing a tensor axis.
        """
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class Layout:
    """
    Contiguous placement of named nodes on one tensor axis.

    Slots follow the node order given to ``build_layout`` and
    tile the axis without gaps, so ``n_units`` is the axis
    length. With every width at ``DEFAULT_NODE_WIDTH`` this is a
    one-unit-per-node vector and ``slot(name).start`` is the
    node's column index. Wider nodes occupy a contiguous block.

    Parameters
    ----------
    slots : tuple[NodeSlot, ...]
        Node slots in axis order.

    Raises
    ------
    AssertionError
        From ``internal_error``, if a width is below 1, a name
        repeats, or the slots do not tile the axis contiguously
        from 0.
    """

    slots: tuple[NodeSlot, ...]
    _by_name: dict[str, NodeSlot] = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=dict,
    )
    _by_unit: tuple[NodeSlot, ...] = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        slots = tuple(self.slots)
        object.__setattr__(
            self,
            "slots",
            slots,
        )
        by_name: dict[str, NodeSlot] = {}
        by_unit: list[NodeSlot] = []
        position = 0
        for slot in slots:
            if slot.width < 1:
                raise internal_error(
                    f"layout node {slot.name!r} has width {slot.width}, below 1"
                )
            if slot.start != position:
                raise internal_error(
                    "layout slots do not tile the axis: node "
                    f"{slot.name!r} starts at {slot.start}, "
                    f"expected {position}"
                )
            if slot.name in by_name:
                raise internal_error(
                    f"layout node {slot.name!r} appears more than once"
                )
            by_name[slot.name] = slot
            by_unit.extend([slot] * slot.width)
            position = slot.stop
        object.__setattr__(
            self,
            "_by_name",
            by_name,
        )
        object.__setattr__(
            self,
            "_by_unit",
            tuple(by_unit),
        )

    @property
    def n_units(self) -> int:
        """
        Length of the tensor axis this layout describes.
        """
        if not self.slots:
            return 0
        return self.slots[-1].stop

    @property
    def names(self) -> tuple[str, ...]:
        """
        Node names in axis order, one entry per node.
        """
        return tuple(slot.name for slot in self.slots)

    def slot(self, name: str) -> NodeSlot:
        """
        Return the slot of ``name``.
        """
        try:
            return self._by_name[name]
        except KeyError:
            raise internal_error(
                f"node {name!r} is not in this layout"
            ) from None

    def start_of(self, name: str) -> int:
        """
        Return the first unit index of ``name``.
        """
        return self.slot(name).start

    def units_of(self, names: Sequence[str]) -> tuple[int, ...]:
        """
        Return every unit of each named node, in ``names`` order.

        Units of one node are ascending. A repeated name repeats
        its units, and empty ``names`` gives ``()``. With width-1
        nodes this is each name's column index.
        """
        units: list[int] = []
        for name in names:
            slot = self.slot(name)
            units.extend(
                range(
                    slot.start,
                    slot.stop,
                )
            )
        return tuple(units)

    def slot_at(self, start: int) -> NodeSlot:
        """
        Return the slot that begins at unit index ``start``.

        With width-1 nodes this turns a stored column index back
        into the owning node; with wider nodes it turns a block
        start into the whole block.
        """
        for slot in self.slots:
            if slot.start == start:
                return slot
        raise internal_error(f"no layout node begins at unit index {start}")

    def slot_containing(self, unit: int) -> NodeSlot:
        """
        Return the slot whose units slice contains ``unit``.

        ``slot_at`` requires a block start; this accepts any
        unit index inside the block.
        """
        n_units = len(self._by_unit)
        if unit < 0 or unit >= n_units:
            raise internal_error(
                f"unit index {unit} is out of the layout's range [0, {n_units})"
            )
        return self._by_unit[unit]

    def widths(self) -> tuple[int, ...]:
        """
        Unit count per node, in axis order.
        """
        return tuple(slot.width for slot in self.slots)

    def unit_names(self) -> list[str]:
        """
        Owning node name per unit, one entry per unit.

        Equal to ``list(self.names)`` while every node is one
        unit wide.
        """
        names: list[str] = []
        for slot in self.slots:
            names.extend([slot.name] * slot.width)
        return names


def build_layout(
    names: Sequence[str],
    widths: Sequence[int] | None = None,
) -> Layout:
    """
    Place ``names`` on an axis, in order, without gaps.

    Parameters
    ----------
    names
        Node names in axis order.
    widths
        Units per node. ``None`` gives every node
        ``DEFAULT_NODE_WIDTH``.

    Returns
    -------
    Layout
        Placement of every name.

    Raises
    ------
    AssertionError
        From ``internal_error``, if ``widths`` has a different
        length than ``names``, or the resulting slots are not a
        valid layout.
    """
    ordered = list(names)
    if widths is None:
        sizes = [DEFAULT_NODE_WIDTH] * len(ordered)
    else:
        sizes = list(widths)
        if len(sizes) != len(ordered):
            raise internal_error(
                f"build_layout got {len(sizes)} width(s) for "
                f"{len(ordered)} node name(s)"
            )
    slots: list[NodeSlot] = []
    start = 0
    for name, width in zip(
        ordered,
        sizes,
    ):
        slots.append(
            NodeSlot(
                name=name,
                start=start,
                width=width,
            )
        )
        start += width
    return Layout(slots=tuple(slots))


def concat_layouts(
    layouts: Sequence[Layout],
) -> Layout:
    """
    Lay several axes end to end on one axis.

    This is the source axis of a hop that reads more than one
    layer: the layouts keep their internal order and each one
    starts where the previous stopped, so a node's slot in the
    result is its slot in its own layer shifted by the widths of
    everything in front of it.

    Parameters
    ----------
    layouts
        Layouts to concatenate, in axis order.

    Returns
    -------
    Layout
        One layout over every node of every input, in order.

    Raises
    ------
    AssertionError
        From ``internal_error``, if a name appears in more than
        one input layout.
    """
    slots: list[NodeSlot] = []
    start = 0
    for layout in layouts:
        for slot in layout.slots:
            slots.append(
                NodeSlot(
                    name=slot.name,
                    start=start + slot.start,
                    width=slot.width,
                )
            )
        start += layout.n_units
    return Layout(slots=tuple(slots))


def hop_axis_layouts(
    layer_nodes: Sequence[Sequence[str]],
    layer_widths: Sequence[Sequence[int]],
    source_layers: Sequence[int],
    target_layer: int,
) -> tuple[Layout, Layout]:
    """
    Build the source and target layouts of one hop.

    The source axis is the source-layer layouts concatenated, the
    same axis ``gather_hop_inputs`` assembles. The target axis
    is that hop's target layer.

    Parameters
    ----------
    layer_nodes
        Node names at each depth, one name per node.
    layer_widths
        Width of each name in ``layer_nodes``, same nesting.
    source_layers
        Depths the hop reads, as on ``Hop.source_layers``.
    target_layer
        Depth the hop produces, as on ``Hop.target_layer``.

    Returns
    -------
    tuple of (Layout, Layout)
        Concatenated source layout, then target-layer layout.
    """
    source_layout = concat_layouts(
        [
            build_layout(
                layer_nodes[layer],
                layer_widths[layer],
            )
            for layer in source_layers
        ]
    )
    target_layout = build_layout(
        layer_nodes[target_layer],
        layer_widths[target_layer],
    )
    return source_layout, target_layout


def packed_indices_for_named_edge(
    source_index: Sequence[int],
    target_index: Sequence[int],
    source_layout: Layout,
    target_layout: Layout,
    source: str,
    target: str,
) -> tuple[int, ...]:
    """
    Return packed slots whose owning names match one edge.

    Walks the stored pairs in live order and does not re-sort.
    A unit belongs to a named node via
    ``Layout.slot_containing``, not a unit index into a
    node-name list.

    Parameters
    ----------
    source_index
        Source-axis unit of each live pair.
    target_index
        Target-axis unit of each live pair. Same length as
        ``source_index``.
    source_layout
        Layout of the source axis those columns address.
    target_layout
        Layout of the target axis those rows address.
    source
        Source node name, already a ``str``.
    target
        Target node name, already a ``str``.

    Returns
    -------
    tuple of int
        Indices into ``source_index`` / ``target_index`` whose
        owning names are ``(source, target)``. Empty if that
        named edge is not among these pairs.
    """
    indices: list[int] = []
    for index, (source_unit, target_unit) in enumerate(
        zip(
            source_index,
            target_index,
            strict=True,
        )
    ):
        source_name = source_layout.slot_containing(
            source_unit,
        ).name
        target_name = target_layout.slot_containing(
            target_unit,
        ).name
        if source_name == source and target_name == target:
            indices.append(index)
    return tuple(indices)


def edges_by_slot(
    slots_of_edge: Iterable[tuple[tuple[str, str], Sequence[int]]],
    n_slots: int,
) -> list[tuple[str, str]]:
    """
    Return the named edge that owns each packed slot, in slot order.

    The inverse of a map from named edge to its packed slots, such
    as a spec's edge-location cache. Every slot in
    ``range(n_slots)`` must belong to exactly one named edge; a
    table that labels weights by slot would otherwise name a
    weight after the wrong edge.

    Parameters
    ----------
    slots_of_edge
        ``((source, target), slots)`` for every named edge of one
        packed array.
    n_slots
        Length of that packed array.

    Returns
    -------
    list of tuple of (str, str)
        ``(source, target)`` of slot ``i`` at position ``i``.

    Raises
    ------
    AssertionError
        From ``internal_error``, if a slot is out of range, owned
        by two edges, or owned by none.
    """
    owners: list[tuple[str, str] | None] = [None] * n_slots
    for edge, slots in slots_of_edge:
        for slot in slots:
            if slot < 0 or slot >= n_slots:
                raise internal_error(
                    f"edge {edge[0]} -> {edge[1]} owns slot {slot}, out of "
                    f"the packed range [0, {n_slots})"
                )
            if owners[slot] is not None:
                raise internal_error(
                    f"packed slot {slot} is owned by two named edges"
                )
            owners[slot] = edge
    edges: list[tuple[str, str]] = []
    for slot, owner in enumerate(owners):
        if owner is None:
            raise internal_error(
                f"packed slot {slot} is owned by no named edge"
            )
        edges.append(owner)
    return edges


def resolve_edge_names(
    source: object,
    target: object,
    known_names: Collection[str],
    listed_in: str,
) -> tuple[str, str]:
    """
    Match an edge pair the way parse matches node names.

    Converts with ``str(...)``. Empty names and names that are
    not in ``known_names`` raise ``Kpnn2Error`` naming the pair
    as ``{source} -> {target}`` and the first bad name, with the
    hints ``resolve_node_name`` gives.

    Parameters
    ----------
    source
        Source node name, or a value converted with ``str``.
    target
        Target node name, or a value converted with ``str``.
    known_names
        Node names that exist on the spec.
    listed_in
        The spec field a caller reads for the valid names, such
        as ``"spec.nodes"``, quoted in the message.

    Returns
    -------
    tuple of str
        ``(source_name, target_name)`` after ``str(...)``.

    Raises
    ------
    Kpnn2Error
        If a name is empty or is not in ``known_names``.
    """
    source_name = str(source)
    target_name = str(target)
    for node_name in (source_name, target_name):
        error = _node_name_error(
            node_name,
            known_names,
            listed_in,
            f"{source_name} -> {target_name}",
        )
        if error is not None:
            raise error
    return source_name, target_name


def resolve_edge_pairs(
    sources: Sequence[str],
    targets: Sequence[str],
    known_names: Iterable[str],
    known_edges: Container[tuple[str, str]],
    listed_in: str,
) -> list[tuple[str, str]]:
    """
    Match several named edges the way parse matches node names.

    The many-edge form of ``resolve_edge_names``: edge ``i`` is
    ``sources[i] -> targets[i]``. Checks, in order: equal lengths;
    every name as ``resolve_node_names`` checks it (an empty name,
    then every unknown name in one message); then every pair that
    is not a named edge, each once, sorted, in one message.

    Parameters
    ----------
    sources
        Source node name of each edge, already ``str``, in order.
    targets
        Target node name of each edge, already ``str``, same order.
    known_names
        Node names that exist on the spec.
    known_edges
        ``(source, target)`` pairs of the spec's named edges.
    listed_in
        The spec field a caller reads for the valid names, such
        as ``"spec.nodes"``, quoted in the unknown-name message.

    Returns
    -------
    list of tuple of (str, str)
        ``(source, target)`` of each edge, in order. Repeats are
        kept.

    Raises
    ------
    Kpnn2Error
        If the lengths differ, a name is empty, a name is not in
        ``known_names``, or a pair is not in ``known_edges``.
    """
    if len(sources) != len(targets):
        raise caller_error(
            "'sources' and 'targets' must have the same length: edge i "
            "is sources[i] -> targets[i].",
            got=f"{len(sources)} source(s) and {len(targets)} target(s)",
        )
    resolve_node_names(
        [*sources, *targets],
        known_names,
        listed_in,
    )
    pairs = list(
        zip(
            sources,
            targets,
            strict=True,
        )
    )
    missing = sorted({pair for pair in pairs if pair not in known_edges})
    if missing:
        raise no_edge_error(
            missing,
            known_edges,
        )
    return pairs


def no_edge_error(
    pairs: Sequence[tuple[str, str]],
    known_edges: Container[tuple[str, str]],
) -> Kpnn2Error:
    """
    Build the error for name pairs that are no named edge.

    Both names of each pair are nodes. One pair puts it in the
    headline, several go in ``Got``. A pair whose reverse is an
    edge gets a fix that says so, since a swapped source and
    target is the likely mistake.

    Parameters
    ----------
    pairs
        ``(source, target)`` pairs that are not in ``known_edges``,
        in the order to name them.
    known_edges
        ``(source, target)`` pairs of the spec's named edges.

    Returns
    -------
    Kpnn2Error
        Error that names the pairs and points at
        ``spec.to_edgelist()``.
    """
    labels = [f"{source} -> {target}" for source, target in pairs]
    swapped = [
        label
        for label, (source, target) in zip(
            labels,
            pairs,
            strict=True,
        )
        if (target, source) in known_edges
    ]
    if len(pairs) == 1:
        if swapped:
            source, target = pairs[0]
            fix = (
                f"Swap source and target: the edge {target} -> {source} exists."
            )
        else:
            fix = None
        return caller_error(
            f"No edge {labels[0]}.",
            expected="a row of spec.to_edgelist()",
            fix=fix,
        )
    if swapped:
        fix = (
            f"Swap source and target for {listed(swapped)}: the "
            "reverse is an edge."
        )
    else:
        fix = None
    return caller_error(
        "No edge(s) for some source-target pairs.",
        got=listed(labels),
        expected="rows of spec.to_edgelist()",
        fix=fix,
    )


def resolve_node_name(
    name: object,
    known_names: Collection[str],
    listed_in: str,
) -> str:
    """
    Match a node name the way parse matches node names.

    Converts with ``str(...)``. An empty name and a name that is
    not in ``known_names`` raise ``Kpnn2Error``, in that order.

    Parameters
    ----------
    name
        Node name, or a value converted with ``str``.
    known_names
        Node names that exist on the spec.
    listed_in
        The spec field a caller reads for the valid names, such
        as ``"spec.nodes"``, quoted in the message.

    Returns
    -------
    str
        The node name after ``str(...)``.

    Raises
    ------
    Kpnn2Error
        If the name is empty or is not in ``known_names``.
    """
    node_name = str(name)
    error = _node_name_error(
        node_name,
        known_names,
        listed_in,
        None,
    )
    if error is not None:
        raise error
    return node_name


def resolve_node_names(
    names: Iterable[object],
    known_names: Iterable[str],
    listed_in: str,
) -> list[str]:
    """
    Match several node names the way parse matches node names.

    The many-name form of ``resolve_node_name``. Converts each
    name with ``str(...)``. An empty name raises first. Otherwise
    every name that is not in ``known_names`` is reported in one
    ``Kpnn2Error``, each once, sorted.

    Parameters
    ----------
    names
        Node names, or values converted with ``str``, in order.
        Repeats are allowed.
    known_names
        Node names that exist on the spec. Read once into a set.
    listed_in
        The spec field a caller reads for the valid names, such
        as ``"spec.nodes"``, quoted in the message.

    Returns
    -------
    list of str
        The node names after ``str(...)``, in ``names`` order.

    Raises
    ------
    Kpnn2Error
        If a name is empty, or if any name is not in
        ``known_names``.
    """
    node_names = [str(name) for name in names]
    if "" in node_names:
        raise caller_error(_EMPTY_NODE_NAME)
    known = frozenset(known_names)
    unknown = sorted({name for name in node_names if name not in known})
    if unknown:
        candidates, fix = unknown_name_hints(
            unknown,
            len(set(node_names)),
            known,
            None,
        )
        raise caller_error(
            "Unknown node name(s).",
            got=listed(repr(name) for name in unknown),
            expected=f"names in {listed_in}",
            candidates=candidates,
            fix=fix,
        )
    return node_names


def unknown_name_hints(
    unknown: Sequence[str],
    n_given: int,
    known_names: Iterable[str],
    argument: str | None,
) -> tuple[str | None, str | None]:
    """
    Return the ``Did you mean`` and ``Fix`` values for unknown names.

    Close valid names are suggested, unless none of several given
    names is valid: then they are likely another kind of
    identifier, such as Ensembl IDs for gene symbols, and the fix
    says to check them instead.

    Parameters
    ----------
    unknown
        The names that are not valid, in the order the message
        lists them.
    n_given
        How many distinct names the caller gave.
    known_names
        The valid names to suggest from.
    argument
        The argument that held the names, quoted in the fix, or
        ``None`` when the message names no argument.

    Returns
    -------
    tuple of (str or None, str or None)
        ``(candidates, fix)``; at most one is not ``None``.
    """
    if len(unknown) == n_given > 1:
        where = "" if argument is None else f" in '{argument}'"
        fix = (
            f"None of the names{where} is a node; check that they use "
            "the same identifiers as the edgelist."
        )
        return None, fix
    return (
        close_matches(
            unknown,
            known_names,
        ),
        None,
    )


def _node_name_error(
    node_name: str,
    known_names: Collection[str],
    listed_in: str,
    edge: str | None,
) -> Kpnn2Error | None:
    """
    Return the error for a name that names no node, or ``None``.

    ``edge`` is ``"source -> target"`` when the name is half of a
    pair. The valid names are not listed, because a graph may have
    thousands; ``Expected`` points at ``listed_in`` instead.
    """
    if node_name != "" and node_name in known_names:
        return None
    prefix = "" if edge is None else f"No edge {edge}: "
    if node_name == "":
        if edge is None:
            return caller_error(_EMPTY_NODE_NAME)
        return caller_error(f"{prefix}a node name is empty.")
    candidates, _ = unknown_name_hints(
        [node_name],
        1,
        known_names,
        None,
    )
    if edge is None:
        headline = "Unknown node name."
    else:
        headline = f"{prefix}unknown node name."
    return caller_error(
        headline,
        got=repr(node_name),
        expected=f"a name in {listed_in}",
        candidates=candidates,
    )


def iter_block_pairs(
    source: NodeSlot,
    target: NodeSlot,
) -> Iterator[tuple[int, int]]:
    """
    Yield unit-index pairs of one named edge.

    Order is target-unit outer, source-unit inner, matching
    ``fill_block`` writing ``mask[target.units, source.units]``.
    Each yielded pair is ``(source_unit, target_unit)``.
    """
    for target_unit in range(target.start, target.stop):
        for source_unit in range(source.start, source.stop):
            yield (
                source_unit,
                target_unit,
            )


def fill_block(
    mask: torch.Tensor,
    target: NodeSlot,
    source: NodeSlot,
) -> None:
    """
    Mark one edge as connected in a mask.

    Writes ``1.0`` into every unit pair of the edge, that is the
    ``(target.width, source.width)`` block. With one-unit nodes
    that block is the single entry
    ``mask[target.start, source.start]``.
    """
    mask[target.units, source.units] = 1.0


def dense_mask_from_indices(
    source_index: Sequence[int],
    target_index: Sequence[int],
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """
    Allocate a dense float32 mask from packed unit indices.

    Shape is ``(out_features, in_features)``. Each live edge
    sets ``1.0`` at ``[target_index[i], source_index[i]]``.
    Every call returns a fresh tensor.
    """
    mask = torch.zeros(
        (
            out_features,
            in_features,
        ),
        dtype=torch.float32,
    )
    source = torch.as_tensor(
        source_index,
        dtype=torch.int64,
    )
    target = torch.as_tensor(
        target_index,
        dtype=torch.int64,
    )
    if source.shape != target.shape:
        raise internal_error(
            f"packed mask got {source.numel()} source and "
            f"{target.numel()} target index(es)"
        )
    mask[target, source] = 1.0
    return mask


def expand_columns(
    values: np.ndarray,
    layout: Layout,
) -> np.ndarray:
    """
    Repeat each node column across the units it owns.

    Column ``i`` of ``values`` must belong to ``layout.slots[i]``.
    Returns ``values`` unchanged while every node is one unit
    wide.
    """
    widths = layout.widths()
    if all(width == DEFAULT_NODE_WIDTH for width in widths):
        return values
    return np.repeat(
        values,
        widths,
        axis=1,
    )
