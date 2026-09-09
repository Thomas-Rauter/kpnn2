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
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from ._errors import Kpnn2Error

DEFAULT_NODE_WIDTH = 1


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
    Kpnn2Error
        If a width is below 1, a name repeats, or the slots do
        not tile the axis contiguously from 0.
    """

    slots: tuple[NodeSlot, ...]
    _by_name: dict[str, NodeSlot] = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        slots = tuple(self.slots)
        object.__setattr__(
            self,
            "slots",
            slots,
        )
        by_name: dict[str, NodeSlot] = {}
        position = 0
        for slot in slots:
            if slot.width < 1:
                raise Kpnn2Error(
                    f"Node {slot.name!r} must own at least one "
                    f"unit. Got width {slot.width}."
                )
            if slot.start != position:
                raise Kpnn2Error(
                    "Node slots must tile the axis without gaps. "
                    f"Node {slot.name!r} starts at {slot.start}, "
                    f"expected {position}."
                )
            if slot.name in by_name:
                raise Kpnn2Error(f"Duplicate node name in layout: {slot.name}.")
            by_name[slot.name] = slot
            position = slot.stop
        object.__setattr__(
            self,
            "_by_name",
            by_name,
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
            raise Kpnn2Error(f"Unknown node name in layout: {name}.") from None

    def start_of(self, name: str) -> int:
        """
        Return the first unit index of ``name``.
        """
        return self.slot(name).start

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
        raise Kpnn2Error(f"No node begins at unit index {start}.")

    def slot_containing(self, unit: int) -> NodeSlot:
        """
        Return the slot whose units slice contains ``unit``.

        ``slot_at`` requires a block start; this accepts any
        unit index inside the block.
        """
        if unit < 0 or unit >= self.n_units:
            raise Kpnn2Error(
                f"Unit index {unit} is out of range [0, {self.n_units})."
            )
        for slot in self.slots:
            if slot.start <= unit < slot.stop:
                return slot
        raise Kpnn2Error(f"No node owns unit index {unit}.")

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
    Kpnn2Error
        If ``widths`` has a different length than ``names``, or
        the resulting slots are not a valid layout.
    """
    ordered = list(names)
    if widths is None:
        sizes = [DEFAULT_NODE_WIDTH] * len(ordered)
    else:
        sizes = list(widths)
        if len(sizes) != len(ordered):
            raise Kpnn2Error(
                "'widths' must have one entry per node. Expected "
                f"{len(ordered)}, got {len(sizes)}."
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
    Kpnn2Error
        If a name appears in more than one input layout.
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
    for source, target in zip(
        source_index,
        target_index,
        strict=True,
    ):
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
