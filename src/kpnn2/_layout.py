"""
Unit placement for graph nodes on a tensor axis.

Every graph node owns a contiguous slice of units on the last
axis of a tensor. In v1 every slice has width
``DEFAULT_NODE_WIDTH == 1``, so a node's slice start is its
column index and block expansion writes a single mask entry.

Routing index arithmetic through this module is what keeps node
width additive: give ``build_layout`` real widths and mask
construction, input alignment, skip indexing, and attribution
naming follow without changes at their call sites.
"""

from collections.abc import Sequence
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
        Number of units the node owns. Always
        ``DEFAULT_NODE_WIDTH`` in v1.
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
    node's column index.

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
