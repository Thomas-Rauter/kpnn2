"""
Structural blueprint for an edgelist-defined DAG.
"""

from dataclasses import dataclass
from itertools import accumulate

import pandas as pd
from torch import Tensor

from ._mask_tensor import as_mask_tensor


@dataclass(frozen=True)
class Hop:
    """
    Every edge entering one layer, as one mask.

    One entry of ``LayeredSpec.hops``, never built by hand: a hop
    is exactly what a single ``MaskedLinear`` computes. Its mask
    holds **all** parents of ``target_layer``, so a skip edge is
    an ordinary one in it rather than a term added later, and no
    edge can be dropped. Columns are the source layers
    concatenated, the axis ``gather_hop_inputs`` assembles. The
    mask is a plain, writable tensor.

    Parameters
    ----------
    target_layer : int
        Depth of the layer this hop produces. Always at least 1;
        layer 0 has no parents. ``LayeredSpec.hops[i]`` has
        ``target_layer == i + 1``.
    source_layers : tuple[int, ...]
        Depths this hop reads, ascending, each one below
        ``target_layer``. Only layers that really feed the
        target appear, and ``target_layer - 1`` is always one of
        them. A single entry is a plain adjacent hop; ``hops[0]``
        is always ``(0,)``, so an ``align_inputs`` tensor feeds
        it with no gathering.
    source_dims : tuple[int, ...]
        Units contributed by each entry of ``source_layers``,
        same order. Their sum is ``mask.shape[1]``.
    source_nodes : tuple[str, ...]
        Node names of the mask columns, source layers
        concatenated in ``source_layers`` order. One name per
        node; with one unit per node that is one name per
        column.
    mask : torch.Tensor
        Connectivity of shape
        ``(layer_dims[target_layer], sum(source_dims))``, dtype
        float32, matching ``nn.Linear.weight``. Rows are named by
        ``layer_nodes[target_layer]`` and columns by
        ``source_nodes``; an entry is ``1.0`` when the original
        edgelist has an edge from the node naming that column to
        the node naming that row, and ``0.0`` otherwise. Treat it
        as read-only: it is a plain tensor, so writing to it
        silently changes the wiring this record describes.

    See Also
    --------
    LayeredSpec : Holds ``hops``, one per layer after the first.
    gather_hop_inputs : Builds the tensor whose columns this mask
        expects.
    MaskedLinear : Applies one hop, on its own copy of the mask.
    Skip : Metadata for the edges in this mask that span layers.

    Notes
    -----
    Every edgelist edge is a one in exactly one hop mask, the one
    of its target layer, so the ones summed over all hops give
    the edge count and applying a hop applies every parent of its
    layer at once.

    To locate one source layer's block inside the mask, add the
    widths in front of it:

    ``offset = sum(source_dims[:source_layers.index(layer)])``

    ``column_offsets`` does that for you.

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
    >>> hop.mask.tolist()
    [[1.0, 1.0]]
    """

    target_layer: int
    source_layers: tuple[int, ...]
    source_dims: tuple[int, ...]
    source_nodes: tuple[str, ...]
    mask: Tensor

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
            "mask",
            as_mask_tensor(self.mask),
        )

    @property
    def column_offsets(self) -> tuple[int, ...]:
        """
        First mask column of each entry of ``source_layers``.

        Same length and order as ``source_layers``. Add a node's
        index inside its own layer to get its mask column.
        """
        return tuple(
            accumulate(
                self.source_dims[:-1],
                initial=0,
            )
        )


@dataclass(frozen=True)
class Skip:
    """
    One original edge whose endpoints are more than one layer apart.

    Metadata, not a second computation: the edge is already a one
    in ``LayeredSpec.hops[target_layer - 1].mask``, exactly like an
    adjacent edge, so nothing has to add it back later and nothing
    can forget to. Read ``LayeredSpec.skips`` to inspect which
    prior-knowledge edges span layers; a forward pass never reads
    it. ``parse_layered`` builds these, never the caller.

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
        and it names the hop carrying it, ``hops[target_layer - 1]``.
    source_index : int
        Index of ``source`` inside ``layer_nodes[source_layer]``.
        This is a position in that layer alone, not a column of the
        hop mask, whose columns are several layers concatenated;
        see Examples for the shift.
    target_index : int
        Index of ``target`` inside ``layer_nodes[target_layer]``,
        which is also its row in ``hops[target_layer - 1].mask``,
        since a hop mask has one row per unit of its own layer.

    See Also
    --------
    Hop : The mask this edge is already a one in, alongside every
        other parent of ``target_layer``.
    LayeredSpec : Holds ``skips``, empty when no edge spans layers.
    parse_layered : Builds the spec these records come from.

    Notes
    -----
    Every original edge with a depth gap greater than 1 is recorded
    once; adjacent edges never are. Membership changes nothing about
    how the edge is computed: its weight, the unit bias, and the
    fan-in the degree-aware initialization uses all stay on the
    target layer's ``MaskedLinear``. Expanding a skip into dummy
    neurons is not the intended use.

    Examples
    --------
    Locate a skip inside the hop mask that carries it:

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
    >>> hop = spec.hops[skip.target_layer - 1]
    >>> offset = hop.column_offsets[
    ...     hop.source_layers.index(skip.source_layer)
    ... ]
    >>> hop.mask[skip.target_index, offset + skip.source_index].item()
    1.0
    """

    source: str
    target: str
    source_layer: int
    target_layer: int
    source_index: int
    target_index: int


@dataclass(frozen=True)
class LayeredSpec:
    """
    Frozen blueprint from ``parse_layered``.

    Depth-ranked wiring for one knowledge-primed network: every
    node sits at a layer, and one ``Hop`` per layer after the
    first holds every edge entering it, skips included. Build one
    ``MaskedLinear`` per hop, and use the name tuples to label
    tensors. It is structure only — no ``nn.Module``, no
    parameters — and the masks are plain, writable tensors.
    ``AdjacencySpec`` is the packed alternative.

    Parameters
    ----------
    input_nodes : tuple[str, ...]
        In-degree 0 names, alphabetical. This is the column order of
        tensors returned by ``align_inputs``.
    output_nodes : tuple[str, ...]
        Out-degree 0 names, alphabetical. A terminal node below
        maximum depth belongs here too, so this is not the same
        tuple as ``layer_nodes[-1]``.
    hidden_nodes : tuple[str, ...]
        Names that are neither input nor output, alphabetical.
    layer_nodes : tuple[tuple[str, ...], ...]
        ``layer_nodes[i]`` is the names at depth ``i``, alphabetical.
        Index 0 is the input layer. Depth is longest path from the
        inputs, and there are always at least two layers.
    layer_dims : tuple[int, ...]
        ``layer_dims[i] == len(layer_nodes[i])``: the unit width of
        each layer, one unit per node.
    hops : tuple[Hop, ...]
        One hop per layer after the first:
        ``len(hops) == len(layer_nodes) - 1`` and
        ``hops[i].target_layer == i + 1``. ``hops[i].mask`` holds
        **every** edge entering layer ``i + 1``, adjacent and
        skip alike, over the concatenated source layers.
        ``hops[0]`` always reads layer 0 only.
    skips : tuple[Skip, ...]
        Original edges with depth gap greater than 1, as metadata.
        Each one is already a one in
        ``hops[target_layer - 1].mask``; this list only says which
        edges span layers, and is empty when none do.

    See Also
    --------
    parse_layered : Builds this spec from a ``source`` / ``target``
        edgelist.
    AdjacencySpec : Packed sibling layout, for cycles, self-loops,
        or one shared state vector instead of depths.
    gather_hop_inputs : Assembles one hop's input from the layer
        tensors produced so far.
    MaskedLinear : Consumes ``hops[i].mask`` as one layer.

    Notes
    -----
    Fields cannot be reassigned and sequences are tuples, so the
    structure itself is fixed. The mask tensors are plain
    float32 tensors and are not write-protected; treat them as
    read-only and rebuild from the edgelist to change wiring.
    ``MaskedLinear`` clones the mask into a non-persistent
    buffer independent of ``spec.hops[i].mask``, so a layer built
    earlier keeps its own connectivity either way.

    Because a hop mask carries every parent of its target, the
    per-row degree ``MaskedLinear`` initializes from is the real
    fan-in of that unit, skips included.

    ``to_edgelist()``, ``to_dict()`` with ``from_dict()``, and
    ``fingerprint`` are the supported interchange; each
    round-trips through ``parse_layered``. Pickle and
    ``torch.save`` of the dataclass are not.

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
    >>> spec.hops[0].source_layers, spec.hops[0].mask.tolist()
    ((0,), [[1.0]])
    >>> spec.hops[1].source_layers, spec.hops[1].mask.tolist()
    ((0, 1), [[1.0, 1.0]])
    >>> spec.skips[0].source, spec.skips[0].target
    ('A', 'C')
    >>> spec.skips[0].source_layer, spec.skips[0].target_layer
    (0, 2)
    """

    input_nodes: tuple[str, ...]
    output_nodes: tuple[str, ...]
    hidden_nodes: tuple[str, ...]
    layer_nodes: tuple[tuple[str, ...], ...]
    layer_dims: tuple[int, ...]
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
        follow the hop masks in canonical order: sorted
        lexicographically by ``(source, target)``, one row per
        original edge, names as strings. Extra columns from the
        DataFrame that was parsed are not reproduced.

        ``parse_layered`` on this table reconstructs the same
        node lists, hops, and hop masks. Skip tuple order
        follows these sorted rows rather than the original
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

    def to_dict(self) -> dict:
        """
        Return this spec as a JSON-safe tagged dict.

        Keys are ``kpnn2_spec`` (integer ``1``), ``layout``
        (``"layered"``), and ``edges`` (list of
        ``[source, target]`` lists in the same order as
        ``to_edgelist()`` rows). The returned dict is new on
        every call.

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
        ``payload["edges"]``. Hops and masks are not assembled
        by hand. Extra unknown keys are ignored.

        Parameters
        ----------
        payload : dict
            A dict with ``kpnn2_spec``, ``layout``, and
            ``edges``. ``layout`` must be ``"layered"``.

        Returns
        -------
        LayeredSpec
            The parsed spec.

        Raises
        ------
        Kpnn2Error
            If ``payload`` is not a dict; ``kpnn2_spec`` is
            missing or not ``1``; ``layout`` is missing, not a
            known layout, or is ``"adjacency"``; or ``edges``
            is missing or not a sequence of two nonempty names.

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
