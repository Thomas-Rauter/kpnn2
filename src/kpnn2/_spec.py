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

    A hop is what a single ``MaskedLinear`` computes. Its mask
    covers **all** parents of ``target_layer``, whether they sit
    in the layer directly below or several layers back, so a
    skip edge is an ordinary one in this mask rather than a
    separate term added later. That is what makes an edge
    impossible to lose: apply the hop and every parent is
    applied with it.

    The mask columns are the source layers concatenated in
    ascending order, which is the axis
    ``kpnn2.gather_hop_inputs`` builds.

    Parameters
    ----------
    target_layer : int
        Depth of the layer this hop produces. Always at least 1;
        layer 0 has no parents.
    source_layers : tuple[int, ...]
        Depths this hop reads, ascending, each one below
        ``target_layer``. Only layers that really feed the
        target appear, and ``target_layer - 1`` is always one of
        them. A hop with a single entry is a plain adjacent hop.
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
        float32, matching ``nn.Linear.weight``. Entry
        ``[target_index, column]`` is ``1.0`` for an original
        edgelist edge and ``0.0`` otherwise. Treat it as
        read-only: it is a plain tensor, so writing to it
        silently changes the wiring this record describes.

    Notes
    -----
    To locate one source layer's block inside the mask, add the
    widths in front of it:

    ``offset = sum(source_dims[:source_layers.index(layer)])``

    ``column_offsets`` does that for you.

    Examples
    --------
    A chain ``A -> H -> C`` plus the skip ``A -> C``:

    >>> import pandas as pd
    >>> import kpnn2 as k2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> spec = k2.parse_layered(edgelist)
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

    This is **metadata**, not a separate computation. The edge
    itself is a one in ``LayeredSpec.hops[target_layer - 1].mask``,
    exactly like an adjacent edge, so nothing has to add it back
    later and nothing can forget to. Read ``skips`` to report or
    inspect which prior-knowledge edges span layers; do not
    expand them into dummy neurons.

    Parameters
    ----------
    source : str
        Source node name.
    target : str
        Target node name.
    source_layer : int
        Depth of ``source`` in ``LayeredSpec.layer_nodes``.
    target_layer : int
        Depth of ``target``. Always satisfies
        ``target_layer - source_layer > 1``.
    source_index : int
        Column index of ``source`` in
        ``layer_nodes[source_layer]``.
    target_index : int
        Column index of ``target`` in
        ``layer_nodes[target_layer]``.
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

    Structure only: not an ``nn.Module`` and no parameters. One
    ``Hop`` per layer after the first, and one ``MaskedLinear``
    per hop is the whole model wiring.

    Parameters
    ----------
    input_nodes : tuple[str, ...]
        In-degree 0 names, alphabetical. This is the column order of
        tensors returned by ``align_inputs``.
    output_nodes : tuple[str, ...]
        Out-degree 0 names, alphabetical.
    hidden_nodes : tuple[str, ...]
        Names that are neither input nor output, alphabetical.
    layer_nodes : tuple[tuple[str, ...], ...]
        ``layer_nodes[i]`` is the names at depth ``i``, alphabetical.
        Index 0 is the input layer.
    layer_dims : tuple[int, ...]
        ``layer_dims[i] == len(layer_nodes[i])``.
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
        edges span layers.

    Notes
    -----
    Fields cannot be reassigned and sequences are tuples, so the
    structure itself is fixed. The mask tensors are plain
    float32 tensors and are not write-protected; treat them as
    read-only and rebuild from the edgelist to change wiring.
    ``MaskedLinear`` clones the mask into a non-persistent
    buffer independent of ``spec.hops[i].mask``, so a layer built
    earlier keeps its own connectivity either way.

    ``to_edgelist()`` returns the original edges as a two-column
    ``source`` / ``target`` DataFrame, rows sorted
    lexicographically. ``parse_layered`` on that table
    reconstructs the same node lists, hops, and hop masks.

    ``to_dict()`` returns a JSON-safe tagged dict
    (``kpnn2_spec``, ``layout``, ``edges``). ``from_dict``
    rebuilds this spec by calling ``parse_layered``.
    ``fingerprint`` is the SHA-256 of that canonical JSON.
    Pickle / ``torch.save`` of the dataclass is not the
    supported interchange.

    Because a hop mask carries every parent of its target, the
    per-row degree ``MaskedLinear`` initializes from is the real
    fan-in of that unit, skips included.

    Examples
    --------
    Inspect layers, a hop, and a skip after parsing:

    >>> import pandas as pd
    >>> import kpnn2 as k2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> spec = k2.parse_layered(edgelist)
    >>> spec.layer_nodes
    (('A',), ('H',), ('C',))
    >>> spec.layer_dims
    (1, 1, 1)
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
        >>> import kpnn2 as k2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["H", "A", "A"],
        ...         "target": ["C", "C", "H"],
        ...     }
        ... )
        >>> spec = k2.parse_layered(edgelist)
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
        >>> import kpnn2 as k2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["A", "H"],
        ...         "target": ["H", "C"],
        ...     }
        ... )
        >>> spec = k2.parse_layered(edgelist)
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
        >>> import kpnn2 as k2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["A", "H"],
        ...         "target": ["H", "C"],
        ...     }
        ... )
        >>> spec = k2.parse_layered(edgelist)
        >>> roundtrip = k2.LayeredSpec.from_dict(spec.to_dict())
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
        >>> import kpnn2 as k2
        >>> edgelist = pd.DataFrame(
        ...     {
        ...         "source": ["A", "H"],
        ...         "target": ["H", "C"],
        ...     }
        ... )
        >>> spec = k2.parse_layered(edgelist)
        >>> len(spec.fingerprint)
        64
        >>> (
        ...     spec.fingerprint
        ...     == k2.parse_layered(spec.to_edgelist()).fingerprint
        ... )
        True
        """
        from ._serialize import spec_fingerprint

        return spec_fingerprint(self)
