"""
Structural blueprint for an edgelist-defined node network.
"""

from dataclasses import dataclass

import pandas as pd
from torch import Tensor

from ._layout import build_layout, dense_mask_from_indices


@dataclass(frozen=True)
class AdjacencySpec:
    """
    Frozen blueprint from ``parse_adjacency``.

    Packed wiring for one knowledge-primed network: every node is
    a unit of one alphabetical state vector, and every edge an
    index pair, so cycles and self-loops are ordinary. Structure
    only — no ``nn.Module``, no parameters, no stored square — and
    the update is yours. Input nodes have no incoming edges, so
    writing the inputs into the state each step is required.
    ``LayeredSpec`` is the depth-ranked alternative.

    Parameters
    ----------
    nodes : tuple[str, ...]
        Every node name, alphabetical. This is the unit order of
        the state vector and the row and column order of
        ``to_mask()``.
    input_nodes : tuple[str, ...]
        In-degree 0 names, alphabetical. This is the column order
        of tensors returned by ``align_inputs``, which is
        narrower than ``nodes`` unless every node is an input.
    output_nodes : tuple[str, ...]
        Out-degree 0 names, alphabetical.
    hidden_nodes : tuple[str, ...]
        Names that are neither input nor output, alphabetical.
        Empty when every node is an input or an output.
    source_index : tuple[int, ...]
        For each original edge, the column in ``nodes`` (the
        source). Same length as ``target_index`` and as the
        edge count. Order is canonical: lexicographic by
        ``(source name, target name)``, identical to
        ``to_edgelist()`` row order. Cycles and self-loops are
        included.
    target_index : tuple[int, ...]
        For each original edge, the row in ``nodes`` (the
        target). A dense square would have ``1.0`` at
        ``[target_index[i], source_index[i]]``.
    input_index : tuple[int, ...]
        Position of each ``input_nodes`` name in ``nodes``, same
        order. Scatter an ``align_inputs`` tensor into the state
        vector along these columns.
    output_index : tuple[int, ...]
        Position of each ``output_nodes`` name in ``nodes``, same
        order. Read the network's outputs from the state vector
        along these columns.

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
    align_inputs : Orders a named DataFrame onto ``input_nodes``.

    Notes
    -----
    Fields cannot be reassigned and sequences are tuples, so the
    structure itself is fixed. There is no ``mask`` field and no
    densifying ``mask`` property. ``to_mask()`` allocates a
    fresh dense square on every call; mutating that tensor does
    not change this spec. ``MaskedLinear(spec.to_mask())``
    clones the square into a non-persistent buffer, so a layer
    built earlier keeps its own connectivity.

    ``align_inputs`` returns ``len(input_nodes)`` columns, which
    is not the state width. Scatter that tensor into the
    ``n``-wide state vector with ``input_index``. Input rows of
    ``to_mask()`` are all zeros, so under the degree-aware init
    of ``MaskedLinear`` and ``PackedLinear`` those units stay
    zero: writing the inputs in is required, not cosmetic.

    Depth does not exist in this layout, so there is no
    ``layer_nodes``, ``layer_dims``, ``hops``, or ``skips``, and
    ``gather_hop_inputs`` does not accept this spec. An edge
    that would span layers is already an ordinary index pair.
    This is not a one-layer ``LayeredSpec``; the layout is your
    choice, and a DAG is valid input to either parser.

    ``to_edgelist()``, ``to_dict()`` with ``from_dict()``, and
    ``fingerprint`` are the supported interchange; each
    round-trips through ``parse_adjacency``, cycle edges and
    self-loops included. Pickle and ``torch.save`` of the
    dataclass are not.

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
    >>> tuple(spec.to_mask().shape)
    (4, 4)
    >>> spec.to_mask()[0].tolist()
    [0.0, 1.0, 1.0, 0.0]
    """

    nodes: tuple[str, ...]
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

    def to_mask(self) -> Tensor:
        """
        Allocate a dense float32 square from the packed edges.

        Shape is ``(n, n)`` with ``n`` from the node layout
        (``len(nodes)`` at width 1). The result starts at zeros;
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
        layout = build_layout(self.nodes)
        n_units = layout.n_units
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
        node lists, packed indices, and input/output indices.

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

    def to_dict(self) -> dict:
        """
        Return this spec as a JSON-safe tagged dict.

        Keys are ``kpnn2_spec`` (integer ``1``), ``layout``
        (``"adjacency"``), and ``edges`` (list of
        ``[source, target]`` lists in the same order as
        ``to_edgelist()`` rows, including cycle edges and
        self-loops). The returned dict is new on every call.

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
        ``payload["edges"]``. Packed indices are not assembled
        by hand. Extra unknown keys are ignored.

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
            known layout, or is ``"layered"``; or ``edges`` is
            missing or not a sequence of two nonempty names.

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
