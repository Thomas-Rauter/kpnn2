"""
Structural blueprint for an edgelist-defined node network.
"""

from dataclasses import dataclass

import pandas as pd
import torch
from torch import Tensor

from ._layout import build_layout


@dataclass(frozen=True)
class AdjacencySpec:
    """
    Frozen blueprint from ``parse_adjacency``.

    Structure only: not an ``nn.Module`` and no parameters. Every
    node lives in one state vector and connectivity is packed as
    source/target index tuples, so the graph may contain cycles
    and self-loops. There is no stored square mask.

    There are no depths here: no ``layer_nodes``, no ``hops``
    tuple, and no ``skips``. This is not a one-layer
    ``LayeredSpec``. Use ``MaskedLinear(spec.to_mask())`` for
    one state update and write the recurrence in ``forward()``.

    Parameters
    ----------
    nodes : tuple[str, ...]
        Every node name, alphabetical. This is the unit order of
        the state vector and the row and column order of
        ``to_mask()``.
    input_nodes : tuple[str, ...]
        In-degree 0 names, alphabetical. This is the column order
        of tensors returned by ``align_inputs``.
    output_nodes : tuple[str, ...]
        Out-degree 0 names, alphabetical.
    hidden_nodes : tuple[str, ...]
        Names that are neither input nor output, alphabetical.
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
        Position of each ``input_nodes`` name in ``nodes``.
    output_index : tuple[int, ...]
        Position of each ``output_nodes`` name in ``nodes``.

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
    of ``MaskedLinear`` they stay zero: writing the inputs in
    is required, not cosmetic.

    ``to_edgelist()`` returns the original edges as a
    two-column ``source`` / ``target`` DataFrame, rows sorted
    lexicographically, including cycle edges and self-loops.
    ``parse_adjacency`` on that table reconstructs this spec's
    node lists, packed indices, and input/output indices.

    ``to_dict()`` returns a JSON-safe tagged dict
    (``kpnn2_spec``, ``layout``, ``edges``). ``from_dict``
    rebuilds this spec by calling ``parse_adjacency``.
    ``fingerprint`` is the SHA-256 of that canonical JSON.
    Pickle / ``torch.save`` of the dataclass is not the
    supported interchange.

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
        mask = torch.zeros(
            (
                n_units,
                n_units,
            ),
            dtype=torch.float32,
        )
        for source, target in zip(
            self.source_index,
            self.target_index,
        ):
            mask[target, source] = 1.0
        return mask

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
