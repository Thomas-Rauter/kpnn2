"""
Structural blueprint for an edgelist-defined DAG.
"""

from dataclasses import dataclass

from torch import Tensor

from ._frozen_mask import freeze_mask


@dataclass(frozen=True)
class Skip:
    """
    One original edge whose endpoints are more than one layer apart.

    Adjacent edges (depth gap exactly 1) live in ``GraphSpec.masks``,
    not here. A skip is metadata for a residual add in user
    ``forward()`` code. It is not expanded into a dummy neuron.

    Parameters
    ----------
    source : str
        Source node name.
    target : str
        Target node name.
    source_layer : int
        Depth of ``source`` in ``GraphSpec.layer_nodes``.
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
class GraphSpec:
    """
    Frozen blueprint from ``parse_edgelist``.

    Structure only: not an ``nn.Module`` and no parameters. Use
    ``masks`` for ``MaskedLinear`` hops and ``skips`` for residual
    adds in ``forward()``.

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
    masks : tuple[torch.Tensor, ...]
        Adjacent-hop masks, dtype float32.
        ``len(masks) == len(layer_nodes) - 1``. ``masks[i]`` is the
        hop from layer ``i`` to ``i + 1`` and has shape
        ``(layer_dims[i + 1], layer_dims[i])``, matching
        ``nn.Linear.weight``. Entry ``[target_index, source_index]``
        is ``1.0`` for an original edge with depth gap exactly 1,
        otherwise ``0.0``. Skip edges are not written into these
        tensors. In-place writes, ``out=`` into the tensor, and
        numpy aliases of stored storage are rejected.
    skips : tuple[Skip, ...]
        Original edges with depth gap greater than 1. Each record
        has ``source``, ``target``, ``source_layer``,
        ``target_layer``, ``source_index``, and ``target_index``.
        Wire these in ``forward()`` as a residual add. Do not expand
        them into pseudo-nodes.

    Notes
    -----
    Fields cannot be reassigned. Sequences are tuples. Mask
    tensors reject in-place writes and ``out=`` writes
    (``Kpnn2Error``). ``numpy()`` is not a writable view of
    stored storage. ``MaskedLinear`` clones the mask into a
    non-persistent buffer independent of ``spec.masks``.

    Examples
    --------
    Inspect layers, a mask, and a skip after parsing:

    >>> import pandas as pd
    >>> import kpnn2 as k2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> spec = k2.parse_edgelist(edgelist)
    >>> spec.layer_nodes
    (('A',), ('H',), ('C',))
    >>> spec.layer_dims
    (1, 1, 1)
    >>> spec.masks[0].tolist()
    [[1.0]]
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
    masks: tuple[Tensor, ...]
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
            "masks",
            tuple(freeze_mask(mask) for mask in self.masks),
        )
        object.__setattr__(
            self,
            "skips",
            tuple(self.skips),
        )
