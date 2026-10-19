"""
Build interpretable neural networks in PyTorch from a graph of
named entities, with checks that every attribution is reported
under the right name.

``kpnn2`` turns a source/target edgelist into a spec so you write
ordinary PyTorch. Pick the layout yourself: ``parse_layered``
ranks a DAG into a ``LayeredSpec``, one ``Hop`` per layer whose
packed indices carry every edge entering it, skips included,
while ``parse_adjacency`` puts every node in one state vector
with packed edge indices (``AdjacencySpec``). That packed layout
allows cycles and self-loops; a DAG is valid too. Build
``PackedLinear.from_hop(hop)`` or
``PackedLinear.from_adjacency(spec)``, or
``MaskedLinear(hop.to_mask())`` / ``MaskedLinear(spec.to_mask())``
for the dense path. An ``AdjacencySpec`` can also feed
``PackedMultiheadAttention.from_adjacency(spec, ...)``; that is a
contraction primitive you assemble, not a compiler and not a
ready-made Transformer.

It is not a graph compiler: there is no ready-made model object
and no training loop.
"""

from ._adjacency_spec import AdjacencySpec
from ._aggregation import (
    aggregate_node_attributions,
    list_aggregation_methods,
)
from ._align import align_inputs
from ._attributions import map_node_attributions
from ._errors import Kpnn2Error
from ._gather import gather_hop_inputs, scatter_hop_outputs
from ._masked_linear import MaskedLinear
from ._packed_linear import PackedLinear
from ._packed_multihead_attention import PackedMultiheadAttention
from ._parse import parse_layered
from ._parse_adjacency import parse_adjacency
from ._spec import Hop, LayeredSpec, Skip

__version__ = "0.3.0"

__all__ = [
    "parse_layered",
    "parse_adjacency",
    "LayeredSpec",
    "Hop",
    "Skip",
    "AdjacencySpec",
    "MaskedLinear",
    "PackedLinear",
    "PackedMultiheadAttention",
    "gather_hop_inputs",
    "scatter_hop_outputs",
    "align_inputs",
    "map_node_attributions",
    "aggregate_node_attributions",
    "list_aggregation_methods",
    "Kpnn2Error",
    "__version__",
]
