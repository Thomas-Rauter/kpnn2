"""
Build sparsely connected PyTorch neural networks from a named
edgelist, using native nn.Module layers.

``kpnn2`` turns a source/target edgelist into a spec so you write
ordinary PyTorch with ``MaskedLinear`` and ``SkipAdd``. Pick the
layout yourself: ``parse_layered`` ranks a DAG into a
``LayeredSpec`` (adjacent masks, layer names, skip list), while
``parse_adjacency`` puts every node in one state vector with a
single square mask (``AdjacencySpec``) and allows cycles.

It is not a graph compiler: there is no ready-made model object
and no training loop.
"""

from ._adjacency_spec import AdjacencySpec
from ._align import align_inputs
from ._attributions import map_node_attributions
from ._errors import Kpnn2Error
from ._masked_linear import MaskedLinear
from ._parse import parse_layered
from ._parse_adjacency import parse_adjacency
from ._skip_add import SkipAdd
from ._spec import LayeredSpec, Skip

__version__ = "0.1.0"

__all__ = [
    "parse_layered",
    "parse_adjacency",
    "LayeredSpec",
    "Skip",
    "AdjacencySpec",
    "MaskedLinear",
    "SkipAdd",
    "align_inputs",
    "map_node_attributions",
    "Kpnn2Error",
    "__version__",
]
