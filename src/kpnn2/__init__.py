"""
Sparse named neural nets from an edgelist, as PyTorch primitives.

``kpnn2`` turns a source/target edgelist into a ``GraphSpec``
(adjacent masks, layer names, skip list) so you write ordinary
PyTorch with ``MaskedLinear``. It is not a graph compiler:
there is no ready-made model object and no training loop.
"""

from .align_inputs import align_inputs
from .errors import Kpnn2Error
from .graph_spec import GraphSpec, Skip
from .map_node_attributions import map_node_attributions
from .masked_linear import MaskedLinear
from .parse_edgelist import parse_edgelist

__version__ = "0.1.0"

__all__ = [
    "parse_edgelist",
    "GraphSpec",
    "Skip",
    "MaskedLinear",
    "align_inputs",
    "map_node_attributions",
    "Kpnn2Error",
    "__version__",
]
