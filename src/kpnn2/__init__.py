"""
Build sparsely connected PyTorch neural networks from a named
edgelist, using native nn.Module layers.

``kpnn2`` turns a source/target edgelist into a ``LayeredSpec``
(adjacent masks, layer names, skip list) so you write ordinary
PyTorch with ``MaskedLinear`` and ``SkipAdd``. It is not a
graph compiler: there is no ready-made model object and no
training loop.
"""

from .align_inputs import align_inputs
from .errors import Kpnn2Error
from .layered_spec import LayeredSpec, Skip
from .map_node_attributions import map_node_attributions
from .masked_linear import MaskedLinear
from .parse_layered import parse_layered
from .skip_add import SkipAdd

__version__ = "0.1.0"

__all__ = [
    "parse_layered",
    "LayeredSpec",
    "Skip",
    "MaskedLinear",
    "SkipAdd",
    "align_inputs",
    "map_node_attributions",
    "Kpnn2Error",
    "__version__",
]
