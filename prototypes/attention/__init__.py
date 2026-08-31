"""
Packed multi-head attention over an ``AdjacencySpec`` edge list.

Prototype only: not exported from ``kpnn2``. Uses public
``parse_adjacency`` packed indices; does not change the spec.
"""

from .packed_multihead_attention import PackedMultiheadAttention

__all__ = [
    "PackedMultiheadAttention",
]
