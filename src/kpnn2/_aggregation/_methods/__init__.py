"""
Registered aggregation methods.

Importing this package runs each method's decorator. Add a new
method by writing a module here and importing it below.
"""

from ._rauter_mangano_2026 import (
    rauter_mangano_2026,
    rauter_mangano_2026_legacy,
)

__all__ = [
    "rauter_mangano_2026",
    "rauter_mangano_2026_legacy",
]
