"""
Values that stand in for a flag but are not a ``bool``.

Every public flag argument rejects each of them with
``Kpnn2Error``; a flag is never read by truthiness.
"""

import re

import numpy
import pytest

NON_BOOL_FLAGS = [
    pytest.param(
        "False",
        id="str",
    ),
    pytest.param(
        0,
        id="zero",
    ),
    pytest.param(
        1,
        id="one",
    ),
    pytest.param(
        None,
        id="None",
    ),
    pytest.param(
        numpy.bool_(True),
        id="numpy_bool",
    ),
]


def non_bool_match(name: str) -> str:
    """
    Pattern for the error a non-``bool`` ``name`` raises.

    Names the argument and holds ``Got``, so the received value
    follows.
    """
    return re.escape(f"'{name}' must be True or False. Got ")
