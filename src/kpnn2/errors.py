"""
Public exception type for kpnn2.
"""


class Kpnn2Error(Exception):
    """
    User-facing failure from the public ``kpnn2`` API.

    Raised for invalid edgelists, illegal ``GraphSpec`` operations,
    bad ``MaskedLinear`` masks, and input or attribution tensors
    that do not match the spec.

    Examples
    --------
    >>> import pandas as pd
    >>> import kpnn2 as k2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A"],
    ...         "target": ["A"],
    ...     }
    ... )
    >>> k2.parse_edgelist(edgelist)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    Kpnn2Error: Edgelist contains 1 self-loop(s). ...
    """
