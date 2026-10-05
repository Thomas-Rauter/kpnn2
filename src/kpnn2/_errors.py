"""
Public exception type and internal-check error factory for kpnn2.

``Kpnn2Error`` is for caller mistakes. ``internal_error`` builds
the ``AssertionError`` for a failed internal consistency check,
which means kpnn2 has a bug.
"""

_ISSUES_URL = "https://github.com/Thomas-Rauter/kpnn2/issues"


class Kpnn2Error(Exception):
    """
    User-facing failure from the public ``kpnn2`` API.

    Raised for invalid edgelists, illegal ``LayeredSpec`` operations,
    bad ``MaskedLinear`` masks, saved activations that do not match
    the hop they are gathered for, input or attribution tensors that
    do not match the spec, and invalid attribution aggregation
    calls.

    Examples
    --------
    >>> import pandas as pd
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A"],
    ...         "target": ["A"],
    ...     }
    ... )
    >>> kpnn2.parse_layered(edgelist)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    Kpnn2Error: Edgelist contains 1 self-loop(s): A. ...
    """


def internal_error(what: str) -> AssertionError:
    """
    Build the error for a failed internal consistency check.

    Call sites raise it explicitly. ``assert`` is not used,
    because ``python -O`` strips it.

    Parameters
    ----------
    what : str
        The invariant that failed, without a trailing period.

    Returns
    -------
    AssertionError
        Error whose message says the failure is a kpnn2 bug and
        asks for a report at the issue tracker.
    """
    from kpnn2 import __version__

    return AssertionError(
        f"kpnn2 internal check failed: {what}. This is a bug in "
        f"kpnn2 {__version__}, not a problem with your call. "
        f"Please report it with this message at {_ISSUES_URL}."
    )
