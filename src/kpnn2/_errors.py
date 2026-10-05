"""
Errors and caller warnings that kpnn2 raises.

``Kpnn2Error`` is for caller mistakes. ``internal_error`` builds
the ``AssertionError`` for a failed internal consistency check,
which means kpnn2 has a bug. ``warn_at_caller`` issues a warning
that points at the caller's line, not at kpnn2's own code.
"""

import os
import sys
import warnings
from types import FrameType

_ISSUES_URL = "https://github.com/Thomas-Rauter/kpnn2/issues"
# Frames from files under this directory belong to kpnn2.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


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


def warn_at_caller(
    message: str,
    category: type[Warning],
) -> None:
    """
    Warn at the first frame outside kpnn2: the caller's own line.

    A fixed ``stacklevel`` is right for one call depth only, and
    ``warnings.warn(skip_file_prefixes=...)`` needs Python 3.12.
    This walks up out of the package instead, so a warning raised
    under ``from_dict`` and one raised by ``parse_layered`` both
    point at the line that called kpnn2.

    Parameters
    ----------
    message : str
        The warning text.
    category : type of Warning
        The warning class, such as ``UserWarning`` or
        ``FutureWarning``.
    """
    # Level 1 is this function, which calls warnings.warn.
    frame: FrameType | None = sys._getframe(0)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(
        _PACKAGE_DIR
    ):
        frame = frame.f_back
        level += 1
    warnings.warn(
        message,
        category,
        stacklevel=level,
    )
