"""
Errors and caller warnings that kpnn2 raises.

``Kpnn2Error`` is for caller mistakes; ``caller_error`` builds it.
``internal_error`` builds the ``AssertionError`` for a failed
internal consistency check, which means kpnn2 has a bug.
``warn_at_caller`` issues a warning that points at the caller's
line, not at kpnn2's own code.

All three messages share one format: a headline, then optional
field lines in a fixed order (see ``caller_error``). ``listed``
and ``close_matches`` build field values that name items.
"""

import difflib
import os
import sys
import warnings
from collections.abc import Iterable, Sequence
from types import FrameType

_ISSUES_URL = "https://github.com/Thomas-Rauter/kpnn2/issues"
# Frames from files under this directory belong to kpnn2.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
# Items a listed value shows before ", and N more".
_LIST_LIMIT = 10
# Unknown names that get suggestions, and suggestions per name.
_SUGGESTED_NAMES = 5
_SUGGESTIONS = 3
# Every character str.splitlines() breaks on, escaped, so a name
# that holds one cannot split a headline or field line.
_LINE_BREAK_ESCAPES = {
    ord(char): repr(char)[1:-1]
    for char in "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"
}


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
    Kpnn2Error: Edgelist contains 1 self-loop(s), and parse_layered() ...
    """


def caller_error(
    headline: str,
    *,
    got: str | None = None,
    expected: str | None = None,
    candidates: str | None = None,
    fix: str | None = None,
) -> Kpnn2Error:
    """
    Build the error for a caller mistake.

    Call sites raise it explicitly. The message is the headline,
    then one line for each field given, in this order::

        <headline>
          Got: <got>
          Expected: <expected>
          Did you mean: <candidates>
          Fix: <fix>

    A line break inside any part is escaped, so each part stays
    one line.

    Parameters
    ----------
    headline : str
        What is wrong, ending with a period. It may add a short
        reason.
    got : str, optional
        The received value, or the part of the input that breaks
        the rule. No trailing period.
    expected : str, optional
        What would be valid, when the headline does not say it.
        No trailing period.
    candidates : str, optional
        Close valid names, as ``close_matches`` returns them.
    fix : str, optional
        What to do, ending with a period.

    Returns
    -------
    Kpnn2Error
        Error whose message has the standard format.
    """
    return Kpnn2Error(
        _message(
            headline,
            got,
            expected,
            candidates,
            fix,
        )
    )


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
        Error whose headline says the failure is a kpnn2 bug if
        every spec came from a parser or ``from_dict``, and whose
        ``Fix`` line asks for a report at the issue tracker. The
        hedge is there because a spec built or changed by hand is
        unsupported and can trip an internal check.
    """
    from kpnn2 import __version__

    # Rendered without _message's checks, which raise this error.
    return AssertionError(
        _render(
            f"kpnn2 internal check failed: {what}. This is a bug in "
            f"kpnn2 {__version__} if every spec came from a parser or "
            "from_dict().",
            (
                (
                    "Fix",
                    f"Report it with this message at {_ISSUES_URL}. A spec, "
                    "Hop, or Skip built or changed by hand is unsupported: "
                    "rebuild it with parse_layered(), parse_adjacency(), or "
                    "from_dict().",
                ),
            ),
        )
    )


def warn_at_caller(
    headline: str,
    category: type[Warning],
    *,
    got: str | None = None,
    expected: str | None = None,
    candidates: str | None = None,
    fix: str | None = None,
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
    headline : str
        What is odd, ending with a period, as in ``caller_error``.
    category : type of Warning
        The warning class, such as ``UserWarning`` or
        ``FutureWarning``.
    got : str, optional
        As in ``caller_error``.
    expected : str, optional
        As in ``caller_error``.
    candidates : str, optional
        As in ``caller_error``.
    fix : str, optional
        As in ``caller_error``.
    """
    message = _message(
        headline,
        got,
        expected,
        candidates,
        fix,
    )
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


def listed(
    items: Iterable[object],
    separator: str = ", ",
) -> str:
    """
    Join items with commas, showing at most ten.

    Parameters
    ----------
    items : iterable of object
        The items, in the order to show them. Each is shown
        through ``str()``.
    separator : str, default ", "
        What goes between two items. Items that hold commas
        themselves, such as one entry per problem, use ``"; "``.

    Returns
    -------
    str
        ``"A, B, C"``, or the first ten items followed by
        ``", and N more"``.
    """
    values = list(items)
    shown = separator.join(str(item) for item in values[:_LIST_LIMIT])
    hidden = len(values) - _LIST_LIMIT
    if hidden > 0:
        return f"{shown}{separator}and {hidden} more"
    return shown


def close_matches(
    unknown: Sequence[str],
    known: Iterable[str],
) -> str | None:
    """
    Suggest valid names close to names that are not valid.

    A name that matches valid names when case is ignored gets
    those names. Any other name gets up to three that ``difflib``
    rates as close. Only the first five unknown names are looked
    up, so a long list of misses stays cheap. The caller decides;
    nothing is picked for it.

    Parameters
    ----------
    unknown : sequence of str
        The names that are not valid, in the order the message
        lists them.
    known : iterable of str
        The valid names.

    Returns
    -------
    str or None
        The value for a ``Did you mean`` line: ``"'TP53'"`` for
        one unknown name, ``"'TP53' for 'TP35'; 'MYC' for
        'MCY'"`` for several, or ``None`` when no valid name is
        close.
    """
    names = list(known)
    by_case: dict[str, list[str]] = {}
    for name in names:
        by_case.setdefault(
            name.casefold(),
            [],
        ).append(name)
    pairs = []
    for name in unknown[:_SUGGESTED_NAMES]:
        found = by_case.get(name.casefold()) or difflib.get_close_matches(
            name,
            names,
            n=_SUGGESTIONS,
        )
        if found:
            quoted = ", ".join(repr(match) for match in found[:_SUGGESTIONS])
            pairs.append((name, quoted))
    if not pairs:
        return None
    if len(unknown) == 1:
        return pairs[0][1]
    return "; ".join(f"{quoted} for {name!r}" for name, quoted in pairs)


def _message(
    headline: str,
    got: str | None,
    expected: str | None,
    candidates: str | None,
    fix: str | None,
) -> str:
    """Check the parts of a message, then render them."""
    if not headline.endswith("."):
        raise internal_error(
            f"message headline {headline!r} does not end with a period"
        )
    fields = (
        ("Got", got),
        ("Expected", expected),
        ("Did you mean", candidates),
        ("Fix", fix),
    )
    # The headline is quoted, so the report names the message, and a
    # caller mistake that reaches here keeps what it was about.
    for label, value in fields:
        if value == "":
            raise internal_error(
                f'the {label} line is empty in the message "{headline}"'
            )
    if fix is not None and not fix.endswith("."):
        raise internal_error(
            f"the Fix line {fix!r} does not end with a period in the "
            f'message "{headline}"'
        )
    return _render(
        headline,
        fields,
    )


def _render(
    headline: str,
    fields: tuple[tuple[str, str | None], ...],
) -> str:
    """Join the headline and the given fields, one line each."""
    lines = [headline.translate(_LINE_BREAK_ESCAPES)]
    for label, value in fields:
        if value is not None:
            escaped = value.translate(_LINE_BREAK_ESCAPES)
            lines.append(f"  {label}: {escaped}")
    return "\n".join(lines)
