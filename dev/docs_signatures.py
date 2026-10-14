"""Put each argument of a wrapped API signature on its own line.

mkdocstrings formats every reference signature with Black at the
``line_length`` in ``mkdocs.yml``. When a signature is too long for
one line but its arguments fit on one indented line, Black keeps them
together there. The source puts each argument on its own line, so
this MkDocs hook formats a wrapped signature with two or more
arguments again, with a magic trailing comma. mkdocstrings has no
option for this, so the hook wraps its private ``_format_signature``.
If that function moves, the build stops instead of rendering the
old layout.
"""

from __future__ import annotations

from typing import Any

_OPENING = "([{"
_CLOSING = ")]}"
_WRAPPED = "_kpnn2_one_argument_per_line"


def with_trailing_comma(signature: str) -> str | None:
    """Add a comma after the last argument of ``signature``.

    Return ``None`` when the argument list holds fewer than two
    arguments or already ends with a comma.
    """
    start = signature.find("(")
    if start < 0:
        return None
    depth = 0
    commas = 0
    for index in range(start, len(signature)):
        char = signature[index]
        if char in _OPENING:
            depth += 1
        elif char in _CLOSING:
            depth -= 1
            if depth == 0:
                arguments = signature[start + 1 : index].rstrip()
                if commas == 0 or arguments.endswith(","):
                    return None
                return (
                    signature[: start + 1] + arguments + "," + signature[index:]
                )
        elif char == "," and depth == 1:
            commas += 1
    return None


def on_config(config: Any) -> None:
    """MkDocs hook: wrap the mkdocstrings signature formatter."""
    from mkdocstrings_handlers.python._internal import rendering

    format_signature = rendering._format_signature
    if getattr(format_signature, _WRAPPED, False):
        return

    def one_argument_per_line(
        name: Any,
        signature: str,
        line_length: int,
    ) -> str:
        formatted = format_signature(
            name,
            signature,
            line_length,
        )
        exploded = with_trailing_comma(signature)
        if "\n" not in formatted or exploded is None:
            return formatted
        return format_signature(
            name,
            exploded,
            line_length,
        )

    setattr(one_argument_per_line, _WRAPPED, True)
    rendering._format_signature = one_argument_per_line
