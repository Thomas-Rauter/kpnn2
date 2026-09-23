"""Serve Home figures from the local tree in the docs build.

``README.md`` links its raster figures by absolute raw GitHub URL on
``main``, because PyPI cannot resolve relative image paths. A figure
that is new or changed on a branch does not exist on ``main`` until
it is pushed, so ``mkdocs serve`` would show a broken image. This
MkDocs hook points those URLs at the local ``docs/figures/`` copy for
the docs build only; ``README.md`` keeps the absolute URL for PyPI.
"""

from __future__ import annotations

from typing import Any

RAW_FIGURES = (
    "https://raw.githubusercontent.com/Thomas-Rauter/kpnn2/main/docs/figures/"
)
_LOCAL_FIGURES = "figures/"
_HOME = "index.md"


def on_page_markdown(
    markdown: str,
    *,
    page: Any,
    config: Any,
    files: Any,
) -> str:
    """MkDocs hook: use local figures on Home, which includes README."""
    if page.file.src_uri != _HOME:
        return markdown
    return markdown.replace(
        RAW_FIGURES,
        _LOCAL_FIGURES,
    )
