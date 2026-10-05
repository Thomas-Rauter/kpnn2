"""Serve Home links and figures from the local tree in the docs build.

GitHub and PyPI show ``README.md`` but cannot resolve its relative
paths. So ``README.md`` links docs pages by absolute URL on the
website's ``latest`` version, and raster figures by raw GitHub URL on
``main``. This MkDocs hook maps both back to local paths when MkDocs
builds Home, which includes ``README.md``. Links then stay inside the
docs version the reader is browsing, ``mkdocs build --strict``
validates their pages and anchors, and a figure that is new on a
branch shows before it is pushed. ``README.md`` keeps the absolute
URLs. A site link that matches no docs page stops the build.
"""

from __future__ import annotations

import re
from typing import Any

SITE_LATEST = "https://thomas-rauter.github.io/kpnn2/latest/"
RAW_FIGURES = (
    "https://raw.githubusercontent.com/Thomas-Rauter/kpnn2/main/docs/figures/"
)
_LOCAL_FIGURES = "figures/"
_HOME = "index.md"
_SITE_LINK = re.compile(
    re.escape(SITE_LATEST) + r"([^\s()<>\"'#]*)(#[^\s()<>\"']*)?"
)


def on_page_markdown(
    markdown: str,
    *,
    page: Any,
    config: Any,
    files: Any,
) -> str:
    """MkDocs hook: map the README URLs on Home to local paths."""
    if page.file.src_uri != _HOME:
        return markdown
    markdown = markdown.replace(
        RAW_FIGURES,
        _LOCAL_FIGURES,
    )
    sources = {file.url: file.src_uri for file in files.documentation_pages()}

    def to_source(match: re.Match[str]) -> str:
        path, anchor = match.group(1), match.group(2) or ""
        source = sources.get(path or "./")
        if source is None:
            raise ValueError(
                f"README.md links {match.group(0)}, but no docs page is "
                f"served at {path!r}. Link a page under docs/ by its "
                f"URL on the site."
            )
        return source + anchor

    return _SITE_LINK.sub(
        to_source,
        markdown,
    )
