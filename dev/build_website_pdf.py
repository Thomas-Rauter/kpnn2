"""Render the documentation website as one PDF.

The print-site plugin joins every page in the nav into one page,
and headless Chrome prints that page to ``out/kpnn2_docs.pdf``
(gitignored). The reference pages for single names are not in the
site nav, so the PDF lists them after ``reference/api.md``, in the
order that page links them, each titled by its public name.
``mkdocs.yml`` stays unchanged.
Notebooks show their stored outputs; run
``python dev/docs_notebooks.py`` first if those are stale. It
needs the ``docs`` extra and a local Chrome or Chromium.

    python dev/build_website_pdf.py
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mkdocs.commands.build import build
from mkdocs.config import load_config
from mkdocs.utils.yaml import yaml_load

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = _ROOT / "mkdocs.yml"
_DOCS = _ROOT / "docs"
_API_PAGE = "reference/api.md"
_TARGET = _ROOT / "out" / "kpnn2_docs.pdf"
_BROWSERS = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
)
# A link to a public name on the API page: [`Hop`][kpnn2.Hop].
_API_LINK = re.compile(r"\]\[kpnn2\.(\w+)\]")
# The mkdocstrings line that renders one public name on its page.
_RENDERS = re.compile(
    r"^::: kpnn2\.(\w+)$",
    re.MULTILINE,
)
_PRINT_SITE = {
    "print-site": {
        "add_cover_page": True,
        # The numbers restart in every notebook cell, so they mislead.
        "enumerate_headings": False,
        "print_page_title": "kpnn2 documentation",
    },
}
# Virtual milliseconds Chrome lets the page's scripts run before it
# prints, so that MathJax has typeset the notebook formulas.
_SCRIPT_TIME_MS = 60_000


def _reference_pages() -> list[dict[str, str]]:
    """List one page per name, in the order the API page links them."""
    pages = {}
    for path in sorted((_DOCS / "reference").glob("*.md")):
        match = _RENDERS.search(path.read_text())
        if match is not None:
            pages[match.group(1)] = path.relative_to(_DOCS).as_posix()
    linked = _API_LINK.findall((_DOCS / _API_PAGE).read_text())
    methods = sorted((_DOCS / "reference" / "aggregation").glob("*.md"))
    return [
        *({name: pages[name]} for name in dict.fromkeys(linked)),
        *({path.stem: path.relative_to(_DOCS).as_posix()} for path in methods),
    ]


def _with_reference(nav: list[Any]) -> list[Any]:
    """Return ``nav`` with every reference page after the API page."""
    result = []
    for item in nav:
        page = item
        if isinstance(item, dict):
            ((title, page),) = item.items()
            if isinstance(page, list):
                item = {title: _with_reference(page)}
        result.append(item)
        if page == _API_PAGE:
            result.extend(_reference_pages())
    return result


def _hide_plugin_order_warning(record: logging.LogRecord) -> bool:
    # The hooks of mkdocs.yml count as plugins after print-site, so
    # print-site warns that it is not last, and asks for an edit to
    # mkdocs.yml that the PDF does not need.
    return "'print-site' should be defined" not in record.getMessage()


def _browser() -> str:
    for name in _BROWSERS:
        path = shutil.which(name)
        if path is not None:
            return path
    raise SystemExit(f"No browser found; tried {', '.join(_BROWSERS)}.")


def write_pdf(target: Path = _TARGET) -> Path:
    """Build the one-page site and print it to ``target``."""
    raw = yaml_load(_CONFIG.read_text())
    logging.getLogger("mkdocs.plugins").addFilter(_hide_plugin_order_warning)
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with tempfile.TemporaryDirectory() as site_dir:
        config = load_config(
            config_file=str(_CONFIG),
            site_dir=site_dir,
            nav=_with_reference(raw["nav"]),
            plugins=[
                *raw["plugins"],
                _PRINT_SITE,
            ],
        )
        config.plugins.on_startup(
            command="build",
            dirty=False,
        )
        try:
            build(config)
        finally:
            config.plugins.on_shutdown()
        page = Path(site_dir) / "print_page" / "index.html"
        subprocess.run(
            [
                _browser(),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--virtual-time-budget={_SCRIPT_TIME_MS}",
                f"--print-to-pdf={target}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    return target


def main() -> None:
    print(f"Wrote {write_pdf()}")


if __name__ == "__main__":
    main()
