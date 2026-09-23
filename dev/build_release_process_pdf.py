"""Render the release procedure as a PDF.

``dev/version_release_process.md`` is the source. This script
writes ``dev/version_release_process.pdf`` next to it, with
copyable command blocks and highlighted notes. GitHub-style alerts
(``> [!WARNING]``) become coloured boxes. It needs Python-Markdown
and pymdownx (the ``docs`` extra) and a local Chrome or Chromium.

    python dev/build_release_process_pdf.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

_DEV = Path(__file__).resolve().parent
_SOURCE = _DEV / "version_release_process.md"
_TARGET = _DEV / "version_release_process.pdf"
_BROWSERS = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
)
# One alert: "> [!KIND]" plus the "> " lines that follow it. Python-
# Markdown would merge two alerts separated by a blank line into one
# blockquote, so each alert becomes its own div before conversion.
_ALERT = re.compile(
    r"^> \[!(NOTE|IMPORTANT|WARNING)\]\n((?:>.*\n?)*)",
    re.MULTILINE,
)
_QUOTE_MARK = re.compile(
    r"^> ?",
    re.MULTILINE,
)
_CSS = """
@page {
  size: A4;
  margin: 16mm 15mm;
}
body {
  font-family: "Liberation Sans", "DejaVu Sans", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #1f2328;
}
h1 {
  font-size: 22pt;
  margin: 0 0 8pt;
  padding-bottom: 6pt;
  border-bottom: 2px solid #ee4c2c;
}
h2 {
  font-size: 14pt;
  margin: 18pt 0 6pt;
  padding-bottom: 3pt;
  border-bottom: 1px solid #d0d7de;
  break-after: avoid;
}
p,
li {
  margin: 4pt 0;
}
a {
  color: #0969da;
  overflow-wrap: anywhere;
}
code {
  font-family: "Liberation Mono", "DejaVu Sans Mono", monospace;
  font-size: 9pt;
  background: #eff1f3;
  border-radius: 3px;
  padding: 0 3px;
  overflow-wrap: anywhere;
}
pre {
  background: #f6f8fa;
  border: 1px solid #d0d7de;
  border-left: 4px solid #57606a;
  border-radius: 4px;
  padding: 7pt 9pt;
  margin: 6pt 0 8pt;
  white-space: pre;
  break-inside: avoid;
}
pre code {
  background: none;
  padding: 0;
  font-size: 8.5pt;
  overflow-wrap: normal;
}
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 9.5pt;
  margin: 6pt 0;
}
th,
td {
  border: 1px solid #d0d7de;
  padding: 4pt 6pt;
  text-align: left;
  vertical-align: top;
}
th {
  background: #f6f8fa;
}
td code {
  white-space: nowrap;
  overflow-wrap: normal;
}
li.task-list-item {
  list-style-type: none;
}
li.task-list-item input {
  margin: 0 5pt 0 -16pt;
  vertical-align: middle;
}
.alert {
  border-left: 4px solid;
  border-radius: 4px;
  padding: 5pt 10pt;
  margin: 10pt 0;
  break-inside: avoid;
}
.alert p {
  margin: 3pt 0;
}
.alert-title {
  font-weight: bold;
  text-transform: uppercase;
  font-size: 9pt;
  letter-spacing: 0.04em;
}
.alert-note {
  border-color: #0969da;
  background: #ddf4ff;
}
.alert-note .alert-title {
  color: #0969da;
}
.alert-important {
  border-color: #8250df;
  background: #fbefff;
}
.alert-important .alert-title {
  color: #8250df;
}
.alert-warning {
  border-color: #bf8700;
  background: #fff8c5;
}
.alert-warning .alert-title {
  color: #9a6700;
}
"""


def _alert(match: re.Match[str]) -> str:
    kind = match.group(1).lower()
    text = _QUOTE_MARK.sub(
        "",
        match.group(2),
    )
    return (
        f'<div class="alert alert-{kind}" markdown="1">\n'
        f'<p class="alert-title">{kind}</p>\n\n'
        f"{text}\n</div>\n"
    )


def render_html(text: str) -> str:
    """Convert the procedure's Markdown to a standalone HTML page."""
    body = markdown.markdown(
        _ALERT.sub(
            _alert,
            text,
        ),
        extensions=[
            "tables",
            "sane_lists",
            "md_in_html",
            "pymdownx.superfences",
            "pymdownx.tasklist",
        ],
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>kpnn2 release procedure</title>"
        f"<style>{_CSS}</style></head>"
        f"<body>{body}</body></html>\n"
    )


def _browser() -> str:
    for name in _BROWSERS:
        path = shutil.which(name)
        if path is not None:
            return path
    raise SystemExit(f"No browser found; tried {', '.join(_BROWSERS)}.")


def write_pdf(
    source: Path = _SOURCE,
    target: Path = _TARGET,
) -> Path:
    """Render ``source`` to ``target`` with headless Chrome."""
    html = render_html(source.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "release_process.html"
        page.write_text(html)
        subprocess.run(
            [
                _browser(),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
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
