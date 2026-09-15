"""The Concepts page is the vocabulary contract; these pin its links.

``mkdocs build --strict`` validates anchors in ``.md`` pages and in
``README.md`` (via ``index.md``), but never in notebooks: mkdocs-jupyter
renders those outside the markdown pipeline. See **Docs terminology** in
``CONTEXT.md``.
"""

import json
import re
from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_REPO = _DOCS.parent
_CONCEPTS = _DOCS / "concepts.md"

# Defined on the landing page because it is that central, so nothing
# needs to link to it. See **Docs terminology** in CONTEXT.md.
_ALLOWED_ORPHANS = {"edgelist"}


def _anchors() -> set[str]:
    headings = re.findall(r"^## (.+)$", _CONCEPTS.read_text(), re.M)
    return {
        re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
        for h in headings
    }


def _pages() -> list[Path]:
    pages = [_REPO / "README.md"]
    pages += sorted(_DOCS.glob("*.md"))
    pages += sorted(_DOCS.glob("*.ipynb"))
    pages += sorted(_DOCS.glob("literature/*.ipynb"))
    return [p for p in pages if p != _CONCEPTS]


def _prose(path: Path) -> str:
    if path.suffix != ".ipynb":
        return path.read_text()
    data = json.loads(path.read_text())
    return "\n".join(
        "".join(cell["source"])
        for cell in data["cells"]
        if cell["cell_type"] == "markdown"
    )


@pytest.mark.parametrize("path", [p for p in _pages() if p.suffix == ".ipynb"])
def test_notebook_concepts_links_resolve(path: Path) -> None:
    anchors = _anchors()
    assert anchors, "docs/concepts.md has no ## headings"
    used = re.findall(r"concepts/#([^)\s]+)", _prose(path))
    missing = sorted(set(used) - anchors)
    assert not missing, f"{path.name}: {missing}"


def test_every_concepts_entry_is_reachable() -> None:
    text = "\n".join(_prose(p) for p in _pages())
    used = set(re.findall(r"concepts(?:/|\.md)#([a-z0-9-]+)", text))
    orphans = sorted(_anchors() - used - _ALLOWED_ORPHANS)
    assert not orphans, f"Concepts entries nothing links to: {orphans}"
