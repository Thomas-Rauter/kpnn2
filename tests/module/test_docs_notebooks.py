"""Docs notebooks must be valid enough for mkdocs-jupyter."""

import importlib.util
import json
from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_REPO = _DOCS.parent
_SCRIPT = _REPO / "scripts" / "docs_notebooks.py"

_spec = importlib.util.spec_from_file_location(
    "docs_notebooks_script",
    _SCRIPT,
)
assert _spec is not None
assert _spec.loader is not None
_docs_notebooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_docs_notebooks)
_iter_notebooks = _docs_notebooks._iter_notebooks


def _all_notebooks() -> list[Path]:
    return _iter_notebooks(
        _DOCS,
        include_literature=True,
    )


def test_stream_outputs_have_name() -> None:
    notebooks = _all_notebooks()
    assert notebooks
    missing = []
    for path in notebooks:
        data = json.loads(path.read_text())
        for i, cell in enumerate(data.get("cells", [])):
            outputs = cell.get("outputs") or []
            for j, output in enumerate(outputs):
                if output.get("output_type") != "stream":
                    continue
                if "name" in output:
                    continue
                rel = path.relative_to(_DOCS)
                missing.append(f"{rel} cell {i} output {j}")
    assert not missing, missing


def test_default_execute_skips_literature() -> None:
    tutorials = _iter_notebooks(
        _DOCS,
        include_literature=False,
    )
    literature = [
        path
        for path in _all_notebooks()
        if "literature" in path.relative_to(_DOCS).parts
    ]
    assert tutorials
    for path in tutorials:
        assert "literature" not in path.relative_to(_DOCS).parts
    assert literature
    for path in literature:
        assert path not in tutorials


def test_literature_notebooks_keep_outputs() -> None:
    literature = sorted((_DOCS / "literature").glob("*.ipynb"))
    assert literature, "expected docs/literature/*.ipynb"
    empty = []
    for path in literature:
        data = json.loads(path.read_text())
        code_cells = [
            cell
            for cell in data.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        assert code_cells, path.name
        if not any(cell.get("outputs") for cell in code_cells):
            empty.append(path.name)
    assert not empty, empty


@pytest.mark.parametrize("path", _all_notebooks())
def test_notebooks_are_nbformat_v4(path: Path) -> None:
    data = json.loads(path.read_text())
    assert data.get("nbformat") == 4, path.name
