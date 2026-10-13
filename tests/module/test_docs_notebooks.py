"""Docs notebooks must be valid enough for mkdocs-jupyter."""

import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_REPO = _DOCS.parent
_SCRIPT = _REPO / "dev" / "docs_notebooks.py"

_spec = importlib.util.spec_from_file_location(
    "docs_notebooks_script",
    _SCRIPT,
)
assert _spec is not None
assert _spec.loader is not None
_docs_notebooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_docs_notebooks)
_iter_notebooks = _docs_notebooks._iter_notebooks
_requested_notebooks = _docs_notebooks._requested_notebooks
clear_execution_metadata = _docs_notebooks.clear_execution_metadata
strip_object_addresses = _docs_notebooks.strip_object_addresses

_TUTORIAL = _DOCS / "skip-edges.ipynb"
_LITERATURE = _DOCS / "literature" / "fortelny-bock-2020.ipynb"


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


def test_requested_tutorial_is_the_only_one_executed() -> None:
    assert _requested_notebooks(
        [_TUTORIAL, _TUTORIAL],
        _DOCS,
        literature=False,
        fix_only=False,
    ) == [_TUTORIAL]


def test_requested_literature_needs_the_flag_to_execute() -> None:
    with pytest.raises(
        SystemExit,
        match="frozen literature notebook. Pass --literature",
    ):
        _requested_notebooks(
            [_LITERATURE],
            _DOCS,
            literature=False,
            fix_only=False,
        )
    for literature, fix_only in [(True, False), (False, True)]:
        assert _requested_notebooks(
            [_LITERATURE],
            _DOCS,
            literature=literature,
            fix_only=fix_only,
        ) == [_LITERATURE]


def test_literature_flag_rejects_a_requested_tutorial() -> None:
    with pytest.raises(
        SystemExit,
        match="is not under docs/literature/",
    ):
        _requested_notebooks(
            [_TUTORIAL],
            _DOCS,
            literature=True,
            fix_only=False,
        )


def test_requested_path_must_be_a_docs_notebook(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.ipynb"
    outside.write_text("{}\n")
    cases = [
        (_DOCS / "missing.ipynb", "Not a notebook file"),
        (_DOCS / "concepts.md", "Not a notebook file"),
        (outside, "Not a docs notebook under"),
    ]
    for path, message in cases:
        with pytest.raises(
            SystemExit,
            match=message,
        ):
            _requested_notebooks(
                [path],
                _DOCS,
                literature=False,
                fix_only=True,
            )


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


def test_clear_execution_metadata(
    tmp_path: Path,
) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 3,
                "id": "abc123",
                "metadata": {
                    "execution": {
                        "iopub.execute_input": "t0",
                        "iopub.status.idle": "t1",
                    }
                },
                "source": ["print(1)\n"],
                "outputs": [
                    {
                        "output_type": "execute_result",
                        "execution_count": 3,
                        "data": {"text/plain": "1"},
                        "metadata": {},
                    },
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["1\n"],
                    },
                ],
            }
        ],
    }
    path = tmp_path / "toy.ipynb"
    path.write_text(json.dumps(nb, indent=1) + "\n")
    assert clear_execution_metadata(path) is True
    assert clear_execution_metadata(path) is False
    data = json.loads(path.read_text())
    cell = data["cells"][0]
    assert cell["execution_count"] is None
    assert "execution" not in cell["metadata"]
    assert cell["id"] == "abc123"
    assert cell["outputs"][0]["execution_count"] is None
    assert cell["outputs"][0]["data"] == {"text/plain": "1"}
    assert cell["outputs"][1]["text"] == ["1\n"]


def test_strip_object_addresses(
    tmp_path: Path,
) -> None:
    svg = ["<svg>drawing</svg>\n"]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "abc123",
                "metadata": {},
                "source": ["drawing\n"],
                "outputs": [
                    {
                        "output_type": "execute_result",
                        "execution_count": None,
                        "data": {
                            "image/svg+xml": svg,
                            "text/plain": [
                                "<graphviz.graphs.Digraph at 0x7ca47c563cb0>"
                            ],
                        },
                        "metadata": {},
                    },
                    {
                        "output_type": "display_data",
                        "data": {"text/plain": "<Thing at 0xABC123>"},
                        "metadata": {},
                    },
                ],
            }
        ],
    }
    path = tmp_path / "toy.ipynb"
    path.write_text(json.dumps(nb, indent=1) + "\n")
    assert strip_object_addresses(path) is True
    assert strip_object_addresses(path) is False
    outputs = json.loads(path.read_text())["cells"][0]["outputs"]
    assert outputs[0]["data"]["text/plain"] == ["<graphviz.graphs.Digraph>"]
    assert outputs[0]["data"]["image/svg+xml"] == svg
    assert outputs[1]["data"]["text/plain"] == "<Thing>"


def test_kernel_env_fixes_plot_date_and_ids(
    tmp_path: Path,
) -> None:
    rc_file = tmp_path / "matplotlibrc"
    env = _docs_notebooks._kernel_env(rc_file)
    assert env["SOURCE_DATE_EPOCH"] == _docs_notebooks._SOURCE_DATE_EPOCH
    assert env["MATPLOTLIBRC"] == str(rc_file)


def test_tutorial_outputs_hold_no_run_specific_values() -> None:
    """A re-run of unchanged code must leave a tutorial notebook unchanged.

    Literature notebooks are frozen as published and are not checked.
    """
    epoch_date = datetime.fromtimestamp(
        int(_docs_notebooks._SOURCE_DATE_EPOCH),
        timezone.utc,
    ).isoformat()
    tutorials = _iter_notebooks(
        _DOCS,
        include_literature=False,
    )
    assert tutorials
    found = []
    for path in tutorials:
        raw = path.read_text()
        for date in re.findall(r"<dc:date>(.*?)</dc:date>", raw):
            if date != epoch_date:
                found.append(f"{path.name}: SVG date {date}")
        for cell in json.loads(raw).get("cells", []):
            for output in cell.get("outputs") or []:
                text = output.get("data", {}).get("text/plain", "")
                text = "".join(text) if isinstance(text, list) else text
                if _docs_notebooks._OBJECT_ADDRESS.search(text):
                    found.append(f"{path.name}: address in {text!r}")
    assert not found, found
