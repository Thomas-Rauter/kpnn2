"""Docs notebooks must be valid enough for mkdocs-jupyter."""

import json
from pathlib import Path

_DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_stream_outputs_have_name() -> None:
    notebooks = sorted(_DOCS.glob("*.ipynb"))
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
                missing.append(f"{path.name} cell {i} output {j}")
    assert not missing, missing
