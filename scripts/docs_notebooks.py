"""Execute and repair documentation notebooks.

Stream outputs need ``name`` (``stdout`` / ``stderr``). Editors often
drop it; mkdocs-jupyter then fails validation. This module repairs
that on ``mkdocs`` pre-build, and is the supported execute path:

    python scripts/docs_notebooks.py
    python scripts/docs_notebooks.py --fix-only
    python scripts/docs_notebooks.py --literature

Tutorial notebooks under ``docs/*.ipynb`` are executed in CI.
Notebooks under ``docs/literature/`` are frozen reproductions:
repair them, but do not execute them unless ``--literature`` is
set. Docs CI never passes that flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_LITERATURE_DIR = "literature"


def _is_literature(
    path: Path,
    docs_dir: Path,
) -> bool:
    try:
        relative = path.relative_to(docs_dir)
    except ValueError:
        return False
    return _LITERATURE_DIR in relative.parts


def _iter_notebooks(
    docs_dir: Path,
    *,
    include_literature: bool,
) -> list[Path]:
    notebooks = []
    for path in sorted(docs_dir.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        if not include_literature and _is_literature(
            path,
            docs_dir,
        ):
            continue
        notebooks.append(path)
    return notebooks


def _detect_indent(text: str) -> int:
    for line in text.splitlines()[1:8]:
        stripped = line.lstrip(" ")
        if stripped and line != stripped:
            return len(line) - len(stripped)
    return 1


def repair_stream_names(path: Path) -> bool:
    """Add missing stream ``name`` fields. Return True if the file changed."""
    raw = path.read_text()
    data = json.loads(raw)
    changed = False
    for cell in data.get("cells", []):
        outputs = cell.get("outputs")
        if not outputs:
            continue
        for index, output in enumerate(outputs):
            if not isinstance(output, dict):
                continue
            if output.get("output_type") != "stream":
                continue
            if "name" in output:
                continue
            repaired = {"name": "stdout"}
            repaired.update(output)
            outputs[index] = repaired
            changed = True
    if not changed:
        return False
    indent = _detect_indent(raw)
    text = json.dumps(
        data,
        indent=indent,
        ensure_ascii=False,
    )
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text)
    return True


def on_pre_build(config: dict) -> None:
    """MkDocs hook: repair notebooks before jupyter conversion."""
    docs_dir = Path(str(config["docs_dir"]))
    notebooks = _iter_notebooks(
        docs_dir,
        include_literature=True,
    )
    for path in notebooks:
        if repair_stream_names(path):
            print(f"Repaired stream output names in {path.name}")


def _ensure_venv_kernel() -> None:
    from ipykernel.kernelspec import install

    install(
        user=False,
        kernel_name="python3",
        prefix=sys.prefix,
    )


def execute_notebook(path: Path) -> None:
    import nbformat
    from nbclient import NotebookClient
    from nbformat.validator import normalize

    nb = nbformat.read(
        path,
        as_version=4,
    )
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={
            "metadata": {
                "path": str(path.parent),
            },
        },
    )
    client.execute()
    _changes, nb = normalize(nb)
    nbformat.validate(nb)
    nbformat.write(nb, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute docs notebooks with this interpreter "
            "and repair invalid stream outputs."
        ),
    )
    parser.add_argument(
        "--fix-only",
        action="store_true",
        help="Repair stream names; do not execute.",
    )
    parser.add_argument(
        "--literature",
        action="store_true",
        help=(
            "Execute frozen notebooks under docs/literature/ "
            "instead of the tutorial notebooks."
        ),
    )
    args = parser.parse_args(argv)
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    if args.fix_only:
        notebooks = _iter_notebooks(
            docs_dir,
            include_literature=True,
        )
    elif args.literature:
        notebooks = [
            path
            for path in _iter_notebooks(
                docs_dir,
                include_literature=True,
            )
            if _is_literature(path, docs_dir)
        ]
    else:
        notebooks = _iter_notebooks(
            docs_dir,
            include_literature=False,
        )
    if not notebooks:
        raise SystemExit(f"No notebooks in {docs_dir}")
    if args.fix_only:
        for path in notebooks:
            if repair_stream_names(path):
                print(f"Repaired {path}")
            else:
                print(f"OK {path}")
        return
    _ensure_venv_kernel()
    for path in notebooks:
        print(f"Executing {path}")
        execute_notebook(path)
        repair_stream_names(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
