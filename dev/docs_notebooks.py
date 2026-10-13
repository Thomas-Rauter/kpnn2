"""Execute and repair documentation notebooks.

Stream outputs need ``name`` (``stdout`` / ``stderr``). Editors often
drop it; mkdocs-jupyter then fails validation. This module repairs
that on ``mkdocs`` pre-build, and is the supported execute path:

    python dev/docs_notebooks.py
    python dev/docs_notebooks.py docs/skip-edges.ipynb
    python dev/docs_notebooks.py --fix-only
    python dev/docs_notebooks.py --literature

With notebook paths, only those notebooks are executed (or, with
``--fix-only``, repaired). Without paths, every notebook of the
chosen kind is.

After execute, cell ``execution_count`` values and Jupyter
``metadata.execution`` timestamps are cleared so a re-run with
unchanged outputs does not dirty the working tree. Outputs and
cell ids are kept. For the same reason the kernel runs with a
fixed ``SOURCE_DATE_EPOCH`` and Matplotlib ``svg.hashsalt``, so a
plot is written with the same SVG date and ids every time, and
memory addresses are dropped from plain-text object reprs such as
``<graphviz.graphs.Digraph at 0x...>``.

Tutorial notebooks under ``docs/*.ipynb`` are executed in CI.
Notebooks under ``docs/literature/`` are frozen reproductions:
repair them, but do not execute them unless ``--literature`` is
set. Docs CI never passes that flag.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

_LITERATURE_DIR = "literature"

# Matplotlib stamps each SVG with the render time unless
# SOURCE_DATE_EPOCH is set, and salts its SVG ids with a random
# value unless svg.hashsalt is set.
_SOURCE_DATE_EPOCH = "0"
_SVG_HASHSALT = "kpnn2-docs"

# The address in a default repr like <module.Class at 0x7f...>.
_OBJECT_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+>")


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


def _requested_notebooks(
    paths: list[Path],
    docs_dir: Path,
    *,
    literature: bool,
    fix_only: bool,
) -> list[Path]:
    """Resolve notebook paths named on the command line.

    Each path must be an ``.ipynb`` file under ``docs_dir``. To
    execute, a literature notebook needs ``literature=True``, and
    ``literature=True`` takes literature notebooks only. Repair
    (``fix_only``) takes either. A repeated path is kept once.
    """
    notebooks: list[Path] = []
    for raw in paths:
        path = raw.resolve()
        if path.suffix != ".ipynb" or not path.is_file():
            raise SystemExit(f"Not a notebook file: {raw}")
        if not path.is_relative_to(docs_dir):
            raise SystemExit(f"Not a docs notebook under {docs_dir}: {raw}")
        frozen = _is_literature(
            path,
            docs_dir,
        )
        if not fix_only and frozen and not literature:
            raise SystemExit(
                f"{raw} is a frozen literature notebook. Pass --literature "
                "to execute it."
            )
        if not fix_only and literature and not frozen:
            raise SystemExit(
                f"{raw} is not under docs/literature/. --literature "
                "executes literature notebooks only."
            )
        if path not in notebooks:
            notebooks.append(path)
    return notebooks


def _detect_indent(text: str) -> int:
    for line in text.splitlines()[1:8]:
        stripped = line.lstrip(" ")
        if stripped and line != stripped:
            return len(line) - len(stripped)
    return 1


def _write_notebook(
    path: Path,
    data: dict,
    indent: int,
) -> None:
    text = json.dumps(
        data,
        indent=indent,
        ensure_ascii=False,
    )
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text)


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
    _write_notebook(
        path,
        data,
        _detect_indent(raw),
    )
    return True


def clear_execution_metadata(path: Path) -> bool:
    """Clear execution counts and timestamps. Keep outputs and ids.

    Return True if the file changed.
    """
    raw = path.read_text()
    data = json.loads(raw)
    changed = False
    for cell in data.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        metadata = cell.get("metadata")
        if isinstance(metadata, dict) and "execution" in metadata:
            del metadata["execution"]
            changed = True
        outputs = cell.get("outputs")
        if not outputs:
            continue
        for output in outputs:
            if not isinstance(output, dict):
                continue
            if output.get("execution_count") is not None:
                output["execution_count"] = None
                changed = True
    if not changed:
        return False
    _write_notebook(
        path,
        data,
        _detect_indent(raw),
    )
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
    with tempfile.TemporaryDirectory() as rc_dir:
        rc_file = Path(rc_dir) / "matplotlibrc"
        rc_file.write_text(f"svg.hashsalt: {_SVG_HASHSALT}\n")
        client.execute(env=_kernel_env(rc_file))
    _changes, nb = normalize(nb)
    nbformat.validate(nb)
    nbformat.write(nb, path)


def _kernel_env(rc_file: Path) -> dict[str, str]:
    """Return this process's environment plus reproducible-plot settings.

    ``rc_file`` is a ``matplotlibrc`` that sets ``svg.hashsalt``. It
    replaces any personal Matplotlib config, so local runs match CI.
    """
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = _SOURCE_DATE_EPOCH
    env["MATPLOTLIBRC"] = str(rc_file)
    return env


def strip_object_addresses(path: Path) -> bool:
    """Drop memory addresses from plain-text object reprs.

    An object without its own text form, such as a Graphviz drawing,
    is stored as ``<module.Class at 0x...>`` next to its rich output,
    and the address differs on every run. Return True if the file
    changed.
    """
    raw = path.read_text()
    data = json.loads(raw)
    changed = False
    for cell in data.get("cells", []):
        for output in cell.get("outputs") or []:
            bundle = output.get("data")
            if not isinstance(bundle, dict) or "text/plain" not in bundle:
                continue
            text = bundle["text/plain"]
            lines = [text] if isinstance(text, str) else text
            stripped = [_OBJECT_ADDRESS.sub(">", line) for line in lines]
            if stripped == lines:
                continue
            bundle["text/plain"] = (
                stripped[0] if isinstance(text, str) else stripped
            )
            changed = True
    if not changed:
        return False
    _write_notebook(
        path,
        data,
        _detect_indent(raw),
    )
    return True


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
        help=(
            "Repair stream names and clear execution metadata; do not execute."
        ),
    )
    parser.add_argument(
        "--literature",
        action="store_true",
        help=(
            "Execute frozen notebooks under docs/literature/ "
            "instead of the tutorial notebooks."
        ),
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help=(
            "Notebooks to process, such as docs/skip-edges.ipynb. "
            "Omit to process every notebook of the chosen kind."
        ),
    )
    args = parser.parse_args(argv)
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    if args.notebooks:
        notebooks = _requested_notebooks(
            args.notebooks,
            docs_dir,
            literature=args.literature,
            fix_only=args.fix_only,
        )
    elif args.fix_only:
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
            repaired = repair_stream_names(path)
            cleared = clear_execution_metadata(path)
            if repaired or cleared:
                print(f"Repaired {path}")
            else:
                print(f"OK {path}")
        return
    _ensure_venv_kernel()
    for path in notebooks:
        print(f"Executing {path}")
        execute_notebook(path)
        repair_stream_names(path)
        clear_execution_metadata(path)
        strip_object_addresses(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
