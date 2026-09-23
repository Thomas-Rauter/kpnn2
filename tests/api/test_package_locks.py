"""Syntax locks for src/kpnn2. Strings and comments are ignored."""

import ast
import sys
from pathlib import Path

from tests.api.test_public_api import _COMPILER_LEFTOVERS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "kpnn2"

_LOCKS = (
    "sparse",
    "compiler",
    "print",
    "seed",
    "device",
    "dependency",
)

# Pinned here. Reading pyproject.toml would let a new
# dependency allow its own import.
_CORE_DEPENDENCIES = (
    "numpy",
    "pandas",
    "torch",
    "xarray",
)

_GLOBAL_SEEDS = frozenset(
    {
        "numpy.random.seed",
        "random.seed",
        "torch.manual_seed",
    }
)

_COMPILER_NAMES = frozenset(_COMPILER_LEFTOVERS)

_FORBIDDEN = """\
import random
import numpy as np
import torch
import torch.sparse
import captum

def compile_graph():
    return None

class CompileArtifact:
    pass

def demo(x):
    print(x)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    x.cuda()
    x.to("cuda")
    torch.device("cuda")
    torch.set_default_device("cuda")
    return torch.sparse.sum(x)
"""

_ALLOWED = """\
import numpy
import pandas
import torch
import xarray
from kpnn2 import __version__

def copy_device(x, weight, scores, source, target, generator):
    \"\"\"torch.sparse is not imported.\"\"\"
    # torch.sparse
    y = x.to(dtype=torch.float32)
    z = weight.to(dtype=torch.float32)
    generator.manual_seed(1)
    return (
        y,
        z,
        x.device,
        weight.device,
        scores.device,
        source.device,
        target.device,
        __version__,
    )
"""


def _aliases(tree: ast.AST) -> dict[str, str]:
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name
                found[bound] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                if module:
                    found[bound] = f"{module}.{alias.name}"
                else:
                    found[bound] = alias.name
    return found


def _dotted(
    node: ast.expr,
    aliases: dict[str, str],
) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _dotted(
            node.value,
            aliases,
        )
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _is_cuda_text(value: object) -> bool:
    return isinstance(value, str) and (
        value == "cuda" or value.startswith("cuda:")
    )


def _cuda_text(call: ast.Call) -> str | None:
    nodes = list(call.args)
    for keyword in call.keywords:
        nodes.append(keyword.value)
    for node in nodes:
        if isinstance(node, ast.Constant) and _is_cuda_text(node.value):
            return str(node.value)
    return None


def _dependency_allowed(module: str) -> bool:
    top = module.split(".", 1)[0]
    if top in _CORE_DEPENDENCIES or top == "kpnn2":
        return True
    return top in sys.stdlib_module_names


def _blank() -> dict[str, list[tuple[int, str]]]:
    return {lock: [] for lock in _LOCKS}


def _scan_tree(
    tree: ast.AST,
    rel: str,
) -> dict[str, list[str]]:
    aliases = _aliases(tree)
    found = _blank()

    def add(
        lock: str,
        lineno: int,
        detail: str,
    ) -> None:
        found[lock].append((lineno, detail))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _dependency_allowed(alias.name):
                    add(
                        "dependency",
                        node.lineno,
                        f"import {alias.name}",
                    )
                if alias.name == "torch.sparse" or alias.name.startswith(
                    "torch.sparse."
                ):
                    add(
                        "sparse",
                        node.lineno,
                        "torch.sparse",
                    )
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if module and not _dependency_allowed(module):
                add(
                    "dependency",
                    node.lineno,
                    f"import {module}",
                )
            if module == "torch.sparse" or module.startswith("torch.sparse."):
                add(
                    "sparse",
                    node.lineno,
                    "torch.sparse",
                )
            for alias in node.names:
                if alias.name == "*":
                    continue
                full = f"{module}.{alias.name}" if module else alias.name
                if module == "torch" and (
                    alias.name == "sparse" or alias.name.startswith("sparse.")
                ):
                    add(
                        "sparse",
                        node.lineno,
                        "torch.sparse",
                    )
                if full in _GLOBAL_SEEDS:
                    add(
                        "seed",
                        node.lineno,
                        full,
                    )
        elif isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            if node.name in _COMPILER_NAMES:
                add(
                    "compiler",
                    node.lineno,
                    node.name,
                )
        elif isinstance(node, ast.Call):
            _scan_call(
                node,
                aliases,
                add,
            )
        elif isinstance(node, ast.Attribute):
            dotted = _dotted(
                node,
                aliases,
            )
            if dotted == "torch.sparse":
                add(
                    "sparse",
                    node.lineno,
                    "torch.sparse",
                )

    reported: dict[str, list[str]] = {}
    for lock in _LOCKS:
        lines: list[str] = []
        seen: set[tuple[int, str]] = set()
        for lineno, detail in sorted(found[lock]):
            item = (lineno, detail)
            if item in seen:
                continue
            seen.add(item)
            lines.append(f"{rel}:{lineno}: {detail}")
        reported[lock] = lines
    return reported


def _scan_call(
    node: ast.Call,
    aliases: dict[str, str],
    add,
) -> None:
    dotted = _dotted(
        node.func,
        aliases,
    )
    if dotted in {"print", "builtins.print"}:
        add(
            "print",
            node.lineno,
            "print",
        )
    if dotted in _GLOBAL_SEEDS:
        add(
            "seed",
            node.lineno,
            dotted or "",
        )
    cuda = _cuda_text(node)
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "cuda":
        add(
            "device",
            node.lineno,
            ".cuda()",
        )
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "to"
        and cuda is not None
    ):
        add(
            "device",
            node.lineno,
            f'.to("{cuda}")',
        )
    if dotted == "torch.device" and cuda is not None:
        add(
            "device",
            node.lineno,
            f'torch.device("{cuda}")',
        )
    if dotted == "torch.set_default_device" or (
        isinstance(func, ast.Attribute) and func.attr == "set_default_device"
    ):
        add(
            "device",
            node.lineno,
            "set_default_device",
        )


def _scan_source(
    source: str,
    rel: str,
) -> dict[str, list[str]]:
    tree = ast.parse(
        source,
        filename=rel,
    )
    return _scan_tree(
        tree,
        rel,
    )


def _src_findings() -> dict[str, list[str]]:
    merged = {lock: [] for lock in _LOCKS}
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        found = _scan_source(
            path.read_text(encoding="utf-8"),
            rel,
        )
        for lock in _LOCKS:
            merged[lock].extend(found[lock])
    return merged


def test_src_does_not_use_torch_sparse():
    assert _src_findings()["sparse"] == []


def test_src_does_not_define_compiler_symbols():
    assert _src_findings()["compiler"] == []


def test_src_does_not_call_print():
    assert _src_findings()["print"] == []


def test_src_does_not_seed_global_rng():
    assert _src_findings()["seed"] == []


def test_src_does_not_select_a_device():
    assert _src_findings()["device"] == []


def test_src_imports_only_pinned_dependencies():
    assert _src_findings()["dependency"] == []


def test_scanner_flags_forbidden_syntax_and_allows_copies():
    bad = _scan_source(
        _FORBIDDEN,
        "<forbidden>",
    )
    assert bad["sparse"]
    assert any("compile_graph" in hit for hit in bad["compiler"])
    assert any("CompileArtifact" in hit for hit in bad["compiler"])
    assert bad["print"]
    assert any("torch.manual_seed" in hit for hit in bad["seed"])
    assert any("numpy.random.seed" in hit for hit in bad["seed"])
    assert any("random.seed" in hit for hit in bad["seed"])
    assert any(".cuda()" in hit for hit in bad["device"])
    assert any('.to("cuda")' in hit for hit in bad["device"])
    assert any('torch.device("cuda")' in hit for hit in bad["device"])
    assert any("set_default_device" in hit for hit in bad["device"])
    assert any("captum" in hit for hit in bad["dependency"])
    assert not any("torch" in hit for hit in bad["dependency"])

    good = _scan_source(
        _ALLOWED,
        "<allowed>",
    )
    assert good == {lock: [] for lock in _LOCKS}
