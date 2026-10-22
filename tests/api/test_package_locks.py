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
    "assert",
    "message",
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

# Messages are built only here, so every error and warning shares
# one format (dev/src_guidelines.txt, section 3).
_MESSAGE_BUILDERS = "src/kpnn2/_errors.py"

# Modules that still build messages by hand. Each step of the
# message-format migration converts some and removes them here.
_NOT_YET_CONVERTED = frozenset(
    {
        "src/kpnn2/_adjacency_spec.py",
        "src/kpnn2/_aggregation/_bind.py",
        "src/kpnn2/_aggregation/_dispatch.py",
        "src/kpnn2/_aggregation/_methods/_rauter_mangano_2026.py",
        "src/kpnn2/_aggregation/_registry.py",
        "src/kpnn2/_align.py",
        "src/kpnn2/_attributions.py",
        "src/kpnn2/_constraint.py",
        "src/kpnn2/_gather.py",
        "src/kpnn2/_generator.py",
        "src/kpnn2/_identity.py",
        "src/kpnn2/_layout.py",
        "src/kpnn2/_masked_linear.py",
        "src/kpnn2/_packed_index.py",
        "src/kpnn2/_packed_multihead_attention.py",
        "src/kpnn2/_parse.py",
        "src/kpnn2/_serialize.py",
        "src/kpnn2/_spec.py",
        "src/kpnn2/_validate.py",
    }
)

_FORBIDDEN = """\
import random
import warnings
import numpy as np
import torch
import torch.sparse
import captum
from kpnn2 import Kpnn2Error

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
    assert x is not None
    warnings.warn("odd")
    if x == 0:
        raise Kpnn2Error("bad")
    if x == 1:
        raise AssertionError("bug")
    return torch.sparse.sum(x)
"""

_ALLOWED = """\
import numpy
import pandas
import torch
import xarray
from kpnn2 import __version__
from kpnn2._errors import Kpnn2Error, caller_error

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

def check(x):
    if isinstance(x, Kpnn2Error):
        raise caller_error("x is wrong.")
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


def _message_call(dotted: str) -> str | None:
    name = dotted.rsplit(".", 1)[-1]
    if name in {"Kpnn2Error", "AssertionError"}:
        return name
    if dotted == "warnings.warn":
        return dotted
    return None


def _hit_path(hit: str) -> str:
    return hit.split(":", 1)[0]


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
        if isinstance(node, ast.Assert):
            add(
                "assert",
                node.lineno,
                "assert",
            )
        elif isinstance(node, ast.Import):
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
    message = None if dotted is None else _message_call(dotted)
    if message is not None:
        add(
            "message",
            node.lineno,
            message,
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


def test_src_does_not_use_assert():
    assert _src_findings()["assert"] == []


def test_src_builds_messages_only_in_the_errors_module():
    allowed = _NOT_YET_CONVERTED | {_MESSAGE_BUILDERS}
    hits = [
        hit
        for hit in _src_findings()["message"]
        if _hit_path(hit) not in allowed
    ]
    assert hits == []


def test_not_yet_converted_modules_still_build_messages():
    paths = {_hit_path(hit) for hit in _src_findings()["message"]}
    assert sorted(_NOT_YET_CONVERTED - paths) == []


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
    assert any("assert" in hit for hit in bad["assert"])
    assert any("warnings.warn" in hit for hit in bad["message"])
    assert any("Kpnn2Error" in hit for hit in bad["message"])
    assert any("AssertionError" in hit for hit in bad["message"])

    good = _scan_source(
        _ALLOWED,
        "<allowed>",
    )
    assert good == {lock: [] for lock in _LOCKS}
