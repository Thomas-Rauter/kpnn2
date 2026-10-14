"""Wrapped API signatures show one argument per line.

``dev/docs_signatures.py`` adds a magic trailing comma to a wrapped
signature with two or more arguments, so Black puts each argument on
its own line, as the source does.
"""

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "dev" / "docs_signatures.py"

_spec = importlib.util.spec_from_file_location(
    "docs_signatures_script",
    _SCRIPT,
)
assert _spec is not None and _spec.loader is not None
_hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hook)


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        (
            "(names: object, spec: LayeredSpec | AdjacencySpec) -> np.ndarray",
            "(names: object, spec: LayeredSpec | AdjacencySpec,) -> np.ndarray",
        ),
        (
            "(saved: Mapping[int, Tensor], hop: Hop) -> Tensor",
            "(saved: Mapping[int, Tensor], hop: Hop,) -> Tensor",
        ),
        (
            "(names: object, *, layer: int) -> tuple[int, ...]",
            "(names: object, *, layer: int,) -> tuple[int, ...]",
        ),
        (
            "(x: object, **kwargs: object) -> None",
            "(x: object, **kwargs: object,) -> None",
        ),
    ],
)
def test_adds_comma_after_last_argument(
    signature: str,
    expected: str,
) -> None:
    assert _hook.with_trailing_comma(signature) == expected


@pytest.mark.parametrize(
    "signature",
    [
        "() -> None",
        "(names: object) -> tuple[int, ...]",
        "(coords: Mapping[str, Sequence] | None = None) -> None",
        "(a: int, b: int,) -> None",
        "-> None",
    ],
)
def test_leaves_short_or_exploded_lists_alone(
    signature: str,
) -> None:
    assert _hook.with_trailing_comma(signature) is None
