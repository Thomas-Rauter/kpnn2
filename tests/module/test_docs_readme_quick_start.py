"""The README quick start, executed.

Every ``python`` block under "## Quick start" in ``README.md`` runs
in page order in one namespace, so the landing page cannot show code
that no longer works. A ``text`` block right after a code block is
that block's printed output and is compared with what the code
prints: words exactly, numbers within a small tolerance, so a torch
release that moves the last digit does not fail the suite. The
claims around the code are pinned too: the trained model relies on
``hidden_signal`` far more than on ``hidden_noise``, and Figure 1
draws the scores the page shows.

The last block imports Captum, which is in the ``dev`` extra.
"""

import ast
import contextlib
import io
import math
import random
import re
from pathlib import Path

import numpy
import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_README = _REPO / "README.md"
_FIG_GEN = _REPO / "docs" / "fig_gen" / "kpnn2_overview.py"

_HEADING = "## Quick start"
# A code block, and the text block that shows its output if one
# follows with nothing but whitespace in between.
_BLOCK = re.compile(
    r"```python\n(.*?)\n```(?:\s*```text\n(.*?)\n```)?",
    re.S,
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_SCORE_LINE = re.compile(r"^(\w+)\s+(-?\d+\.\d+)$", re.M)
_ABS_TOL = 0.1
# The prose says the model relies on hidden_signal and not on
# hidden_noise.
_MIN_RATIO = 5.0


def _section() -> str:
    text = _README.read_text()
    start = text.find(_HEADING)
    assert start != -1, (
        f"README.md has no {_HEADING!r} heading. If the section was "
        "renamed or removed, update _HEADING or drop this module."
    )
    end = text.find("\n## ", start + len(_HEADING))
    return text[start:] if end == -1 else text[start:end]


def _blocks() -> list[tuple[str, str]]:
    blocks = _BLOCK.findall(_section())
    assert blocks, "The quick start has no python blocks."
    return blocks


def _scores(output: str) -> dict[str, float]:
    return {name: float(value) for name, value in _SCORE_LINE.findall(output)}


def _shown_scores() -> dict[str, float]:
    shown = _blocks()[-1][1]
    assert shown, "The last quick-start block shows no output."
    return _scores(shown)


@pytest.fixture(scope="module")
def printed() -> list[str]:
    random.seed(42)
    numpy.random.seed(42)
    torch.manual_seed(42)
    namespace: dict[str, object] = {"__name__": "readme_quick_start"}
    outputs = []
    for code, _ in _blocks():
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, "README.md", "exec"), namespace)  # noqa: S102
        outputs.append(buffer.getvalue())
    return outputs


def test_every_block_with_output_shows_what_it_prints(
    printed: list[str],
) -> None:
    for index, ((_, shown), actual) in enumerate(
        zip(
            _blocks(),
            printed,
        )
    ):
        if not shown:
            continue
        shown_words = shown.split()
        actual_words = actual.split()
        assert len(shown_words) == len(actual_words), (
            f"block {index}: page shows {shown!r}, code prints {actual!r}"
        )
        for want, got in zip(
            shown_words,
            actual_words,
        ):
            if _NUMBER.fullmatch(want) and _NUMBER.fullmatch(got):
                assert math.isclose(
                    float(want),
                    float(got),
                    abs_tol=_ABS_TOL,
                ), f"block {index}: page shows {want}, code prints {got}"
            else:
                assert want == got, f"block {index}: {want!r} != {got!r}"


def test_trained_model_relies_on_hidden_signal(printed: list[str]) -> None:
    scores = _scores(printed[-1])
    assert set(scores) == {"hidden_noise", "hidden_signal"}
    assert scores["hidden_signal"] > _MIN_RATIO * scores["hidden_noise"]


def test_figure_draws_the_scores_the_page_shows() -> None:
    tree = ast.parse(_FIG_GEN.read_text())
    drawn = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SCORES"
        ):
            drawn = ast.literal_eval(node.value)
    assert drawn is not None, f"No SCORES assignment in {_FIG_GEN}"
    assert drawn == _shown_scores()
