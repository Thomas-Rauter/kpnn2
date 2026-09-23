"""The homepage "Why not custom PyTorch?" comparison, executed.

Both fenced blocks are lifted out of ``README.md`` and run, so the
claims the section makes -- that the hand-written module and the
``kpnn2`` one build the same network, and that both reject the same
malformed edgelists -- are checked rather than asserted in prose.
So is the silent failure below them: after the prior update the
prose names, an old checkpoint loads into the hand-written module
without complaint and brings the old masks with it, while the
``kpnn2`` module refuses it. Editing either block re-runs it here.
Renaming or moving the section, dropping a column label, or changing
the number of fenced blocks fails these tests rather than silently
skipping them.

The snippets read ``pathway_prior.csv``, a file that does not exist in
the repo: it stands for the reader's own prior. The fixture writes it
from the edge list in ``docs/fig_gen/custom_pytorch_pathway.py``, so
the graph under test is the one Figure 3 draws. That generator imports
graphviz, which is in the ``docs`` extra and not in ``dev``, so the
edge list is read statically instead of imported.
"""

import ast
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from kpnn2 import Kpnn2Error, parse_layered

_REPO = Path(__file__).resolve().parents[2]
_README = _REPO / "README.md"
_FIG_GEN = _REPO / "docs" / "fig_gen" / "custom_pytorch_pathway.py"

_HEADING = "## Why not custom PyTorch?"
_CSV_NAME = "pathway_prior.csv"
_COLUMNS = ["source", "target"]

# One "**Label**" per column, each followed by exactly one block.
_BLOCK = re.compile(
    r"\*\*(Custom PyTorch|kpnn2)\*\*\s*\n+```python\n(.*?)\n```",
    re.S,
)

_NUMBER_WORDS = {
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}

_GOOD = [("a", "h"), ("b", "h"), ("h", "c")]

# "`gene_n3` now regulates `tf_stat` instead of `tf_nfkb`", read as
# (source, new target, old target).
_REVISION = re.compile(r"`(\w+)` now regulates `(\w+)` instead of `(\w+)`")

# Each edge list both columns must refuse. The snippet raises
# ValueError and parse_layered raises Kpnn2Error; only that both
# refuse is pinned, not the wording.
_REJECTED = {
    "missing column": pd.DataFrame({"source": ["a"], "tgt": ["b"]}),
    "missing name": pd.DataFrame(_GOOD + [(None, "z")], columns=_COLUMNS),
    "empty name": pd.DataFrame(_GOOD + [("", "z")], columns=_COLUMNS),
    "self loop": pd.DataFrame(_GOOD + [("z", "z")], columns=_COLUMNS),
    "duplicate edge": pd.DataFrame(_GOOD + [("a", "h")], columns=_COLUMNS),
    "cycle": pd.DataFrame(
        [("in", "a"), ("a", "b"), ("b", "a"), ("b", "out")],
        columns=_COLUMNS,
    ),
}


def _section() -> str:
    text = _README.read_text()
    start = text.find(_HEADING)
    assert start != -1, (
        f"README.md has no {_HEADING!r} heading. If the section was "
        "renamed or removed, update _HEADING or drop this module."
    )
    end = text.find("\n## ", start + len(_HEADING))
    return text[start:] if end == -1 else text[start:end]


def _columns() -> dict[str, str]:
    found = {label: code for label, code in _BLOCK.findall(_section())}
    assert set(found) == {"Custom PyTorch", "kpnn2"}, (
        "Expected one '**Custom PyTorch**' and one '**kpnn2**' block "
        f"in the comparison; found {sorted(found)}."
    )
    return found


def _figure_pairs() -> list[tuple[str, str]]:
    """Evaluate only the PAIRS statements, so graphviz is not needed."""
    tree = ast.parse(_FIG_GEN.read_text())
    body: list[ast.stmt] = []
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == "PAIRS" for t in targets):
            body.append(node)
    assert body, f"No PAIRS assignment found in {_FIG_GEN}"
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102 - static slice of a repo file, no input
        compile(ast.Module(body=body, type_ignores=[]), str(_FIG_GEN), "exec"),
        namespace,
    )
    return namespace["PAIRS"]


def _run(code: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "readme_snippet"}
    exec(compile(code, "README.md", "exec"), namespace)  # noqa: S102
    return namespace


def _flat_section() -> str:
    """The section with hard wraps flattened, for matching prose."""
    return re.sub(r"\s+", " ", _section())


def _revised_pairs() -> list[tuple[str, str]]:
    """Apply the prior update that the silent-failure prose names."""
    match = _REVISION.search(_flat_section())
    assert match, (
        "The section no longer names the revised interaction as "
        "'`source` now regulates `new` instead of `old`'; update "
        "this test or the prose."
    )
    source, new, old = match.groups()
    pairs = _figure_pairs()
    assert (source, old) in pairs, f"{source} -> {old} is not in Figure 3"
    assert (source, new) not in pairs, f"{source} -> {new} already exists"
    return [pair for pair in pairs if pair != (source, old)] + [(source, new)]


def _build(
    code: str,
    pairs: list[tuple[str, str]],
    path: Path,
) -> dict[str, Any]:
    pd.DataFrame(pairs, columns=_COLUMNS).to_csv(path, index=False)
    torch.manual_seed(42)
    return _run(code)


@pytest.fixture
def prior_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / _CSV_NAME
    pd.DataFrame(_figure_pairs(), columns=_COLUMNS).to_csv(path, index=False)
    monkeypatch.chdir(tmp_path)
    return path


def test_comparison_has_one_block_per_column() -> None:
    columns = _columns()
    assert "parse_layered" in columns["kpnn2"]
    assert "import kpnn2" not in columns["Custom PyTorch"]


def test_both_columns_run_and_build_the_same_network(
    prior_csv: Path,
) -> None:
    left = _run(_columns()["Custom PyTorch"])
    right = _run(_columns()["kpnn2"])

    spec = right["spec"]
    assert [tuple(layer) for layer in left["layers"]] == [
        tuple(layer) for layer in spec.layer_nodes
    ]

    hand_masks = [mask for _, mask in left["masks"]]
    assert len(hand_masks) == len(spec.hops)
    for index, (hand, hop) in enumerate(zip(hand_masks, spec.hops)):
        assert torch.equal(hand, hop.to_mask().float()), f"hop {index}"
        assert list(left["masks"][index][0]) == list(hop.source_layers)

    x = torch.randn(4, len(left["layers"][0]))
    assert left["model"](x).shape == right["model"](x).shape


@pytest.mark.parametrize("name", sorted(_REJECTED))
def test_both_columns_reject_the_same_edgelists(
    name: str, prior_csv: Path
) -> None:
    _REJECTED[name].to_csv(prior_csv, index=False)
    with pytest.raises(ValueError):
        _run(_columns()["Custom PyTorch"])
    with pytest.raises(Kpnn2Error):
        parse_layered(pd.read_csv(prior_csv))


def test_prose_names_the_number_of_rejected_edgelists() -> None:
    # The prose is hard-wrapped, so match against flattened text.
    flat = _flat_section()
    match = re.search(r"reject the same (\w+) malformed edgelists", flat)
    assert match, (
        "The section no longer states how many malformed edge lists "
        "both columns reject; update this test or the prose."
    )
    claimed = _NUMBER_WORDS.get(match.group(1))
    assert claimed == len(_REJECTED), (
        f"README claims {match.group(1)} rejected edge lists, "
        f"but {len(_REJECTED)} are pinned here."
    )


def test_revised_prior_moves_no_node(prior_csv: Path) -> None:
    code = _columns()["kpnn2"]
    old = _build(code, _figure_pairs(), prior_csv)["spec"]
    new = _build(code, _revised_pairs(), prior_csv)["spec"]
    assert new.layer_nodes == old.layer_nodes
    assert new.fingerprint != old.fingerprint


def test_hand_written_model_loads_the_old_masks_silently(
    prior_csv: Path,
) -> None:
    code = _columns()["Custom PyTorch"]
    old = _build(code, _figure_pairs(), prior_csv)
    new = _build(code, _revised_pairs(), prior_csv)
    # The Parameters share storage with the script's masks, so copy
    # the new wiring before the load overwrites it.
    new_masks = [mask.clone() for _, mask in new["masks"]]

    result = new["model"].load_state_dict(old["model"].state_dict())

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert repr(result) in _section()
    loaded = list(new["model"].masks)
    old_masks = [mask for _, mask in old["masks"]]
    assert all(torch.equal(a, b) for a, b in zip(loaded, old_masks))
    assert not all(torch.equal(a, b) for a, b in zip(loaded, new_masks))


@pytest.mark.parametrize("identity", [True, False])
def test_kpnn2_model_refuses_the_old_checkpoint(
    identity: bool,
    prior_csv: Path,
) -> None:
    code = _columns()["kpnn2"]
    if not identity:
        # The index digest alone must still catch the rewiring.
        assert "identity=spec.fingerprint," in code
        code = code.replace("identity=spec.fingerprint,", "")
    old = _build(code, _figure_pairs(), prior_csv)
    new = _build(code, _revised_pairs(), prior_csv)

    with pytest.raises(Kpnn2Error) as info:
        new["model"].load_state_dict(old["model"].state_dict())

    if identity:
        assert f"Kpnn2Error: {info.value}" in _flat_section()
