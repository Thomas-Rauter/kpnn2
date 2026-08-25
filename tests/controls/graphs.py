"""
Edgelist builders for computational-control tests.

``two_tower_feedforward`` builds two structurally matched subgraphs
from one construction applied twice, so degree sequences and depths
cannot drift. ``dead_edge_graph`` and ``skip_edge_graph`` isolate
pinned-zero hops and skip residuals. ``wide_layer_live_middle_graph``
and ``wide_layer_live_gap_graph`` put live hidden units at non-zero
or gapped tensor indices in one layer. Edgelists are source/target
only; intended zero pins live on ``dead_edges``, not parser columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

Edge = tuple[str, str]

PREDICTION_OUTPUT = "prediction"
DECOY_OUTPUT = "decoy_readout"


@dataclass(frozen=True)
class TwoTowerGraph:
    """
    Two matched towers, with the names each tower owns.

    The builder does not decide which tower is important.
    """

    edgelist: pd.DataFrame
    tower_a_features: tuple[str, ...]
    tower_b_features: tuple[str, ...]
    tower_a_nodes: tuple[str, ...]
    tower_b_nodes: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class ImportanceGraph:
    """
    A graph plus declared structural labels and intended zero pins.

    ``dead_edges`` are still rows in ``edgelist``. Tests pin those
    weights to 0 after building the module.
    """

    edgelist: pd.DataFrame
    important_features: tuple[str, ...]
    unimportant_features: tuple[str, ...]
    important_nodes: tuple[str, ...]
    unimportant_nodes: tuple[str, ...]
    outputs: tuple[str, ...]
    dead_edges: frozenset[Edge]


def two_tower_feedforward(
    *,
    n_features_per_tower: int = 3,
    depth: int = 3,
    hidden_width: int | None = None,
    shared_output: bool = True,
) -> TwoTowerGraph:
    """
    Build two matched feedforward towers.

    Each tower has ``n_features_per_tower`` inputs and ``depth``
    hidden levels. Hidden width defaults to the feature count.

    With ``shared_output`` both towers end at ``"prediction"``.
    Without it, tower B ends at ``"decoy_readout"`` and has no live
    path to ``"prediction"``.
    """
    _validate_tower_size(n_features_per_tower)
    if depth < 1:
        raise ValueError(f"'depth' must be at least 1, got {depth}.")
    width = n_features_per_tower if hidden_width is None else hidden_width
    if width < 1:
        raise ValueError(f"'hidden_width' must be at least 1, got {width}.")
    tower_b_output = PREDICTION_OUTPUT if shared_output else DECOY_OUTPUT
    dense = hidden_width is not None
    a_features, a_nodes, a_edges = _feedforward_tower(
        prefix="a",
        n_features=n_features_per_tower,
        hidden_width=width,
        depth=depth,
        output=PREDICTION_OUTPUT,
        dense=dense,
    )
    b_features, b_nodes, b_edges = _feedforward_tower(
        prefix="b",
        n_features=n_features_per_tower,
        hidden_width=width,
        depth=depth,
        output=tower_b_output,
        dense=dense,
    )
    outputs = (
        (PREDICTION_OUTPUT,)
        if shared_output
        else (PREDICTION_OUTPUT, DECOY_OUTPUT)
    )
    return TwoTowerGraph(
        edgelist=_make_edgelist([*a_edges, *b_edges]),
        tower_a_features=a_features,
        tower_b_features=b_features,
        tower_a_nodes=a_nodes,
        tower_b_nodes=b_nodes,
        outputs=outputs,
    )


def dead_edge_graph() -> ImportanceGraph:
    """
    Two identical branches to prediction; one first hop is dead.

    Intended pin (applied in tests, not in the edgelist):
      (dead_in, dead_h1) = 0, all other edges = 1.

    ``dead_h1`` and ``dead_h2`` stay in the graph. They cannot carry
    signal once that hop is pinned at zero.
    """
    edges: list[Edge] = [
        ("signal_in", "live_h1"),
        ("live_h1", "live_h2"),
        ("live_h2", PREDICTION_OUTPUT),
        ("dead_in", "dead_h1"),
        ("dead_h1", "dead_h2"),
        ("dead_h2", PREDICTION_OUTPUT),
    ]
    return ImportanceGraph(
        edgelist=_make_edgelist(edges),
        important_features=("signal_in",),
        unimportant_features=("dead_in",),
        important_nodes=("live_h1", "live_h2"),
        unimportant_nodes=("dead_h1", "dead_h2"),
        outputs=(PREDICTION_OUTPUT,),
        dead_edges=frozenset({("dead_in", "dead_h1")}),
    )


def skip_edge_graph() -> ImportanceGraph:
    """
    Live chain, live skip residual, and matched dead controls.

    ``skip_in -> prediction`` has depth gap > 1 (a Skip, not a mask
    entry). ``dead_skip_in -> prediction`` is the same span with
    intended pin 0. ``dead_in -> dead_h`` is a dead adjacent hop.

    Intended pins:
      (dead_skip_in, prediction) = 0
      (dead_in, dead_h) = 0
      all other edges = 1.

    This tests skip residuals, not pseudo-nodes.
    """
    edges: list[Edge] = [
        ("chain_in", "chain_h1"),
        ("chain_h1", "chain_h2"),
        ("chain_h2", PREDICTION_OUTPUT),
        ("skip_in", PREDICTION_OUTPUT),
        ("dead_skip_in", PREDICTION_OUTPUT),
        ("dead_in", "dead_h"),
        ("dead_h", PREDICTION_OUTPUT),
    ]
    return ImportanceGraph(
        edgelist=_make_edgelist(edges),
        important_features=("chain_in", "skip_in"),
        unimportant_features=("dead_in", "dead_skip_in"),
        important_nodes=("chain_h1", "chain_h2"),
        unimportant_nodes=("dead_h",),
        outputs=(PREDICTION_OUTPUT,),
        dead_edges=frozenset(
            {
                ("dead_skip_in", PREDICTION_OUTPUT),
                ("dead_in", "dead_h"),
            }
        ),
    )


def wide_layer_live_middle_graph() -> ImportanceGraph:
    """
    Three same-depth hiddens; the live unit is tensor index 1.

    Alphabetical hidden order is
    ``("alpha_dead", "mid_live", "zed_dead")``. Mapping names to
    the wrong index (live at 0) would fail ranking tests.

    ``live_in -> mid_live -> prediction`` is the live path.
    ``dead_in`` feeds both dead hiddens, which end at
    ``decoy_readout`` (not attributed). No intended zero pins.
    """
    edges: list[Edge] = [
        ("live_in", "mid_live"),
        ("mid_live", PREDICTION_OUTPUT),
        ("dead_in", "alpha_dead"),
        ("dead_in", "zed_dead"),
        ("alpha_dead", DECOY_OUTPUT),
        ("zed_dead", DECOY_OUTPUT),
    ]
    return ImportanceGraph(
        edgelist=_make_edgelist(edges),
        important_features=("live_in",),
        unimportant_features=("dead_in",),
        important_nodes=("mid_live",),
        unimportant_nodes=("alpha_dead", "zed_dead"),
        outputs=(PREDICTION_OUTPUT, DECOY_OUTPUT),
        dead_edges=frozenset(),
    )


def wide_layer_live_gap_graph() -> ImportanceGraph:
    """
    Same-depth hiddens in live, dead, live tensor order.

    Alphabetical hidden order is
    ``("left_live", "mid_dead", "right_live")``. An off-by-one
    slice of the live units would smear scores onto ``mid_dead``.

    Both live hiddens feed ``prediction``. ``mid_dead`` feeds
    ``decoy_readout`` only. No intended zero pins.
    """
    edges: list[Edge] = [
        ("live_in", "left_live"),
        ("live_in", "right_live"),
        ("left_live", PREDICTION_OUTPUT),
        ("right_live", PREDICTION_OUTPUT),
        ("dead_in", "mid_dead"),
        ("mid_dead", DECOY_OUTPUT),
    ]
    return ImportanceGraph(
        edgelist=_make_edgelist(edges),
        important_features=("live_in",),
        unimportant_features=("dead_in",),
        important_nodes=("left_live", "right_live"),
        unimportant_nodes=("mid_dead",),
        outputs=(PREDICTION_OUTPUT, DECOY_OUTPUT),
        dead_edges=frozenset(),
    )


def multi_output_graph() -> ImportanceGraph:
    """
    Two outputs that share one hidden node and each own another.

    Attribute only ``output_1``: ``only_2_*`` cannot influence it.
    No intended zero pins.
    """
    edges: list[Edge] = [
        ("shared_in", "shared_h"),
        ("shared_h", "output_1"),
        ("shared_h", "output_2"),
        ("only_1_in", "only_1_h"),
        ("only_1_h", "output_1"),
        ("only_2_in", "only_2_h"),
        ("only_2_h", "output_2"),
    ]
    return ImportanceGraph(
        edgelist=_make_edgelist(edges),
        important_features=("shared_in", "only_1_in"),
        unimportant_features=("only_2_in",),
        important_nodes=("shared_h", "only_1_h"),
        unimportant_nodes=("only_2_h",),
        outputs=("output_1", "output_2"),
        dead_edges=frozenset(),
    )


def _feedforward_tower(
    *,
    prefix: str,
    n_features: int,
    hidden_width: int,
    depth: int,
    output: str,
    dense: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...], list[Edge]]:
    """
    Build one feedforward tower: features, hidden names, and edges.
    """
    features = tuple(f"{prefix}_in_{index}" for index in range(n_features))
    levels = [
        tuple(f"{prefix}_h{level}_{index}" for index in range(hidden_width))
        for level in range(1, depth + 1)
    ]
    edges: list[Edge] = []
    previous: tuple[str, ...] = features
    for level_nodes in levels:
        edges.extend(
            _inter_level_edges(
                previous,
                level_nodes,
                dense=dense,
            )
        )
        previous = level_nodes
    for source in previous:
        edges.append((source, output))
    nodes = tuple(name for level in levels for name in level)
    return features, nodes, edges


def _inter_level_edges(
    sources: tuple[str, ...],
    targets: tuple[str, ...],
    *,
    dense: bool,
) -> list[Edge]:
    """
    Connect one layer to the next, densely or with two-neighbor wrap.
    """
    if dense or len(sources) != len(targets):
        return [(source, target) for source in sources for target in targets]
    edges: list[Edge] = []
    n_targets = len(targets)
    for index, source in enumerate(sources):
        edges.append((source, targets[index]))
        edges.append((source, targets[(index + 1) % n_targets]))
    return edges


def _validate_tower_size(n_features_per_tower: int) -> None:
    """
    Reject a one-feature tower that would collapse ring wiring.
    """
    if n_features_per_tower < 2:
        raise ValueError(
            "'n_features_per_tower' must be at least 2, got "
            f"{n_features_per_tower}."
        )


def _make_edgelist(edges: Sequence[Edge]) -> pd.DataFrame:
    """
    Create a deduplicated source/target DataFrame.
    """
    edgelist = pd.DataFrame(
        list(edges),
        columns=["source", "target"],
    )
    return edgelist.drop_duplicates().reset_index(drop=True)
