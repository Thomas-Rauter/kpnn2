"""
Rewired-prior control: the named graph must contain the true paths.

One matched-tower scenario. Labels are linear in tower A.
``G_true`` wires both towers to ``prediction``. ``G_broken`` keeps
the same feature and hidden names, but tower A ends at
``decoy_readout`` and has no live path to ``prediction``. Tower B
may still reach the task output.

Measured on seed 42: G_true val ROC-AUC = 1.000; G_broken = 0.461.
Marked integration/slow.
"""

from __future__ import annotations

import pytest

from tests.controls.graphs import (
    DECOY_OUTPUT,
    PREDICTION_OUTPUT,
    TwoTowerGraph,
)
from tests.controls.ground_truth import (
    assert_all_structurally_live,
    solve_structural_ground_truth,
)
from tests.controls.training import (
    MIN_VAL_ROC_AUC,
    broken_matched_towers,
    linear_tower_simulator,
    matched_linear_towers,
    train_matched_linear_towers,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_SEED = 42
# Same ceiling as shuffled-label Tier 3 (val ROC-AUC < 0.6).
_MAX_CHANCE_ROC_AUC = 0.6


def test_rewired_prior_cannot_learn_without_true_paths() -> None:
    """
    Training on A-labels needs a live A path to ``prediction``.
    """
    graph_true = matched_linear_towers()
    graph_broken = broken_matched_towers()
    _assert_matched_names(
        graph_true,
        graph_broken,
    )
    _assert_true_prior_is_live(graph_true)
    _assert_broken_prior_cuts_tower_a(graph_broken)

    simulate = linear_tower_simulator(graph_true)
    run_true = train_matched_linear_towers(
        seed=_SEED,
        simulate=simulate,
        graph=graph_true,
    )
    run_broken = train_matched_linear_towers(
        seed=_SEED,
        simulate=simulate,
        graph=graph_broken,
    )
    reports = (
        f"G_true val_roc_auc={run_true.val_roc_auc:.3f} "
        f"G_broken val_roc_auc={run_broken.val_roc_auc:.3f}"
    )
    assert run_true.val_roc_auc >= MIN_VAL_ROC_AUC, (
        "learnability PRECONDITION failed: G_true did not fit "
        "tower-A labels (val ROC-AUC="
        f"{run_true.val_roc_auc:.3f}, need >= {MIN_VAL_ROC_AUC}). "
        "The rewired-prior check is not meaningful if the matched "
        f"graph cannot learn.\n{reports}"
    )
    assert run_broken.val_roc_auc < _MAX_CHANCE_ROC_AUC, (
        "rewired prior still learned tower-A labels, so a model "
        "can fit without a live path from the causal features to "
        "prediction (val ROC-AUC="
        f"{run_broken.val_roc_auc:.3f}, need < "
        f"{_MAX_CHANCE_ROC_AUC}).\n{reports}"
    )


def _assert_matched_names(
    graph_true: TwoTowerGraph,
    graph_broken: TwoTowerGraph,
) -> None:
    """
    G_broken must reuse G_true tower names; only the last hop changes.
    """
    assert graph_true.tower_a_features == graph_broken.tower_a_features
    assert graph_true.tower_b_features == graph_broken.tower_b_features
    assert graph_true.tower_a_nodes == graph_broken.tower_a_nodes
    assert graph_true.tower_b_nodes == graph_broken.tower_b_nodes
    assert graph_true.outputs == (PREDICTION_OUTPUT,)
    assert graph_broken.outputs == (
        PREDICTION_OUTPUT,
        DECOY_OUTPUT,
    )


def _assert_true_prior_is_live(graph: TwoTowerGraph) -> None:
    """
    Both towers must reach ``prediction`` on G_true.
    """
    names = [
        *graph.tower_a_features,
        *graph.tower_a_nodes,
        *graph.tower_b_features,
        *graph.tower_b_nodes,
    ]
    assert_all_structurally_live(
        edgelist=graph.edgelist,
        dead_edges=(),
        attributed_outputs=(PREDICTION_OUTPUT,),
        names=names,
    )


def _assert_broken_prior_cuts_tower_a(graph: TwoTowerGraph) -> None:
    """
    Tower A must be dead to ``prediction``; tower B may stay live.
    """
    ground_truth = solve_structural_ground_truth(
        edgelist=graph.edgelist,
        dead_edges=(),
        attributed_outputs=(PREDICTION_OUTPUT,),
    )
    live_a_features = sorted(
        name
        for name in graph.tower_a_features
        if name in ground_truth.important_features
    )
    live_a_nodes = sorted(
        name
        for name in graph.tower_a_nodes
        if name in ground_truth.important_nodes
    )
    assert not live_a_features and not live_a_nodes, (
        "G_broken still has a live path from tower A to "
        f"{PREDICTION_OUTPUT}: features={live_a_features}, "
        f"nodes={live_a_nodes}."
    )
    assert_all_structurally_live(
        edgelist=graph.edgelist,
        dead_edges=(),
        attributed_outputs=(PREDICTION_OUTPUT,),
        names=[
            *graph.tower_b_features,
            *graph.tower_b_nodes,
        ],
    )
