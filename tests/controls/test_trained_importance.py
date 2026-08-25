"""
Tier 2: trained importance on matched towers.

Important and unimportant groups are structurally matched and both
wired to ``prediction``. After a learnability gate (held-out
ROC-AUC >= MIN_VAL_ROC_AUC, currently 0.95), autograd scores must
rank the data-generating tower above the decoy tower. Hidden
scores are gradient×input at each node's layer.

Marked integration/slow. Unique scenario-id checks are fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.controls.diagnostics import score_report
from tests.controls.metrics import (
    median_abs_scores,
    separation_passes,
)
from tests.controls.training import (
    MIN_SEEDS_PASSING,
    MIN_VAL_ROC_AUC,
    SEEDS,
    TRAINED_SCENARIO_IDS,
    TrainedRun,
    autograd_feature_table,
    autograd_hidden_tables,
    train_matched_linear_towers,
    trained_separations,
)


@dataclass(frozen=True)
class _SeedOutcome:
    """One training seed's gate, importance, and diagnostics."""

    seed: int
    val_roc_auc: float
    passed_gate: bool
    passed_importance: bool
    min_auc: float
    reports: str
    score_tables: str


def test_trained_scenario_ids_are_unique() -> None:
    ids = list(TRAINED_SCENARIO_IDS)
    assert len(ids) == len(set(ids)), (
        "TRAINED_SCENARIO_IDS contains duplicate ids: "
        f"{sorted(id_ for id_ in ids if ids.count(id_) > 1)}."
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "scenario_id",
    TRAINED_SCENARIO_IDS,
    ids=list(TRAINED_SCENARIO_IDS),
)
def test_trained_importance_recovers_the_data_generating_group(
    scenario_id: str,
) -> None:
    """Attribution must follow the labels, not the matched topology."""
    outcomes = [_run_seed(scenario_id, seed) for seed in SEEDS]
    n_passing = sum(
        outcome.passed_gate and outcome.passed_importance
        for outcome in outcomes
    )
    gate_failures = [
        outcome.seed for outcome in outcomes if not outcome.passed_gate
    ]
    formatted = _format_outcomes(outcomes)
    assert not gate_failures, (
        f"Scenario '{scenario_id}': learnability PRECONDITION failed "
        f"on seeds {gate_failures} (need held-out ROC-AUC >= "
        f"{MIN_VAL_ROC_AUC}). Importance was not counted for those "
        f"seeds.\n{formatted}"
    )
    worst = min(
        outcomes,
        key=lambda item: (item.passed_importance, item.min_auc),
    )
    assert n_passing >= MIN_SEEDS_PASSING, (
        f"Scenario '{scenario_id}': importance criterion passed on "
        f"{n_passing}/{len(SEEDS)} seeds, need at least "
        f"{MIN_SEEDS_PASSING}.\n{formatted}\n"
        f"Worst seed {worst.seed} score tables:\n{worst.score_tables}"
    )


def _run_seed(
    scenario_id: str,
    seed: int,
) -> _SeedOutcome:
    """Train one seed and measure feature and hidden separations."""
    trained = train_matched_linear_towers(seed=seed)
    if trained.val_roc_auc < MIN_VAL_ROC_AUC:
        return _SeedOutcome(
            seed=seed,
            val_roc_auc=trained.val_roc_auc,
            passed_gate=False,
            passed_importance=False,
            min_auc=0.0,
            reports=(
                f"seed={seed} val_roc_auc={trained.val_roc_auc:.3f} "
                "LEARNABILITY GATE FAILED"
            ),
            score_tables="(not computed; learnability gate failed)",
        )
    separations = trained_separations(trained)
    passed = all(separation_passes(item) for item in separations)
    min_auc = min(item.auc for item in separations)
    reports = " | ".join(
        (
            f"{item.label} auc={item.auc:.3f} "
            f"ratio={item.ratio:.3f} "
            f"cons={item.per_example_consistency:.3f}"
        )
        for item in separations
    )
    return _SeedOutcome(
        seed=seed,
        val_roc_auc=trained.val_roc_auc,
        passed_gate=True,
        passed_importance=passed,
        min_auc=min_auc,
        reports=(
            f"seed={seed} val_roc_auc={trained.val_roc_auc:.3f} "
            f"importance={'PASS' if passed else 'FAIL'} {reports}"
        ),
        score_tables=_score_tables(
            scenario_id,
            trained,
        ),
    )


def _score_tables(
    scenario_id: str,
    trained: TrainedRun,
) -> str:
    """Named median-|score| tables for CI diagnosis."""
    graph = trained.graph
    feature_table = autograd_feature_table(
        trained.model,
        trained.spec,
        trained.x_eval_df,
    )
    feature_scores = median_abs_scores(
        feature_table,
        [*graph.tower_a_features, *graph.tower_b_features],
        label="features",
    )
    lines = [
        score_report(
            scenario_id=scenario_id,
            label="features",
            scores=feature_scores,
            important=graph.tower_a_features,
            unimportant=graph.tower_b_features,
        )
    ]
    hidden_tables = autograd_hidden_tables(
        trained.model,
        trained.spec,
        trained.x_eval_df,
    )
    for layer, table in hidden_tables.items():
        columns = set(table.columns)
        important = [name for name in graph.tower_a_nodes if name in columns]
        unimportant = [name for name in graph.tower_b_nodes if name in columns]
        if not important or not unimportant:
            continue
        scores = median_abs_scores(
            table,
            [*important, *unimportant],
            label=f"layer_{layer}",
        )
        lines.append(
            score_report(
                scenario_id=scenario_id,
                label=f"layer_{layer}",
                scores=scores,
                important=important,
                unimportant=unimportant,
            )
        )
    return "\n".join(lines)


def _format_outcomes(outcomes: list[_SeedOutcome]) -> str:
    """Join every seed's one-line reports."""
    return "\n".join(outcome.reports for outcome in outcomes)
