"""
Tier 2: trained importance on matched towers.

Important and unimportant groups are structurally matched and both
wired to ``prediction``. After a per-scenario learnability gate
(held-out ROC-AUC >= ``min_val_roc_auc``, 0.95 for every current
id), autograd scores must rank the data-generating tower above
the decoy tower.

``linear_feedforward`` is a linear no-bias control. It scores
features and hidden layers with gradient×input, AUC = 1.0,
max_ratio = 0.2, and per-example consistency >= 0.9.

``relu_feedforward`` uses ReLU and bias on the same labels.
Measured ReLU-specific scoring (still strict; gate unchanged):

- Features only. Hidden gradient×input is not counted: inactive
  important ReLU units have median score 0 (ratio inf), and
  positive bias keeps decoy units active so tower-sum hidden
  ratios stay above 0.2.
- Per-example consistency >= 0.6 (not 0.9). ReLU zeros some
  example-wise input grads; median feature ranking still requires
  AUC = 1.0 and max_ratio = 0.2.

``relu_product_feedforward`` uses ReLU and bias on the product of
the two tower-A features. Feature count stays 2. Hidden width is
8 so a ReLU net can represent the product and the 0.95 gate
holds. Scoring is features only. Measured product cuts (still
strict; median AUC stays 1.0): max_ratio = 0.45 and consistency
>= 0.5.

All three ids share ``SEEDS`` (42-71). A seed counts only if
the gate and the importance cuts both pass. The bar is a
per-scenario rate floor ~2.5 SE below a 30-seed measurement at
80 epochs, not a 4/5 majority on a lucky window. Learnability
is part of that combined count; a single gated-out seed does
not fail the test.

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
    SEEDS,
    TRAINED_SCENARIO_IDS,
    TrainedRun,
    autograd_feature_table,
    autograd_hidden_tables,
    train_trained_scenario,
    trained_scenario,
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
    scenario = trained_scenario(scenario_id)
    outcomes = [_run_seed(scenario_id, seed) for seed in SEEDS]
    n_passing = sum(
        outcome.passed_gate and outcome.passed_importance
        for outcome in outcomes
    )
    n_gate = sum(outcome.passed_gate for outcome in outcomes)
    formatted = _format_outcomes(outcomes)
    worst = min(
        outcomes,
        key=lambda item: (
            item.passed_gate and item.passed_importance,
            item.min_auc,
        ),
    )
    assert n_passing >= scenario.min_seeds_passing, (
        f"Scenario '{scenario_id}': combined learnability and "
        f"importance passed on {n_passing}/{len(SEEDS)} seeds "
        f"(gate {n_gate}/{len(SEEDS)}), need at least "
        f"{scenario.min_seeds_passing}.\n{formatted}\n"
        f"Worst seed {worst.seed} score tables:\n{worst.score_tables}"
    )


def _run_seed(
    scenario_id: str,
    seed: int,
) -> _SeedOutcome:
    """Train one seed and measure feature and hidden separations."""
    scenario = trained_scenario(scenario_id)
    trained = train_trained_scenario(
        scenario_id,
        seed=seed,
    )
    if trained.val_roc_auc < scenario.min_val_roc_auc:
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
    if not scenario.include_hidden:
        separations = [item for item in separations if item.label == "features"]
    passed = all(
        separation_passes(
            item,
            min_per_example_consistency=scenario.min_per_example_consistency,
            max_ratio=scenario.max_ratio,
        )
        for item in separations
    )
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
    important_features = trained.important_features
    feature_table = autograd_feature_table(
        trained.model,
        trained.spec,
        trained.x_eval_df,
    )
    feature_scores = median_abs_scores(
        feature_table,
        [*important_features, *graph.tower_b_features],
        label="features",
    )
    lines = [
        score_report(
            scenario_id=scenario_id,
            label="features",
            scores=feature_scores,
            important=important_features,
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
