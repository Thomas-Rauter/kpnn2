"""
Tier 3: negative controls for the trained importance criterion.

These do not add graphs. They break what Tier 2 is supposed to
detect. Marked integration/slow.
"""

from __future__ import annotations

import pytest

from tests.controls.metrics import (
    DEFAULT_MAX_RATIO,
    clears_tier2_margin,
    separation_passes,
)
from tests.controls.simulate import shuffled_labels
from tests.controls.training import (
    MIN_VAL_ROC_AUC,
    linear_tower_simulator,
    matched_linear_towers,
    train_matched_linear_towers,
    trained_separations,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_SHUFFLED_SEED = 42
_MAX_CHANCE_ROC_AUC = 0.6
_NULL_AUC = 0.5
# Small groups (2 vs 2 features, 2 vs 2 hidden nodes) live on a
# coarse AUC grid, so chance rankings can sit well off 0.5.
_NULL_AUC_TOLERANCE = 0.30


def _shuffled_simulator(graph):
    """Permute labels after the linear simulator draws data."""
    original = linear_tower_simulator(graph)

    def simulate(
        rng,
        n_samples: int,
    ):
        features, labels = original(
            rng,
            n_samples,
        )
        return features, shuffled_labels(
            labels,
            rng,
        )

    return simulate


def _format_separations(separations) -> str:
    """One-line diagnostics for assertion messages."""
    return " | ".join(
        (
            f"{item.label} auc={item.auc:.3f} "
            f"ratio={item.ratio:.3f} "
            f"null={item.null_threshold:.3g} "
            f"floor={item.important_floor:.3g}"
        )
        for item in separations
    )


def test_shuffled_labels_produce_no_separation() -> None:
    """
    Training on permuted labels must not separate tower A from B.

    Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M.,
    and Kim, B. "Sanity Checks for Saliency Maps." NeurIPS 2018.
    """
    graph = matched_linear_towers()
    trained = train_matched_linear_towers(
        seed=_SHUFFLED_SEED,
        simulate=_shuffled_simulator(graph),
    )
    separations = trained_separations(trained)
    reports = (
        f"val_roc_auc={trained.val_roc_auc:.3f} "
        f"{_format_separations(separations)}"
    )
    assert trained.val_roc_auc < _MAX_CHANCE_ROC_AUC, (
        "learnability PRECONDITION failed: the model learned shuffled "
        f"labels (val ROC-AUC={trained.val_roc_auc:.3f}, need < "
        f"{_MAX_CHANCE_ROC_AUC}). The no-separation check is not "
        f"meaningful if the model still fits the permuted task.\n"
        f"{reports}"
    )
    auc_floor = _NULL_AUC - _NULL_AUC_TOLERANCE
    auc_ceiling = _NULL_AUC + _NULL_AUC_TOLERANCE
    off_chance = [
        item
        for item in separations
        if not (auc_floor <= item.auc <= auc_ceiling)
    ]
    cleared_null = [
        item
        for item in separations
        if clears_tier2_margin(
            item,
            max_ratio=DEFAULT_MAX_RATIO,
        )
    ]
    criterion_hits = [item for item in separations if separation_passes(item)]
    assert not off_chance and not cleared_null and not criterion_hits, (
        "shuffled labels still produced importance separation, so "
        "the Tier 2 criterion is following topology rather than "
        "the data.\n"
        f"off-chance AUC (outside {_NULL_AUC} ± "
        f"{_NULL_AUC_TOLERANCE}): {[item.label for item in off_chance]}. "
        f"cleared null by Tier 2 margin: "
        f"{[item.label for item in cleared_null]}. "
        f"sites that still satisfy the Tier 2 criterion: "
        f"{[item.label for item in criterion_hits]}.\n"
        f"{reports}"
    )


def test_swapped_ground_truth_fails_the_criterion() -> None:
    """
    Mutation test: swapping important and unimportant must fail Tier 2.
    """
    trained = train_matched_linear_towers(seed=_SHUFFLED_SEED)
    graph = trained.graph
    separations = trained_separations(
        trained,
        important_features=graph.tower_b_features,
        unimportant_features=graph.tower_a_features,
        important_nodes=graph.tower_b_nodes,
        unimportant_nodes=graph.tower_a_nodes,
    )
    reports = (
        f"val_roc_auc={trained.val_roc_auc:.3f} "
        f"{_format_separations(separations)}"
    )
    assert trained.val_roc_auc >= MIN_VAL_ROC_AUC, (
        "learnability PRECONDITION failed "
        f"(val ROC-AUC={trained.val_roc_auc:.3f}, need >= "
        f"{MIN_VAL_ROC_AUC}). The swap check needs a model that "
        f"learned.\n{reports}"
    )
    passed = all(separation_passes(item) for item in separations)
    assert not passed, (
        "the Tier 2 criterion still passed after the important and "
        "unimportant groups were swapped, so the assertion is not "
        f"sensitive to the ground truth.\n{reports}"
    )
