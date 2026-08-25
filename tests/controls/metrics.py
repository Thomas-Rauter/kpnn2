"""
Rank-based separation for trained-tier scores.

ROC-AUC of median |score| against the ground-truth label is
scale-free. A margin/null check rejects ranking with a negligible
gap: every important score must clear a robust null from the decoy
group, and max(decoy) / min(important) must stay below max_ratio.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_NULL_K = 5.0
DEFAULT_NULL_FLOOR = 1e-9
DEFAULT_MAX_RATIO = 0.2
DEFAULT_MIN_AUC = 1.0
DEFAULT_MIN_PER_EXAMPLE = 0.9


@dataclass(frozen=True)
class Separation:
    """
    How cleanly one table separated important from unimportant names.
    """

    label: str
    auc: float
    important_floor: float
    unimportant_ceiling: float
    ratio: float
    null_threshold: float
    per_example_consistency: float
    n_important: int
    n_unimportant: int


def median_abs_scores(
    table: pd.DataFrame,
    names: Sequence[str],
    *,
    label: str = "table",
) -> pd.Series:
    """
    Median absolute score of each named column.
    """
    _validate_columns(
        table,
        names,
        label=label,
    )
    return pd.Series(
        {name: float(table[name].abs().median()) for name in names},
        dtype=float,
    )


def robust_null_threshold(
    values: Sequence[float],
    *,
    k: float = DEFAULT_NULL_K,
    floor: float = DEFAULT_NULL_FLOOR,
) -> float:
    """
    Null cutoff ``median + k * MAD`` from the unimportant group.
    """
    if len(values) == 0:
        raise ValueError(
            "'values' must be non-empty to calibrate a null threshold."
        )
    series = pd.Series(
        list(values),
        dtype=float,
    )
    median = float(series.median())
    mad = float((series - median).abs().median())
    return median + k * max(mad, floor)


def per_example_consistency(
    table: pd.DataFrame,
    *,
    important: Sequence[str],
    unimportant: Sequence[str],
) -> float:
    """
    Fraction of rows where min(important) > max(unimportant).
    """
    _validate_groups(
        table,
        important=important,
        unimportant=unimportant,
        label="table",
    )
    if table.empty:
        raise ValueError("'table' must contain at least one example.")
    important_floor = table.loc[:, list(important)].abs().min(axis=1)
    unimportant_ceiling = table.loc[:, list(unimportant)].abs().max(axis=1)
    return float((important_floor > unimportant_ceiling).mean())


def compute_separation(
    table: pd.DataFrame,
    *,
    label: str,
    important: Sequence[str],
    unimportant: Sequence[str],
) -> Separation:
    """
    ROC-AUC and margin of median |score| vs ground-truth groups.
    """
    _validate_groups(
        table,
        important=important,
        unimportant=unimportant,
        label=label,
    )
    important_scores = median_abs_scores(
        table,
        important,
        label=label,
    )
    unimportant_scores = median_abs_scores(
        table,
        unimportant,
        label=label,
    )
    group_labels = [1] * len(important) + [0] * len(unimportant)
    group_scores = list(important_scores) + list(unimportant_scores)
    important_floor = float(important_scores.min())
    unimportant_ceiling = float(unimportant_scores.max())
    return Separation(
        label=label,
        auc=float(
            roc_auc_score(
                group_labels,
                group_scores,
            )
        ),
        important_floor=important_floor,
        unimportant_ceiling=unimportant_ceiling,
        ratio=_ratio(
            unimportant_ceiling=unimportant_ceiling,
            important_floor=important_floor,
        ),
        null_threshold=robust_null_threshold(list(unimportant_scores)),
        per_example_consistency=per_example_consistency(
            table,
            important=important,
            unimportant=unimportant,
        ),
        n_important=len(important),
        n_unimportant=len(unimportant),
    )


def separation_passes(
    separation: Separation,
    *,
    min_auc: float = DEFAULT_MIN_AUC,
    max_ratio: float = DEFAULT_MAX_RATIO,
    min_per_example_consistency: float = DEFAULT_MIN_PER_EXAMPLE,
) -> bool:
    """
    Return whether AUC, ratio, consistency, and null tests all pass.
    """
    return (
        separation.auc >= min_auc
        and (separation.per_example_consistency >= min_per_example_consistency)
        and clears_tier2_margin(
            separation,
            max_ratio=max_ratio,
        )
    )


def clears_tier2_margin(
    separation: Separation,
    *,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> bool:
    """
    Return whether important scores clear the decoy null by max_ratio.

    This is the margin/null part of ``separation_passes``, not a
    second metric.
    """
    if separation.important_floor <= separation.null_threshold:
        return False
    if math.isinf(separation.ratio):
        return False
    return separation.ratio <= max_ratio


def _ratio(
    *,
    unimportant_ceiling: float,
    important_floor: float,
) -> float:
    """
    Return max(decoy) / min(important), or inf if the floor is 0.
    """
    if important_floor == 0.0:
        return math.inf
    return unimportant_ceiling / important_floor


def _validate_groups(
    table: pd.DataFrame,
    *,
    important: Sequence[str],
    unimportant: Sequence[str],
    label: str,
) -> None:
    """
    Reject empty or overlapping groups.
    """
    if not important:
        raise ValueError(f"'{label}': the important group is empty.")
    if not unimportant:
        raise ValueError(f"'{label}': the unimportant group is empty.")
    overlap = sorted(set(important) & set(unimportant))
    if overlap:
        raise ValueError(f"'{label}': names in both groups: {overlap}.")
    _validate_columns(
        table,
        [*important, *unimportant],
        label=label,
    )


def _validate_columns(
    table: pd.DataFrame,
    names: Sequence[str],
    *,
    label: str,
) -> None:
    """
    Require each name to appear exactly once.
    """
    counts = Counter(table.columns)
    missing = sorted(name for name in names if counts[name] == 0)
    if missing:
        raise ValueError(
            f"'{label}': missing columns {missing}. "
            f"Available: {sorted(table.columns)}."
        )
    duplicated = sorted(name for name in names if counts[name] > 1)
    if duplicated:
        raise ValueError(f"'{label}': duplicated columns {duplicated}.")
