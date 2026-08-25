"""
Data simulators for trained-tier control tests.

Features are independent and centered. Attribution is not causal
inference: a decoy correlated with a causal feature has no
well-defined ground-truth-unimportant label.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def independent_normal_features(
    rng: np.random.Generator,
    n_samples: int,
    *,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    Draw independent standard-normal features, one column per name.
    """
    if n_samples < 1:
        raise ValueError(f"'n_samples' must be positive, got {n_samples}.")
    if not feature_names:
        raise ValueError("'feature_names' must name at least one feature.")
    values = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_samples, len(feature_names)),
    )
    return pd.DataFrame(
        values,
        columns=list(feature_names),
        index=[f"sample_{index}" for index in range(n_samples)],
    )


def linear_logit_labels(
    x: pd.DataFrame,
    *,
    causal_features: Sequence[str],
    coefficients: Sequence[float],
) -> np.ndarray:
    """
    Label from a linear combination of the causal features only.

    Every other column is ignored. Threshold is 0, which is balanced
    for centered features.
    """
    _validate_causal_features(
        x,
        causal_features,
    )
    if len(coefficients) != len(causal_features):
        raise ValueError(
            "'coefficients' must have one entry per causal feature, "
            f"got {len(coefficients)} for {len(causal_features)} "
            "features."
        )
    weights = np.asarray(
        coefficients,
        dtype=np.float64,
    )
    logit = x.loc[:, list(causal_features)].to_numpy() @ weights
    return _as_labels(logit)


def product_logit_labels(
    x: pd.DataFrame,
    *,
    causal_features: Sequence[str],
) -> np.ndarray:
    """
    Label from the product of exactly two causal features.
    """
    _validate_causal_features(
        x,
        causal_features,
    )
    if len(causal_features) != 2:
        raise ValueError(
            "'causal_features' must name exactly two features for a "
            f"product label, got {len(causal_features)}."
        )
    first, second = causal_features
    logit = x[first].to_numpy() * x[second].to_numpy()
    return _as_labels(logit)


def shuffled_labels(
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Permute rows, preserving class balance, for negative controls.
    """
    labels = np.asarray(
        y,
        dtype=np.float32,
    )
    if labels.ndim != 2 or labels.shape[1] != 1:
        raise ValueError(
            f"'y' must have shape (n_samples, 1), got {labels.shape}."
        )
    order = rng.permutation(labels.shape[0])
    return labels[order].copy()


def _as_labels(logit: np.ndarray) -> np.ndarray:
    """
    Threshold a score at zero; return float32 labels of shape (n, 1).
    """
    return (logit > 0.0).astype(np.float32).reshape(-1, 1)


def _validate_causal_features(
    x: pd.DataFrame,
    causal_features: Sequence[str],
) -> None:
    """
    Require every causal name to exist in ``x``.
    """
    if not causal_features:
        raise ValueError("'causal_features' must name at least one feature.")
    missing = sorted(set(causal_features) - set(x.columns))
    if missing:
        raise ValueError(
            "These causal features are missing from the data: "
            f"{missing}. Available columns: {sorted(x.columns)}."
        )
