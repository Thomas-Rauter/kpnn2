"""Rauter and Mangano 2026 binary class-difference scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..._errors import Kpnn2Error
from .._bind import (
    LABEL_COORD,
    LAYER_COORD,
    NODE_DIM,
    OBSERVATION_DIM,
    SEED_DIM,
    bind_labels,
    require_named_dims,
)
from .._registry import register_aggregation_method

_METHOD_NAME = "rauter_mangano_2026"
_SIGN_REFERENCES = frozenset({"per_seed", "seed_mean"})
_NEAR_TIE_EPS = 1e-12
_DEFAULT_TIE_TOLERANCE = 0.05

_RAUTER_MANGANO_DESCRIPTION = (
    "Binary signed class-mean difference: magnitude |mean_1 - "
    "mean_0|, sign of the class farther from zero."
)
_RAUTER_MANGANO_REFERENCES = ("Rauter and Mangano, 2026",)


@register_aggregation_method(
    name=_METHOD_NAME,
    status="recommended",
    description=_RAUTER_MANGANO_DESCRIPTION,
    references=_RAUTER_MANGANO_REFERENCES,
    added_in="0.2.0",
)
def rauter_mangano_2026(
    attributions: xr.DataArray,
    labels: object | None,
    *,
    class_0: object | None = None,
    class_1: object | None = None,
    sign_reference: str = "per_seed",
    tie_tolerance: float = _DEFAULT_TIE_TOLERANCE,
) -> xr.Dataset:
    """
    Score nodes by signed class-mean difference, ``eps * |D|``.

    Binary classification only. For each seed ``s`` and node
    ``i`` (a missing ``seed`` dim is one seed):

    1. ``mean_c`` is the mean attribution over observations of
       class ``c`` (``c`` in {0, 1}).
    2. ``D = mean_1 - mean_0``.
    3. ``eps = +1`` if ``|mean_1| >= |mean_0|``, else ``-1``
       (ties go to class 1).
    4. ``score = eps * |D|``.

    The magnitude at this step is the absolute class-mean
    difference. The sign is the class whose mean attribution
    is farther from the all-zero baseline (positive: class 1,
    negative: class 0), including nodes with ``D < 0`` that
    counteract class separation.

    On one seed, and with ``sign_reference="seed_mean"``,
    ``abs_score`` equals ``|D|`` (or the seed-averaged
    ``|D|``), so ranking by ``abs_score`` does not depend on
    ``eps``. With the default ``"per_seed"``, ``score`` is
    the mean of per-seed scores, so mixed per-seed signs can
    shrink ``abs_score``.

    ``sign_reference`` controls folding across seeds:

    - ``"per_seed"`` (default): compute ``score`` per seed,
      then average.
    - ``"seed_mean"``: average ``mean_0`` and ``mean_1``
      across seeds, then compute ``D``, ``eps``, and
      ``score`` once. This avoids shrinking nodes whose
      per-seed sign varies.

    ``mean_class0``, ``mean_class1``, and
    ``class_difference`` are always seed-averaged.
    ``near_tie`` is true when the absolute class means
    differ by less than relative ``tie_tolerance`` (default
    0.05); in those cases the sign is unreliable.

    This function is not exported. Pass
    ``method="rauter_mangano_2026"`` to
    ``aggregate_node_attributions``.

    Parameters
    ----------
    attributions : xarray.DataArray
        Output of ``map_node_attributions``, or an array
        with the same named dims. Requires ``observation``
        and ``node``, accepts optional ``seed``, and rejects
        any other dim (reduce or ``.rename`` first).
        Concatenate trained models with
        ``xr.concat(..., dim="seed")``. A scalar ``layer``
        coordinate is copied through when present. The
        array is read, never modified.
    labels : 1-d array or pandas.Series
        Observation-to-class mapping aligned to
        ``observation``. A numpy array or sequence is
        paired in order and must be as long as that axis.
        A pandas Series is reindexed to the observation
        coordinate; missing or extra ids raise.
    class_0 : scalar
        Label value for class 0. Required even when labels
        are already ``0`` / ``1``.
    class_1 : scalar
        Label value for class 1. Must differ from
        ``class_0``.
    sign_reference : {"per_seed", "seed_mean"}, optional
        How to fold the ``seed`` dim. Default
        ``"per_seed"``.
    tie_tolerance : float, optional
        Relative tolerance for ``near_tie``. Default 0.05.
        Must be ``>= 0``.

    Returns
    -------
    xarray.Dataset
        One value per node (the same ``node`` axis as the
        input, including a repeated name when a node is
        wider than 1). Variables: ``score``, ``abs_score``,
        ``mean_class0``, ``mean_class1``,
        ``class_difference``, ``sign`` (+1 or -1;
        ``score == 0`` is +1), ``n_seeds``,
        ``sign_consistency`` (fraction of seeds whose
        per-seed ``eps`` matches the final sign),
        ``counteracting`` (seed-averaged ``D < 0``),
        ``near_tie``.

    Raises
    ------
    Kpnn2Error
        If ``labels`` or kwargs are invalid, or the
        DataArray dims do not match this method.

    See Also
    --------
    aggregate_node_attributions : Public dispatcher; pass
        ``method="rauter_mangano_2026"``.
    list_aggregation_methods : Registry table.

    Notes
    -----
    Sample labels (which row is class 0 or 1) are not
    Captum's ``target=`` (which output class was explained).
    If a Captum ``class`` dim is still on the array, select
    one class before calling.
    """
    _check_class_mapping(
        class_0,
        class_1,
    )
    _check_sign_reference(sign_reference)
    _check_tie_tolerance(tie_tolerance)
    require_named_dims(
        attributions,
        required=(OBSERVATION_DIM, NODE_DIM),
        optional=(SEED_DIM,),
    )
    scored = bind_labels(
        attributions,
        labels,
    )
    is0, is1 = _binary_masks(
        scored,
        class_0=class_0,
        class_1=class_1,
    )
    mean0 = scored.where(is0).mean(OBSERVATION_DIM)
    mean1 = scored.where(is1).mean(OBSERVATION_DIM)
    has_seed = SEED_DIM in scored.dims
    if has_seed:
        n_seeds = int(scored.sizes[SEED_DIM])
        mean0_bar = mean0.mean(SEED_DIM)
        mean1_bar = mean1.mean(SEED_DIM)
    else:
        n_seeds = 1
        mean0_bar = mean0
        mean1_bar = mean1
    eps_s, score_s = _signed_scores(
        mean0,
        mean1,
    )
    class_difference = mean1_bar - mean0_bar
    if sign_reference == "seed_mean":
        _, score = _signed_scores(
            mean0_bar,
            mean1_bar,
        )
    elif has_seed:
        score = score_s.mean(SEED_DIM)
    else:
        score = score_s
    sign = xr.where(
        score > 0,
        1,
        xr.where(score < 0, -1, 1),
    ).astype(np.int64)
    if has_seed:
        sign_consistency = (eps_s == sign).mean(SEED_DIM)
    else:
        sign_consistency = xr.ones_like(
            score,
            dtype=np.float64,
        )
    abs0 = np.abs(mean0_bar)
    abs1 = np.abs(mean1_bar)
    numer = np.abs(abs1 - abs0)
    denom = np.maximum(
        np.maximum(abs0, abs1),
        _NEAR_TIE_EPS,
    )
    near_tie = (numer / denom) < float(tie_tolerance)
    data_vars = {
        "score": score,
        "abs_score": np.abs(score),
        "mean_class0": mean0_bar,
        "mean_class1": mean1_bar,
        "class_difference": class_difference,
        "sign": sign,
        "sign_consistency": sign_consistency,
        "counteracting": class_difference < 0,
        "near_tie": near_tie,
        "n_seeds": n_seeds,
    }
    coords: dict[str, object] = {}
    if NODE_DIM in scored.coords:
        coords[NODE_DIM] = scored.coords[NODE_DIM]
    if LAYER_COORD in scored.coords and LAYER_COORD not in scored.dims:
        coords[LAYER_COORD] = scored.coords[LAYER_COORD]
    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
    )


def _signed_scores(
    mean0: xr.DataArray,
    mean1: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return ``eps`` and ``score = eps * |D|`` for class means."""
    difference = mean1 - mean0
    eps = xr.where(
        np.abs(mean1) >= np.abs(mean0),
        1.0,
        -1.0,
    )
    score = eps * np.abs(difference)
    return eps, score


def _check_class_mapping(
    class_0: object | None,
    class_1: object | None,
) -> None:
    """Require two distinct class-label values."""
    if class_0 is None or class_1 is None:
        raise Kpnn2Error(
            f"Method {_METHOD_NAME!r} requires 'class_0' and "
            "'class_1' to map label values onto class 0 and "
            "class 1."
        )
    if class_0 == class_1:
        raise Kpnn2Error("'class_0' and 'class_1' must be distinct.")


def _check_sign_reference(sign_reference: str) -> None:
    """Accept only the two seed-folding rules."""
    if sign_reference not in _SIGN_REFERENCES:
        raise Kpnn2Error("'sign_reference' must be 'per_seed' or 'seed_mean'.")


def _check_tie_tolerance(tie_tolerance: object) -> None:
    """Require a non-negative float tolerance."""
    if isinstance(tie_tolerance, bool) or not isinstance(
        tie_tolerance,
        (int, float),
    ):
        raise Kpnn2Error("'tie_tolerance' must be a float.")
    if float(tie_tolerance) < 0:
        raise Kpnn2Error("'tie_tolerance' must be >= 0.")


def _binary_masks(
    scored: xr.DataArray,
    *,
    class_0: object,
    class_1: object,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Boolean masks for the two classes along observation."""
    label_coord = scored.coords[LABEL_COORD]
    values = np.asarray(label_coord.values)
    if np.any(np.asarray(pd.isna(values), dtype=bool)):
        raise Kpnn2Error("'labels' must not contain missing values.")
    unique = np.unique(values)
    unique_list = unique.tolist()
    if len(unique_list) > 2:
        extra_str = ", ".join(repr(v) for v in unique_list)
        raise Kpnn2Error(
            f"Method {_METHOD_NAME!r} supports binary "
            "classification only. 'labels' has more than "
            f"two classes: {extra_str}."
        )
    is0 = label_coord == class_0
    is1 = label_coord == class_1
    unmatched = ~(np.asarray(is0.values) | np.asarray(is1.values))
    if np.any(unmatched):
        extras = np.unique(values[unmatched])
        extra_str = ", ".join(repr(v) for v in extras.tolist())
        raise Kpnn2Error(
            f"Method {_METHOD_NAME!r} supports binary "
            "classification only. 'labels' must contain only "
            f"class_0={class_0!r} and class_1={class_1!r}. "
            f"Unknown label(s): {extra_str}."
        )
    if not bool(np.any(np.asarray(is0.values))):
        raise Kpnn2Error(
            "Both class_0 and class_1 must appear in 'labels'. "
            f"class_0={class_0!r} is missing."
        )
    if not bool(np.any(np.asarray(is1.values))):
        raise Kpnn2Error(
            "Both class_0 and class_1 must appear in 'labels'. "
            f"class_1={class_1!r} is missing."
        )
    return is0, is1
