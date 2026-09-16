"""Rauter and Mangano 2026 binary class-difference scores."""

from __future__ import annotations

import math
from numbers import Real

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

    Binary classification only. This function is not exported.
    Pass ``method="rauter_mangano_2026"`` to
    ``aggregate_node_attributions``.

    Write ``a[s, o, i]`` for the attribution at seed ``s``,
    observation ``o``, and node ``i``. A missing ``seed`` dim
    is ``S = 1``. ``C_0`` and ``C_1`` are the observations
    whose labels equal ``class_0`` and ``class_1``. The
    class means omit NaN attributions (xarray
    ``skipna=True``).

    For each seed and node, the class-c mean is the mean
    attribution over observations of that class:

    ``mu_c(s, i) = mean_{o in C_c} a[s, o, i]``
    for ``c`` in {0, 1}.

    Seed ``s`` is valid for node ``i`` when both
    ``mu_0(s, i)`` and ``mu_1(s, i)`` are not NaN. For an
    invalid seed both are set to NaN. ``V(i)`` is the set
    of valid seeds. Every average over seeds below,
    written ``(1/S) * sum_s``, runs over ``V(i)`` only.
    Without NaN, ``V(i)`` holds all ``S`` seeds.

    The class-mean difference is

    ``D(s, i) = mu_1(s, i) - mu_0(s, i)``.

    The sign ``eps`` is ``+1`` when class 1's mean is at
    least as far from 0 as class 0's mean, else ``-1``
    (equality goes to class 1):

    ``eps(s, i) = +1 if |mu_1(s, i)| >= |mu_0(s, i)|``
    ``            -1 otherwise``.

    The per-seed score is that sign times the absolute
    difference:

    ``score(s, i) = eps(s, i) * |D(s, i)|``.

    Seed-averaged class means are always

    ``mu_c_bar(i) = (1/S) * sum_s mu_c(s, i)``,

    and

    ``class_difference(i) = mu_1_bar(i) - mu_0_bar(i)``.

    With no ``seed`` dim, ``mu_c_bar = mu_c``.

    ``sign_reference`` selects how ``score`` is folded over
    seeds. For ``"per_seed"`` (default), average the
    per-seed scores:

    ``score(i) = (1/S) * sum_s score(s, i)``.

    For ``"seed_mean"``, compute ``D``, ``eps``, and
    ``score`` once from the seed-averaged class means:

    ``D_bar(i) = mu_1_bar(i) - mu_0_bar(i)``
    ``eps_bar(i) = +1 if |mu_1_bar(i)| >= |mu_0_bar(i)|``
    ``             -1 otherwise``
    ``score(i) = eps_bar(i) * |D_bar(i)|``.

    The two folding rules are the same when ``S = 1``. They
    can differ when ``S > 1``.

    The remaining outputs are

    ``abs_score(i) = |score(i)|``

    ``sign(i) = +1 if score(i) > 0``
    ``          -1 if score(i) < 0``
    ``          +1 if score(i) = 0``
    ``           0 if score(i) is NaN``

    ``n_seeds = S``

    ``sign_consistency(i)``
    ``    = (1/S) * sum_s 1[eps(s, i) == sign(i)]``

    With ``S = 1``, ``sign_consistency(i) = 1``.

    When ``V(i)`` is empty, ``score``, ``abs_score``,
    ``mean_class0``, ``mean_class1``, ``class_difference``,
    and ``sign_consistency`` are NaN, ``sign`` is 0, and
    ``counteracting`` and ``near_tie`` are False.
    ``n_seeds`` counts every seed, valid or not.

    ``counteracting(i) = [class_difference(i) < 0]``

    ``u(i) = abs(|mu_1_bar(i)| - |mu_0_bar(i)|)``
    ``v(i) = max(|mu_0_bar(i)|, |mu_1_bar(i)|, 1e-12)``
    ``near_tie(i) = [u(i) / v(i) < tie_tolerance]``

    Default ``tie_tolerance`` is 0.05.

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
        are already ``0`` / ``1``. Arrays are rejected.
    class_1 : scalar
        Label value for class 1. Must differ from
        ``class_0``. Arrays are rejected.
    sign_reference : {"per_seed", "seed_mean"}, optional
        Folding rule for ``score`` over seeds, as above.
        Default ``"per_seed"``.
    tie_tolerance : float, optional
        Threshold in the ``near_tie`` formula. Default
        0.05. Any real number (numpy scalars included) that
        is finite and ``>= 0``.

    Returns
    -------
    xarray.Dataset
        One value per node (the same ``node`` axis as the
        input, including a repeated name when a node is
        wider than 1). Variables ``score``, ``abs_score``,
        ``mean_class0`` (``mu_0_bar``), ``mean_class1``
        (``mu_1_bar``), ``class_difference``, ``sign``,
        ``n_seeds``, ``sign_consistency``,
        ``counteracting``, and ``near_tie``, as defined
        above.

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
    mean0 = scored.isel({OBSERVATION_DIM: is0}).mean(OBSERVATION_DIM)
    mean1 = scored.isel({OBSERVATION_DIM: is1}).mean(OBSERVATION_DIM)
    # A seed counts for a node only when both class means exist.
    valid = mean0.notnull() & mean1.notnull()
    mean0 = mean0.where(valid)
    mean1 = mean1.where(valid)
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
    sign = (
        xr.where(score < 0, -1, 1)
        .where(
            score.notnull(),
            0,
        )
        .astype(np.int64)
    )
    agrees = (eps_s == sign).astype(np.float64).where(eps_s.notnull())
    if has_seed:
        sign_consistency = agrees.mean(SEED_DIM)
    else:
        sign_consistency = agrees
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
    """Return ``eps`` and ``score = eps * |D|``, NaN where ``D`` is."""
    difference = mean1 - mean0
    eps = xr.where(
        np.abs(mean1) >= np.abs(mean0),
        1.0,
        -1.0,
    ).where(difference.notnull())
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
    for name, value in (("class_0", class_0), ("class_1", class_1)):
        if np.ndim(value) != 0:
            raise Kpnn2Error(f"'{name}' must be a scalar label value.")
    if class_0 == class_1:
        raise Kpnn2Error("'class_0' and 'class_1' must be distinct.")


def _check_sign_reference(sign_reference: str) -> None:
    """Accept only the two seed-folding rules."""
    if sign_reference not in _SIGN_REFERENCES:
        raise Kpnn2Error("'sign_reference' must be 'per_seed' or 'seed_mean'.")


def _check_tie_tolerance(tie_tolerance: object) -> None:
    """Require a finite, non-negative real tolerance."""
    if isinstance(tie_tolerance, (bool, np.bool_)) or not isinstance(
        tie_tolerance,
        Real,
    ):
        raise Kpnn2Error("'tie_tolerance' must be a real number.")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise Kpnn2Error("'tie_tolerance' must be finite and >= 0.")


def _binary_masks(
    scored: xr.DataArray,
    *,
    class_0: object,
    class_1: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for the two classes along observation."""
    label_coord = scored.coords[LABEL_COORD]
    values = np.asarray(label_coord.values)
    if np.any(np.asarray(pd.isna(values), dtype=bool)):
        raise Kpnn2Error("'labels' must not contain missing values.")
    # pd.unique hashes instead of sorting, so mixed label types work.
    unique_list = pd.unique(values).tolist()
    if len(unique_list) > 2:
        extra_str = ", ".join(repr(v) for v in unique_list)
        raise Kpnn2Error(
            f"Method {_METHOD_NAME!r} supports binary "
            "classification only. 'labels' has more than "
            f"two classes: {extra_str}."
        )
    is0 = np.asarray((label_coord == class_0).values, dtype=bool)
    is1 = np.asarray((label_coord == class_1).values, dtype=bool)
    unmatched = ~(is0 | is1)
    if np.any(unmatched):
        extras = pd.unique(values[unmatched])
        extra_str = ", ".join(repr(v) for v in extras.tolist())
        raise Kpnn2Error(
            f"Method {_METHOD_NAME!r} supports binary "
            "classification only. 'labels' must contain only "
            f"class_0={class_0!r} and class_1={class_1!r}. "
            f"Unknown label(s): {extra_str}."
        )
    if not is0.any():
        raise Kpnn2Error(
            "Both class_0 and class_1 must appear in 'labels'. "
            f"class_0={class_0!r} is missing."
        )
    if not is1.any():
        raise Kpnn2Error(
            "Both class_0 and class_1 must appear in 'labels'. "
            f"class_1={class_1!r} is missing."
        )
    return is0, is1
