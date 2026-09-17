"""Rauter and Mangano 2026 winner-minus-loser node scores."""

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
    require_unique_nodes,
)
from .._registry import register_aggregation_method

_METHOD_NAME = "rauter_mangano_2026"
_FLAG_NAME = "mean_class1_below_class0"

_RAUTER_MANGANO_DESCRIPTION = (
    "Binary: class mean with the larger absolute value minus "
    "the other class mean, averaged over seeds."
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
    correct_sign: bool = False,
) -> xr.Dataset:
    """
    Score each node by winner minus loser of its class means.

    Binary classification only. This function is not exported.
    Pass ``method="rauter_mangano_2026"`` to
    ``aggregate_node_attributions``.

    Write ``a[s, o, i]`` for the attribution at seed ``s``,
    observation ``o``, and node ``i``, with seeds
    ``s = 1, ..., S``. A missing ``seed`` dim is ``S = 1``.
    ``O_0`` and ``O_1`` are the observations whose labels
    equal ``class_0`` and ``class_1``.

    Step 1. For each seed and node, the class-c mean is the
    mean attribution over the observations of that class:

    ``mu_c(s, i) = mean_{o in O_c} a[s, o, i]``
    for ``c`` in {0, 1}.

    Step 2. The winner ``w(s, i)`` is the class mean with the
    larger absolute value, and the loser ``l(s, i)`` is the
    other class mean. On a tie, class 1 is the winner:

    ``(w, l) = (mu_1, mu_0)  if |mu_1(s, i)| >= |mu_0(s, i)|``
    ``(w, l) = (mu_0, mu_1)  otherwise``.

    The score of node ``i`` under seed ``s`` is the winner
    minus the loser, both with their original signs:

    ``r(s, i) = w(s, i) - l(s, i)``.

    Step 3. The score of node ``i`` is the mean over seeds:

    ``score(i) = (1/S) * sum_s r(s, i)``.

    Flag. For each seed and node, the method records whether
    the class 1 mean is below the class 0 mean:

    ``f(s, i) = [mu_1(s, i) < mu_0(s, i)]``.

    Exactly when ``f(s, i)`` is True, ``r(s, i)`` has the
    opposite sign to ``eps(s, i) * |mu_1(s, i) - mu_0(s, i)|``,
    where ``eps(s, i) = +1`` if class 1 is the winner and
    ``-1`` otherwise. With ``correct_sign=True``, Step 3
    averages ``-r(s, i)`` instead of ``r(s, i)`` wherever
    ``f(s, i)`` is True, so the score becomes

    ``score(i) = (1/S) * sum_s eps(s, i) * |mu_1(s, i) - mu_0(s, i)|``.

    The flag is computed and returned either way.

    Missing values: a NaN attribution is left out of its
    class mean. Seed ``s`` counts for node ``i`` only when
    both ``mu_0(s, i)`` and ``mu_1(s, i)`` exist; Step 3 then
    averages over those seeds only, and ``f(s, i)`` is False
    for the other seeds. A node with no such seed has
    ``score`` NaN. Without NaN, every seed counts.

    Parameters
    ----------
    attributions : xarray.DataArray
        Output of ``map_node_attributions``, or an array
        with the same named dims. Requires ``observation``
        and ``node``, accepts optional ``seed``, and rejects
        any other dim (reduce or ``.rename`` first). Node
        names must be unique: reduce the units of a node
        wider than 1 to one column first.
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
    correct_sign : bool, optional
        If True, flip the sign of ``r(s, i)`` wherever
        ``f(s, i)`` is True before Step 3, as above. Default
        False, which is the winner-minus-loser score.

    Returns
    -------
    xarray.Dataset
        Two variables. ``score`` has one value per node (the
        same ``node`` axis as the input).
        ``mean_class1_below_class0`` is the boolean flag
        ``f(s, i)``, with dims ``(seed, node)``, or ``(node,)``
        when the input has no ``seed`` dim.

    Raises
    ------
    Kpnn2Error
        If ``labels`` or kwargs are invalid, the DataArray
        dims do not match this method, or a node name
        repeats.

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

    ``mean_class1_below_class0`` compares the class means of
    the attributions as given. If the attributions explain
    the class 1 output (for example Captum ``target=None`` on
    a single logit), True means that, in that seed, the node
    pushes class 1 observations less toward class 1 than
    class 0 observations.
    """
    _check_class_mapping(
        class_0,
        class_1,
    )
    _check_correct_sign(correct_sign)
    require_named_dims(
        attributions,
        required=(OBSERVATION_DIM, NODE_DIM),
        optional=(SEED_DIM,),
    )
    require_unique_nodes(attributions)
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
    per_seed = xr.where(
        np.abs(mean1) >= np.abs(mean0),
        mean1 - mean0,
        mean0 - mean1,
    ).where(valid)
    below = (mean1 < mean0) & valid
    if correct_sign:
        per_seed = xr.where(
            below,
            -per_seed,
            per_seed,
        )
    if SEED_DIM in scored.dims:
        score = per_seed.mean(SEED_DIM)
    else:
        score = per_seed
    coords: dict[str, object] = {}
    if NODE_DIM in scored.coords:
        coords[NODE_DIM] = scored.coords[NODE_DIM]
    if LAYER_COORD in scored.coords and LAYER_COORD not in scored.dims:
        coords[LAYER_COORD] = scored.coords[LAYER_COORD]
    return xr.Dataset(
        data_vars={
            "score": score,
            _FLAG_NAME: below,
        },
        coords=coords,
    )


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


def _check_correct_sign(correct_sign: object) -> None:
    """Require a boolean ``correct_sign``."""
    if not isinstance(correct_sign, (bool, np.bool_)):
        raise Kpnn2Error("'correct_sign' must be a bool.")


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
