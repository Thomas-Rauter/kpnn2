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

    Missing values: a NaN attribution is left out of its
    class mean. Seed ``s`` counts for node ``i`` only when
    both ``mu_0(s, i)`` and ``mu_1(s, i)`` exist; Step 3 then
    averages over those seeds only. A node with no such seed
    has ``score`` NaN. Without NaN, every seed counts.

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

    Returns
    -------
    xarray.Dataset
        One variable, ``score``, with one value per node (the
        same ``node`` axis as the input).

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
    """
    _check_class_mapping(
        class_0,
        class_1,
    )
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
        data_vars={"score": score},
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
