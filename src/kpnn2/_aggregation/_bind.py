"""Bind observation labels and check named dims."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .._errors import Kpnn2Error

OBSERVATION_DIM = "observation"
NODE_DIM = "node"
SEED_DIM = "seed"
LABEL_COORD = "label"
LAYER_COORD = "layer"


def require_named_dims(
    attributions: xr.DataArray,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> None:
    """Require named dims and reject any leftover dim."""
    dims = set(attributions.dims)
    missing = [name for name in required if name not in dims]
    if missing:
        listed = ", ".join(repr(name) for name in missing)
        raise Kpnn2Error(
            f"Attribution DataArray is missing required dim(s): {listed}."
        )
    allowed = set(required) | set(optional)
    extra = sorted(dims - allowed)
    if extra:
        listed = ", ".join(repr(name) for name in extra)
        allowed_listed = ", ".join(repr(name) for name in sorted(allowed))
        raise Kpnn2Error(
            "Attribution DataArray has unexpected dim(s): "
            f"{listed}. Reduce or rename them first. "
            f"Allowed dims: {allowed_listed}."
        )


def bind_labels(
    attributions: xr.DataArray,
    labels: object,
    *,
    observation_dim: str = OBSERVATION_DIM,
) -> xr.DataArray:
    """
    Attach ``labels`` as coord ``label`` on the observation dim.

    A numpy array or sequence is paired in order. A pandas
    Series is reindexed to the observation coordinate.
    """
    if labels is None:
        raise Kpnn2Error("'labels' is required for this aggregation method.")
    if isinstance(labels, xr.DataArray):
        raise Kpnn2Error(
            "'labels' must be a 1-d array or pandas Series, not a DataArray."
        )
    if isinstance(labels, pd.DataFrame):
        raise Kpnn2Error(
            "'labels' must be a 1-d array or pandas Series, not a DataFrame."
        )
    if observation_dim not in attributions.dims:
        raise Kpnn2Error(
            f"Attribution DataArray must have an {observation_dim!r} dim."
        )
    n_obs = int(attributions.sizes[observation_dim])
    if isinstance(labels, pd.Series):
        aligned = _align_series(
            labels,
            attributions.coords[observation_dim],
            observation_dim=observation_dim,
        )
    else:
        aligned = np.asarray(labels)
        if aligned.ndim != 1:
            raise Kpnn2Error("'labels' must be 1-dimensional.")
        if int(aligned.shape[0]) != n_obs:
            raise Kpnn2Error(
                "'labels' length must match the "
                f"{observation_dim!r} dim. Expected {n_obs}, "
                f"got {int(aligned.shape[0])}."
            )
    return attributions.assign_coords({LABEL_COORD: (observation_dim, aligned)})


def _align_series(
    labels: pd.Series,
    observation_coord: xr.DataArray,
    *,
    observation_dim: str,
) -> np.ndarray:
    """Reindex a Series onto the observation coordinate."""
    obs_index = pd.Index(observation_coord.values)
    if obs_index.has_duplicates:
        raise Kpnn2Error(
            f"The {observation_dim!r} coordinate must be unique "
            "to align a pandas Series of labels."
        )
    extra = labels.index.difference(obs_index)
    missing = obs_index.difference(labels.index)
    if len(extra) > 0 or len(missing) > 0:
        extra_str = ", ".join(repr(v) for v in extra.tolist())
        missing_str = ", ".join(repr(v) for v in missing.tolist())
        raise Kpnn2Error(
            "'labels' index does not match the "
            f"{observation_dim!r} coordinate. Missing: "
            f"{missing_str or '(none)'}. Extra: "
            f"{extra_str or '(none)'}."
        )
    return labels.reindex(obs_index).to_numpy()
