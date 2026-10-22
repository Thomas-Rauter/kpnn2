"""Bind observation labels and check named dims."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .._errors import caller_error, close_matches, listed
from .._validate import describe

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
    allowed = set(required) | set(optional)
    missing = [name for name in required if name not in dims]
    if missing:
        raise caller_error(
            "Attribution DataArray is missing required dim(s): "
            f"{listed(repr(name) for name in missing)}.",
            got=_dims_text(attributions),
            candidates=close_matches(
                missing,
                [
                    name
                    for name in attributions.dims
                    if isinstance(name, str) and name not in allowed
                ],
            ),
        )
    extra = sorted(dims - allowed)
    if extra:
        allowed_names = listed(repr(name) for name in sorted(allowed))
        raise caller_error(
            "Attribution DataArray has unexpected dim(s).",
            got=listed(repr(name) for name in extra),
            expected=f"only dims {allowed_names}",
            candidates=close_matches(
                [name for name in extra if isinstance(name, str)],
                sorted(allowed - dims),
            ),
            fix="Reduce or rename the unexpected dims first.",
        )


def require_unique_nodes(attributions: xr.DataArray) -> None:
    """Reject a ``node`` coordinate that repeats a name."""
    if NODE_DIM not in attributions.coords:
        return
    names = pd.Index(attributions.coords[NODE_DIM].values)
    if names.has_duplicates:
        repeated = names[names.duplicated()].unique().tolist()
        raise caller_error(
            f"The {NODE_DIM!r} coordinate repeats node names.",
            got=listed(repr(name) for name in repeated),
            fix="Reduce the units to one column per node first; a node "
            "wider than 1, or a hop axis, has one column per unit.",
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
        raise caller_error(
            "'labels' is required for this aggregation method.",
            fix="Pass labels= with one label per observation.",
        )
    if isinstance(labels, (xr.DataArray, pd.DataFrame)):
        raise caller_error(
            "'labels' must be a 1-d array or pandas Series.",
            got=describe(labels),
        )
    if observation_dim not in attributions.dims:
        raise caller_error(
            f"Attribution DataArray must have an {observation_dim!r} dim.",
            got=_dims_text(attributions),
            candidates=close_matches(
                [observation_dim],
                [name for name in attributions.dims if isinstance(name, str)],
            ),
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
            raise caller_error(
                "'labels' must be 1-dimensional.",
                got=f"shape {tuple(aligned.shape)}",
            )
        if int(aligned.shape[0]) != n_obs:
            raise caller_error(
                f"'labels' length must match the {observation_dim!r} dim.",
                got=f"length {int(aligned.shape[0])}",
                expected=f"length {n_obs}",
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
        raise caller_error(
            f"The {observation_dim!r} coordinate must be unique to align a "
            "pandas Series of labels.",
            got=f"repeated {_repeated_text(obs_index)}",
            fix="Make the coordinate unique, or pass 'labels' as an array, "
            "which is paired by position.",
        )
    if labels.index.has_duplicates:
        raise caller_error(
            "'labels' index must be unique to align it to the "
            f"{observation_dim!r} coordinate.",
            got=f"repeated {_repeated_text(labels.index)}",
        )
    extra = labels.index.difference(obs_index).tolist()
    missing = obs_index.difference(labels.index).tolist()
    if extra or missing:
        groups = []
        if missing:
            groups.append(f"missing {listed(repr(v) for v in missing)}")
        if extra:
            groups.append(f"extra {listed(repr(v) for v in extra)}")
        raise caller_error(
            "'labels' index does not match the "
            f"{observation_dim!r} coordinate.",
            got="; ".join(groups),
            candidates=close_matches(
                [v for v in extra if isinstance(v, str)],
                [v for v in missing if isinstance(v, str)],
            ),
        )
    return labels.reindex(obs_index).to_numpy()


def _repeated_text(index: pd.Index) -> str:
    """List the values an index repeats, each once."""
    return listed(repr(v) for v in index[index.duplicated()].unique().tolist())


def _dims_text(attributions: xr.DataArray) -> str:
    """List the dims of a DataArray for a ``Got`` line."""
    return f"dims {listed(repr(name) for name in attributions.dims)}"
