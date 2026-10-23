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
    """
    Require named dims and reject any leftover dim.

    Missing and unexpected dims are found together, and both go
    into one message, whose suggestions name an allowed dim for
    each unexpected one.
    """
    dims = set(attributions.dims)
    allowed = set(required) | set(optional)
    missing = [name for name in required if name not in dims]
    extra = sorted(dims - allowed)
    if missing and not extra:
        raise caller_error(
            "'attributions' is missing required dim(s): "
            f"{listed(repr(name) for name in missing)}.",
            got=_dims_text(attributions),
        )
    if extra:
        allowed_names = listed(repr(name) for name in sorted(allowed))
        extra_names = listed(repr(name) for name in extra)
        if missing:
            headline = (
                "'attributions' has unexpected dim(s) and is missing "
                "required dim(s)."
            )
            got = (
                f"unexpected {extra_names}; missing "
                f"{listed(repr(name) for name in missing)}"
            )
        else:
            headline = "'attributions' has unexpected dim(s)."
            got = extra_names
        raise caller_error(
            headline,
            got=got,
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
            "wider than 1 has one column per unit.",
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
    if isinstance(labels, xr.DataArray):
        raise caller_error(
            "'labels' must be a 1-d array or pandas Series, not a DataArray.",
            got=describe(labels),
            fix="Pass labels.to_series(), which aligns the labels by their "
            "coordinate.",
        )
    if isinstance(labels, pd.DataFrame):
        raise caller_error(
            "'labels' must be a 1-d array or pandas Series, not a DataFrame.",
            got=describe(labels),
            fix="Pass the column that holds the labels, as a Series.",
        )
    if observation_dim not in attributions.dims:
        # A close dim is the caller's own, so it is a rename, not a
        # name to suggest.
        close = [
            name
            for name in attributions.dims
            if isinstance(name, str)
            and close_matches(
                [observation_dim],
                [name],
            )
            is not None
        ]
        if len(close) == 1:
            fix = f"Rename dim {close[0]!r} to {observation_dim!r}."
        elif close:
            fix = (
                f"Rename one of dims {listed(repr(name) for name in close)} "
                f"to {observation_dim!r}."
            )
        else:
            fix = None
        raise caller_error(
            f"'attributions' is missing required dim(s): {observation_dim!r}.",
            got=_dims_text(attributions),
            fix=fix,
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
            # A set, dict, or str converts to a 0-d array; its shape
            # would describe the conversion, not what was passed.
            raise caller_error(
                "'labels' must be 1-dimensional.",
                got=(
                    describe(labels)
                    if aligned.ndim == 0
                    else f"shape {tuple(aligned.shape)}"
                ),
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
        extra_names = [v for v in extra if isinstance(v, str)]
        missing_names = [v for v in missing if isinstance(v, str)]
        folded = {v.casefold() for v in missing_names}
        # No shared value, and no case-only typo: the index likely
        # holds another kind of identifier, such as sample IDs for
        # the default 0..n-1 coordinate.
        if obs_index.intersection(labels.index).empty and not any(
            v.casefold() in folded for v in extra_names
        ):
            candidates = None
            fix = (
                "No index value names an observation; check the kind of "
                "identifier, or name the observations with coords= in "
                "map_node_attributions()."
            )
        else:
            candidates = close_matches(
                extra_names,
                missing_names,
            )
            fix = None
        raise caller_error(
            "'labels' index does not match the "
            f"{observation_dim!r} coordinate.",
            got="; ".join(groups),
            candidates=candidates,
            fix=fix,
        )
    return labels.reindex(obs_index).to_numpy()


def _repeated_text(index: pd.Index) -> str:
    """List the values an index repeats, each once."""
    return listed(repr(v) for v in index[index.duplicated()].unique().tolist())


def _dims_text(attributions: xr.DataArray) -> str:
    """List the dims of a DataArray for a ``Got`` line."""
    return f"dims {listed(repr(name) for name in attributions.dims)}"
