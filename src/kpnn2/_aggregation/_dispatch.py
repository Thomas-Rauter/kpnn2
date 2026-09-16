"""Dispatch mapped attributions to a registered aggregation method."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from .._errors import Kpnn2Error
from ._registry import (
    AggregationMethod,
    lookup_aggregation_method,
)

_DEFAULT_METHOD = "rauter_mangano_2026"


def aggregate_node_attribution(
    attributions: xr.DataArray,
    labels: object | None = None,
    *,
    method: str = _DEFAULT_METHOD,
    **method_kwargs: object,
) -> xr.Dataset:
    """
    Reduce mapped node attributions with a registered method.

    ``map_node_attributions`` names a tensor; this call folds
    observations (and optional seeds) according to ``method``.
    The dispatcher does not know about binary classes or seeds:
    it looks up the name, applies the method's status warnings,
    calls the registered function, and stores the method name,
    the parameters actually used, and the ``kpnn2`` version on
    the result. Adding a method does not change this function.

    The default method ``rauter_mangano_2026`` is binary
    classification only. For each seed ``s``, layer, and node
    ``i`` (a missing ``seed`` dim is one seed):

    1. ``mean_c`` is the mean attribution over observations of
       class ``c`` (``c`` in {0, 1}).
    2. ``D = mean_1 - mean_0``.
    3. ``eps = +1`` if ``|mean_1| >= |mean_0|``, else ``-1``
       (ties go to class 1).
    4. ``score = eps * |D|``.

    The magnitude is always the absolute class-mean difference.
    The sign is the class whose mean attribution is farther from
    the all-zero baseline (positive: class 1, negative: class 0),
    including nodes with ``D < 0`` that counteract class
    separation. Ranking by ``abs_score`` is independent of that
    sign convention.

    ``sign_reference`` controls folding across seeds:

    - ``"per_seed"`` (default): compute ``score`` per seed, then
      average.
    - ``"seed_mean"``: average ``mean_0`` and ``mean_1`` across
      seeds, then compute ``D``, ``eps``, and ``score`` once.
      This avoids shrinking nodes whose per-seed sign varies.

    ``mean_class0``, ``mean_class1``, and ``class_difference``
    are always seed-averaged. ``near_tie`` is true when the
    absolute class means differ by less than relative
    ``tie_tolerance`` (default 0.05); in those cases the sign is
    unreliable.

    Parameters
    ----------
    attributions : xarray.DataArray
        Output of ``map_node_attributions``, or an array with the
        same named dims. ``rauter_mangano_2026`` requires
        ``observation`` and ``node``, accepts optional ``seed``,
        and rejects any other dim (reduce or ``.rename`` first).
        Concatenate trained models with
        ``xr.concat(..., dim="seed")``. A scalar ``layer``
        coordinate is copied through when present. The array is
        read, never modified.
    labels : 1-d array or pandas.Series, optional
        Observation-to-class mapping aligned to ``observation``.
        A numpy array or sequence is paired in order and must be
        as long as that axis. A pandas Series is reindexed to
        the observation coordinate; missing or extra ids raise.
        Required by ``rauter_mangano_2026``. Methods that do not
        use labels leave this as ``None``.
    method : str, optional
        Registered method name. Default
        ``"rauter_mangano_2026"``. See
        ``list_aggregation_methods()``.
    **method_kwargs
        Forwarded to the method. ``rauter_mangano_2026`` requires
        ``class_0`` and ``class_1`` (the label values for class 0
        and class 1; pass them even when labels are already
        ``0``/``1``). Optional ``sign_reference``
        (``"per_seed"`` or ``"seed_mean"``) and
        ``tie_tolerance``.

    Returns
    -------
    xarray.Dataset
        One value per node (the same ``node`` axis as the input,
        including a repeated name when a node is wider than 1).
        Variables: ``score``, ``abs_score``, ``mean_class0``,
        ``mean_class1``, ``class_difference``, ``sign`` (+1 or
        -1; ``score == 0`` is +1), ``n_seeds``,
        ``sign_consistency`` (fraction of seeds whose per-seed
        ``eps`` matches the final sign), ``counteracting``
        (seed-averaged ``D < 0``), ``near_tie``. Attributes
        ``method``, ``method_params``, and ``kpnn2_version``.
        ``.attrs`` is dropped by a CSV round-trip. For a table,
        ``.to_dataframe().reset_index()``.

    Raises
    ------
    Kpnn2Error
        If ``attributions`` is not a DataArray; ``method`` is
        unknown (the message lists callable methods) or
        ``removed``; ``labels`` or method kwargs are invalid
        for the selected method; or the DataArray dims do not
        match that method.

    Warns
    -----
    UserWarning
        If ``method`` is ``experimental``: results may change.
    FutureWarning
        If ``method`` is ``deprecated``: names the version and
        the replacement.

    See Also
    --------
    map_node_attributions : Names the node axis of a layer
        tensor; call that first.
    list_aggregation_methods : Registry table (name, status,
        references, versions).

    Notes
    -----
    Captum is not imported here. Sample labels (which row is
    class 0 or 1) are not Captum's ``target=`` (which output
    class was explained). If a Captum ``class`` dim is still on
    the array, select one class before calling.

    The deprecated method ``rauter_mangano_2026_legacy`` uses
    ``score = eps * D``. It has the same magnitude and a
    reversed sign when ``D < 0``. Keep it only to reproduce
    earlier results.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import kpnn2
    >>> da = xr.DataArray(
    ...     [[1.0, 0.0], [0.0, 2.0]],
    ...     dims=("observation", "node"),
    ...     coords={"node": ["A", "B"]},
    ... )
    >>> out = kpnn2.aggregate_node_attribution(
    ...     da,
    ...     labels=np.array([0, 1]),
    ...     class_0=0,
    ...     class_1=1,
    ... )
    >>> out["score"].values.tolist()
    [-1.0, 2.0]
    >>> out.attrs["method"]
    'rauter_mangano_2026'
    """
    if not isinstance(attributions, xr.DataArray):
        raise Kpnn2Error("'attributions' must be an xarray.DataArray.")
    if not isinstance(method, str):
        raise Kpnn2Error("'method' must be a str.")
    entry = lookup_aggregation_method(method)
    _emit_status_signal(entry)
    if entry.func is None:
        raise Kpnn2Error(_removed_message(entry))
    params = _bound_method_params(
        entry.func,
        attributions,
        labels,
        method_kwargs,
    )
    result = entry.func(
        attributions,
        labels,
        **method_kwargs,
    )
    if not isinstance(result, xr.Dataset):
        raise Kpnn2Error(
            f"Aggregation method {method!r} must return an xarray.Dataset."
        )
    stamped = result.copy(deep=False)
    stamped.attrs = dict(result.attrs)
    stamped.attrs["method"] = method
    stamped.attrs["method_params"] = params
    stamped.attrs["kpnn2_version"] = _kpnn2_version()
    return stamped


def _emit_status_signal(entry: AggregationMethod) -> None:
    """Warn or reject according to the registry status."""
    status = entry.status
    if status == "experimental":
        warnings.warn(
            f"Aggregation method {entry.name!r} is "
            "experimental; results may change.",
            UserWarning,
            stacklevel=3,
        )
        return
    if status == "deprecated":
        if entry.deprecation_message:
            message = entry.deprecation_message
        else:
            version = entry.deprecated_in
            replacement = entry.replacement
            message = (
                f"Aggregation method {entry.name!r} was "
                f"deprecated in kpnn2 {version}. Use "
                f"{replacement!r} instead."
            )
        warnings.warn(
            message,
            FutureWarning,
            stacklevel=3,
        )
        return
    if status == "removed":
        raise Kpnn2Error(_removed_message(entry))


def _removed_message(entry: AggregationMethod) -> str:
    """Build the error for a removed method."""
    message = (
        f"Aggregation method {entry.name!r} was removed in "
        f"kpnn2 {entry.removed_in}."
    )
    if entry.replacement:
        message += f" Use {entry.replacement!r} instead."
    return message


def _bound_method_params(
    func: Callable[..., Any],
    attributions: xr.DataArray,
    labels: object | None,
    method_kwargs: dict[str, object],
) -> dict[str, object]:
    """Bind kwargs and defaults, dropping the data arguments."""
    signature = inspect.signature(func)
    try:
        bound = signature.bind(
            attributions,
            labels,
            **method_kwargs,
        )
    except TypeError as exc:
        raise Kpnn2Error(str(exc)) from exc
    bound.apply_defaults()
    params: dict[str, object] = {}
    for key, value in bound.arguments.items():
        if key in {"attributions", "labels"}:
            continue
        params[key] = _attr_value(value)
    return params


def _attr_value(value: object) -> object:
    """Store a method parameter in Dataset attrs."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _kpnn2_version() -> str:
    """Read ``kpnn2.__version__`` after the package is imported."""
    from kpnn2 import __version__

    return __version__
