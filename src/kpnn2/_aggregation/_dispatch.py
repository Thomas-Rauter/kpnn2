"""Dispatch mapped attributions to a registered aggregation method."""

from __future__ import annotations

import inspect
import json
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


def aggregate_node_attributions(
    attributions: xr.DataArray,
    labels: object | None = None,
    *,
    method: str = _DEFAULT_METHOD,
    **method_kwargs: object,
) -> xr.Dataset:
    """
    Reduce mapped node attributions with a registered method.

    ``map_node_attributions`` names a tensor; this call folds
    it according to ``method``. The dispatcher looks up the
    name, applies the method's status warnings, calls the
    registered function, and stores the method name, the
    parameters actually used, and the ``kpnn2`` version on
    the result. Adding a method does not change this
    function.

    Parameters
    ----------
    attributions : xarray.DataArray
        Output of ``map_node_attributions``, or an array with
        the same named dims. Required dims depend on the
        method. The array is read, never modified.
    labels : 1-d array or pandas.Series, optional
        Forwarded to the method. Some methods require it.
    method : str, optional
        Registered method name. Default
        ``"rauter_mangano_2026"``. See
        ``list_aggregation_methods()``.
    **method_kwargs
        Forwarded to the method.

    Returns
    -------
    xarray.Dataset
        The method's per-node result, with attributes
        ``method``, ``method_params``, and ``kpnn2_version``
        added here. ``method_params`` is a JSON string of
        the parameters actually used, defaults included;
        read it with ``json.loads``. Every attribute is a
        string, so ``.to_netcdf`` keeps them. ``.attrs`` is
        dropped by a CSV round-trip. For a table,
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
    Captum is not imported here. Method formulas, required
    dims, and extra kwargs are documented on each method's
    reference page.

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
    >>> out = kpnn2.aggregate_node_attributions(
    ...     da,
    ...     labels=np.array([0, 1]),
    ...     class_0=0,
    ...     class_1=1,
    ... )
    >>> out["score"].values.tolist()
    [1.0, 2.0]
    >>> out.attrs["method"]
    'rauter_mangano_2026'
    >>> import json
    >>> json.loads(out.attrs["method_params"])
    {'class_0': 0, 'class_1': 1}
    """
    if not isinstance(attributions, xr.DataArray):
        raise Kpnn2Error("'attributions' must be an xarray.DataArray.")
    if not isinstance(method, str):
        raise Kpnn2Error("'method' must be a str.")
    entry = lookup_aggregation_method(method)
    if entry.func is None:
        raise Kpnn2Error(_removed_message(entry))
    _emit_status_signal(entry)
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
    stamped.attrs["method"] = method
    stamped.attrs["method_params"] = params
    stamped.attrs["kpnn2_version"] = _kpnn2_version()
    return stamped


def _emit_status_signal(entry: AggregationMethod) -> None:
    """Warn for an experimental or deprecated method."""
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


def _removed_message(entry: AggregationMethod) -> str:
    """Build the error for a removed method."""
    return (
        f"Aggregation method {entry.name!r} was removed in "
        f"kpnn2 {entry.removed_in}. Use {entry.replacement!r} "
        "instead."
    )


def _bound_method_params(
    func: Callable[..., Any],
    attributions: xr.DataArray,
    labels: object | None,
    method_kwargs: dict[str, object],
) -> str:
    """Bind kwargs and defaults as JSON, without the data arguments."""
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
    # The first two parameters receive the data, whatever their names.
    data_names = set(list(signature.parameters)[:2])
    params = {
        key: value
        for key, value in bound.arguments.items()
        if key not in data_names
    }
    return json.dumps(
        params,
        default=_json_value,
    )


def _json_value(value: object) -> object:
    """Encode a method parameter that ``json`` cannot."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def _kpnn2_version() -> str:
    """Read ``kpnn2.__version__`` after the package is imported."""
    from kpnn2 import __version__

    return __version__
