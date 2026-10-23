"""Dispatch mapped attributions to a registered aggregation method."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from .._errors import (
    Kpnn2Error,
    caller_error,
    close_matches,
    internal_error,
    listed,
    warn_at_caller,
)
from .._validate import describe
from ._registry import (
    AggregationMethod,
    lookup_aggregation_method,
)

_DEFAULT_METHOD = "rauter_mangano_2026"
_VARIADIC = (
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
)


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
        ``"rauter_mangano_2026"``, which is not yet
        implemented and raises ``Kpnn2Error``. See
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
        unknown (the message lists callable methods),
        ``removed``, or not yet implemented; ``labels`` or
        method kwargs are invalid for the selected method; or
        the DataArray dims do not match that method.

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
    reference page. The default method is not yet
    implemented.
    """
    if not isinstance(attributions, xr.DataArray):
        raise caller_error(
            "'attributions' must be an xarray.DataArray.",
            got=describe(attributions),
            fix="Name the scores with map_node_attributions() first.",
        )
    if not isinstance(method, str):
        raise caller_error(
            "'method' must be a str.",
            got=describe(method),
            fix="See list_aggregation_methods() for the names.",
        )
    entry = lookup_aggregation_method(method)
    if entry.func is None:
        raise caller_error(
            f"Aggregation method {entry.name!r} was removed in kpnn2 "
            f"{entry.removed_in}.",
            fix=f"Pass method={entry.replacement!r} instead.",
        )
    _emit_status_signal(entry)
    params = _bound_method_params(
        method,
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
        raise internal_error(
            f"aggregation method {method!r} returned "
            f"{type(result).__name__}, not an xarray.Dataset"
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
        warn_at_caller(
            f"Aggregation method {entry.name!r} is experimental; results "
            "may change.",
            UserWarning,
            fix="Pass a recommended or supported method from "
            "list_aggregation_methods() to stop this warning.",
        )
        return
    if status == "deprecated":
        # A method's own deprecation message ends the headline as its
        # reason, so the headline still names the method and version.
        headline = (
            f"Aggregation method {entry.name!r} was deprecated in kpnn2 "
            f"{entry.deprecated_in}."
        )
        if entry.deprecation_message:
            headline = f"{headline} {entry.deprecation_message}"
        warn_at_caller(
            headline,
            FutureWarning,
            fix=f"Pass method={entry.replacement!r} instead.",
        )


def _bound_method_params(
    method: str,
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
        raise _keyword_error(
            method,
            signature,
            attributions,
            labels,
            method_kwargs,
        ) from exc
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


def _keyword_error(
    method: str,
    signature: inspect.Signature,
    attributions: xr.DataArray,
    labels: object | None,
    method_kwargs: dict[str, object],
) -> Kpnn2Error:
    """Name the keywords that do not bind and the required ones left out."""
    unexpected = []
    bindable = {}
    for key, value in method_kwargs.items():
        # One keyword at a time, so inspect decides which ones bind.
        try:
            signature.bind_partial(
                attributions,
                labels,
                **{key: value},
            )
        except TypeError:
            unexpected.append(key)
        else:
            bindable[key] = value
    bound = signature.bind_partial(
        attributions,
        labels,
        **bindable,
    )
    missing = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.default is parameter.empty
        and parameter.kind not in _VARIADIC
        and parameter.name not in bound.arguments
    ]
    groups = []
    if unexpected:
        groups.append(f"unexpected {listed(repr(key) for key in unexpected)}")
    if missing:
        groups.append(f"missing {listed(repr(name) for name in missing)}")
    accepted = _keyword_arguments(signature)
    return caller_error(
        f"Keyword arguments do not match aggregation method {method!r}.",
        got="; ".join(groups),
        expected=(
            f"keyword arguments {listed(repr(name) for name in accepted)}"
            if accepted
            else "no keyword arguments"
        ),
        candidates=close_matches(
            unexpected,
            accepted,
        ),
    )


def _keyword_arguments(signature: inspect.Signature) -> list[str]:
    """Name the keyword arguments a method takes after the data two."""
    return [
        parameter.name
        for parameter in list(signature.parameters.values())[2:]
        if parameter.kind not in _VARIADIC
    ]


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
