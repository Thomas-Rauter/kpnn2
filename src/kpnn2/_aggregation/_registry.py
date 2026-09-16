"""Registry for node-attribution aggregation methods."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from .._errors import Kpnn2Error

_VALID_STATUSES = frozenset(
    {
        "recommended",
        "supported",
        "experimental",
        "deprecated",
        "removed",
    }
)
_STATUS_ORDER = {
    "recommended": 0,
    "supported": 1,
    "experimental": 2,
    "deprecated": 3,
    "removed": 4,
}

AggregationFn = Callable[..., "object"]


@dataclass(frozen=True)
class AggregationMethod:
    """One registered aggregation method."""

    name: str
    status: str
    description: str
    references: tuple[str, ...]
    added_in: str
    deprecated_in: str | None
    removed_in: str | None
    replacement: str | None
    deprecation_message: str | None
    func: AggregationFn | None


_REGISTRY: dict[str, AggregationMethod] = {}


def register_aggregation_method(
    name: str,
    status: str,
    description: str,
    references: Sequence[str],
    added_in: str,
    deprecated_in: str | None = None,
    removed_in: str | None = None,
    replacement: str | None = None,
    deprecation_message: str | None = None,
) -> Callable[[AggregationFn], AggregationFn]:
    """
    Register an aggregation method under ``name``.

    Adding a method is writing one function with the shared
    ``(attributions, labels, **kwargs)`` signature, decorating
    it, and importing the module from
    ``kpnn2._aggregation._methods``. The dispatcher does not
    change.
    """
    if status not in _VALID_STATUSES:
        allowed = ", ".join(sorted(_VALID_STATUSES))
        raise ValueError(
            f"Unknown aggregation status {status!r}. Valid statuses: {allowed}."
        )
    if status in {"deprecated", "removed"} and not replacement:
        raise ValueError(
            f"Aggregation method {name!r} with status "
            f"{status!r} requires 'replacement'."
        )
    if status == "deprecated" and not deprecated_in:
        raise ValueError(
            f"Aggregation method {name!r} with status "
            "'deprecated' requires 'deprecated_in'."
        )
    if status == "removed" and not removed_in:
        raise ValueError(
            f"Aggregation method {name!r} with status "
            "'removed' requires 'removed_in'."
        )

    def decorator(func: AggregationFn) -> AggregationFn:
        if name in _REGISTRY:
            raise ValueError(
                f"Aggregation method {name!r} is already registered."
            )
        _REGISTRY[name] = AggregationMethod(
            name=name,
            status=status,
            description=description,
            references=tuple(references),
            added_in=added_in,
            deprecated_in=deprecated_in,
            removed_in=removed_in,
            replacement=replacement,
            deprecation_message=deprecation_message,
            func=None if status == "removed" else func,
        )
        return func

    return decorator


def unregister_aggregation_method(name: str) -> None:
    """Drop a registry entry. Tests use this to undo a dummy method."""
    _REGISTRY.pop(name, None)


def lookup_aggregation_method(name: str) -> AggregationMethod:
    """Return the registry entry for ``name``."""
    entry = _REGISTRY.get(name)
    if entry is None:
        available = callable_method_names()
        if available:
            listed = ", ".join(available)
        else:
            listed = "(none)"
        raise Kpnn2Error(
            f"Unknown aggregation method {name!r}. Available methods: {listed}."
        )
    return entry


def callable_method_names() -> list[str]:
    """Return registered names that can still be called, sorted."""
    return sorted(
        name for name, entry in _REGISTRY.items() if entry.status != "removed"
    )


def list_aggregation_methods() -> pd.DataFrame:
    """
    Return one row per registered aggregation method.

    The table includes methods whose status is ``removed``, so
    callers can see the replacement. It is not the list used
    when an unknown ``method`` name is rejected; that lists
    only callable methods.

    Returns
    -------
    pandas.DataFrame
        Columns ``name``, ``status``, ``description``,
        ``references``, ``added_in``, ``deprecated_in``,
        ``removed_in``, and ``replacement``. ``references``
        is a semicolon-separated string. Version columns
        that do not apply are empty. Row order is status
        (recommended first, then supported, experimental,
        deprecated, removed), then name.

    See Also
    --------
    aggregate_node_attribution : Dispatches to a registered
        method.
    """
    rows = sorted(
        _REGISTRY.values(),
        key=lambda entry: (
            _STATUS_ORDER.get(entry.status, 99),
            entry.name,
        ),
    )
    return pd.DataFrame(
        {
            "name": [entry.name for entry in rows],
            "status": [entry.status for entry in rows],
            "description": [entry.description for entry in rows],
            "references": ["; ".join(entry.references) for entry in rows],
            "added_in": [entry.added_in for entry in rows],
            "deprecated_in": [entry.deprecated_in or "" for entry in rows],
            "removed_in": [entry.removed_in or "" for entry in rows],
            "replacement": [entry.replacement or "" for entry in rows],
        }
    )
