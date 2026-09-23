"""Placeholder for the Rauter and Mangano 2026 method."""

from __future__ import annotations

import xarray as xr

from ..._errors import Kpnn2Error
from .._registry import register_aggregation_method

_METHOD_NAME = "rauter_mangano_2026"
_NOT_IMPLEMENTED = (
    f"Aggregation method {_METHOD_NAME!r} is not yet implemented."
)
_DESCRIPTION = "Not yet implemented."
_REFERENCES = ("Rauter and Mangano, 2026",)


@register_aggregation_method(
    name=_METHOD_NAME,
    status="recommended",
    description=_DESCRIPTION,
    references=_REFERENCES,
    added_in="0.2.0",
)
def rauter_mangano_2026(
    attributions: xr.DataArray,
    labels: object | None,
    **kwargs: object,
) -> xr.Dataset:
    """
    Raise until the Rauter and Mangano 2026 method exists.

    This function is not exported. Pass
    ``method="rauter_mangano_2026"`` to
    ``aggregate_node_attributions``. The name stays
    registered so the method can be filled in later.
    Calling it raises ``Kpnn2Error``.

    Parameters
    ----------
    attributions : xarray.DataArray
        Output of ``map_node_attributions``, or an array
        with the same named dims. Ignored.
    labels : 1-d array or pandas.Series, optional
        Forwarded by the dispatcher. Ignored.
    **kwargs
        Method parameters. Ignored.

    Returns
    -------
    xarray.Dataset
        Not returned. The call raises.

    Raises
    ------
    Kpnn2Error
        Always. The method is not yet implemented.

    See Also
    --------
    aggregate_node_attributions : Public dispatcher; pass
        ``method="rauter_mangano_2026"``.
    list_aggregation_methods : Registry table.
    """
    del attributions, labels, kwargs
    raise Kpnn2Error(_NOT_IMPLEMENTED)
