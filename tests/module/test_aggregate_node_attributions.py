import json

import numpy as np
import pytest
import xarray as xr

from kpnn2 import (
    Kpnn2Error,
    __version__,
    aggregate_node_attributions,
    list_aggregation_methods,
)
from kpnn2._aggregation._registry import (
    register_aggregation_method,
    unregister_aggregation_method,
)


def _da(data, *, nodes=None, layer=None):
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 2:
        dims = ("observation", "node")
        n_nodes = array.shape[1]
        coords = {
            "node": nodes or [f"n{i}" for i in range(n_nodes)],
        }
    elif array.ndim == 3:
        dims = ("seed", "observation", "node")
        n_nodes = array.shape[2]
        coords = {
            "seed": np.arange(array.shape[0]),
            "node": nodes or [f"n{i}" for i in range(n_nodes)],
        }
    else:
        raise AssertionError(array.ndim)
    da = xr.DataArray(
        array,
        dims=dims,
        coords=coords,
    )
    if layer is not None:
        da = da.assign_coords(layer=layer)
    return da


@pytest.fixture
def register_dummy():
    """Register test methods and always drop them afterwards."""
    names = []

    def register(name, *, status="supported", **options):
        names.append(name)
        return register_aggregation_method(
            name=name,
            status=status,
            description=f"Test method {name}.",
            references=(),
            added_in="0.0.0",
            **options,
        )

    yield register
    for name in names:
        unregister_aggregation_method(name)


def _mean_dataset(attributions, labels):
    del labels
    return xr.Dataset({"score": attributions.mean("observation")})


@pytest.mark.filterwarnings("error")
def test_default_method_is_not_yet_implemented():
    da = _da(
        [[1.0], [2.0]],
        nodes=["n"],
    )
    da.attrs["origin"] = "test"
    before = da.copy(deep=True)
    with pytest.raises(
        Kpnn2Error,
        match="not yet implemented",
    ):
        aggregate_node_attributions(
            da,
            np.array([0, 1]),
            class_0=0,
            class_1=1,
        )
    xr.testing.assert_identical(da, before)


def test_method_signature_is_checked_at_registration(register_dummy):
    def keyword_labels(attributions, *, labels=None):
        return xr.Dataset()

    def star_args(attributions, labels, *extra):
        return xr.Dataset()

    for bad in (keyword_labels, star_args):
        with pytest.raises(ValueError, match="Aggregation method"):
            register_dummy(f"_dummy_{bad.__name__}")(bad)


def test_deprecated_method_emits_future_warning(register_dummy):
    name = "_dummy_deprecated"
    register_dummy(
        name,
        status="deprecated",
        deprecated_in="0.2.0",
        replacement="rauter_mangano_2026",
    )(_mean_dataset)
    with pytest.warns(FutureWarning, match="deprecated in"):
        aggregate_node_attributions(
            _da([[1.0], [0.0]], nodes=["n"]),
            np.array([0, 1]),
            method=name,
        )


def test_removed_method_raises_and_names_replacement(register_dummy):
    name = "_dummy_removed"

    @register_dummy(
        name,
        status="removed",
        removed_in="0.2.0",
        replacement="rauter_mangano_2026",
    )
    def _never_called(attributions, labels, **kwargs):
        raise AssertionError("removed method must not run")

    with pytest.raises(
        Kpnn2Error,
        match="Use 'rauter_mangano_2026' instead",
    ):
        aggregate_node_attributions(
            _da([[1.0], [0.0]], nodes=["n"]),
            np.array([0, 1]),
            method=name,
        )
    table = list_aggregation_methods()
    assert name in set(table["name"])


def test_unknown_method_lists_available_names():
    da = _da(
        [[1.0], [0.0]],
        nodes=["n"],
    )
    with pytest.raises(
        Kpnn2Error,
        match="Available methods: .*rauter_mangano_2026",
    ):
        aggregate_node_attributions(
            da,
            np.array([0, 1]),
            method="not_a_method",
        )


def test_registered_dummy_is_dispatched_and_listed(register_dummy):
    name = "_dummy_sum"

    @register_dummy(name)
    def dummy(attributions, labels, *, scale=1.0):
        del labels
        score = scale * attributions.mean("observation")
        return xr.Dataset({"score": score})

    out = aggregate_node_attributions(
        _da([[1.0, 3.0], [2.0, 4.0]], nodes=["a", "b"]),
        np.array([0, 1]),
        method=name,
        scale=2.0,
    )
    np.testing.assert_allclose(
        out["score"].values,
        [3.0, 7.0],
    )
    table = list_aggregation_methods()
    row = table[table["name"] == name].iloc[0]
    assert row["status"] == "supported"
    assert row["added_in"] == "0.0.0"
    assert out.attrs["method"] == name
    assert json.loads(out.attrs["method_params"]) == {"scale": 2.0}
    assert out.attrs["kpnn2_version"] == __version__


def test_method_must_return_a_dataset(register_dummy):
    name = "_dummy_array"

    @register_dummy(name)
    def dummy(attributions, labels):
        del labels
        return attributions.mean("observation")

    with pytest.raises(
        Kpnn2Error,
        match="must return an xarray.Dataset",
    ):
        aggregate_node_attributions(
            _da([[1.0], [0.0]], nodes=["n"]),
            np.array([0, 1]),
            method=name,
        )


def test_experimental_method_emits_user_warning(register_dummy):
    name = "_dummy_experimental"
    register_dummy(name, status="experimental")(_mean_dataset)
    with pytest.warns(UserWarning, match="experimental"):
        aggregate_node_attributions(
            _da([[1.0], [0.0]], nodes=["n"]),
            np.array([0, 1]),
            method=name,
        )


def test_list_aggregation_methods_includes_default():
    table = list_aggregation_methods()
    assert list(table.columns) == [
        "name",
        "status",
        "description",
        "references",
        "added_in",
        "deprecated_in",
        "removed_in",
        "replacement",
    ]
    names = set(table["name"])
    assert "rauter_mangano_2026" in names
    recommended = table[table["status"] == "recommended"]
    assert list(recommended["name"]) == ["rauter_mangano_2026"]
    row = recommended.iloc[0]
    assert row["description"] == "Not yet implemented."
    assert row["references"] == "Rauter and Mangano, 2026"


def test_rejects_non_dataarray():
    with pytest.raises(
        Kpnn2Error,
        match="xarray.DataArray",
    ):
        aggregate_node_attributions(
            np.array([[1.0], [0.0]]),
            np.array([0, 1]),
        )


def test_netcdf_round_trip_keeps_attrs(tmp_path, register_dummy):
    name = "_dummy_netcdf"

    @register_dummy(name)
    def dummy(attributions, labels):
        del labels
        return xr.Dataset(
            {"score": attributions.mean("observation")},
        )

    out = aggregate_node_attributions(
        _da(
            [[1.0], [2.0]],
            nodes=["n"],
        ),
        np.array([0, 1]),
        method=name,
    )
    path = tmp_path / "scores.nc"
    out.to_netcdf(path)
    with xr.open_dataset(path) as loaded:
        assert loaded.attrs["method"] == name
        assert loaded.attrs["kpnn2_version"] == __version__
        assert "method_params" in loaded.attrs


def test_method_params_skip_data_arguments_by_position(register_dummy):
    name = "_dummy_renamed"
    marker = object()

    @register_dummy(name)
    def dummy(data, y, *, weights=None, token=None):
        del y, weights, token
        result = xr.Dataset({"score": data.mean("observation")})
        result.attrs["note"] = "kept"
        return result

    out = aggregate_node_attributions(
        _da([[1.0], [0.0]], nodes=["n"]),
        np.array([0, 1]),
        method=name,
        weights=np.array([1, 2]),
        token=marker,
    )
    assert json.loads(out.attrs["method_params"]) == {
        "weights": [1, 2],
        "token": repr(marker),
    }
    assert out.attrs["note"] == "kept"


def test_deprecation_message_requires_deprecated_status(register_dummy):
    with pytest.raises(
        ValueError,
        match="not 'deprecated'",
    ):
        register_dummy(
            "_dummy_bad_message",
            deprecation_message="Not deprecated.",
        )


def test_method_name_must_be_a_string():
    with pytest.raises(Kpnn2Error, match="'method' must be a str"):
        aggregate_node_attributions(
            _da([[1.0], [2.0]], nodes=["n"]),
            np.array([0, 1]),
            method=None,
        )
