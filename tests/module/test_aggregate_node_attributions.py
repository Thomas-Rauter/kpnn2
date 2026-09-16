import json

import numpy as np
import pandas as pd
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


def _agg(da, labels, **kwargs):
    return aggregate_node_attributions(
        da,
        labels,
        class_0=kwargs.pop("class_0", 0),
        class_1=kwargs.pop("class_1", 1),
        **kwargs,
    )


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


def _paper_scores(attributions, labels):
    """Loop form of the paper algorithm on a (seed, obs, node) array."""
    n_seeds, _, n_nodes = attributions.shape
    scores = np.zeros((n_seeds, n_nodes))
    for s in range(n_seeds):
        for i in range(n_nodes):
            mean0 = attributions[s, labels == 0, i].mean()
            mean1 = attributions[s, labels == 1, i].mean()
            if abs(mean1) >= abs(mean0):
                scores[s, i] = mean1 - mean0
            else:
                scores[s, i] = mean0 - mean1
    return scores.mean(axis=0)


@pytest.mark.parametrize(
    ("mean0", "mean1", "expected"),
    [
        (1.0, 5.0, 4.0),
        (5.0, 1.0, 4.0),
        (-5.0, -1.0, -4.0),
        (-1.0, -5.0, -4.0),
        (2.0, -1.0, 3.0),
        (1.0, -3.0, -4.0),
        (-3.0, 5.0, 8.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_score_is_winner_minus_loser(mean0, mean1, expected):
    data = np.array(
        [
            [mean0 - 1.0],
            [mean0 + 1.0],
            [mean1 - 2.0],
            [mean1 + 2.0],
        ]
    )
    out = _agg(
        _da(data, nodes=["x"]),
        np.array([0, 0, 1, 1]),
    )
    assert out["score"].item() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("mean0", "mean1", "expected"),
    [
        (-1.0, 1.0, 2.0),
        (1.0, -1.0, -2.0),
    ],
)
def test_tie_goes_to_class_1(mean0, mean1, expected):
    out = _agg(
        _da([[mean0], [mean1]], nodes=["tied"]),
        np.array([0, 1]),
    )
    assert out["score"].item() == pytest.approx(expected)


def test_score_matches_loop_form_of_the_algorithm():
    np.random.seed(42)
    attributions = np.random.normal(size=(4, 9, 6))
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1])
    out = _agg(
        _da(attributions),
        labels,
    )
    np.testing.assert_allclose(
        out["score"].values,
        _paper_scores(attributions, labels),
    )


def test_seed_scores_are_averaged():
    # Seed scores: 4 - 0 = 4, 3 - 1 = 2, -3 - 0 = -3 → mean 1
    data = np.stack(
        [
            np.array([[0.0], [4.0]]),
            np.array([[3.0], [1.0]]),
            np.array([[0.0], [-3.0]]),
        ],
        axis=0,
    )
    out = _agg(
        _da(data, nodes=["n"]),
        np.array([0, 1]),
    )
    assert out["score"].item() == pytest.approx(1.0)


def test_returns_only_the_score():
    out = _agg(
        _da([[1.0], [2.0]], nodes=["n"]),
        np.array([0, 1]),
    )
    assert list(out.data_vars) == ["score"]
    assert out["score"].dims == ("node",)


def test_nan_attribution_is_left_out_of_its_class_mean():
    data = np.array(
        [
            [1.0],
            [np.nan],
            [4.0],
        ]
    )
    out = _agg(
        _da(data, nodes=["n"]),
        np.array([0, 0, 1]),
    )
    assert out["score"].item() == pytest.approx(3.0)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("with_seed", [False, True])
def test_node_without_class_mean_scores_nan(with_seed):
    data = np.array(
        [
            [np.nan, 1.0],
            [np.nan, 2.0],
        ]
    )
    if with_seed:
        data = np.stack([data, data], axis=0)
    out = _agg(
        _da(data, nodes=["missing", "ok"]),
        np.array([0, 1]),
    )
    assert np.isnan(out.sel(node="missing")["score"].item())
    assert out.sel(node="ok")["score"].item() == pytest.approx(1.0)


def test_seed_missing_a_class_mean_is_left_out():
    # Seed 1 has no class-0 mean, so seed 0 alone decides.
    data = np.stack(
        [
            np.array([[1.0], [3.0]]),
            np.array([[np.nan], [-5.0]]),
        ],
        axis=0,
    )
    out = _agg(
        _da(data, nodes=["n"]),
        np.array([0, 1]),
    )
    assert out["score"].item() == pytest.approx(2.0)


def test_repeated_node_names_are_rejected():
    with pytest.raises(Kpnn2Error, match="repeats name"):
        _agg(
            _da([[1.0, 2.0], [3.0, 4.0]], nodes=["wide", "wide"]),
            np.array([0, 1]),
        )


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


def test_more_than_two_classes_raises():
    da = _da(
        [[1.0], [0.0], [2.0]],
        nodes=["n"],
    )
    with pytest.raises(
        Kpnn2Error,
        match="more than two classes",
    ):
        _agg(
            da,
            np.array([0, 1, 2]),
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


def test_series_labels_align_to_observation_coord():
    da = _da(
        [[1.0], [2.0], [10.0]],
        nodes=["n"],
    )
    da = da.assign_coords(observation=["s0", "s1", "s2"])
    labels = pd.Series(
        [1, 0, 0],
        index=["s2", "s0", "s1"],
    )
    out = _agg(da, labels)
    # After reindex: class 0 mean 1.5, class 1 mean 10 → 8.5.
    # Pairing in order would give class 0 mean 6, class 1 mean 1.
    assert out["score"].item() == pytest.approx(8.5)


def test_series_labels_reject_index_mismatch():
    da = _da(
        [[1.0], [0.0]],
        nodes=["n"],
    )
    da = da.assign_coords(observation=["s0", "s1"])
    labels = pd.Series(
        [0, 1],
        index=["s0", "other"],
    )
    with pytest.raises(
        Kpnn2Error,
        match="does not match",
    ):
        _agg(da, labels)


def test_unexpected_dim_is_rejected():
    da = _da(
        [[1.0], [0.0]],
        nodes=["n"],
    )
    da = da.expand_dims(step=[0])
    with pytest.raises(
        Kpnn2Error,
        match="unexpected dim",
    ):
        _agg(
            da,
            np.array([0, 1]),
        )


def test_missing_class_mapping_raises():
    da = _da(
        [[1.0], [0.0]],
        nodes=["n"],
    )
    with pytest.raises(
        Kpnn2Error,
        match="requires 'class_0' and 'class_1'",
    ):
        aggregate_node_attributions(
            da,
            np.array([0, 1]),
        )


def test_copies_scalar_layer_coordinate():
    da = _da(
        [[1.0], [2.0]],
        nodes=["n"],
        layer=2,
    )
    out = _agg(
        da,
        np.array([0, 1]),
    )
    assert int(out.coords["layer"]) == 2


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


def test_default_method_stamps_attrs():
    out = _agg(
        _da(
            [[1.0], [2.0]],
            nodes=["n"],
        ),
        np.array([0, 1]),
    )
    assert out.attrs["method"] == "rauter_mangano_2026"
    assert json.loads(out.attrs["method_params"]) == {
        "class_0": 0,
        "class_1": 1,
    }


def test_rejects_non_dataarray():
    with pytest.raises(
        Kpnn2Error,
        match="xarray.DataArray",
    ):
        aggregate_node_attributions(
            np.array([[1.0], [0.0]]),
            np.array([0, 1]),
            class_0=0,
            class_1=1,
        )


def test_netcdf_round_trip_keeps_attrs(tmp_path):
    out = _agg(
        _da(
            [[1.0], [2.0]],
            nodes=["n"],
        ),
        np.array([0, 1]),
        class_0=np.int64(0),
    )
    path = tmp_path / "scores.nc"
    out.to_netcdf(path)
    with xr.open_dataset(path) as loaded:
        assert loaded.attrs["method"] == "rauter_mangano_2026"
        assert loaded.attrs["kpnn2_version"] == __version__
        params = json.loads(loaded.attrs["method_params"])
    assert params["class_0"] == 0


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


def test_string_labels_and_list_labels():
    data = [[1.0], [3.0], [0.0]]
    out = _agg(
        _da(data, nodes=["n"]),
        ["ctrl", "case", "ctrl"],
        class_0="ctrl",
        class_1="case",
    )
    assert out["score"].item() == pytest.approx(2.5)


def test_mixed_type_labels_are_supported():
    labels = np.array([0, "case"], dtype=object)
    out = _agg(
        _da([[1.0], [3.0]], nodes=["n"]),
        labels,
        class_1="case",
    )
    assert out["score"].item() == pytest.approx(2.0)


def test_mixed_type_labels_outside_the_pair_raise():
    labels = np.array([0, "case", "other"], dtype=object)
    with pytest.raises(Kpnn2Error, match="more than two classes"):
        _agg(
            _da([[1.0], [3.0], [2.0]], nodes=["n"]),
            labels,
            class_1="case",
        )


def test_integer_attributions_are_averaged_as_floats():
    da = _da([[1.0], [2.0], [4.0]], nodes=["n"]).astype(np.int64)
    out = _agg(da, np.array([0, 0, 1]))
    assert out["score"].item() == pytest.approx(2.5)


def test_input_is_not_modified():
    da = _da([[1.0], [2.0]], nodes=["n"])
    da.attrs["origin"] = "test"
    before = da.copy(deep=True)
    _agg(da, np.array([0, 1]))
    xr.testing.assert_identical(da, before)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"class_0": 1, "class_1": 1}, "must be distinct"),
        ({"class_0": np.array([0])}, "'class_0' must be a scalar"),
        ({"class_1": [1, 2]}, "'class_1' must be a scalar"),
        ({"class_1": 2}, "class_1=2"),
        ({"unknown": 1}, "unexpected keyword argument 'unknown'"),
        (
            {"sign_reference": "per_seed"},
            "unexpected keyword argument 'sign_reference'",
        ),
    ],
)
def test_invalid_method_arguments_raise(kwargs, match):
    with pytest.raises(Kpnn2Error, match=match):
        _agg(
            _da([[1.0], [2.0]], nodes=["n"]),
            np.array([0, 1]),
            **kwargs,
        )


def test_missing_class_in_labels_raises():
    with pytest.raises(Kpnn2Error, match="class_1=1 is missing"):
        _agg(
            _da([[1.0], [2.0]], nodes=["n"]),
            np.array([0, 0]),
        )


def test_method_name_must_be_a_string():
    with pytest.raises(Kpnn2Error, match="'method' must be a str"):
        aggregate_node_attributions(
            _da([[1.0], [2.0]], nodes=["n"]),
            np.array([0, 1]),
            method=None,
        )
