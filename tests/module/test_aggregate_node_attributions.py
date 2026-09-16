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


def test_toy_linear_abs_score_matches_weight_times_mean_gap():
    weights = np.array([1.5, -2.0, 0.5])
    mu0 = np.array([1.0, 2.0, 0.0])
    mu1 = np.array([3.0, 2.0, 4.0])
    n_per_class = 4
    x0 = np.broadcast_to(mu0, (n_per_class, 3))
    x1 = np.broadcast_to(mu1, (n_per_class, 3))
    features = np.vstack([x0, x1])
    attributions = features * weights
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    out = _agg(
        _da(attributions, nodes=["a", "b", "c"]),
        labels,
    )
    expected = np.abs(weights * (mu1 - mu0))
    np.testing.assert_allclose(
        out["abs_score"].values,
        expected,
    )


def test_negative_d_sign_is_the_class_farther_from_zero():
    # mean0=2, mean1=-1 → D=-3, |m0| > |m1|, eps=-1
    data = np.array(
        [
            [2.0],
            [2.0],
            [-1.0],
            [-1.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    out = _agg(
        _da(
            data,
            nodes=["x"],
        ),
        labels,
    )
    assert out["score"].item() == pytest.approx(-3.0)
    assert out["sign"].item() == -1
    assert bool(out["counteracting"].item()) is True


def test_tie_goes_to_class_1_and_near_tie_is_flagged():
    data = np.array(
        [
            [-1.0],
            [1.0],
        ]
    )
    out = _agg(
        _da(data, nodes=["tied"]),
        np.array([0, 1]),
    )
    assert out["sign"].item() == 1
    assert out["score"].item() == pytest.approx(2.0)
    assert bool(out["near_tie"].item()) is True


def test_near_tie_relative_tolerance():
    data = np.array(
        [
            [1.0],
            [1.03],
        ]
    )
    out = _agg(
        _da(data, nodes=["close"]),
        np.array([0, 1]),
        tie_tolerance=0.05,
    )
    assert bool(out["near_tie"].item()) is True
    far = np.array(
        [
            [1.0],
            [2.0],
        ]
    )
    out_far = _agg(
        _da(far, nodes=["far"]),
        np.array([0, 1]),
        tie_tolerance=0.05,
    )
    assert bool(out_far["near_tie"].item()) is False


def test_per_seed_and_seed_mean_differ_when_scores_vary():
    # Seed 0: m0=0, m1=4 → score = 4
    # Seed 1: m0=0, m1=-2 → score = 2
    # per_seed mean = 3; seed_mean |D| = 1
    seed0 = np.array(
        [
            [0.0],
            [4.0],
        ]
    )
    seed1 = np.array(
        [
            [0.0],
            [-2.0],
        ]
    )
    data = np.stack([seed0, seed1], axis=0)
    labels = np.array([0, 1])
    da = _da(
        data,
        nodes=["n"],
    )
    per_seed = _agg(
        da,
        labels,
        sign_reference="per_seed",
    )
    seed_mean = _agg(
        da,
        labels,
        sign_reference="seed_mean",
    )
    assert per_seed["score"].item() == pytest.approx(3.0)
    assert seed_mean["score"].item() == pytest.approx(1.0)
    assert int(per_seed["n_seeds"].item()) == 2
    assert seed_mean["sign_consistency"].item() == pytest.approx(1.0)


def test_per_seed_sign_consistency_counts_disagreeing_seeds():
    # eps per seed: +1, -1, +1 → mean score (4 - 2 + 3) / 3 = 5/3
    data = np.stack(
        [
            np.array([[0.0], [4.0]]),
            np.array([[3.0], [1.0]]),
            np.array([[0.0], [3.0]]),
        ],
        axis=0,
    )
    out = _agg(
        _da(data, nodes=["n"]),
        np.array([0, 1]),
    )
    assert out["score"].item() == pytest.approx(5.0 / 3.0)
    assert out["sign"].item() == 1
    assert out["sign_consistency"].item() == pytest.approx(2.0 / 3.0)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("sign_reference", ["per_seed", "seed_mean"])
def test_node_without_class_mean_is_nan_with_neutral_flags(sign_reference):
    data = np.array(
        [
            [np.nan, 1.0],
            [np.nan, 2.0],
        ]
    )
    out = _agg(
        _da(data, nodes=["missing", "ok"]),
        np.array([0, 1]),
        sign_reference=sign_reference,
    )
    missing = out.sel(node="missing")
    for name in (
        "score",
        "abs_score",
        "mean_class0",
        "mean_class1",
        "class_difference",
        "sign_consistency",
    ):
        assert np.isnan(missing[name].item()), name
    assert missing["sign"].item() == 0
    assert out["sign"].dtype == np.int64
    assert bool(missing["counteracting"].item()) is False
    assert bool(missing["near_tie"].item()) is False
    ok = out.sel(node="ok")
    assert ok["score"].item() == pytest.approx(1.0)
    assert ok["sign"].item() == 1
    assert ok["sign_consistency"].item() == pytest.approx(1.0)


@pytest.mark.parametrize("sign_reference", ["per_seed", "seed_mean"])
def test_seed_missing_a_class_mean_is_left_out(sign_reference):
    # Seed 1 has no class-0 mean, so its class-1 value -5 is
    # dropped too and seed 0 alone decides every output.
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
        sign_reference=sign_reference,
    )
    assert out["score"].item() == pytest.approx(2.0)
    assert out["mean_class0"].item() == pytest.approx(1.0)
    assert out["mean_class1"].item() == pytest.approx(3.0)
    assert out["sign_consistency"].item() == pytest.approx(1.0)
    assert int(out["n_seeds"].item()) == 2


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
        [[1.0], [0.0]],
        nodes=["n"],
    )
    da = da.assign_coords(observation=["s0", "s1"])
    labels = pd.Series(
        [1, 0],
        index=["s1", "s0"],
    )
    out = _agg(da, labels)
    # After reindex, s0 is class 0 (attr 1), s1 is class 1 (attr 0)
    assert out["mean_class0"].item() == pytest.approx(1.0)
    assert out["mean_class1"].item() == pytest.approx(0.0)


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


def test_no_seed_dim_sets_n_seeds_one():
    out = _agg(
        _da(
            [[1.0], [2.0]],
            nodes=["n"],
        ),
        np.array([0, 1]),
    )
    assert int(out["n_seeds"].item()) == 1
    assert out["sign_consistency"].item() == pytest.approx(1.0)
    assert out.attrs["method"] == "rauter_mangano_2026"
    assert json.loads(out.attrs["method_params"]) == {
        "class_0": 0,
        "class_1": 1,
        "sign_reference": "per_seed",
        "tie_tolerance": 0.05,
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
    assert params["sign_reference"] == "per_seed"


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
    assert out["mean_class0"].item() == pytest.approx(0.5)
    assert out["mean_class1"].item() == pytest.approx(3.0)


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
    assert out["mean_class0"].item() == pytest.approx(1.5)
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
        ({"sign_reference": "median"}, "'sign_reference' must be"),
        ({"tie_tolerance": "0.1"}, "must be a real number"),
        ({"tie_tolerance": True}, "must be a real number"),
        ({"tie_tolerance": -0.1}, "finite and >= 0"),
        ({"tie_tolerance": float("nan")}, "finite and >= 0"),
        ({"tie_tolerance": float("inf")}, "finite and >= 0"),
        ({"unknown": 1}, "unexpected keyword argument 'unknown'"),
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


@pytest.mark.parametrize(
    "tie_tolerance",
    [np.float32(0.1), np.int64(0), 0],
)
def test_numpy_and_int_tie_tolerance_are_accepted(tie_tolerance):
    out = _agg(
        _da([[1.0], [2.0]], nodes=["n"]),
        np.array([0, 1]),
        tie_tolerance=tie_tolerance,
    )
    params = json.loads(out.attrs["method_params"])
    assert params["tie_tolerance"] == pytest.approx(float(tie_tolerance))


def test_method_name_must_be_a_string():
    with pytest.raises(Kpnn2Error, match="'method' must be a str"):
        aggregate_node_attributions(
            _da([[1.0], [2.0]], nodes=["n"]),
            np.array([0, 1]),
            method=None,
        )
