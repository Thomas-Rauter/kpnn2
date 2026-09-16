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


def test_deprecated_method_emits_future_warning():
    name = "_dummy_deprecated"

    @register_aggregation_method(
        name=name,
        status="deprecated",
        description="Test deprecated method.",
        references=(),
        added_in="0.0.0",
        deprecated_in="0.2.0",
        replacement="rauter_mangano_2026",
    )
    def dummy(attributions, labels):
        del labels
        return xr.Dataset(
            {"score": attributions.mean("observation")},
        )

    da = _da(
        [[1.0], [0.0]],
        nodes=["n"],
    )
    try:
        with pytest.warns(FutureWarning, match="deprecated in"):
            aggregate_node_attributions(
                da,
                np.array([0, 1]),
                method=name,
            )
    finally:
        unregister_aggregation_method(name)


def test_removed_method_raises_and_names_replacement():
    name = "_dummy_removed"

    @register_aggregation_method(
        name=name,
        status="removed",
        description="Test removed method.",
        references=(),
        added_in="0.0.0",
        removed_in="0.2.0",
        replacement="rauter_mangano_2026",
    )
    def _never_called(attributions, labels, **kwargs):
        raise AssertionError("removed method must not run")

    da = _da(
        [[1.0], [0.0]],
        nodes=["n"],
    )
    try:
        with pytest.raises(
            Kpnn2Error,
            match="Use 'rauter_mangano_2026' instead",
        ):
            aggregate_node_attributions(
                da,
                np.array([0, 1]),
                method=name,
            )
        table = list_aggregation_methods()
        assert name in set(table["name"])
    finally:
        unregister_aggregation_method(name)


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


def test_registered_dummy_is_dispatched_and_listed():
    name = "_dummy_sum"

    @register_aggregation_method(
        name=name,
        status="supported",
        description="Test dummy sum.",
        references=("dummy",),
        added_in="0.0.0",
    )
    def dummy(attributions, labels, *, scale=1.0):
        del labels
        score = scale * attributions.mean("observation")
        return xr.Dataset({"score": score})

    da = _da(
        [[1.0, 3.0], [2.0, 4.0]],
        nodes=["a", "b"],
    )
    try:
        out = aggregate_node_attributions(
            da,
            np.array([0, 1]),
            method=name,
            scale=2.0,
        )
        np.testing.assert_allclose(
            out["score"].values,
            [3.0, 7.0],
        )
        table = list_aggregation_methods()
        assert name in set(table["name"])
        row = table[table["name"] == name].iloc[0]
        assert row["status"] == "supported"
        assert row["added_in"] == "0.0.0"
        assert out.attrs["method"] == name
        assert out.attrs["method_params"]["scale"] == 2.0
        assert out.attrs["kpnn2_version"] == __version__
    finally:
        unregister_aggregation_method(name)


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


def test_experimental_method_emits_user_warning():
    name = "_dummy_experimental"

    @register_aggregation_method(
        name=name,
        status="experimental",
        description="Test experimental method.",
        references=(),
        added_in="0.0.0",
    )
    def dummy(attributions, labels):
        del labels
        return xr.Dataset({"score": attributions.mean("observation")})

    da = _da(
        [[1.0], [0.0]],
        nodes=["n"],
    )
    try:
        with pytest.warns(UserWarning, match="experimental"):
            aggregate_node_attributions(
                da,
                np.array([0, 1]),
                method=name,
            )
    finally:
        unregister_aggregation_method(name)


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
    assert out.attrs["method_params"]["sign_reference"] == ("per_seed")


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
