import pandas as pd
import pytest
import torch
import xarray as xr

from kpnn2 import Kpnn2Error, map_node_attributions, parse_layered


def _tiny_spec():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "A", "H"],
            "target": ["H", "E", "C"],
        }
    )
    return parse_layered(edgelist)


def test_map_node_attributions_uses_layer_node_names():
    spec = _tiny_spec()
    assert spec.layer_nodes[1] == ("E", "H")
    attributions = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=torch.float32,
    )

    da = map_node_attributions(
        attributions,
        spec,
        1,
    )

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("observation", "node")
    assert da["node"].values.tolist() == ["E", "H"]
    assert int(da.coords["layer"]) == 1
    assert da.sel(node="E").values.tolist() == pytest.approx([0.1, 0.3])
    assert da.sel(node="H").values.tolist() == pytest.approx([0.2, 0.4])


def test_map_node_attributions_rejects_wrong_width():
    spec = _tiny_spec()
    attributions = torch.tensor(
        [
            [0.1, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(
        Kpnn2Error,
        match="wrong number of units",
    ):
        map_node_attributions(
            attributions,
            spec,
            1,
        )


def test_map_node_attributions_rejects_bad_layer_index():
    spec = _tiny_spec()
    attributions = torch.tensor(
        [
            [0.1],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(
        Kpnn2Error,
        match="layer",
    ):
        map_node_attributions(
            attributions,
            spec,
            3,
        )


def test_map_node_attributions_labels_extra_dims():
    spec = _tiny_spec()
    attributions = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
        ],
        dtype=torch.float32,
    )

    da = map_node_attributions(
        attributions=attributions,
        spec=spec,
        layer=1,
        dims=("observation", "class", "node"),
        coords={"class": ["neg", "pos"]},
    )

    assert da.dims == ("observation", "class", "node")
    assert list(da.coords["class"].values) == ["neg", "pos"]
    long = da.to_dataframe(name="score").reset_index()
    assert set(long.columns) == {
        "observation",
        "class",
        "node",
        "layer",
        "score",
    }
    assert long.shape[0] == 8
    row = long[
        (long["observation"] == 0)
        & (long["class"] == "pos")
        & (long["node"] == "H")
    ]
    assert float(row["score"].iloc[0]) == pytest.approx(0.4)


def test_map_node_attributions_stacks_step_tensors():
    spec = _tiny_spec()
    step0 = torch.tensor(
        [[0.1, 0.2]],
        dtype=torch.float32,
    )
    step1 = torch.tensor(
        [[0.3, 0.4]],
        dtype=torch.float32,
    )

    da = map_node_attributions(
        attributions=(step0, step1),
        spec=spec,
        layer=1,
    )

    assert da.dims == ("step", "observation", "node")
    assert list(da.coords["step"].values) == [0, 1]
    assert da.sel(
        step=1,
        observation=0,
        node="H",
    ).item() == pytest.approx(0.4)


def test_map_node_attributions_requires_dims_for_3d():
    spec = _tiny_spec()
    attributions = torch.zeros(2, 2, 2)

    with pytest.raises(
        Kpnn2Error,
        match="dims=",
    ):
        map_node_attributions(
            attributions,
            spec,
            1,
        )


def test_map_node_attributions_rejects_coords_for_node():
    spec = _tiny_spec()
    attributions = torch.zeros(1, 2)

    with pytest.raises(
        Kpnn2Error,
        match="must not include",
    ):
        map_node_attributions(
            attributions=attributions,
            spec=spec,
            layer=1,
            coords={"node": ["x", "y"]},
        )


def test_map_node_attributions_labels_1d_tensor():
    spec = _tiny_spec()
    da = map_node_attributions(
        attributions=torch.tensor(
            [0.1, 0.2],
            dtype=torch.float32,
        ),
        spec=spec,
        layer=1,
    )

    assert da.dims == ("node",)
    assert da["node"].values.tolist() == ["E", "H"]
    assert da.sel(node="H").item() == pytest.approx(0.2)


def test_map_node_attributions_stacks_1d_step_tensors():
    spec = _tiny_spec()
    da = map_node_attributions(
        attributions=(
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.3, 0.4]),
        ),
        spec=spec,
        layer=1,
    )

    assert da.dims == ("step", "node")
    assert da.sel(
        step=1,
        node="H",
    ).item() == pytest.approx(0.4)


def _invalid_map_cases():
    spec = _tiny_spec()
    scores = torch.zeros(1, 2)
    stacked_3d = (
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
    )
    return [
        pytest.param(
            scores,
            object(),
            1,
            {},
            "LayeredSpec",
            id="non_layered_spec",
        ),
        pytest.param(
            scores,
            spec,
            True,
            {},
            "must be an int",
            id="layer_is_bool",
        ),
        pytest.param(
            {"scores": scores},
            spec,
            1,
            {},
            "torch.Tensor or a sequence",
            id="not_tensor_or_sequence",
        ),
        pytest.param(
            [],
            spec,
            1,
            {},
            "must not be empty",
            id="empty_sequence",
        ),
        pytest.param(
            [scores, 1.0],
            spec,
            1,
            {},
            "Each item",
            id="sequence_item_not_tensor",
        ),
        pytest.param(
            (scores, torch.zeros(2, 2)),
            spec,
            1,
            {},
            "same shape",
            id="mismatched_shapes",
        ),
        pytest.param(
            stacked_3d,
            spec,
            1,
            {},
            "Pass dims=",
            id="stacked_3d_needs_dims",
        ),
        pytest.param(
            scores,
            spec,
            1,
            {"dims": (0, "node")},
            "sequence of strings",
            id="dims_not_strings",
        ),
        pytest.param(
            scores,
            spec,
            1,
            {"dims": ("node",)},
            "length must match",
            id="dims_wrong_length",
        ),
        pytest.param(
            scores,
            spec,
            1,
            {"dims": ("layer", "node")},
            "must not include 'layer'",
            id="dims_include_layer",
        ),
        pytest.param(
            scores,
            spec,
            1,
            {"dims": ("node", "node")},
            "must be unique",
            id="dims_not_unique",
        ),
        pytest.param(
            scores,
            spec,
            1,
            {"dims": ("observation", "class")},
            "exactly once",
            id="dims_missing_node",
        ),
        pytest.param(
            scores,
            spec,
            1,
            {"coords": ["obs"]},
            "must be a mapping",
            id="coords_not_mapping",
        ),
        pytest.param(
            scores,
            spec,
            1,
            {"coords": {"foo": [0]}},
            "unknown dim",
            id="coords_unknown_dim",
        ),
        pytest.param(
            torch.zeros(2, 2),
            spec,
            1,
            {"coords": {"observation": ["a"]}},
            "length must be 2",
            id="coords_wrong_length",
        ),
    ]


@pytest.mark.parametrize(
    "attributions, spec, layer, extra, match",
    _invalid_map_cases(),
)
def test_map_node_attributions_rejects_invalid_arguments(
    attributions,
    spec,
    layer,
    extra,
    match,
):
    with pytest.raises(
        Kpnn2Error,
        match=match,
    ):
        map_node_attributions(
            attributions,
            spec,
            layer,
            **extra,
        )
