import pandas as pd
import pytest
import torch
import xarray as xr

from kpnn2 import map_node_attributions, parse_edgelist
from kpnn2.errors import Kpnn2Error


def _tiny_spec():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "A", "H"],
            "target": ["H", "E", "C"],
        }
    )
    return parse_edgelist(edgelist)


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
