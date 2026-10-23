import inspect
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from kpnn2 import (
    Kpnn2Error,
    map_node_attributions,
    parse_adjacency,
    parse_layered,
)


def _tiny_spec():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "A", "H"],
            "target": ["H", "E", "C"],
        }
    )
    return parse_layered(
        edgelist,
        outputs=["C", "E"],
    )


def _tiny_adjacency_spec():
    """
    Input x feeds a two-node cycle; a feeds output y.
    """
    edgelist = pd.DataFrame(
        {
            "source": ["x", "a", "b", "a"],
            "target": ["a", "b", "a", "y"],
        }
    )
    return parse_adjacency(edgelist)


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
    assert da["layer"].dims == ("node",)
    assert da["layer"].dtype == np.int64
    assert da["layer"].values.tolist() == [1, 1]
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
        match=re.escape(
            "The 'node' axis of 'attributions' must have one entry per "
            "unit of layer 1.\n"
            "  Got: 3 (axis 1 of shape (1, 3))\n"
            "  Expected: 2 (spec.layer_dims[1])"
        ),
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


def test_map_node_attributions_stores_bfloat16_as_float32():
    spec = _tiny_spec()
    attributions = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    original = attributions.detach().float()
    da = map_node_attributions(
        attributions,
        spec,
        1,
    )

    assert torch.as_tensor(da.values).dtype is torch.float32
    torch.testing.assert_close(
        torch.as_tensor(da.values),
        original,
        rtol=1e-5,
        atol=1e-5,
    )
    assert attributions.dtype is torch.bfloat16
    da.values[0, 0] = 9.0
    torch.testing.assert_close(
        attributions.detach().float(),
        original,
        rtol=1e-5,
        atol=1e-5,
    )


def test_map_node_attributions_copies_cpu_input():
    spec = _tiny_spec()
    attributions = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    da = map_node_attributions(
        attributions,
        spec,
        1,
    )

    assert torch.from_numpy(da.values).data_ptr() != attributions.data_ptr()
    da.values[0, 0] = 9.0
    assert attributions[0, 0].item() == pytest.approx(0.1)
    with torch.no_grad():
        attributions[1, 1] = -3.0
    assert da.values[1, 1] == pytest.approx(0.4)


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
            "'attributions' must be a torch.Tensor or a list or tuple of "
            "tensors.\n"
            "  Got: dict of length 1",
            id="not_tensor_or_sequence",
        ),
        pytest.param(
            [],
            spec,
            1,
            {},
            "'attributions' must hold at least one tensor.",
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
            "'dims' must have one name per axis of 'attributions'.\n"
            "  Got: length 1\n"
            "  Expected: length 2, for shape (1, 2)",
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
            "Unknown dim name(s) in 'coords'.\n"
            "  Got: 'foo'\n"
            "  Expected: dim names 'observation'",
            id="coords_unknown_dim",
        ),
        pytest.param(
            torch.zeros(2, 2),
            spec,
            1,
            {"coords": {"observation": ["a"]}},
            "'coords[\"observation\"]' must have one label per position on "
            "the 'observation' axis.\n"
            "  Got: length 1\n"
            "  Expected: length 2",
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
        match=re.escape(match),
    ):
        map_node_attributions(
            attributions,
            spec,
            layer,
            **extra,
        )


def test_map_node_attributions_names_adjacency_state_vector():
    spec = _tiny_adjacency_spec()
    assert spec.nodes == ("a", "b", "x", "y")
    attributions = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=torch.float32,
    )

    da = map_node_attributions(
        attributions,
        spec,
    )

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("observation", "node")
    assert da["node"].values.tolist() == ["a", "b", "x", "y"]
    assert da.sel(node="a").values.tolist() == pytest.approx([0.1, 0.5])
    assert da.sel(node="y").values.tolist() == pytest.approx([0.4, 0.8])


def test_map_node_attributions_adjacency_has_no_layer_coord():
    spec = _tiny_adjacency_spec()

    da = map_node_attributions(
        torch.zeros(2, 4),
        spec,
    )

    assert "layer" not in da.coords
    assert "layer" not in da.dims


def test_map_node_attributions_adjacency_accepts_1d_scores():
    spec = _tiny_adjacency_spec()

    da = map_node_attributions(
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        spec,
    )

    assert da.dims == ("node",)
    assert da["node"].values.tolist() == ["a", "b", "x", "y"]
    assert "layer" not in da.coords


def test_map_node_attributions_adjacency_stacks_steps():
    spec = _tiny_adjacency_spec()
    step0 = torch.zeros(2, 4)
    step1 = torch.ones(2, 4)

    da = map_node_attributions(
        [step0, step1],
        spec,
    )

    assert da.dims == ("step", "observation", "node")
    assert da["node"].values.tolist() == ["a", "b", "x", "y"]
    assert list(da.coords["step"].values) == [0, 1]
    assert "layer" not in da.coords
    assert da.sel(
        step=1,
        node="x",
    ).values.tolist() == pytest.approx([1.0, 1.0])


def test_map_node_attributions_adjacency_keeps_extra_coords():
    spec = _tiny_adjacency_spec()

    da = map_node_attributions(
        torch.zeros(2, 4),
        spec,
        coords={"observation": ["s1", "s2"]},
    )

    assert list(da.coords["observation"].values) == ["s1", "s2"]
    assert "layer" not in da.coords


def test_map_node_attributions_adjacency_rejects_wrong_width():
    spec = _tiny_adjacency_spec()

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "The 'node' axis of 'attributions' must have one entry per "
            "unit of the state vector.\n"
            "  Got: 3 (axis 1 of shape (2, 3))\n"
            "  Expected: 4 (spec.state_dim)"
        ),
    ):
        map_node_attributions(
            torch.zeros(2, 3),
            spec,
        )


def test_map_node_attributions_adjacency_inputs_axis():
    spec = _tiny_adjacency_spec()
    scores = torch.tensor(
        [
            [0.5],
            [1.5],
        ],
        dtype=torch.float32,
    )

    da = map_node_attributions(
        scores,
        spec,
        axis="inputs",
    )

    assert da["node"].values.tolist() == ["x"]
    assert da.sel(node="x").values.tolist() == pytest.approx(
        [0.5, 1.5],
    )
    assert "layer" not in da.coords
    assert len(spec.input_index) == 1


def test_map_node_attributions_adjacency_inputs_name_each_input_once():
    spec = parse_adjacency(
        pd.DataFrame(
            {
                "source": ["x", "w", "a", "b", "a"],
                "target": ["a", "a", "b", "a", "y"],
            }
        ),
        widths={"a": 2},
    )
    scores = torch.tensor(
        [[0.25, 0.75]],
        dtype=torch.float32,
    )

    da = map_node_attributions(
        scores,
        spec,
        axis="inputs",
    )

    assert spec.input_index == (3, 4)
    assert da["node"].values.tolist() == ["w", "x"]
    assert "layer" not in da.coords


def test_map_node_attributions_does_not_infer_adjacency_inputs():
    spec = _tiny_adjacency_spec()

    with pytest.raises(
        Kpnn2Error,
        match=re.escape("\n  Expected: 4 (spec.state_dim)"),
    ):
        map_node_attributions(
            torch.zeros(2, 1),
            spec,
        )


def test_map_node_attributions_layered_inputs_axis_matches_layer_zero():
    spec = parse_layered(
        pd.DataFrame(
            {
                "source": ["A", "B", "H"],
                "target": ["H", "H", "C"],
            }
        ),
        widths={"H": 2},
    )
    scores = torch.tensor(
        [[0.1, 0.2]],
        dtype=torch.float32,
    )

    by_axis = map_node_attributions(
        scores,
        spec,
        axis="inputs",
    )
    by_layer = map_node_attributions(
        scores,
        spec,
        0,
    )

    xr.testing.assert_equal(
        by_axis,
        by_layer,
    )
    assert by_axis["node"].values.tolist() == ["A", "B"]
    assert by_axis["layer"].values.tolist() == [0, 0]


def test_map_node_attributions_rejects_inputs_axis_with_layer():
    spec = _tiny_spec()

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "A LayeredSpec takes only one of 'layer', 'hop_input', "
            "'hop_output', and 'axis'.\n"
            "  Got: 'layer', 'axis'\n"
        ),
    ):
        map_node_attributions(
            torch.zeros(1, 1),
            spec,
            0,
            axis="inputs",
        )


def test_map_node_attributions_rejects_an_unknown_axis():
    spec = _tiny_adjacency_spec()

    with pytest.raises(
        Kpnn2Error,
        match="'axis' must be 'inputs' or None",
    ) as caught:
        map_node_attributions(
            torch.zeros(2, 4),
            spec,
            axis="state",
        )

    assert "Did you mean" not in str(caught.value)


def test_map_node_attributions_suggests_inputs_for_a_typo():
    spec = _tiny_adjacency_spec()

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'axis' must be 'inputs' or None.\n"
            "  Got: 'input' (str)\n"
            "  Did you mean: 'inputs'"
        ),
    ):
        map_node_attributions(
            torch.zeros(2, 1),
            spec,
            axis="input",
        )


def test_map_node_attributions_adjacency_rejects_a_layer():
    spec = _tiny_adjacency_spec()

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'layer', 'hop_input', and 'hop_output' do not apply to an "
            "AdjacencySpec: it has no layers, only one state vector.\n"
            "  Got: 'layer'\n"
            "  Fix: Omit them to label the node axis with spec.nodes, or "
            "pass axis='inputs' to label the input units."
        ),
    ):
        map_node_attributions(
            torch.zeros(2, 4),
            spec,
            0,
        )


def test_map_node_attributions_layered_requires_layer_or_hop_side():
    spec = _tiny_spec()

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "A LayeredSpec needs one of 'layer', 'hop_input', "
            "'hop_output', or axis='inputs'.\n"
            "  Fix: Pass layer= with a 0-based index into "
            "spec.layer_nodes, or a Hop from spec.hops as hop_output= "
            "(what its module returns) or hop_input= (what its module "
            "reads)."
        ),
    ):
        map_node_attributions(
            torch.zeros(2, 2),
            spec,
        )


def test_map_node_attributions_spec_error_names_both_spec_types():
    with pytest.raises(
        Kpnn2Error,
        match="LayeredSpec",
    ) as caught:
        map_node_attributions(
            torch.zeros(2, 2),
            object(),
        )

    message = str(caught.value)
    assert "LayeredSpec" in message
    assert "AdjacencySpec" in message


def test_map_node_attributions_repeats_wide_layered_names():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )
    spec = parse_layered(
        edgelist,
        widths={"H": 3},
    )
    scores = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    da = map_node_attributions(
        scores,
        spec,
        1,
    )
    assert da["node"].values.tolist() == ["H", "H", "H"]
    assert da.shape[-1] == spec.layer_dims[1]
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "\n  Got: 1 (axis 1 of shape (2, 1))\n"
            "  Expected: 3 (spec.layer_dims[1])"
        ),
    ):
        map_node_attributions(
            torch.zeros(2, 1),
            spec,
            1,
        )


def test_map_node_attributions_wide_axis_selection_contract():
    """
    Pin what the Returns section says about a non-unique index.
    """
    edgelist = pd.DataFrame(
        {
            "source": ["A", "A", "A", "K", "H", "B"],
            "target": ["K", "H", "B", "C", "C", "C"],
        }
    )
    spec = parse_layered(
        edgelist,
        widths={"H": 3},
    )
    assert spec.layer_nodes[1] == ("B", "H", "K")
    scores = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ]
    )
    da = map_node_attributions(
        scores,
        spec,
        1,
    )

    assert da["node"].values.tolist() == ["B", "H", "H", "H", "K"]
    np.testing.assert_allclose(
        da.sel(node="H").values,
        [[2.0, 3.0, 4.0], [7.0, 8.0, 9.0]],
    )
    picked = da.sel(node=da["node"].isin(["B", "K"]))
    assert picked["node"].values.tolist() == ["B", "K"]
    folded = da.groupby("node").mean()
    assert folded["node"].values.tolist() == ["B", "H", "K"]
    np.testing.assert_allclose(
        folded.values,
        [[1.0, 3.0, 5.0], [6.0, 8.0, 10.0]],
    )
    with pytest.raises(pd.errors.InvalidIndexError):
        da.sel(node=["B", "K"])
    with pytest.raises(ValueError):
        da.reindex(node=["B", "K"])
    with pytest.raises(ValueError):
        da + da.isel(node=[0, 4])
    reversed_h = da.isel(node=[0, 3, 2, 1, 4])
    np.testing.assert_allclose(
        (da - reversed_h).sel(node="H").values,
        [[-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]],
    )


def _skip_spec():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )
    return parse_layered(edgelist)


def test_map_node_attributions_names_skip_hop_source_axis():
    spec = _skip_spec()
    hop = spec.hops[1]
    scores = torch.tensor(
        [0.1, 0.2],
        dtype=torch.float32,
    )

    da = map_node_attributions(
        scores,
        spec,
        hop_input=hop,
    )

    assert da["node"].values.tolist() == list(hop.source_nodes)
    assert hop.in_features == len(hop.source_nodes)
    assert da.shape[-1] == hop.in_features
    assert da["layer"].dims == ("node",)
    assert da["layer"].values.tolist() == [0, 1]
    assert "layer" not in da.dims
    assert da.sel(node="A").item() == pytest.approx(0.1)
    assert da.sel(node="H").item() == pytest.approx(0.2)


@pytest.mark.parametrize(
    "concat_kwargs",
    [
        pytest.param(
            {
                "coords": "different",
                "compat": "equals",
            },
            id="different_equals",
        ),
        pytest.param(
            {
                "coords": "minimal",
                "compat": "override",
            },
            id="minimal_override",
        ),
    ],
)
def test_map_node_attributions_layer_survives_concat(concat_kwargs):
    """
    Each unit keeps its depth when results join along ``node``.

    A scalar ``layer`` lost it: ``xr.concat`` kept the first
    array's value for every unit, and ``minimal`` is the default
    xarray announces.
    """
    spec = _skip_spec()
    inputs = map_node_attributions(
        torch.zeros(2, 1),
        spec,
        0,
    )
    skip_input = map_node_attributions(
        torch.zeros(2, 2),
        spec,
        hop_input=spec.hops[1],
    )
    top = map_node_attributions(
        torch.zeros(2, 1),
        spec,
        2,
    )

    joined = xr.concat(
        [inputs, skip_input, top],
        dim="node",
        **concat_kwargs,
    )

    assert joined["node"].values.tolist() == ["A", "A", "H", "C"]
    assert joined["layer"].values.tolist() == [0, 0, 1, 2]


def test_map_node_attributions_repeats_wide_hop_source_names():
    edgelist = pd.DataFrame(
        {
            "source": ["x", "A", "H", "A"],
            "target": ["A", "H", "C", "C"],
        }
    )
    spec = parse_layered(
        edgelist,
        widths={"A": 2, "H": 3},
    )
    hop = spec.hops[2]
    scores = torch.arange(
        hop.in_features,
        dtype=torch.float32,
    )

    da = map_node_attributions(
        scores,
        spec,
        hop_input=hop,
    )

    assert da["node"].values.tolist() == [
        "A",
        "A",
        "H",
        "H",
        "H",
    ]
    assert da.shape[-1] == hop.in_features
    assert hop.in_features != len(hop.source_nodes)
    assert list(hop.source_nodes) == ["A", "H"]
    assert hop.source_layers == (1, 2)
    assert da["layer"].values.tolist() == [1, 1, 2, 2, 2]


@pytest.mark.parametrize(
    ("layer", "sides", "named"),
    [
        (1, ("hop_input",), "'layer', 'hop_input'"),
        (1, ("hop_output",), "'layer', 'hop_output'"),
        (None, ("hop_input", "hop_output"), "'hop_input', 'hop_output'"),
        (
            1,
            ("hop_input", "hop_output"),
            "'layer', 'hop_input', 'hop_output'",
        ),
    ],
)
def test_map_node_attributions_rejects_more_than_one_axis(
    layer,
    sides,
    named,
):
    spec = _skip_spec()
    hop = spec.hops[1]

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "A LayeredSpec takes only one of 'layer', 'hop_input', "
            "'hop_output', and 'axis'."
        ),
    ) as caught:
        map_node_attributions(
            torch.zeros(1, 2),
            spec,
            layer,
            **{side: hop for side in sides},
        )

    assert f"\n  Got: {named}\n" in str(caught.value)


@pytest.mark.parametrize(
    "side",
    ["hop_input", "hop_output"],
)
def test_map_node_attributions_adjacency_rejects_a_hop(side):
    spec = _tiny_adjacency_spec()
    hop = _skip_spec().hops[0]

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'layer', 'hop_input', and 'hop_output' do not apply to an "
            "AdjacencySpec: it has no layers, only one state vector.\n"
            f"  Got: '{side}'\n"
        ),
    ):
        map_node_attributions(
            torch.zeros(2, 4),
            spec,
            **{side: hop},
        )


@pytest.mark.parametrize(
    "side",
    ["hop_input", "hop_output"],
)
def test_map_node_attributions_rejects_hop_not_in_spec(side):
    spec = _skip_spec()
    hop = spec.hops[1]
    other = replace(
        hop,
        source_index=hop.source_index + (0,),
        target_index=hop.target_index + (0,),
    )

    with pytest.raises(
        Kpnn2Error,
        match=f"'{side}' must match an entry of spec.hops",
    ):
        map_node_attributions(
            torch.zeros(hop.in_features),
            spec,
            **{side: other},
        )


@pytest.mark.parametrize(
    "side",
    ["hop_input", "hop_output"],
)
def test_map_node_attributions_rejects_non_hop(side):
    spec = _skip_spec()

    with pytest.raises(
        Kpnn2Error,
        match=f"'{side}' must be a Hop",
    ):
        map_node_attributions(
            torch.zeros(2),
            spec,
            **{side: 1},
        )


def test_map_node_attributions_hop_input_rejects_wrong_width():
    spec = _skip_spec()
    hop = spec.hops[1]

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "The 'node' axis of 'attributions' must have one entry per "
            "input unit of 'hop_input'.\n"
            "  Got: 3 (axis 0 of shape (3,))\n"
            f"  Expected: {hop.in_features} (hop_input.in_features)"
        ),
    ):
        map_node_attributions(
            torch.zeros(3),
            spec,
            hop_input=hop,
        )


def test_map_node_attributions_hop_output_rejects_input_width():
    spec = _skip_spec()
    hop = spec.hops[1]
    assert hop.in_features != hop.out_features

    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "The 'node' axis of 'attributions' must have one entry per "
            "output unit of 'hop_output'.\n"
            f"  Got: {hop.in_features} (axis 0 of shape "
            f"({hop.in_features},))\n"
            f"  Expected: {hop.out_features} (hop_output.out_features)"
        ),
    ):
        map_node_attributions(
            torch.zeros(hop.in_features),
            spec,
            hop_output=hop,
        )


def _square_hop_spec():
    """
    ``hops[0]`` reads A, B and writes H1, H2: equal widths.
    """
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "A", "B", "H1", "H2"],
            "target": ["H1", "H1", "H2", "H2", "C", "C"],
        }
    )
    return parse_layered(edgelist)


def test_map_node_attributions_hop_output_names_target_layer():
    spec = _skip_spec()
    scores = torch.tensor(
        [
            [0.1],
            [0.2],
        ]
    )

    for index, hop in enumerate(spec.hops):
        width = hop.out_features
        values = torch.arange(
            2 * width,
            dtype=torch.float32,
        ).reshape(2, width)
        by_hop = map_node_attributions(
            values,
            spec,
            hop_output=hop,
        )
        by_layer = map_node_attributions(
            values,
            spec,
            hop.target_layer,
        )

        xr.testing.assert_identical(
            by_hop,
            by_layer,
        )
        assert hop.target_layer == index + 1
        assert by_hop["layer"].values.tolist() == [index + 1] * width
        assert by_hop["node"].values.tolist() == list(
            spec.layer_nodes[index + 1]
        )

    last = map_node_attributions(
        scores,
        spec,
        hop_output=spec.hops[-1],
    )
    assert last["node"].values.tolist() == ["C"]


def test_map_node_attributions_same_width_sides_name_differently():
    spec = _square_hop_spec()
    hop = spec.hops[0]
    assert hop.in_features == hop.out_features
    scores = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    output_side = map_node_attributions(
        scores,
        spec,
        hop_output=hop,
    )
    input_side = map_node_attributions(
        scores,
        spec,
        hop_input=hop,
    )

    assert output_side["node"].values.tolist() == ["H1", "H2"]
    assert input_side["node"].values.tolist() == ["A", "B"]
    assert output_side["layer"].values.tolist() == [1, 1]
    assert input_side["layer"].values.tolist() == [0, 0]
    torch.testing.assert_close(
        torch.as_tensor(output_side.values),
        scores,
    )
    torch.testing.assert_close(
        torch.as_tensor(input_side.values),
        scores,
    )


def test_map_node_attributions_hop_output_repeats_wide_names():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )
    spec = parse_layered(
        edgelist,
        widths={"H": 3},
    )
    hop = spec.hops[0]

    da = map_node_attributions(
        torch.zeros(2, hop.out_features),
        spec,
        hop_output=hop,
    )

    assert da["node"].values.tolist() == ["H", "H", "H"]
    assert da["layer"].values.tolist() == [1, 1, 1]


def test_map_node_attributions_has_no_sideless_hop_keyword():
    parameters = inspect.signature(map_node_attributions).parameters

    assert "hop" not in parameters
    assert parameters["hop_input"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["hop_output"].kind is inspect.Parameter.KEYWORD_ONLY


def _received_value_cases():
    spec = _tiny_spec()
    other = replace(
        spec.hops[1],
        source_index=spec.hops[1].source_index + (0,),
        target_index=spec.hops[1].target_index + (0,),
    )
    scores = torch.zeros(1, 2)
    return [
        pytest.param(
            (
                torch.zeros(1, 3),
                torch.zeros(1, 3),
            ),
            {"layer": 1},
            "The 'node' axis of 'attributions' must have one entry per "
            "unit of layer 1.\n"
            "  Got: 3 (axis 2 of stacked shape (2, 1, 3))\n"
            "  Expected: 2 (spec.layer_dims[1])",
            id="node_axis_stacked",
        ),
        pytest.param(
            scores,
            {
                "spec": "spec",
                "layer": 1,
            },
            "'spec' must be a LayeredSpec or an AdjacencySpec.\n"
            "  Got: 'spec' (str)\n"
            "  Fix: Build one with parse_layered() or parse_adjacency(), or "
            "load a saved one with LayeredSpec.from_dict() or "
            "AdjacencySpec.from_dict().",
            id="spec",
        ),
        pytest.param(
            scores,
            {"axis": "state"},
            "'axis' must be 'inputs' or None.\n  Got: 'state' (str)",
            id="axis",
        ),
        pytest.param(
            scores,
            {"axis": np.array(["inputs"])},
            "'axis' must be 'inputs' or None.\n"
            "  Got: ndarray of shape (1,), dtype <U6",
            id="axis_array",
        ),
        pytest.param(
            scores,
            {"layer": 1.0},
            "'layer' must be an int.\n  Got: 1.0 (float)",
            id="layer_float",
        ),
        pytest.param(
            scores,
            {"layer": True},
            "'layer' must be an int.\n  Got: True (bool)",
            id="layer_bool",
        ),
        pytest.param(
            torch.zeros(2),
            {"hop_input": 1},
            "'hop_input' must be a Hop from spec.hops.\n  Got: 1 (int)",
            id="hop_input_not_hop",
        ),
        pytest.param(
            torch.zeros(2),
            {"hop_output": other},
            "'hop_output' must match an entry of spec.hops: a Hop from "
            "another spec does not match.\n"
            "  Got: a Hop into layer 2 that differs from spec.hops[1]\n"
            "  Fix: Take it from this spec's hops.",
            id="hop_output_not_in_spec",
        ),
        pytest.param(
            torch.zeros(1, 2).numpy(),
            {"layer": 1},
            "'attributions' must be a torch.Tensor or a list or tuple of "
            "tensors.\n"
            "  Got: ndarray of shape (1, 2), dtype float32",
            id="attributions_array",
        ),
        pytest.param(
            [
                scores,
                1.0,
            ],
            {"layer": 1},
            "Each item of 'attributions' must be a torch.Tensor.\n"
            "  Got: 1.0 (float) at position 1",
            id="attributions_item",
        ),
        pytest.param(
            (
                scores,
                scores,
                torch.zeros(2, 2),
            ),
            {"layer": 1},
            "Tensors in 'attributions' must all have the same shape: a list "
            "or tuple is stacked on a new 'step' axis.\n"
            "  Got: shape (1, 2) at position 0 and shape (2, 2) at position "
            "2\n"
            "  Fix: To label several layers or hops, call "
            "map_node_attributions() once per tensor.",
            id="attributions_shapes",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "dims": ("observation", 0),
            },
            "'dims' must be a sequence of strings.\n"
            "  Got: 0 (int) at position 1",
            id="dims_item",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "dims": True,
            },
            "'dims' must be a sequence of strings, one per tensor axis.\n"
            "  Got: True (bool)",
            id="dims_bool",
        ),
        pytest.param(
            torch.zeros(2),
            {
                "layer": 1,
                "dims": "node",
            },
            "'dims' must be a sequence of strings, one per tensor axis, not "
            "a single string.\n"
            "  Got: 'node' (str)\n"
            "  Fix: Wrap it in a list.",
            id="dims_str",
        ),
        pytest.param(
            torch.zeros(2),
            {
                "layer": 1,
                "dims": np.array("node"),
            },
            "'dims' must be a sequence of strings, one per tensor axis.\n"
            "  Got: ndarray of shape (), dtype <U4",
            id="dims_zero_dim_array",
        ),
        pytest.param(
            torch.zeros(2),
            {
                "layer": 1,
                "dims": {"node"},
            },
            "'dims' must be a sequence of strings, one per tensor axis, not "
            "a set: its items are matched by position.\n"
            "  Got: {'node'} (set)\n"
            "  Fix: Pass a list or tuple in the intended order.",
            id="dims_set",
        ),
        pytest.param(
            torch.zeros(1, 1, 1, 2),
            {
                "layer": 1,
                "dims": ("b", "a", "b", "a"),
            },
            "'dims' names must be unique.\n  Got: repeated 'a', 'b'",
            id="dims_repeated",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "dims": ("observation", "class"),
            },
            "'dims' must contain 'node' exactly once.\n"
            "  Got: 'observation', 'class'",
            id="dims_without_node",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "dims": ("observation", "nodes"),
            },
            "'dims' must contain 'node' exactly once.\n"
            "  Got: 'observation', 'nodes'\n"
            "  Did you mean: 'node' for 'nodes'",
            id="dims_node_typo",
        ),
        pytest.param(
            (
                scores,
                scores,
            ),
            {
                "layer": 1,
                "dims": ("observation", "node"),
            },
            "'dims' must have one name per axis of 'attributions'.\n"
            "  Got: length 2\n"
            "  Expected: length 3, for the stacked shape (2, 1, 2)",
            id="dims_length_stacked",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "dims": ("layer", "node"),
            },
            "'dims' must not include 'layer': kpnn2 reserves it for the "
            "depth coordinate along 'node'.",
            id="dims_layer",
        ),
        pytest.param(
            torch.zeros(1, 1, 2),
            {"layer": 1},
            "'dims' is required when 'attributions' has 3 or more axes.\n"
            "  Got: Tensor of shape (1, 1, 2), dtype torch.float32\n"
            "  Fix: Pass dims= with one name per axis, 'node' among them.",
            id="dims_required",
        ),
        pytest.param(
            (
                torch.zeros(1, 1, 2),
                torch.zeros(1, 1, 2),
            ),
            {"layer": 1},
            "'dims' is required when 'attributions' stacks tensors with 3 or "
            "more axes.\n"
            "  Got: tensors of shape (1, 1, 2)\n"
            "  Fix: Pass dims= with one name per axis of the stacked tensor, "
            "the new leading axis included.",
            id="dims_required_stacked",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "coords": ["obs"],
            },
            "'coords' must be a mapping of dim name to labels.\n"
            "  Got: ['obs'] (list)",
            id="coords",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "coords": {"node": ["E", "H"]},
            },
            "'coords' must not include 'node' or 'layer': kpnn2 reserves "
            "them for the node names and depths.\n"
            "  Got: 'node'",
            id="coords_node",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "coords": {"observaton": [0]},
            },
            "Unknown dim name(s) in 'coords'.\n"
            "  Got: 'observaton'\n"
            "  Expected: dim names 'observation'\n"
            "  Did you mean: 'observation'",
            id="coords_key_typo",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "coords": {
                    0: [0],
                    "obs": [0],
                },
            },
            "Unknown dim name(s) in 'coords'.\n"
            "  Got: 0, 'obs'\n"
            "  Expected: dim names 'observation'",
            id="coords_key_not_a_str",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "coords": {"observation": None},
            },
            "'coords[\"observation\"]' must be a sequence of labels, one per "
            "position on that axis.\n"
            "  Got: None",
            id="coords_value_none",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "coords": {"observation": "s"},
            },
            "'coords[\"observation\"]' must be a sequence of labels, one per "
            "position on that axis, not a single string.\n"
            "  Got: 's' (str)\n"
            "  Fix: Wrap it in a list.",
            id="coords_value_str",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "coords": {"observation": {"s"}},
            },
            "'coords[\"observation\"]' must be a sequence of labels, one per "
            "position on that axis, not a set: its items are matched by "
            "position.\n"
            "  Got: {'s'} (set)\n"
            "  Fix: Pass a list or tuple in the intended order.",
            id="coords_value_set",
        ),
        pytest.param(
            scores,
            {
                "layer": 1,
                "coords": {"observation": {0: "s"}},
            },
            "'coords[\"observation\"]' must be a sequence of labels, one per "
            "position on that axis, not a mapping: its items are matched by "
            "position.\n"
            "  Got: {0: 's'} (dict)\n"
            "  Fix: Pass a list or tuple in the intended order.",
            id="coords_value_mapping",
        ),
    ]


@pytest.mark.parametrize(
    ("attributions", "kwargs", "message"),
    _received_value_cases(),
)
def test_map_node_attributions_messages_report_the_received_value(
    attributions,
    kwargs,
    message,
):
    arguments = {"spec": _tiny_spec()}
    arguments.update(kwargs)
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(message),
    ):
        map_node_attributions(
            attributions,
            **arguments,
        )


def test_map_node_attributions_tensor_mapping_is_not_printed():
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("\n  Got: dict of length 1"),
    ) as caught:
        map_node_attributions(
            {"scores": torch.tensor([[123.0, 456.0]])},
            _tiny_spec(),
            1,
        )
    assert "123" not in str(caught.value)


@pytest.mark.parametrize(
    ("attributions", "message"),
    [
        pytest.param(
            torch.tensor(1.0),
            "'attributions' needs a 'node' axis.\n"
            "  Got: Tensor of shape (), dtype torch.float32",
            id="tensor",
        ),
        pytest.param(
            [torch.tensor(1.0)],
            "Tensors in 'attributions' need a 'node' axis.\n"
            "  Got: tensors of shape ()",
            id="stacked",
        ),
    ],
)
def test_map_node_attributions_rejects_a_zero_dimensional_tensor(
    attributions,
    message,
):
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(message),
    ):
        map_node_attributions(
            attributions,
            _tiny_spec(),
            1,
        )


_LAYERED_AXIS_FIX = (
    "  Fix: axis='inputs' names only the model inputs; for another layer "
    "pass layer=, or hop_output= with a Hop from spec.hops."
)
_ADJACENCY_AXIS_FIX = (
    "  Fix: axis='inputs' names only the model inputs; omit axis to name "
    "the whole state vector, output nodes included."
)


@pytest.mark.parametrize(
    ("spec", "axis", "fields"),
    [
        pytest.param(
            _tiny_spec(),
            "outputs",
            [_LAYERED_AXIS_FIX],
            id="outputs_layered",
        ),
        pytest.param(
            _tiny_adjacency_spec(),
            "output",
            [_ADJACENCY_AXIS_FIX],
            id="output_adjacency",
        ),
        pytest.param(
            _tiny_spec(),
            "input",
            [
                "  Did you mean: 'inputs'",
                _LAYERED_AXIS_FIX,
            ],
            id="input_layered",
        ),
    ],
)
def test_axis_message_never_suggests_inputs_for_the_other_end(
    spec,
    axis,
    fields,
):
    # Taking 'inputs' for 'outputs' would label the scores with the
    # wrong end of the model without an error.
    with pytest.raises(Kpnn2Error) as excinfo:
        map_node_attributions(
            torch.zeros(2, 1),
            spec,
            axis=axis,
        )
    assert str(excinfo.value).splitlines() == [
        "'axis' must be 'inputs' or None.",
        f"  Got: {axis!r} (str)",
        *fields,
    ]


@pytest.mark.parametrize(
    ("spec", "kwargs", "fix"),
    [
        pytest.param(
            _tiny_adjacency_spec(),
            {},
            "  Fix: For scores on the input units, pass axis='inputs'.",
            id="state_vector",
        ),
        pytest.param(
            _tiny_spec(),
            {"layer": 1},
            "  Fix: For the output of the module on spec.hops[i], pass "
            "hop_output=spec.hops[i]; layer counts from 0 at the input "
            "layer.",
            id="layer",
        ),
        pytest.param(
            _tiny_spec(),
            {"hop_output": _tiny_spec().hops[0]},
            "  Fix: For scores on what the hop's module reads, pass "
            "hop_input= instead.",
            id="hop_output",
        ),
        pytest.param(
            _tiny_spec(),
            {"hop_input": _tiny_spec().hops[0]},
            "  Fix: For scores on what the hop's module returns, pass "
            "hop_output= instead.",
            id="hop_input",
        ),
        pytest.param(
            _tiny_spec(),
            {"axis": "inputs"},
            None,
            id="inputs",
        ),
    ],
)
def test_node_axis_length_message_names_the_other_side(
    spec,
    kwargs,
    fix,
):
    with pytest.raises(Kpnn2Error) as excinfo:
        map_node_attributions(
            torch.zeros(2, 99),
            spec,
            **kwargs,
        )
    last = str(excinfo.value).splitlines()[-1]
    if fix is None:
        assert last.startswith("  Expected: ")
    else:
        assert last == fix


def test_coords_on_a_node_only_tensor_says_node_is_the_only_dim():
    with pytest.raises(Kpnn2Error) as excinfo:
        map_node_attributions(
            torch.zeros(1),
            _tiny_spec(),
            0,
            coords={"observation": ["s1"]},
        )
    assert str(excinfo.value) == (
        "Unknown dim name(s) in 'coords'.\n"
        "  Got: 'observation'\n"
        "  Expected: no key: the only dim of 'attributions' is 'node'"
    )
