import pandas as pd
import pytest
import torch

from kpnn2 import align_inputs, parse_edgelist
from tests.helpers.layered_net import (
    LayeredNet,
    pin_all_weights,
)


def _independent_paths_edgelist():
    return pd.DataFrame(
        {
            "source": [
                "feature_a",
                "hidden_a",
                "feature_b",
                "hidden_b",
            ],
            "target": [
                "hidden_a",
                "prediction",
                "hidden_b",
                "decoy",
            ],
        }
    )


def _pinned_linear_model(spec):
    model = LayeredNet(
        spec,
        bias=False,
        relu=False,
    )
    pin_all_weights(
        model,
        value=1.0,
    )
    return model


def _features_frame(
    feature_a,
    feature_b,
):
    return pd.DataFrame(
        {
            "feature_b": [feature_b],
            "extra": [0.0],
            "feature_a": [feature_a],
        }
    )


def test_absent_edge_has_zero_forward_influence():
    spec = parse_edgelist(_independent_paths_edgelist())
    model = _pinned_linear_model(spec)
    prediction_idx = spec.output_nodes.index("prediction")
    decoy_idx = spec.output_nodes.index("decoy")
    assert spec.output_nodes == ("decoy", "prediction")

    x_low = align_inputs(
        _features_frame(
            feature_a=2.0,
            feature_b=0.0,
        ),
        spec,
    )
    x_high = align_inputs(
        _features_frame(
            feature_a=2.0,
            feature_b=100.0,
        ),
        spec,
    )
    output_low = model(x_low)
    output_high = model(x_high)

    torch.testing.assert_close(
        output_low[:, prediction_idx],
        output_high[:, prediction_idx],
    )
    assert not torch.allclose(
        output_low[:, decoy_idx],
        output_high[:, decoy_idx],
    )


def test_absent_edge_has_zero_input_gradient():
    spec = parse_edgelist(_independent_paths_edgelist())
    model = _pinned_linear_model(spec)
    prediction_idx = spec.output_nodes.index("prediction")
    feature_a_idx = spec.input_nodes.index("feature_a")
    feature_b_idx = spec.input_nodes.index("feature_b")

    x = align_inputs(
        _features_frame(
            feature_a=2.0,
            feature_b=100.0,
        ),
        spec,
    )
    x.requires_grad_(True)
    prediction = model(x)[:, prediction_idx].sum()
    prediction.backward()

    assert x.grad is not None
    assert x.grad[0, feature_a_idx].item() != 0
    assert x.grad[0, feature_b_idx].item() == pytest.approx(0.0)
