import math

import pytest
import torch

from kpnn2 import Kpnn2Error, MaskedLinear


def test_masked_linear_output_shape():
    torch.manual_seed(42)
    mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    layer = MaskedLinear(mask)
    x = torch.randn(
        4,
        3,
    )
    y = layer(x)
    assert y.shape == (4, 2)


def test_masked_linear_zero_mask_entry_blocks_source():
    mask = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    layer = MaskedLinear(
        mask,
        bias=False,
    )
    with torch.no_grad():
        layer.raw_weight.fill_(1.0)

    x_base = torch.tensor(
        [[1.0, 2.0]],
        dtype=torch.float32,
    )
    x_change_src0 = torch.tensor(
        [[99.0, 2.0]],
        dtype=torch.float32,
    )
    x_change_src1 = torch.tensor(
        [[1.0, 99.0]],
        dtype=torch.float32,
    )

    y_base = layer(x_base)
    y_src0 = layer(x_change_src0)
    y_src1 = layer(x_change_src1)

    assert y_base[0, 0].item() == y_src1[0, 0].item()
    assert y_base[0, 1].item() == y_src0[0, 1].item()
    assert y_base[0, 0].item() != y_src0[0, 0].item()
    assert y_base[0, 1].item() != y_src1[0, 1].item()


def test_masked_linear_bias_true_has_bias_parameter():
    mask = torch.ones(
        2,
        3,
    )
    layer = MaskedLinear(
        mask,
        bias=True,
    )
    assert layer.bias is not None
    assert layer.bias.shape == (2,)
    parameters = dict(layer.named_parameters())
    assert "bias" in parameters


def test_masked_linear_bias_false_has_no_bias_parameter():
    mask = torch.ones(
        2,
        3,
    )
    layer = MaskedLinear(
        mask,
        bias=False,
    )
    assert layer.bias is None
    parameters = dict(layer.named_parameters())
    assert "bias" not in parameters


def test_masked_linear_bias_false_matches_zero_bias():
    torch.manual_seed(42)
    mask = torch.ones(
        2,
        3,
    )
    with_bias = MaskedLinear(
        mask,
        bias=True,
    )
    no_bias = MaskedLinear(
        mask,
        bias=False,
    )
    with torch.no_grad():
        no_bias.raw_weight.copy_(with_bias.raw_weight)
        with_bias.bias.zero_()
    x = torch.randn(
        5,
        3,
    )
    assert torch.allclose(
        with_bias(x),
        no_bias(x),
    )


def test_masked_linear_mask_is_buffer_not_parameter():
    mask = torch.ones(
        2,
        3,
    )
    layer = MaskedLinear(mask)
    buffers = dict(layer.named_buffers())
    parameters = dict(layer.named_parameters())
    assert "mask" in buffers
    assert "mask" not in parameters
    assert buffers["mask"].dtype == torch.float32
    assert "raw_weight" in parameters


def test_masked_linear_rejects_non_tensor_mask():
    with pytest.raises(
        Kpnn2Error,
        match="torch.Tensor",
    ):
        MaskedLinear(
            [[1.0, 0.0]],
        )


def test_masked_linear_rejects_1d_mask():
    mask = torch.ones(3)
    with pytest.raises(
        Kpnn2Error,
        match="2-dimensional",
    ):
        MaskedLinear(mask)


def test_masked_linear_rejects_3d_mask():
    mask = torch.ones(
        2,
        3,
        1,
    )
    with pytest.raises(
        Kpnn2Error,
        match="2-dimensional",
    ):
        MaskedLinear(mask)


def _diag_mask():
    return torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _pinned_raw_weight():
    return torch.tensor(
        [
            [0.5, 0.7],
            [0.3, 0.9],
        ],
        dtype=torch.float32,
    )


def _layer_with_pinned_diag_weights(bias):
    layer = MaskedLinear(
        _diag_mask(),
        bias=bias,
    )
    with torch.no_grad():
        layer.raw_weight.copy_(_pinned_raw_weight())
    return layer


def test_masked_linear_pinned_weights_match_masked_product():
    layer = _layer_with_pinned_diag_weights(bias=False)
    x = torch.tensor(
        [[1.0, 1.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        layer(x),
        torch.tensor(
            [[0.5, 0.9]],
            dtype=torch.float32,
        ),
    )


def test_masked_linear_pinned_weights_plus_bias_match_product():
    layer = _layer_with_pinned_diag_weights(bias=True)
    with torch.no_grad():
        layer.bias.copy_(
            torch.tensor(
                [0.1, -0.2],
                dtype=torch.float32,
            )
        )
    x = torch.tensor(
        [[1.0, 1.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        layer(x),
        torch.tensor(
            [[0.6, 0.7]],
            dtype=torch.float32,
        ),
    )


def test_masked_linear_batch_matches_manual_masked_product():
    layer = _layer_with_pinned_diag_weights(bias=False)
    x = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, -0.5],
        ],
        dtype=torch.float32,
    )
    effective = _pinned_raw_weight() * _diag_mask()
    expected = x @ effective.T
    torch.testing.assert_close(
        layer(x),
        expected,
    )


def test_masked_linear_forward_leaves_masked_out_raw_weight():
    layer = _layer_with_pinned_diag_weights(bias=False)
    x = torch.tensor(
        [[1.0, 1.0]],
        dtype=torch.float32,
    )
    _ = layer(x)
    torch.testing.assert_close(
        layer.raw_weight,
        _pinned_raw_weight(),
    )


def test_masked_linear_mask_is_not_in_state_dict():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    keys = set(layer.state_dict().keys())
    assert "mask" not in keys
    assert "raw_weight" in keys


def test_masked_linear_rejects_in_place_mask_write():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask.fill_(0.0)


def test_masked_linear_rejects_mask_item_assignment():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask[0, 0] = 0.0


def test_masked_linear_rejects_mask_replacement():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask = torch.zeros(
            2,
            3,
        )


def test_masked_linear_device_move_keeps_mask_read_only():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    layer = layer.to(
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask.fill_(0.0)
    x = torch.ones(
        1,
        3,
    )
    y = layer(x)
    assert y.shape == (1, 2)


def test_masked_linear_state_dict_roundtrip_keeps_mask():
    src = MaskedLinear(
        torch.ones(
            2,
            3,
        ),
        bias=False,
    )
    dst = MaskedLinear(
        torch.ones(
            2,
            3,
        ),
        bias=False,
    )
    with torch.no_grad():
        src.raw_weight.fill_(0.5)
    dst.load_state_dict(src.state_dict())
    torch.testing.assert_close(
        dst.raw_weight,
        src.raw_weight,
    )
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        dst.mask.fill_(0.0)


def test_masked_linear_zero_degree_row_stays_zero():
    torch.manual_seed(42)
    mask = torch.tensor(
        [
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    layer = MaskedLinear(mask)
    assert torch.equal(
        layer.raw_weight[1],
        torch.zeros(2),
    )
    assert layer.bias[1].item() == 0.0
    y = layer(
        torch.randn(
            3,
            2,
        )
    )
    assert torch.equal(
        y[:, 1],
        torch.zeros(3),
    )


def test_masked_linear_init_respects_degree_bound():
    torch.manual_seed(42)
    in_features = 32
    mask = torch.zeros(
        1,
        in_features,
    )
    mask[0, 0] = 1.0
    layer = MaskedLinear(mask)
    fan_in = 1
    bound = 1.0 / math.sqrt(fan_in)
    eps = 1e-6
    assert torch.all(layer.raw_weight[0].abs() <= bound + eps)
    assert abs(layer.bias[0].item()) <= bound + eps


def test_masked_linear_init_not_full_in_features():
    torch.manual_seed(42)
    in_features = 32
    mask = torch.zeros(
        1,
        in_features,
    )
    mask[0, 0] = 1.0
    layer = MaskedLinear(mask)
    degree_bound = 1.0 / math.sqrt(1)
    full_width_bound = 1.0 / math.sqrt(in_features)
    eps = 1e-6
    exceeded_full_width = False
    n_resets = 32
    for seed in range(42, 42 + n_resets):
        torch.manual_seed(seed)
        layer.reset_parameters()
        row = layer.raw_weight[0]
        assert torch.all(row.abs() <= degree_bound + eps)
        assert abs(layer.bias[0].item()) <= degree_bound + eps
        if torch.any(row.abs() > full_width_bound):
            exceeded_full_width = True
        if abs(layer.bias[0].item()) > full_width_bound:
            exceeded_full_width = True
    assert exceeded_full_width


def test_masked_linear_init_dense_row_tighter_bound():
    torch.manual_seed(42)
    in_features = 32
    mask = torch.zeros(
        2,
        in_features,
    )
    mask[0, 0] = 1.0
    mask[1] = 1.0
    layer = MaskedLinear(mask)
    sparse_bound = 1.0 / math.sqrt(1)
    dense_bound = 1.0 / math.sqrt(in_features)
    eps = 1e-6
    assert dense_bound < sparse_bound
    assert torch.all(layer.raw_weight[1].abs() <= dense_bound + eps)
    assert abs(layer.bias[1].item()) <= dense_bound + eps


def test_masked_linear_ignores_mutation_of_constructor_tensor():
    mask = torch.ones(
        2,
        3,
    )
    layer = MaskedLinear(
        mask,
        bias=False,
    )
    with torch.no_grad():
        layer.raw_weight.fill_(1.0)
    mask.zero_()
    x = torch.ones(
        1,
        3,
    )
    y = layer(x)
    assert y.shape == (1, 2)
    assert not torch.equal(
        y,
        torch.zeros(
            1,
            2,
        ),
    )
