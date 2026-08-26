import copy
import math

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kpnn2 import Kpnn2Error, MaskedLinear, parse_edgelist
from kpnn2._frozen_mask import FrozenMask


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


def _pinned_bias():
    return torch.tensor(
        [0.1, -0.2],
        dtype=torch.float32,
    )


def _expected_linear(
    layer,
    x,
):
    effective = layer.raw_weight * layer.mask.to(
        dtype=layer.raw_weight.dtype,
        device=layer.raw_weight.device,
    )
    return F.linear(
        x,
        effective,
        layer.bias,
    )


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


def test_masked_linear_rejects_out_kwarg_write():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    before = layer.mask.tolist()
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        torch.add(
            layer.mask,
            1,
            out=layer.mask,
        )
    assert layer.mask.tolist() == before


def test_masked_linear_numpy_cannot_change_mask():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    before = layer.mask.tolist()
    arr = layer.mask.numpy()
    try:
        arr[:] = 0
    except (ValueError, Kpnn2Error):
        pass
    assert layer.mask.tolist() == before
    assert not arr.flags.writeable


def test_masked_linear_rejects_register_buffer_mask():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    before = layer.mask.tolist()
    assert isinstance(layer.mask, FrozenMask)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.register_buffer(
            "mask",
            torch.zeros(
                2,
                3,
            ),
            persistent=False,
        )
    assert layer.mask.tolist() == before
    assert isinstance(layer.mask, FrozenMask)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask.fill_(0.0)


def test_masked_linear_mask_independent_of_graph_spec():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_edgelist(edgelist)
    layer = MaskedLinear(spec.masks[0])
    spec_before = spec.masks[0].tolist()
    layer_before = layer.mask.tolist()
    assert layer.mask is not spec.masks[0]
    assert layer.mask.data_ptr() != spec.masks[0].data_ptr()

    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask.fill_(0.0)
    assert spec.masks[0].tolist() == spec_before
    assert layer.mask.tolist() == layer_before

    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        spec.masks[0].fill_(0.0)
    assert spec.masks[0].tolist() == spec_before
    assert layer.mask.tolist() == layer_before

    for tensor in (layer.mask, spec.masks[0]):
        arr = tensor.numpy()
        try:
            arr[:] = 0
        except (ValueError, Kpnn2Error):
            pass
    assert spec.masks[0].tolist() == spec_before
    assert layer.mask.tolist() == layer_before

    layer = layer.to(
        device="cpu",
        dtype=torch.float32,
    )
    assert spec.masks[0].tolist() == spec_before
    assert layer.mask.tolist() == layer_before
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask.fill_(0.0)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        spec.masks[0].fill_(0.0)


def test_masked_linear_rejects_in_place_mask_write_keeps_values():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    before = layer.mask.tolist()
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask.fill_(0.0)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask[0, 0] = 0.0
    assert layer.mask.tolist() == before


@pytest.mark.parametrize(
    "apply_cast",
    [
        pytest.param(
            lambda layer: layer.half(),
            id="half",
        ),
        pytest.param(
            lambda layer: layer.to(dtype=torch.bfloat16),
            id="bfloat16",
        ),
        pytest.param(
            lambda layer: layer.double(),
            id="double",
        ),
    ],
)
def test_masked_linear_module_dtype_cast_forward(apply_cast):
    layer = _layer_with_pinned_diag_weights(bias=True)
    with torch.no_grad():
        layer.bias.copy_(_pinned_bias())
    layer = apply_cast(layer)
    assert layer.mask.dtype == torch.float32
    x = torch.ones(
        2,
        2,
        dtype=layer.raw_weight.dtype,
        device=layer.raw_weight.device,
    )
    y = layer(x)
    assert y.dtype == layer.raw_weight.dtype
    torch.testing.assert_close(
        y,
        _expected_linear(
            layer,
            x,
        ),
    )


def test_masked_linear_half_keeps_float32_readonly_mask():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    layer = layer.half()
    assert layer.mask.dtype == torch.float32
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask.fill_(0.0)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask[0, 0] = 0.0


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


def test_masked_linear_deepcopy_independent_params_and_frozen_mask():
    layer = _layer_with_pinned_diag_weights(bias=True)
    with torch.no_grad():
        layer.bias.copy_(_pinned_bias())
    before_mask = layer.mask.tolist()
    copied = copy.deepcopy(layer)

    assert copied is not layer
    assert copied.raw_weight is not layer.raw_weight
    assert copied.bias is not layer.bias
    assert layer.mask is not copied.mask
    assert layer.mask.data_ptr() != copied.mask.data_ptr()
    torch.testing.assert_close(
        copied.raw_weight,
        layer.raw_weight,
    )
    torch.testing.assert_close(
        copied.bias,
        layer.bias,
    )
    torch.testing.assert_close(
        copied.mask,
        layer.mask,
    )
    assert copied.mask.dtype == torch.float32
    assert isinstance(
        copied.mask,
        FrozenMask,
    )
    x = torch.tensor(
        [[1.0, 1.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        copied(x),
        layer(x),
    )
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        copied.mask.fill_(0.0)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask.fill_(0.0)
    assert layer.mask.tolist() == before_mask
    assert copied.mask.tolist() == before_mask


def test_module_with_masked_linear_and_graph_spec_deepcopy():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_edgelist(edgelist)

    class Net(nn.Module):
        def __init__(self, spec):
            super().__init__()
            self.lin = MaskedLinear(spec.masks[0])
            self.spec = spec

        def forward(self, x):
            return self.lin(x)

    net = Net(spec)
    with torch.no_grad():
        net.lin.raw_weight.fill_(0.5)
        net.lin.bias.fill_(0.1)
    copied = copy.deepcopy(net)
    assert copied is not net
    assert copied.lin is not net.lin
    assert copied.spec is not net.spec
    assert copied.lin.raw_weight is not net.lin.raw_weight
    x = torch.ones(
        2,
        net.lin.in_features,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        copied(x),
        net(x),
    )
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        copied.lin.mask.fill_(0.0)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        copied.spec.masks[0].fill_(0.0)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        net.lin.mask.fill_(0.0)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        net.spec.masks[0].fill_(0.0)
    assert copied.lin.mask.dtype == torch.float32
    assert copied.spec.masks[0].dtype == torch.float32
    assert isinstance(
        copied.lin.mask,
        FrozenMask,
    )
    assert isinstance(
        copied.spec.masks[0],
        FrozenMask,
    )


def test_masked_linear_deepcopy_keeps_dtype_cast_and_state_dict():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    copied = copy.deepcopy(layer)
    for module in (
        layer,
        copied,
    ):
        assert "mask" not in module.state_dict()
        assert module.mask.dtype == torch.float32
        with pytest.raises(
            Kpnn2Error,
            match="read-only",
        ):
            module.mask.fill_(0.0)

    layer.half()
    copied.double()
    assert layer.mask.dtype == torch.float32
    assert copied.mask.dtype == torch.float32
    assert "mask" not in layer.state_dict()
    assert "mask" not in copied.state_dict()
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        layer.mask.fill_(0.0)
    with pytest.raises(
        Kpnn2Error,
        match="read-only",
    ):
        copied.mask.fill_(0.0)


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
