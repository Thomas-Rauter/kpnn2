import copy
import math

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize

from kpnn2 import Kpnn2Error, MaskedLinear, parse_layered


def _original(layer):
    """
    The trainable tensor behind ``layer.weight``.

    Spelled out in full in the parametrization tests below.
    """
    return layer.parametrizations.weight.original


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
        _original(layer).fill_(1.0)

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
        _original(no_bias).copy_(_original(with_bias))
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
    mask_name = "parametrizations.weight.0.mask"
    assert mask_name in buffers
    assert buffers[mask_name].dtype == torch.float32
    assert mask_name not in parameters
    assert "parametrizations.weight.original" in parameters


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


def _pinned_weight():
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
        _original(layer).copy_(_pinned_weight())
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
    original = _original(layer)
    effective = original * layer.mask.to(
        dtype=original.dtype,
        device=original.device,
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
    effective = _pinned_weight() * _diag_mask()
    expected = x @ effective.T
    torch.testing.assert_close(
        layer(x),
        expected,
    )


def test_masked_linear_forward_leaves_trainable_tensor_alone():
    layer = _layer_with_pinned_diag_weights(bias=False)
    x = torch.tensor(
        [[1.0, 1.0]],
        dtype=torch.float32,
    )
    _ = layer(x)
    torch.testing.assert_close(
        _original(layer),
        _pinned_weight(),
    )


def test_masked_linear_mask_is_not_in_state_dict():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    keys = set(layer.state_dict().keys())
    assert not any(key.endswith("mask") for key in keys)
    assert "parametrizations.weight.original" in keys


def test_masked_linear_mask_is_a_plain_tensor():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    assert type(layer.mask) is torch.Tensor
    assert not layer.mask.requires_grad
    assert layer.mask.is_contiguous()


def test_masked_linear_forward_cast_of_mask_is_free():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    same = layer.mask.to(
        dtype=_original(layer).dtype,
        device=_original(layer).device,
    )
    assert same is layer.mask

    row = layer.mask[0]
    assert row.data_ptr() == layer.mask.data_ptr()

    arr = layer.mask.numpy()
    assert arr.flags.writeable


def test_masked_linear_mask_independent_of_layered_spec():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_layered(edgelist)
    layer = MaskedLinear(spec.masks[0])
    layer_before = layer.mask.tolist()
    assert layer.mask is not spec.masks[0]
    assert layer.mask.data_ptr() != spec.masks[0].data_ptr()

    spec.masks[0].fill_(0.0)
    assert layer.mask.tolist() == layer_before

    layer = layer.to(
        device="cpu",
        dtype=torch.float32,
    )
    assert layer.mask.tolist() == layer_before


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
        dtype=_original(layer).dtype,
        device=_original(layer).device,
    )
    y = layer(x)
    assert y.dtype == _original(layer).dtype
    torch.testing.assert_close(
        y,
        _expected_linear(
            layer,
            x,
        ),
    )


def test_masked_linear_half_keeps_float32_plain_mask():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    layer = layer.half()
    assert layer.mask.dtype == torch.float32
    assert type(layer.mask) is torch.Tensor
    assert _original(layer).dtype == torch.float16
    assert layer.weight.dtype == torch.float16


def test_masked_linear_device_move_keeps_mask_usable():
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
    assert type(layer.mask) is torch.Tensor
    assert layer.mask.dtype == torch.float32
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
        _original(src).fill_(0.5)
    dst.load_state_dict(src.state_dict())
    torch.testing.assert_close(
        _original(dst),
        _original(src),
    )
    assert (
        dst.mask.tolist()
        == torch.ones(
            2,
            3,
        ).tolist()
    )


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
        _original(layer)[1],
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
    assert torch.all(_original(layer)[0].abs() <= bound + eps)
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
        row = _original(layer)[0]
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
    assert torch.all(_original(layer)[1].abs() <= dense_bound + eps)
    assert abs(layer.bias[1].item()) <= dense_bound + eps


def test_masked_linear_deepcopy_independent_params_and_mask():
    layer = _layer_with_pinned_diag_weights(bias=True)
    with torch.no_grad():
        layer.bias.copy_(_pinned_bias())
    before_mask = layer.mask.tolist()
    copied = copy.deepcopy(layer)

    assert copied is not layer
    assert _original(copied) is not _original(layer)
    assert copied.bias is not layer.bias
    assert layer.mask is not copied.mask
    assert layer.mask.data_ptr() != copied.mask.data_ptr()
    torch.testing.assert_close(
        _original(copied),
        _original(layer),
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
    assert type(copied.mask) is torch.Tensor
    x = torch.tensor(
        [[1.0, 1.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        copied(x),
        layer(x),
    )
    copied.mask.fill_(0.0)
    assert layer.mask.tolist() == before_mask


def test_module_with_masked_linear_and_layered_spec_deepcopy():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )
    spec = parse_layered(edgelist)

    class Net(nn.Module):
        def __init__(self, spec):
            super().__init__()
            self.lin = MaskedLinear(spec.masks[0])
            self.spec = spec

        def forward(self, x):
            return self.lin(x)

    net = Net(spec)
    with torch.no_grad():
        _original(net.lin).fill_(0.5)
        net.lin.bias.fill_(0.1)
    copied = copy.deepcopy(net)
    assert copied is not net
    assert copied.lin is not net.lin
    assert copied.spec is not net.spec
    assert _original(copied.lin) is not _original(net.lin)
    x = torch.ones(
        2,
        net.lin.in_features,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        copied(x),
        net(x),
    )
    assert copied.lin.mask.dtype == torch.float32
    assert copied.spec.masks[0].dtype == torch.float32
    assert type(copied.lin.mask) is torch.Tensor
    assert type(copied.spec.masks[0]) is torch.Tensor

    before = net.lin.mask.tolist()
    copied.lin.mask.fill_(0.0)
    copied.spec.masks[0].fill_(0.0)
    assert net.lin.mask.tolist() == before
    assert net.spec.masks[0].tolist() == before


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

    layer.half()
    copied.double()
    assert layer.mask.dtype == torch.float32
    assert copied.mask.dtype == torch.float32
    assert type(layer.mask) is torch.Tensor
    assert type(copied.mask) is torch.Tensor
    assert "mask" not in layer.state_dict()
    assert "mask" not in copied.state_dict()


def test_masked_linear_compiles_without_a_graph_break():
    dynamo = pytest.importorskip("torch._dynamo")
    if not dynamo.is_dynamo_supported():
        pytest.skip("torch.compile is not supported here")
    torch.manual_seed(42)
    mask = (
        torch.rand(
            16,
            24,
        )
        < 0.3
    ).float()
    layer = MaskedLinear(mask)
    x = torch.randn(
        4,
        24,
    )
    expected = layer(x)
    dynamo.reset()
    compiled = torch.compile(
        layer,
        fullgraph=True,
        backend="eager",
    )
    torch.testing.assert_close(
        compiled(x),
        expected,
    )


def test_masked_linear_weight_is_the_effective_masked_product():
    layer = _layer_with_pinned_diag_weights(bias=False)
    expected = _pinned_weight() * _diag_mask()
    torch.testing.assert_close(
        layer.weight,
        expected,
    )
    assert layer.weight.shape == (2, 2)


def test_masked_linear_trainable_tensor_is_the_parametrize_original():
    layer = MaskedLinear(
        _diag_mask(),
        bias=False,
    )
    assert parametrize.is_parametrized(
        layer,
        "weight",
    )
    original = layer.parametrizations.weight.original
    assert isinstance(original, nn.Parameter)
    assert original.shape == (2, 2)
    assert original is _original(layer)
    parameters = dict(layer.named_parameters())
    assert parameters["parametrizations.weight.original"] is original
    assert not isinstance(layer.weight, nn.Parameter)


def test_masked_linear_weight_assignment_writes_the_original():
    layer = MaskedLinear(
        _diag_mask(),
        bias=False,
    )
    before = _original(layer)
    with torch.no_grad():
        layer.weight = torch.full(
            (2, 2),
            2.0,
        )
    assert _original(layer) is before
    torch.testing.assert_close(
        _original(layer),
        torch.full(
            (2, 2),
            2.0,
        ),
    )
    torch.testing.assert_close(
        layer.weight,
        2.0 * _diag_mask(),
    )


def test_masked_linear_weight_assignment_copies_the_given_tensor():
    layer = MaskedLinear(
        _diag_mask(),
        bias=False,
    )
    source = torch.full(
        (2, 2),
        2.0,
    )
    with torch.no_grad():
        layer.weight = source
    source.fill_(9.0)
    torch.testing.assert_close(
        _original(layer),
        torch.full(
            (2, 2),
            2.0,
        ),
    )


def test_masked_linear_in_place_write_to_weight_is_discarded():
    layer = _layer_with_pinned_diag_weights(bias=False)
    with torch.no_grad():
        layer.weight.fill_(7.0)
    torch.testing.assert_close(
        _original(layer),
        _pinned_weight(),
    )


def test_masked_linear_grad_reaches_only_live_edges():
    mask = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    layer = MaskedLinear(
        mask,
        bias=False,
    )
    layer(
        torch.ones(
            3,
            2,
        )
    ).sum().backward()
    grad = _original(layer).grad
    assert grad is not None
    torch.testing.assert_close(
        grad,
        3.0 * mask,
    )


def test_masked_linear_optimizer_step_keeps_blocked_edges_dead():
    torch.manual_seed(42)
    layer = MaskedLinear(
        _diag_mask(),
        bias=False,
    )
    optimizer = torch.optim.SGD(
        layer.parameters(),
        lr=0.1,
        weight_decay=0.5,
    )
    assert optimizer.param_groups[0]["params"][0] is _original(layer)
    x = torch.ones(
        4,
        2,
    )
    loss = (layer(x) - 3.0).pow(2).sum()
    loss.backward()
    optimizer.step()
    blocked = layer.weight * (1.0 - _diag_mask())
    assert torch.equal(
        blocked,
        torch.zeros(
            2,
            2,
        ),
    )


def test_masked_linear_stays_an_instance_of_its_own_class():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        )
    )
    assert isinstance(layer, MaskedLinear)
    assert parametrize.type_before_parametrizations(layer) is MaskedLinear
    assert type(layer) is not MaskedLinear


def test_masked_linear_repr_reports_sizes():
    layer = MaskedLinear(
        torch.ones(
            2,
            3,
        ),
        bias=False,
    )
    text = repr(layer)
    assert "in_features=3" in text
    assert "out_features=2" in text
    assert "bias=False" in text


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
        _original(layer).fill_(1.0)
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
