import torch

from kpnn2 import (
    PackedMultiheadAttention,
    parse_adjacency,
)
from kpnn2._packed_multihead_attention import (
    _packed_attention,
    _padding_participate,
)
from tests.helpers.packed_attention import (
    allow_matrix,
    cyclic_edgelist,
    dense_masked_attention,
    rectangular_indices,
    shape_heads,
    three_cycle_indices,
)


def _assert_close(
    actual,
    expected,
):
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def _index_tensor(index):
    return torch.tensor(
        index,
        dtype=torch.int64,
    )


def _cyclic_indices():
    spec = parse_adjacency(cyclic_edgelist())
    source_index = _index_tensor(spec.source_index)
    target_index = _index_tensor(spec.target_index)
    return source_index, target_index, len(spec.nodes)


def _leaf_qkv(
    query,
    key,
    value,
):
    return (
        query.detach().clone().requires_grad_(True),
        key.detach().clone().requires_grad_(True),
        value.detach().clone().requires_grad_(True),
    )


def _grad_or_zeros(
    grad,
    like,
):
    if grad is None:
        return torch.zeros_like(like)
    return grad


def _sum_loss(mix):
    return mix.sum()


def _packed_backward(
    query,
    key,
    value,
    source_index,
    target_index,
    loss_fn,
    participate=None,
):
    q, k, v = _leaf_qkv(
        query,
        key,
        value,
    )
    mix = _packed_attention(
        q,
        k,
        v,
        source_index,
        target_index,
        dropout_p=0.0,
        training=False,
        participate=participate,
    )
    loss_fn(mix).backward()
    return q, k, v


def _dense_backward(
    query,
    key,
    value,
    source_index,
    target_index,
    loss_fn,
    key_padding_mask=None,
):
    q, k, v = _leaf_qkv(
        query,
        key,
        value,
    )
    mix = dense_masked_attention(
        q,
        k,
        v,
        source_index,
        target_index,
        key_padding_mask=key_padding_mask,
    )
    loss_fn(mix).backward()
    return q, k, v


def _assert_kernel_grads_match(
    query,
    key,
    value,
    source_index,
    target_index,
    loss_fn,
    participate=None,
    key_padding_mask=None,
):
    packed_q, packed_k, packed_v = _packed_backward(
        query,
        key,
        value,
        source_index,
        target_index,
        loss_fn,
        participate=participate,
    )
    dense_q, dense_k, dense_v = _dense_backward(
        query,
        key,
        value,
        source_index,
        target_index,
        loss_fn,
        key_padding_mask=key_padding_mask,
    )
    _assert_close(
        _grad_or_zeros(
            packed_q.grad,
            query,
        ),
        _grad_or_zeros(
            dense_q.grad,
            query,
        ),
    )
    _assert_close(
        _grad_or_zeros(
            packed_k.grad,
            key,
        ),
        _grad_or_zeros(
            dense_k.grad,
            key,
        ),
    )
    _assert_close(
        _grad_or_zeros(
            packed_v.grad,
            value,
        ),
        _grad_or_zeros(
            dense_v.grad,
            value,
        ),
    )
    return packed_q, packed_k, packed_v


def _assert_all_zeros(tensor):
    assert tensor is not None
    assert torch.equal(
        tensor,
        torch.zeros_like(tensor),
    )


def _assert_not_all_zeros(tensor):
    assert tensor is not None
    assert not torch.equal(
        tensor,
        torch.zeros_like(tensor),
    )


def _assert_index_buffers_have_no_grad(layer):
    assert not layer.source_index.requires_grad
    assert not layer.target_index.requires_grad
    assert layer.source_index.grad is None
    assert layer.target_index.grad is None


def test_packed_kernel_grads_match_dense_on_cyclic_spec():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    batch = 4
    num_heads = 2
    head_dim = 4
    embed_dim = num_heads * head_dim
    query = shape_heads(
        torch.randn(
            batch,
            n,
            embed_dim,
        ),
        num_heads,
    )
    key = shape_heads(
        torch.randn(
            batch,
            n,
            embed_dim,
        ),
        num_heads,
    )
    value = shape_heads(
        torch.randn(
            batch,
            n,
            embed_dim,
        ),
        num_heads,
    )
    _assert_kernel_grads_match(
        query,
        key,
        value,
        source_index,
        target_index,
        _sum_loss,
    )


def test_packed_kernel_grads_match_dense_on_three_cycle():
    torch.manual_seed(42)
    source_list, target_list, n = three_cycle_indices()
    source_index = _index_tensor(source_list)
    target_index = _index_tensor(target_list)
    allow = allow_matrix(
        source_index,
        target_index,
        n,
        n,
    )
    assert torch.all(allow.sum(dim=1) > 0)
    batch = 4
    num_heads = 2
    head_dim = 4
    query = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )
    key = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )
    value = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )
    _assert_kernel_grads_match(
        query,
        key,
        value,
        source_index,
        target_index,
        _sum_loss,
    )


def test_dead_pair_sends_zero_grad_into_that_key():
    torch.manual_seed(42)
    source_index, target_index, n = _cyclic_indices()
    allow = allow_matrix(
        source_index,
        target_index,
        n,
        n,
    )
    query_i = None
    key_dead = None
    key_live = None
    for candidate in range(n):
        live_keys = (
            (allow[candidate] > 0)
            .nonzero(
                as_tuple=False,
            )
            .view(-1)
        )
        dead_keys = (
            (allow[candidate] == 0)
            .nonzero(
                as_tuple=False,
            )
            .view(-1)
        )
        if live_keys.numel() >= 2 and dead_keys.numel() >= 1:
            query_i = candidate
            key_live = int(live_keys[0])
            key_dead = int(dead_keys[0])
            break
    assert query_i is not None
    batch = 3
    num_heads = 2
    head_dim = 4
    query = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )
    key = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )
    value = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )
    batch0 = 0
    head0 = 0

    def loss_fn(mix):
        return mix[
            batch0,
            query_i,
            head0,
            :,
        ].sum()

    _, packed_k, packed_v = _assert_kernel_grads_match(
        query,
        key,
        value,
        source_index,
        target_index,
        loss_fn,
    )
    _assert_all_zeros(packed_k.grad[batch0, key_dead])
    _assert_all_zeros(packed_v.grad[batch0, key_dead])
    _assert_not_all_zeros(packed_k.grad[batch0, key_live])
    _assert_not_all_zeros(packed_v.grad[batch0, key_live])


def test_isolated_query_sends_no_grad_into_keys():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    source_index = _index_tensor(spec.source_index)
    target_index = _index_tensor(spec.target_index)
    n = len(spec.nodes)
    isolated = spec.input_index[0]
    allow = allow_matrix(
        source_index,
        target_index,
        n,
        n,
    )
    assert allow[isolated].sum().item() == 0.0
    batch = 3
    num_heads = 2
    head_dim = 4
    query = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )
    key = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )
    value = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )

    def loss_fn(mix):
        return mix[:, isolated].sum()

    packed_q, packed_k, packed_v = _assert_kernel_grads_match(
        query,
        key,
        value,
        source_index,
        target_index,
        loss_fn,
    )
    _assert_all_zeros(
        _grad_or_zeros(
            packed_k.grad,
            key,
        )
    )
    _assert_all_zeros(
        _grad_or_zeros(
            packed_v.grad,
            value,
        )
    )
    packed_q_grad = _grad_or_zeros(
        packed_q.grad,
        query,
    )
    _assert_all_zeros(packed_q_grad[:, isolated])


def test_rectangular_dead_and_live_key_grads():
    torch.manual_seed(42)
    (
        source_list,
        target_list,
        query_features,
        key_features,
    ) = rectangular_indices()
    source_index = _index_tensor(source_list)
    target_index = _index_tensor(target_list)
    allow = allow_matrix(
        source_index,
        target_index,
        query_features,
        key_features,
    )
    assert allow[0, 0].item() == 1.0
    assert allow[0, 1].item() == 1.0
    assert allow[0, 4].item() == 0.0
    assert allow[1].sum().item() == 0.0
    batch = 3
    num_heads = 2
    head_dim = 4
    query = torch.randn(
        batch,
        query_features,
        num_heads,
        head_dim,
    )
    key = torch.randn(
        batch,
        key_features,
        num_heads,
        head_dim,
    )
    value = torch.randn(
        batch,
        key_features,
        num_heads,
        head_dim,
    )

    def loss_query_0(mix):
        return mix[:, 0].sum()

    _, packed_k, packed_v = _assert_kernel_grads_match(
        query,
        key,
        value,
        source_index,
        target_index,
        loss_query_0,
    )
    _assert_all_zeros(packed_k.grad[:, 4])
    _assert_all_zeros(packed_v.grad[:, 4])
    key0_live = not torch.equal(
        packed_k.grad[:, 0],
        torch.zeros_like(packed_k.grad[:, 0]),
    )
    key1_live = not torch.equal(
        packed_k.grad[:, 1],
        torch.zeros_like(packed_k.grad[:, 1]),
    )
    assert key0_live or key1_live

    def loss_query_1(mix):
        return mix[:, 1].sum()

    _, packed_k, packed_v = _assert_kernel_grads_match(
        query,
        key,
        value,
        source_index,
        target_index,
        loss_query_1,
    )
    _assert_all_zeros(
        _grad_or_zeros(
            packed_k.grad,
            key,
        )
    )
    _assert_all_zeros(
        _grad_or_zeros(
            packed_v.grad,
            value,
        )
    )


def test_public_module_projection_grads_are_finite_and_live():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    embed_dim = 8
    num_heads = 2
    layer = PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
    )
    layer.eval()
    batch = 4
    query = torch.randn(
        batch,
        n,
        embed_dim,
        requires_grad=True,
    )
    key = torch.randn(
        batch,
        n,
        embed_dim,
        requires_grad=True,
    )
    value = torch.randn(
        batch,
        n,
        embed_dim,
        requires_grad=True,
    )
    out, _ = layer(
        query,
        key,
        value,
    )
    out.sum().backward()
    for param in (
        layer.q_proj.weight,
        layer.k_proj.weight,
        layer.v_proj.weight,
        layer.out_proj.weight,
        layer.q_proj.bias,
        layer.v_proj.bias,
        layer.out_proj.bias,
    ):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        _assert_not_all_zeros(param.grad)
    # Same add on every key is a softmax-invariant score shift.
    assert layer.k_proj.bias.grad is not None
    assert torch.isfinite(layer.k_proj.bias.grad).all()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key.grad).all()
    assert torch.isfinite(value.grad).all()
    _assert_index_buffers_have_no_grad(layer)


def test_isolated_query_still_trains_out_proj():
    torch.manual_seed(42)
    spec = parse_adjacency(cyclic_edgelist())
    n = len(spec.nodes)
    isolated = spec.input_index[0]
    embed_dim = 8
    num_heads = 2
    layer = PackedMultiheadAttention(
        spec.source_index,
        spec.target_index,
        n,
        n,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
    )
    layer.eval()
    batch = 3
    query = torch.randn(
        batch,
        n,
        embed_dim,
        requires_grad=True,
    )
    key = torch.randn(
        batch,
        n,
        embed_dim,
        requires_grad=True,
    )
    value = torch.randn(
        batch,
        n,
        embed_dim,
        requires_grad=True,
    )
    out, _ = layer(
        query,
        key,
        value,
    )
    out[:, isolated].sum().backward()
    assert layer.out_proj.bias.grad is not None
    _assert_not_all_zeros(layer.out_proj.bias.grad)
    _assert_index_buffers_have_no_grad(layer)
    _assert_all_zeros(
        _grad_or_zeros(
            key.grad,
            key,
        )
    )
    _assert_all_zeros(
        _grad_or_zeros(
            value.grad,
            value,
        )
    )
    _assert_all_zeros(
        _grad_or_zeros(
            layer.k_proj.weight.grad,
            layer.k_proj.weight,
        )
    )
    _assert_all_zeros(
        _grad_or_zeros(
            layer.v_proj.weight.grad,
            layer.v_proj.weight,
        )
    )
    if layer.k_proj.bias is not None:
        _assert_all_zeros(
            _grad_or_zeros(
                layer.k_proj.bias.grad,
                layer.k_proj.bias,
            )
        )
    if layer.v_proj.bias is not None:
        _assert_all_zeros(
            _grad_or_zeros(
                layer.v_proj.bias.grad,
                layer.v_proj.bias,
            )
        )


def test_padding_grads_zero_through_padded_key():
    torch.manual_seed(42)
    source_list, target_list, n = three_cycle_indices()
    source_index = _index_tensor(source_list)
    target_index = _index_tensor(target_list)
    query_i = 0
    padded_key = 2
    remaining_key = 0
    live_sources = [
        source
        for source, target in zip(
            source_list,
            target_list,
        )
        if target == query_i
    ]
    assert live_sources == [padded_key]
    batch = 3
    num_heads = 2
    head_dim = 4
    query = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )
    key = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )
    value = torch.randn(
        batch,
        n,
        num_heads,
        head_dim,
    )
    mask = torch.zeros(
        batch,
        n,
        dtype=torch.bool,
    )
    mask[:, padded_key] = True
    participate = _padding_participate(
        mask,
        source_index,
        (batch,),
        n,
        query.device,
    )

    def loss_padded_query(mix):
        return mix[:, query_i].sum()

    _, packed_k, packed_v = _assert_kernel_grads_match(
        query,
        key,
        value,
        source_index,
        target_index,
        loss_padded_query,
        participate=participate,
        key_padding_mask=mask,
    )
    _assert_all_zeros(
        _grad_or_zeros(
            packed_k.grad,
            key,
        )
    )
    _assert_all_zeros(
        _grad_or_zeros(
            packed_v.grad,
            value,
        )
    )

    _, packed_k, packed_v = _assert_kernel_grads_match(
        query,
        key,
        value,
        source_index,
        target_index,
        _sum_loss,
        participate=participate,
        key_padding_mask=mask,
    )
    _assert_all_zeros(packed_k.grad[:, padded_key])
    _assert_all_zeros(packed_v.grad[:, padded_key])
    _assert_not_all_zeros(packed_v.grad[:, remaining_key])
