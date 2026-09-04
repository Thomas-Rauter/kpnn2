import pandas as pd
import pytest
import torch

import kpnn2
from kpnn2 import parse_adjacency
from tests.helpers import packed_attention as packed_attention_helpers
from tests.helpers.packed_attention import (
    allow_matrix,
    cyclic_edgelist,
    dense_masked_attention,
    rectangular_indices,
    self_loop_edgelist,
    shape_heads,
    three_cycle_indices,
)

_HELPER_NAMES = (
    "allow_matrix",
    "shape_heads",
    "dense_masked_attention",
    "cyclic_edgelist",
    "self_loop_edgelist",
    "three_cycle_indices",
    "rectangular_indices",
    "pin_projections_identity",
)


def test_packed_attention_helpers_are_not_public_api():
    for name in _HELPER_NAMES:
        assert hasattr(
            packed_attention_helpers,
            name,
        )
        assert name not in kpnn2.__all__
        assert not hasattr(
            kpnn2,
            name,
        )


def test_allow_matrix_matches_to_mask():
    spec = parse_adjacency(cyclic_edgelist())
    source_index = torch.tensor(
        spec.source_index,
        dtype=torch.int64,
    )
    target_index = torch.tensor(
        spec.target_index,
        dtype=torch.int64,
    )
    n = len(spec.nodes)
    got = allow_matrix(
        source_index,
        target_index,
        n,
        n,
    )
    assert torch.equal(
        got,
        spec.to_mask(),
    )
    nodes = spec.nodes
    live_row = nodes.index("a")
    live_col = nodes.index("x")
    assert got[live_row, live_col].item() == 1.0
    missing_row = nodes.index("a")
    missing_col = nodes.index("a")
    assert got[missing_row, missing_col].item() == 0.0


def test_one_live_key_mix_equals_value():
    query = torch.zeros(2, 1, 3)
    key = torch.zeros(2, 1, 3)
    value = torch.zeros(2, 1, 3)
    query[1, 0] = torch.tensor([1.0, 0.0, 0.0])
    key[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    value[0, 0] = torch.tensor([2.0, 3.0, 4.0])
    query[0, 0] = torch.tensor([9.0, 9.0, 9.0])
    key[1, 0] = torch.tensor([9.0, 9.0, 9.0])
    value[1, 0] = torch.tensor([9.0, 9.0, 9.0])
    mix = dense_masked_attention(
        query,
        key,
        value,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        mix[1],
        value[0],
    )
    assert torch.equal(
        mix[0],
        torch.zeros_like(mix[0]),
    )


def test_two_live_keys_equal_scores_mean_values():
    query = torch.ones(1, 1, 3)
    key = torch.ones(2, 1, 3)
    value = torch.tensor(
        [
            [[1.0, 2.0, 3.0]],
            [[5.0, 6.0, 7.0]],
        ]
    )
    mix = dense_masked_attention(
        query,
        key,
        value,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int64),
    )
    expected = (value[0] + value[1]) / 2.0
    torch.testing.assert_close(
        mix[0],
        expected,
    )


def test_two_live_keys_one_dominates():
    head_dim = 4
    scale = head_dim**-0.5
    query = torch.ones(1, 1, head_dim)
    key = torch.zeros(2, 1, head_dim)
    key[0, 0] = torch.ones(head_dim) * 2.0
    value = torch.zeros(2, 1, head_dim)
    value[0, 0, 0] = 1.0
    value[1, 0, 1] = 1.0
    mix = dense_masked_attention(
        query,
        key,
        value,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int64),
    )
    scores = torch.tensor(
        [
            float((query[0, 0] * key[0, 0]).sum() * scale),
            float((query[0, 0] * key[1, 0]).sum() * scale),
        ]
    )
    weights = torch.softmax(
        scores,
        dim=0,
    )
    expected = weights[0] * value[0] + weights[1] * value[1]
    torch.testing.assert_close(
        mix[0],
        expected,
    )
    dist_win = torch.norm(mix[0] - value[0])
    dist_lose = torch.norm(mix[0] - value[1])
    assert dist_win < dist_lose


def test_dead_key_does_not_contribute():
    query = torch.ones(1, 1, 2)
    key = torch.tensor(
        [
            [[1.0, 0.0]],
            [[100.0, 100.0]],
        ]
    )
    value = torch.tensor(
        [
            [[3.0, 4.0]],
            [[99.0, 99.0]],
        ]
    )
    source_index = torch.tensor([0], dtype=torch.int64)
    target_index = torch.tensor([0], dtype=torch.int64)
    mix_a = dense_masked_attention(
        query,
        key,
        value,
        source_index,
        target_index,
    )
    key_b = key.clone()
    value_b = value.clone()
    key_b[1] = torch.tensor([[-50.0, 80.0]])
    value_b[1] = torch.tensor([[-7.0, 8.0]])
    mix_b = dense_masked_attention(
        query,
        key_b,
        value_b,
        source_index,
        target_index,
    )
    assert torch.equal(
        mix_a,
        mix_b,
    )


def test_isolated_query_is_zeros_not_nan():
    query = torch.ones(3, 1, 2)
    key = torch.ones(2, 1, 2)
    value = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ]
    )
    mix = dense_masked_attention(
        query,
        key,
        value,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
    )
    isolated = mix[1]
    assert torch.equal(
        isolated,
        torch.zeros_like(isolated),
    )
    assert not torch.isnan(mix).any()
    assert torch.equal(
        mix[2],
        torch.zeros_like(mix[2]),
    )


def test_key_padding_mask_live_and_dead():
    query = torch.ones(2, 1, 2)
    key = torch.ones(2, 1, 2)
    value = torch.tensor(
        [
            [[1.0, 2.0]],
            [[9.0, 9.0]],
        ]
    )
    source_index = torch.tensor([0], dtype=torch.int64)
    target_index = torch.tensor([0], dtype=torch.int64)
    base = dense_masked_attention(
        query,
        key,
        value,
        source_index,
        target_index,
    )
    torch.testing.assert_close(
        base[0],
        value[0],
    )
    pad_live = torch.tensor(
        [True, False],
        dtype=torch.bool,
    )
    mix_pad_live = dense_masked_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        key_padding_mask=pad_live,
    )
    assert torch.equal(
        mix_pad_live[0],
        torch.zeros_like(mix_pad_live[0]),
    )
    assert not torch.isnan(mix_pad_live).any()
    pad_dead = torch.tensor(
        [False, True],
        dtype=torch.bool,
    )
    mix_pad_dead = dense_masked_attention(
        query,
        key,
        value,
        source_index,
        target_index,
        key_padding_mask=pad_dead,
    )
    assert torch.equal(
        mix_pad_dead,
        base,
    )
    with pytest.raises(TypeError):
        dense_masked_attention(
            query,
            key,
            value,
            source_index,
            target_index,
            key_padding_mask=torch.zeros(2),
        )


def test_shape_heads_round_trips():
    projected = torch.arange(
        2 * 4 * 8,
        dtype=torch.float32,
    ).reshape(2, 4, 8)
    headed = shape_heads(
        projected,
        2,
    )
    assert headed.shape == (2, 4, 2, 4)
    round_trip = headed.reshape(projected.shape)
    assert torch.equal(
        round_trip,
        projected,
    )


def test_edgelist_and_index_fixtures():
    spec = parse_adjacency(cyclic_edgelist())
    assert isinstance(
        spec.to_edgelist(),
        pd.DataFrame,
    )
    parse_adjacency(self_loop_edgelist())
    source, target, n = three_cycle_indices()
    assert len(source) == 3
    assert len(target) == 3
    assert n == 3
    assert sorted(target) == list(range(n))
    (
        rect_source,
        rect_target,
        query_features,
        key_features,
    ) = rectangular_indices()
    assert query_features != key_features
    assert 1 not in rect_target
    assert rect_source == [0, 1, 4]
    assert rect_target == [0, 0, 2]
    assert query_features == 3
    assert key_features == 5
