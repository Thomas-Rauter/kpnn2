import copy

import pandas as pd
import pytest
import torch

from kpnn2 import Kpnn2Error, PackedLinear, parse_adjacency, parse_layered
from kpnn2._layout import hop_axis_layouts, iter_block_pairs


def _chain_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )


def _chain_plus_skip():
    return pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )


def _wide_two_parents():
    return pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["H", "H"],
        }
    )


def _cyclic_edgelist():
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a"],
            "target": ["a", "b", "a", "y"],
        }
    )


def test_layered_chain_lookup_is_the_unique_width_one_slot():
    spec = parse_layered(_chain_edgelist())
    hop_index, packed = spec.edge_location(
        "A",
        "H",
    )
    assert hop_index == 0
    assert packed == (0,)
    hop = spec.hops[hop_index]
    assert hop.source_index[packed[0]] == 0
    assert hop.target_index[packed[0]] == 0


def test_layered_skip_is_on_the_later_hop():
    spec = parse_layered(_chain_plus_skip())
    hop_index, packed = spec.edge_location(
        "A",
        "C",
    )
    assert hop_index == 1
    assert packed == (0,)
    adjacent = spec.edge_location(
        "A",
        "H",
    )
    assert adjacent[0] == 0
    assert adjacent[0] != hop_index
    hop = spec.hops[hop_index]
    assert hop.source_index[packed[0]] == 0
    assert hop.target_index[packed[0]] == 0


def test_layered_wide_named_edge_returns_the_full_block():
    spec = parse_layered(
        _wide_two_parents(),
        widths={
            "A": 2,
            "H": 3,
        },
    )
    hop_index, packed = spec.edge_location(
        "A",
        "H",
    )
    assert hop_index == 0
    assert len(packed) == 6
    hop = spec.hops[hop_index]
    source_layout, target_layout = hop_axis_layouts(
        spec.layer_nodes,
        spec.layer_widths,
        hop.source_layers,
        hop.target_layer,
    )
    source_slot = source_layout.slot("A")
    target_slot = target_layout.slot("H")
    expected_pairs = list(
        iter_block_pairs(
            source_slot,
            target_slot,
        )
    )
    got_pairs = [
        (
            hop.source_index[index],
            hop.target_index[index],
        )
        for index in packed
    ]
    assert got_pairs == expected_pairs

    layer = PackedLinear(
        hop.source_index,
        hop.target_index,
        hop.out_features,
        hop.in_features,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.fill_(0.0)
        layer.weight[list(packed)] = 0.5
    assert layer.weight[list(packed)].tolist() == [0.5] * 6
    other = spec.edge_location(
        "B",
        "H",
    )[1]
    assert layer.weight[list(other)].tolist() == [0.0] * len(other)

    mask = hop.to_mask()
    block = mask[
        target_slot.units,
        source_slot.units,
    ]
    assert torch.equal(
        block,
        torch.ones(
            target_slot.width,
            source_slot.width,
        ),
    )
    packed_set = set(packed)
    for index, (source_unit, target_unit) in enumerate(
        zip(
            hop.source_index,
            hop.target_index,
            strict=True,
        )
    ):
        in_block = (
            source_slot.start <= source_unit < source_slot.stop
            and target_slot.start <= target_unit < target_slot.stop
        )
        if index in packed_set:
            assert in_block
            assert mask[target_unit, source_unit].item() == 1.0
        else:
            assert not in_block


def test_adjacency_indices_match_packed_pairs_and_edgelist():
    spec = parse_adjacency(_cyclic_edgelist())
    pairs = list(
        zip(
            spec.source_index,
            spec.target_index,
        )
    )
    table = spec.to_edgelist()
    for row_index, (source, target) in enumerate(
        zip(
            table["source"],
            table["target"],
            strict=True,
        )
    ):
        packed = spec.edge_location(
            source,
            target,
        )
        source_unit = spec.nodes.index(source)
        target_unit = spec.nodes.index(target)
        expected = pairs.index(
            (
                source_unit,
                target_unit,
            )
        )
        assert packed == (expected,)
        assert packed == (row_index,)


def test_missing_pair_and_unknown_name_raise_kpnn2error():
    layered = parse_layered(_chain_plus_skip())
    adjacency = parse_adjacency(_cyclic_edgelist())
    cases = (
        (
            layered,
            "H",
            "A",
        ),
        (
            layered,
            "A",
            "Z",
        ),
        (
            layered,
            "",
            "H",
        ),
        (
            layered,
            "A",
            "",
        ),
        (
            adjacency,
            "y",
            "x",
        ),
        (
            adjacency,
            "z",
            "a",
        ),
        (
            adjacency,
            "",
            "a",
        ),
    )
    for spec, source, target in cases:
        with pytest.raises(
            Kpnn2Error,
            match=rf"{source} -> {target}",
        ) as caught:
            spec.edge_location(
                source,
                target,
            )
        assert "No edge" in str(caught.value)


def _named_edges(spec):
    table = spec.to_edgelist()
    return list(
        zip(
            table["source"].tolist(),
            table["target"].tolist(),
            strict=True,
        )
    )


def test_repeated_layered_lookup_reuses_stored_slots():
    widths = {
        "A": 2,
        "H": 3,
        "C": 2,
    }
    spec = parse_layered(
        _chain_plus_skip(),
        widths=widths,
    )
    twin = parse_layered(
        _chain_plus_skip(),
        widths=widths,
    )
    fingerprint = spec.fingerprint
    pairs = _named_edges(spec)
    first = [
        spec.edge_location(
            source,
            target,
        )
        for source, target in pairs
    ]
    second = [
        spec.edge_location(
            source,
            target,
        )
        for source, target in pairs
    ]
    assert first == second
    assert all(
        left is right
        for left, right in zip(
            first,
            second,
            strict=True,
        )
    )
    hop_index, packed = spec.edge_location(
        "A",
        "H",
    )
    assert hop_index == 0
    assert len(packed) == 6
    assert spec == twin
    assert spec.fingerprint == fingerprint
    assert twin.fingerprint == fingerprint
    copied = copy.deepcopy(spec)
    assert copied.edge_location(
        "A",
        "C",
    ) == spec.edge_location(
        "A",
        "C",
    )


def test_repeated_adjacency_lookup_reuses_stored_slots():
    spec = parse_adjacency(_cyclic_edgelist())
    twin = parse_adjacency(_cyclic_edgelist())
    fingerprint = spec.fingerprint
    pairs = _named_edges(spec)
    first = [
        spec.edge_location(
            source,
            target,
        )
        for source, target in pairs
    ]
    second = [
        spec.edge_location(
            source,
            target,
        )
        for source, target in pairs
    ]
    assert first == second
    assert all(
        left is right
        for left, right in zip(
            first,
            second,
            strict=True,
        )
    )
    assert first == [(index,) for index in range(len(pairs))]
    assert spec == twin
    assert spec.fingerprint == fingerprint
    copied = copy.deepcopy(spec)
    assert copied.edge_location(
        "x",
        "a",
    ) == spec.edge_location(
        "x",
        "a",
    )


def test_adjacency_self_loop_location_is_stable():
    spec = parse_adjacency(
        pd.DataFrame(
            {
                "source": ["x", "a", "a"],
                "target": ["a", "a", "y"],
            }
        )
    )
    packed = spec.edge_location(
        "a",
        "a",
    )
    assert packed == (0,)
    assert (
        spec.edge_location(
            "a",
            "a",
        )
        is packed
    )
    assert spec.source_index[packed[0]] == spec.nodes.index("a")
    assert spec.target_index[packed[0]] == spec.nodes.index("a")


def test_missing_edge_after_a_hit_still_raises():
    layered = parse_layered(_chain_plus_skip())
    adjacency = parse_adjacency(_cyclic_edgelist())
    layered.edge_location(
        "A",
        "H",
    )
    adjacency.edge_location(
        "a",
        "b",
    )
    with pytest.raises(
        Kpnn2Error,
        match=r"No edge H -> A",
    ):
        layered.edge_location(
            "H",
            "A",
        )
    with pytest.raises(
        Kpnn2Error,
        match=r"No edge y -> x",
    ):
        adjacency.edge_location(
            "y",
            "x",
        )


def test_str_matching_accepts_integer_names():
    edgelist = pd.DataFrame(
        {
            "source": [1, 2],
            "target": [2, 3],
        }
    )
    spec = parse_layered(edgelist)
    hop_index, packed = spec.edge_location(
        1,
        2,
    )
    as_strings = spec.edge_location(
        "1",
        "2",
    )
    assert hop_index == as_strings[0]
    assert packed == as_strings[1]
    assert packed == (0,)
