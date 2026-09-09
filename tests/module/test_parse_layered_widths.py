import pandas as pd
import pytest
import torch

from kpnn2 import (
    Kpnn2Error,
    PackedLinear,
    parse_layered,
)


def _chain_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )


def test_widths_none_matches_default_spec():
    edgelist = _chain_edgelist()
    default = parse_layered(edgelist)
    omitted = parse_layered(
        edgelist,
        widths=None,
    )
    empty = parse_layered(
        edgelist,
        widths={},
    )

    assert default.layer_dims == omitted.layer_dims == empty.layer_dims
    assert default.layer_widths == ((1,), (1,), (1,))
    assert omitted.layer_widths == default.layer_widths
    assert empty.layer_widths == default.layer_widths
    for left, right in zip(
        default.hops,
        omitted.hops,
        strict=True,
    ):
        assert left.source_index == right.source_index
        assert left.target_index == right.target_index
    assert default.fingerprint == omitted.fingerprint == empty.fingerprint
    assert set(default.to_dict()) == {
        "kpnn2_spec",
        "layout",
        "edges",
    }


def test_widths_h_two_expands_named_edges_into_blocks():
    spec = parse_layered(
        _chain_edgelist(),
        widths={"H": 2},
    )

    assert spec.layer_nodes == (("A",), ("H",), ("C",))
    assert spec.layer_widths == ((1,), (2,), (1,))
    assert spec.layer_dims == (1, 2, 1)
    assert spec.hops[0].source_nodes == ("A",)
    assert spec.hops[0].in_features == 1
    assert spec.hops[0].out_features == 2
    assert len(spec.hops[0].source_index) == 2
    assert spec.hops[0].to_mask().tolist() == [[1.0], [1.0]]
    assert spec.hops[1].source_nodes == ("H",)
    assert spec.hops[1].in_features == 2
    assert spec.hops[1].out_features == 1
    assert len(spec.hops[1].source_index) == 2
    assert spec.hops[1].to_mask().tolist() == [[1.0, 1.0]]

    layer = PackedLinear(
        spec.hops[0].source_index,
        spec.hops[0].target_index,
        spec.hops[0].out_features,
        spec.hops[0].in_features,
    )
    out = layer(torch.ones(3, 1))
    assert tuple(out.shape) == (3, 2)


def test_widths_missing_keys_default_to_one():
    spec = parse_layered(
        _chain_edgelist(),
        widths={"C": 2},
    )
    assert spec.layer_widths == ((1,), (1,), (2,))
    assert spec.layer_dims == (1, 1, 2)
    assert spec.hops[1].to_mask().tolist() == [[1.0], [1.0]]


def test_widths_unknown_name_errors():
    with pytest.raises(
        Kpnn2Error,
        match="Unknown node name",
    ) as caught:
        parse_layered(
            _chain_edgelist(),
            widths={"Z": 2, "Q": 3},
        )
    message = str(caught.value)
    assert "Q, Z" in message


def test_widths_rejects_bool_zero_and_negative():
    for value in (True, False, 0, -1):
        with pytest.raises(
            Kpnn2Error,
            match="positive int",
        ):
            parse_layered(
                _chain_edgelist(),
                widths={"H": value},
            )


def test_widths_rejects_non_mapping():
    with pytest.raises(
        Kpnn2Error,
        match="mapping of node name",
    ):
        parse_layered(
            _chain_edgelist(),
            widths=[("H", 2)],
        )


def test_edgelist_round_trip_requires_widths():
    spec = parse_layered(
        _chain_edgelist(),
        widths={"H": 2, "C": 2},
    )
    table = spec.to_edgelist()
    restored = parse_layered(
        table,
        widths={"H": 2, "C": 2},
    )
    assert restored.layer_widths == spec.layer_widths
    assert restored.hops[0].source_index == spec.hops[0].source_index
    default = parse_layered(table)
    assert default.layer_widths != spec.layer_widths
    assert default.fingerprint != spec.fingerprint
    assert default.to_dict() != spec.to_dict()
    assert len(table) == 2


def test_skip_stays_one_record_with_block_starts():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )
    spec = parse_layered(
        edgelist,
        widths={"A": 2, "C": 3},
    )
    assert len(spec.skips) == 1
    skip = spec.skips[0]
    assert skip.source == "A"
    assert skip.target == "C"
    assert skip.source_in_layer == 0
    assert skip.target_in_layer == 0
    hop = spec.hops[skip.target_layer - 1]
    assert len(hop.source_index) == 2 * 3 + 1 * 3
