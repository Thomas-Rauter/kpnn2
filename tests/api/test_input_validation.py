"""
Every public entry point turns bad input into a ``Kpnn2Error``.

One table walks each name in ``kpnn2.__all__`` with bad calls: a
wrong type, a non-bool flag, an out-of-range value, a wrong shape,
and a semantic violation, wherever the category applies. Each row
names the text its message must contain: the argument or the
offending name. Type, flag, and value messages must also report
what was received (``Got``). A second table holds valid edge cases
that must not raise, so the first cannot pass by rejecting too
much. A guard fails when a public name has no row and no listed
reason.

Every layer here draws its init from a local generator seeded with
42, so the module never touches the global RNG.
"""

import re
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import kpnn2
from kpnn2 import Kpnn2Error

_TYPE = "type"
_FLAG = "flag"
_VALUE = "value"
_SHAPE = "shape"
_SEMANTIC = "semantic"

# Kinds whose message must report the received value.
_REPORTS_RECEIVED = frozenset(
    {
        _TYPE,
        _FLAG,
        _VALUE,
    }
)

# Public names with no bad-input row, and why. LayeredSpec and
# AdjacencySpec do have rows (their methods and from_dict); like
# Hop and Skip, they are valid only as returned by the parsers or
# from_dict, so direct construction has no row (CONTEXT.md
# **Errors**).
_EXEMPT = {
    "Hop": "valid only as returned by parse_layered; no argument",
    "Skip": "valid only as returned by parse_layered; no argument",
    "list_aggregation_methods": "takes no arguments",
    "Kpnn2Error": "the error type itself",
    "__version__": "a string, not a callable",
}


class _Bad(NamedTuple):
    public_name: str
    label: str
    kind: str
    match: str
    call: Callable[[], object]


class _Valid(NamedTuple):
    public_name: str
    label: str
    call: Callable[[], object]


def _generator():
    return torch.Generator().manual_seed(42)


# A -> H -> C plus the skip A -> C: layers (A,), (H,), (C,). The
# hop into C reads layers 0 and 1, so it is 2 units wide.
_EDGELIST = pd.DataFrame(
    {
        "source": ["A", "A", "H"],
        "target": ["H", "C", "C"],
    }
)
# x -> a <-> b -> ... -> y: a feedback edge, nodes a, b, x, y.
_CYCLE = pd.DataFrame(
    {
        "source": ["x", "a", "b", "a"],
        "target": ["a", "b", "a", "y"],
    }
)
_LAYERED = kpnn2.parse_layered(_EDGELIST)
_ADJACENCY = kpnn2.parse_adjacency(_CYCLE)
_OTHER_HOP = kpnn2.parse_layered(
    pd.DataFrame(
        {
            "source": ["P"],
            "target": ["Q"],
        }
    )
).hops[0]
_TWO_SOURCE_HOP = _LAYERED.hops[1]

_INDEX_INT32 = torch.tensor(
    [0, 1],
    dtype=torch.int32,
)
_INDEX_FLOAT = torch.tensor(
    [0.0, 1.0],
    dtype=torch.float32,
)
_TARGET_INT32 = np.array(
    [0, 0],
    dtype=np.int32,
)
_MASK_COMPLEX = torch.tensor(
    [[1.0, 1.0]],
    dtype=torch.complex64,
)
_LAYER_FLOAT64 = torch.tensor(
    [[1.0], [1.0], [1.0]],
    dtype=torch.float64,
)
_PADDING_FLOAT = torch.tensor([0.0, 0.0])
_PADDING_TOO_LONG = torch.tensor([False, False, False])

_SCORES = kpnn2.map_node_attributions(
    torch.tensor([[1.0], [2.0], [3.0]]),
    _LAYERED,
    layer=0,
)


def _layered(**kwargs):
    return kpnn2.parse_layered(
        _EDGELIST,
        **kwargs,
    )


def _adjacency(**kwargs):
    return kpnn2.parse_adjacency(
        _CYCLE,
        **kwargs,
    )


def _layered_payload(**overrides):
    return kpnn2.LayeredSpec.from_dict(
        {
            **_LAYERED.to_dict(),
            **overrides,
        }
    )


def _adjacency_payload(**overrides):
    return kpnn2.AdjacencySpec.from_dict(
        {
            **_ADJACENCY.to_dict(),
            **overrides,
        }
    )


def _masked(**overrides):
    """A 1-by-2 MaskedLinear with ``overrides`` applied."""
    arguments = {
        "mask": torch.tensor([[1.0, 1.0]]),
        "generator": _generator(),
        **overrides,
    }
    return kpnn2.MaskedLinear(**arguments)


def _packed(**overrides):
    """The PackedLinear of the hop into C, with ``overrides``."""
    arguments = {
        "source_index": [0, 1],
        "target_index": [0, 0],
        "out_features": 1,
        "in_features": 2,
        "generator": _generator(),
        **overrides,
    }
    return kpnn2.PackedLinear(**arguments)


def _attention(**overrides):
    """Two keys into one query, embed_dim 4, 2 heads."""
    arguments = {
        "source_index": [0, 1],
        "target_index": [0, 0],
        "query_features": 1,
        "key_features": 2,
        "embed_dim": 4,
        "num_heads": 2,
        "generator": _generator(),
        **overrides,
    }
    return kpnn2.PackedMultiheadAttention(**arguments)


_MASKED = _masked()
_PACKED = _packed()
_ATTENTION = _attention()


def _attend(**overrides):
    arguments = {
        "query": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        "key": torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
        "value": torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
        **overrides,
    }
    return _ATTENTION(**arguments)


def _gather(saved):
    return kpnn2.gather_hop_inputs(
        saved,
        _TWO_SOURCE_HOP,
    )


def _scatter(tensor):
    return kpnn2.scatter_hop_outputs(
        tensor,
        _TWO_SOURCE_HOP,
    )


def _align(names):
    return kpnn2.align_inputs(
        names,
        _LAYERED,
    )


def _map(**overrides):
    """Three observations of layer 0 (one unit), with ``overrides``."""
    arguments = {
        "attributions": torch.tensor([[1.0], [2.0], [3.0]]),
        "spec": _LAYERED,
        "layer": 0,
        **overrides,
    }
    return kpnn2.map_node_attributions(**arguments)


def _aggregate(**overrides):
    arguments = {
        "attributions": _SCORES,
        **overrides,
    }
    return kpnn2.aggregate_node_attributions(**arguments)


_PARSE_LAYERED_CASES = [
    _Bad(
        "parse_layered",
        "edgelist_not_a_frame",
        _TYPE,
        "'edgelist'",
        lambda: kpnn2.parse_layered({"source": ["A"], "target": ["H"]}),
    ),
    _Bad(
        "parse_layered",
        "widths_not_a_mapping",
        _TYPE,
        "'widths'",
        lambda: _layered(widths=[("H", 2)]),
    ),
    _Bad(
        "parse_layered",
        "width_is_float",
        _TYPE,
        "Width for node 'H'",
        lambda: _layered(widths={"H": 2.0}),
    ),
    _Bad(
        "parse_layered",
        "width_is_zero",
        _VALUE,
        "Width for node 'H'",
        lambda: _layered(widths={"H": 0}),
    ),
    _Bad(
        "parse_layered",
        "ranks_not_a_mapping",
        _TYPE,
        "'ranks'",
        lambda: _layered(ranks=[0, 1, 2]),
    ),
    _Bad(
        "parse_layered",
        "rank_is_bool",
        _TYPE,
        "Rank for node 'A'",
        lambda: _layered(ranks={"A": True, "H": 1, "C": 2}),
    ),
    _Bad(
        "parse_layered",
        "rank_is_negative",
        _VALUE,
        "Rank for node 'A'",
        lambda: _layered(ranks={"A": -1, "H": 1, "C": 2}),
    ),
    _Bad(
        "parse_layered",
        "target_column_missing",
        _SHAPE,
        "Missing: target",
        lambda: kpnn2.parse_layered(
            pd.DataFrame(
                {
                    "source": ["A"],
                    "dst": ["H"],
                }
            )
        ),
    ),
    _Bad(
        "parse_layered",
        "cycle",
        _SEMANTIC,
        "Unranked nodes: A, B, O",
        lambda: kpnn2.parse_layered(
            pd.DataFrame(
                {
                    "source": ["I", "A", "B", "B"],
                    "target": ["A", "B", "A", "O"],
                }
            )
        ),
    ),
    _Bad(
        "parse_layered",
        "self_loop",
        _SEMANTIC,
        "self-loop(s): A",
        lambda: kpnn2.parse_layered(
            pd.DataFrame(
                {
                    "source": ["A", "A"],
                    "target": ["A", "H"],
                }
            )
        ),
    ),
    _Bad(
        "parse_layered",
        "names_differ_only_by_whitespace",
        _SEMANTIC,
        "' A', 'A'",
        lambda: kpnn2.parse_layered(
            pd.DataFrame(
                {
                    "source": [" A", "A"],
                    "target": ["H", "H"],
                }
            )
        ),
    ),
    _Bad(
        "parse_layered",
        "widths_name_unknown_node",
        _SEMANTIC,
        "Unknown node name(s) in 'widths': Z",
        lambda: _layered(widths={"Z": 2}),
    ),
    _Bad(
        "parse_layered",
        "ranks_miss_a_node",
        _SEMANTIC,
        "Missing node name(s) in 'ranks': C, H",
        lambda: _layered(ranks={"A": 0}),
    ),
    _Bad(
        "parse_layered",
        "ranks_put_an_edge_backwards",
        _SEMANTIC,
        "Non-forward edge(s): H -> C",
        lambda: _layered(ranks={"A": 0, "H": 2, "C": 1}),
    ),
]

_PARSE_ADJACENCY_CASES = [
    _Bad(
        "parse_adjacency",
        "edgelist_is_an_array",
        _TYPE,
        "'edgelist'",
        lambda: kpnn2.parse_adjacency(_CYCLE.to_numpy()),
    ),
    _Bad(
        "parse_adjacency",
        "widths_not_a_mapping",
        _TYPE,
        "'widths'",
        lambda: _adjacency(widths=2),
    ),
    _Bad(
        "parse_adjacency",
        "width_is_numpy_bool",
        _TYPE,
        "Width for node 'a'",
        lambda: _adjacency(widths={"a": np.bool_(True)}),
    ),
    _Bad(
        "parse_adjacency",
        "width_is_negative",
        _VALUE,
        "Width for node 'a'",
        lambda: _adjacency(widths={"a": -1}),
    ),
    _Bad(
        "parse_adjacency",
        "no_rows",
        _SHAPE,
        "at least one edge",
        lambda: kpnn2.parse_adjacency(
            pd.DataFrame(
                {
                    "source": [],
                    "target": [],
                }
            )
        ),
    ),
    _Bad(
        "parse_adjacency",
        "missing_name",
        _SEMANTIC,
        "'source': 1 row(s) at index 1",
        lambda: kpnn2.parse_adjacency(
            pd.DataFrame(
                {
                    "source": ["x", None],
                    "target": ["y", "y"],
                }
            )
        ),
    ),
    _Bad(
        "parse_adjacency",
        "names_differ_only_by_whitespace",
        _SEMANTIC,
        "'x', 'x '",
        lambda: kpnn2.parse_adjacency(
            pd.DataFrame(
                {
                    "source": ["x", "x "],
                    "target": ["y", "z"],
                }
            )
        ),
    ),
    _Bad(
        "parse_adjacency",
        "duplicate_edge",
        _SEMANTIC,
        "x -> y",
        lambda: kpnn2.parse_adjacency(
            pd.DataFrame(
                {
                    "source": ["x", "x"],
                    "target": ["y", "y"],
                }
            )
        ),
    ),
    _Bad(
        "parse_adjacency",
        "ring_has_no_input",
        _SEMANTIC,
        "at least one input node",
        lambda: kpnn2.parse_adjacency(
            pd.DataFrame(
                {
                    "source": ["a", "b"],
                    "target": ["b", "a"],
                }
            )
        ),
    ),
]

_LAYERED_SPEC_CASES = [
    _Bad(
        "LayeredSpec",
        "edge_location_absent_edge",
        _SEMANTIC,
        "No edge H -> A",
        lambda: _LAYERED.edge_location(
            "H",
            "A",
        ),
    ),
    _Bad(
        "LayeredSpec",
        "edge_location_unknown_name",
        _SEMANTIC,
        "Unknown node name: Z",
        lambda: _LAYERED.edge_location(
            "A",
            "Z",
        ),
    ),
    _Bad(
        "LayeredSpec",
        "edge_location_empty_name",
        _SEMANTIC,
        "Node name is empty",
        lambda: _LAYERED.edge_location(
            "",
            "H",
        ),
    ),
    _Bad(
        "LayeredSpec",
        "node_units_unknown_name",
        _SEMANTIC,
        "Unknown node name: Z",
        lambda: _LAYERED.node_units("Z"),
    ),
    _Bad(
        "LayeredSpec",
        "node_units_name_with_whitespace",
        _SEMANTIC,
        "Unknown node name:  A",
        lambda: _LAYERED.node_units(" A"),
    ),
    _Bad(
        "LayeredSpec",
        "hop_units_hop_is_an_index",
        _TYPE,
        "'hop'",
        lambda: _LAYERED.hop_units(
            0,
            "A",
        ),
    ),
    _Bad(
        "LayeredSpec",
        "hop_units_hop_from_another_spec",
        _SEMANTIC,
        "'hop' must match an entry of spec.hops",
        lambda: _LAYERED.hop_units(
            _OTHER_HOP,
            "A",
        ),
    ),
    _Bad(
        "LayeredSpec",
        "hop_units_target_name",
        _SEMANTIC,
        "Node H is not on this hop's source axis",
        lambda: _LAYERED.hop_units(
            _LAYERED.hops[0],
            "H",
        ),
    ),
    _Bad(
        "LayeredSpec",
        "from_dict_payload_not_a_dict",
        _TYPE,
        "'payload'",
        lambda: kpnn2.LayeredSpec.from_dict([["A", "H"]]),
    ),
    _Bad(
        "LayeredSpec",
        "from_dict_schema_tag_is_str",
        _TYPE,
        "'kpnn2_spec'",
        lambda: _layered_payload(kpnn2_spec="1"),
    ),
    _Bad(
        "LayeredSpec",
        "from_dict_schema_tag_is_2",
        _VALUE,
        "'kpnn2_spec'",
        lambda: _layered_payload(kpnn2_spec=2),
    ),
    _Bad(
        "LayeredSpec",
        "from_dict_widths_not_a_mapping",
        _TYPE,
        "'widths'",
        lambda: _layered_payload(widths=[1]),
    ),
    _Bad(
        "LayeredSpec",
        "from_dict_ranks_not_a_mapping",
        _TYPE,
        "'ranks'",
        lambda: _layered_payload(ranks=[0, 1, 2]),
    ),
    _Bad(
        "LayeredSpec",
        "from_dict_edge_not_a_pair",
        _SHAPE,
        "edges[0]",
        lambda: _layered_payload(edges=[["A", "H", "C"]]),
    ),
    _Bad(
        "LayeredSpec",
        "from_dict_missing_name",
        _SEMANTIC,
        "'target': 1 row(s) at index 0",
        lambda: _layered_payload(edges=[["A", None]]),
    ),
    _Bad(
        "LayeredSpec",
        "from_dict_adjacency_layout",
        _SEMANTIC,
        "layout 'adjacency'",
        lambda: _layered_payload(layout="adjacency"),
    ),
]

_ADJACENCY_SPEC_CASES = [
    _Bad(
        "AdjacencySpec",
        "edge_location_absent_edge",
        _SEMANTIC,
        "No edge x -> y",
        lambda: _ADJACENCY.edge_location(
            "x",
            "y",
        ),
    ),
    _Bad(
        "AdjacencySpec",
        "edge_location_unknown_name",
        _SEMANTIC,
        "Unknown node name: q",
        lambda: _ADJACENCY.edge_location(
            "x",
            "q",
        ),
    ),
    _Bad(
        "AdjacencySpec",
        "node_units_unknown_name",
        _SEMANTIC,
        "Unknown node name: q",
        lambda: _ADJACENCY.node_units("q"),
    ),
    _Bad(
        "AdjacencySpec",
        "node_units_empty_name",
        _SEMANTIC,
        "Node name is empty",
        lambda: _ADJACENCY.node_units(""),
    ),
    _Bad(
        "AdjacencySpec",
        "from_dict_payload_is_json_text",
        _TYPE,
        "'payload'",
        lambda: kpnn2.AdjacencySpec.from_dict("{}"),
    ),
    _Bad(
        "AdjacencySpec",
        "from_dict_layout_is_int",
        _TYPE,
        "'layout'",
        lambda: _adjacency_payload(layout=5),
    ),
    _Bad(
        "AdjacencySpec",
        "from_dict_edges_is_str",
        _TYPE,
        "'edges'",
        lambda: _adjacency_payload(edges="ab"),
    ),
    _Bad(
        "AdjacencySpec",
        "from_dict_width_is_zero",
        _VALUE,
        "Width for node 'a'",
        lambda: _adjacency_payload(widths={"a": 0}),
    ),
    _Bad(
        "AdjacencySpec",
        "from_dict_layered_layout",
        _SEMANTIC,
        "layout 'layered'",
        lambda: _adjacency_payload(layout="layered"),
    ),
]

_MASKED_LINEAR_CASES = [
    _Bad(
        "MaskedLinear",
        "mask_is_an_array",
        _TYPE,
        "'mask'",
        lambda: _masked(mask=np.array([[1.0, 1.0]])),
    ),
    _Bad(
        "MaskedLinear",
        "mask_is_complex",
        _TYPE,
        "'mask'",
        lambda: _masked(mask=_MASK_COMPLEX),
    ),
    _Bad(
        "MaskedLinear",
        "bias_is_str",
        _FLAG,
        "'bias'",
        lambda: _masked(bias="False"),
    ),
    _Bad(
        "MaskedLinear",
        "bias_is_numpy_bool",
        _FLAG,
        "'bias'",
        lambda: _masked(bias=np.bool_(True)),
    ),
    _Bad(
        "MaskedLinear",
        "identity_is_int",
        _TYPE,
        "'identity'",
        lambda: _masked(identity=5),
    ),
    _Bad(
        "MaskedLinear",
        "constraint_is_str",
        _TYPE,
        "'constraint'",
        lambda: _masked(constraint="softplus"),
    ),
    _Bad(
        "MaskedLinear",
        "generator_is_a_seed",
        _TYPE,
        "'generator'",
        lambda: _masked(generator=42),
    ),
    _Bad(
        "MaskedLinear",
        "mask_entry_is_2",
        _VALUE,
        "'mask' must contain only 0 and 1",
        lambda: _masked(mask=torch.tensor([[2.0]])),
    ),
    _Bad(
        "MaskedLinear",
        "mask_entry_is_nan",
        _VALUE,
        "'mask' must contain only 0 and 1",
        lambda: _masked(mask=torch.tensor([[1.0, float("nan")]])),
    ),
    _Bad(
        "MaskedLinear",
        "mask_is_1d",
        _SHAPE,
        "'mask'",
        lambda: _masked(mask=torch.tensor([1.0, 1.0])),
    ),
    _Bad(
        "MaskedLinear",
        "mask_has_no_row",
        _SHAPE,
        "'mask'",
        lambda: _masked(mask=torch.tensor([[]]).T),
    ),
    _Bad(
        "MaskedLinear",
        "mask_is_all_zero",
        _SEMANTIC,
        "'mask' has no live entry",
        lambda: _masked(mask=torch.tensor([[0.0, 0.0]])),
    ),
    _Bad(
        "MaskedLinear",
        "forward_input_is_a_list",
        _TYPE,
        "MaskedLinear input",
        lambda: _MASKED([[1.0, 1.0]]),
    ),
    _Bad(
        "MaskedLinear",
        "forward_input_too_wide",
        _SHAPE,
        "in_features=2",
        lambda: _MASKED(torch.tensor([[1.0, 1.0, 1.0]])),
    ),
    _Bad(
        "MaskedLinear",
        "forward_input_is_0d",
        _SHAPE,
        "in_features=2",
        lambda: _MASKED(torch.tensor(1.0)),
    ),
    _Bad(
        "MaskedLinear",
        "reset_parameters_generator_is_a_seed",
        _TYPE,
        "'generator'",
        lambda: _MASKED.reset_parameters(42),
    ),
]

_PACKED_LINEAR_CASES = [
    _Bad(
        "PackedLinear",
        "in_features_is_float",
        _TYPE,
        "'in_features'",
        lambda: _packed(in_features=2.0),
    ),
    _Bad(
        "PackedLinear",
        "out_features_is_zero",
        _VALUE,
        "'out_features'",
        lambda: _packed(out_features=0),
    ),
    _Bad(
        "PackedLinear",
        "bias_is_str",
        _FLAG,
        "'bias'",
        lambda: _packed(bias="no"),
    ),
    _Bad(
        "PackedLinear",
        "source_index_is_str",
        _TYPE,
        "'source_index'",
        lambda: _packed(source_index="01"),
    ),
    _Bad(
        "PackedLinear",
        "source_index_holds_floats",
        _TYPE,
        "at position 0",
        lambda: _packed(source_index=[0.0, 1.0]),
    ),
    _Bad(
        "PackedLinear",
        "target_index_is_a_float_tensor",
        _TYPE,
        "'target_index'",
        lambda: _packed(target_index=_INDEX_FLOAT),
    ),
    _Bad(
        "PackedLinear",
        "source_index_is_a_complex_tensor",
        _TYPE,
        "dtype torch.complex64",
        lambda: _packed(source_index=torch.tensor([0j, 1j])),
    ),
    _Bad(
        "PackedLinear",
        "source_index_is_a_set",
        _TYPE,
        "'source_index' must be a 1-dimensional integer tensor or a "
        "sequence of int, not a set",
        lambda: _packed(source_index={1, 0}),
    ),
    _Bad(
        "PackedLinear",
        "constraint_is_a_class",
        _TYPE,
        "'constraint'",
        lambda: _packed(constraint=nn.Softplus),
    ),
    _Bad(
        "PackedLinear",
        "source_index_entry_out_of_range",
        _VALUE,
        "0 <= source_index < in_features",
        lambda: _packed(source_index=[0, 2]),
    ),
    _Bad(
        "PackedLinear",
        "source_index_is_2d",
        _SHAPE,
        "'source_index'",
        lambda: _packed(source_index=torch.tensor([[0, 1]])),
    ),
    _Bad(
        "PackedLinear",
        "index_lengths_differ",
        _SHAPE,
        "'source_index' and 'target_index' must have the same length",
        lambda: _packed(target_index=[0]),
    ),
    _Bad(
        "PackedLinear",
        "indices_are_empty",
        _SHAPE,
        "at least one index",
        lambda: _packed(
            source_index=[],
            target_index=[],
        ),
    ),
    _Bad(
        "PackedLinear",
        "duplicate_pair",
        _SEMANTIC,
        "(0, 0)",
        lambda: _packed(
            source_index=[0, 0],
            target_index=[0, 0],
        ),
    ),
    _Bad(
        "PackedLinear",
        "forward_input_is_an_array",
        _TYPE,
        "PackedLinear input",
        lambda: _PACKED(np.array([[1.0, 1.0]])),
    ),
    _Bad(
        "PackedLinear",
        "forward_input_too_wide",
        _SHAPE,
        "in_features=2",
        lambda: _PACKED(torch.tensor([[1.0, 1.0, 1.0]])),
    ),
    _Bad(
        "PackedLinear",
        "transpose_tie_is_str",
        _FLAG,
        "'tie'",
        lambda: _PACKED.transpose(tie="yes"),
    ),
    _Bad(
        "PackedLinear",
        "transpose_bias_is_int",
        _FLAG,
        "'bias'",
        lambda: _PACKED.transpose(bias=1),
    ),
    _Bad(
        "PackedLinear",
        "transpose_identity_is_int",
        _TYPE,
        "'identity'",
        lambda: _PACKED.transpose(identity=5),
    ),
    _Bad(
        "PackedLinear",
        "transpose_generator_is_a_seed",
        _TYPE,
        "'generator'",
        lambda: _PACKED.transpose(generator=42),
    ),
    _Bad(
        "PackedLinear",
        "reset_parameters_generator_is_a_seed",
        _TYPE,
        "'generator'",
        lambda: _PACKED.reset_parameters(42),
    ),
]

_ATTENTION_CASES = [
    _Bad(
        "PackedMultiheadAttention",
        "embed_dim_is_float",
        _TYPE,
        "'embed_dim'",
        lambda: _attention(embed_dim=4.0),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "num_heads_is_zero",
        _VALUE,
        "'num_heads'",
        lambda: _attention(num_heads=0),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "dropout_is_str",
        _TYPE,
        "'dropout'",
        lambda: _attention(dropout="0.1"),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "dropout_above_1",
        _VALUE,
        "'dropout'",
        lambda: _attention(dropout=1.5),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "dropout_below_0",
        _VALUE,
        "'dropout'",
        lambda: _attention(dropout=-0.1),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "chunk_size_is_zero",
        _VALUE,
        "'chunk_size'",
        lambda: _attention(chunk_size=0),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "bias_is_none",
        _FLAG,
        "'bias'",
        lambda: _attention(bias=None),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "batch_first_is_str",
        _FLAG,
        "'batch_first'",
        lambda: _attention(batch_first="False"),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "add_self_loops_is_int",
        _FLAG,
        "'add_self_loops'",
        lambda: _attention(add_self_loops=1),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "identity_is_bytes",
        _TYPE,
        "'identity'",
        lambda: _attention(identity=b"fingerprint"),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "generator_is_a_seed",
        _TYPE,
        "'generator'",
        lambda: _attention(generator=42),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "target_index_entry_out_of_range",
        _VALUE,
        "0 <= target_index < query_features",
        lambda: _attention(target_index=[0, 1]),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "source_index_is_2d",
        _SHAPE,
        "'source_index'",
        lambda: _attention(source_index=torch.tensor([[0, 1]])),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "target_index_is_a_mapping",
        _TYPE,
        "'target_index' must be a 1-dimensional integer tensor or a "
        "sequence of int, not a mapping",
        lambda: _attention(target_index={0: 0, 1: 0}),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "embed_dim_not_divisible",
        _SEMANTIC,
        "'embed_dim' must be divisible by 'num_heads'",
        lambda: _attention(num_heads=3),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "kdim_differs_from_embed_dim",
        _SEMANTIC,
        "'kdim'",
        lambda: _attention(kdim=8),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "self_loops_on_unequal_axes",
        _SEMANTIC,
        "'add_self_loops' requires query_features == key_features",
        lambda: _attention(add_self_loops=True),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "duplicate_pair",
        _SEMANTIC,
        "(0, 0)",
        lambda: _attention(
            source_index=[0, 0],
            target_index=[0, 0],
        ),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "forward_query_is_a_list",
        _TYPE,
        "'query'",
        lambda: _attend(query=[[1.0, 1.0, 1.0, 1.0]]),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "forward_need_weights_is_str",
        _FLAG,
        "'need_weights'",
        lambda: _attend(need_weights="yes"),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "forward_average_attn_weights_is_int",
        _FLAG,
        "'average_attn_weights'",
        lambda: _attend(average_attn_weights=0),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "forward_padding_mask_is_float",
        _TYPE,
        "'key_padding_mask'",
        lambda: _attend(key_padding_mask=_PADDING_FLOAT),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "forward_query_embed_dim_wrong",
        _SHAPE,
        "'query' last dimension must equal embed_dim",
        lambda: _attend(query=torch.tensor([[1.0, 1.0, 1.0]])),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "forward_key_sequence_too_short",
        _SHAPE,
        "key sequence length must equal key_features",
        lambda: _attend(key=torch.tensor([[1.0, 1.0, 1.0, 1.0]])),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "forward_padding_mask_too_long",
        _SHAPE,
        "'key_padding_mask'",
        lambda: _attend(key_padding_mask=_PADDING_TOO_LONG),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "forward_attn_mask_given",
        _SEMANTIC,
        "'attn_mask' must be None",
        lambda: _attend(attn_mask=torch.tensor([[False, False]])),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "forward_is_causal_true",
        _SEMANTIC,
        "'is_causal' must be False",
        lambda: _attend(is_causal=True),
    ),
    _Bad(
        "PackedMultiheadAttention",
        "reset_parameters_generator_is_a_seed",
        _TYPE,
        "'generator'",
        lambda: _ATTENTION.reset_parameters(42),
    ),
]

_HOP_AXIS_CASES = [
    _Bad(
        "gather_hop_inputs",
        "saved_is_a_list",
        _TYPE,
        "'saved'",
        lambda: _gather([torch.tensor([[1.0]]), torch.tensor([[1.0]])]),
    ),
    _Bad(
        "gather_hop_inputs",
        "hop_is_an_index",
        _TYPE,
        "'hop'",
        lambda: kpnn2.gather_hop_inputs(
            {0: torch.tensor([[1.0]])},
            1,
        ),
    ),
    _Bad(
        "gather_hop_inputs",
        "saved_layer_is_an_array",
        _TYPE,
        "saved[0]",
        lambda: _gather({0: np.array([[1.0]]), 1: torch.tensor([[1.0]])}),
    ),
    _Bad(
        "gather_hop_inputs",
        "saved_layer_too_wide",
        _SHAPE,
        "saved[1] has the wrong number of units",
        lambda: _gather(
            {
                0: torch.tensor([[1.0]]),
                1: torch.tensor([[1.0, 1.0]]),
            }
        ),
    ),
    _Bad(
        "gather_hop_inputs",
        "saved_misses_a_layer",
        _SEMANTIC,
        "saved is missing layer 1",
        lambda: _gather({0: torch.tensor([[1.0]])}),
    ),
    _Bad(
        "gather_hop_inputs",
        "saved_layers_differ_in_dtype",
        _SEMANTIC,
        "saved[1] is torch.float64",
        lambda: _gather(
            {
                0: torch.tensor([[1.0], [1.0], [1.0]]),
                1: _LAYER_FLOAT64,
            }
        ),
    ),
    _Bad(
        "gather_hop_inputs",
        "saved_layers_differ_in_batch_size",
        _SHAPE,
        "saved[1] has shape (3, 1)",
        lambda: _gather(
            {
                0: torch.tensor([[1.0], [1.0]]),
                1: torch.tensor([[1.0], [1.0], [1.0]]),
            }
        ),
    ),
    _Bad(
        "scatter_hop_outputs",
        "tensor_is_an_array",
        _TYPE,
        "'tensor'",
        lambda: _scatter(np.array([[1.0, 1.0]])),
    ),
    _Bad(
        "scatter_hop_outputs",
        "hop_is_a_skip",
        _TYPE,
        "'hop'",
        lambda: kpnn2.scatter_hop_outputs(
            torch.tensor([[1.0, 1.0]]),
            _LAYERED.skips[0],
        ),
    ),
    _Bad(
        "scatter_hop_outputs",
        "tensor_too_wide",
        _SHAPE,
        "tensor has the wrong number of units",
        lambda: _scatter(torch.tensor([[1.0, 1.0, 1.0]])),
    ),
]

_ALIGN_CASES = [
    _Bad(
        "align_inputs",
        "names_is_an_int",
        _TYPE,
        "names",
        lambda: _align(5),
    ),
    _Bad(
        "align_inputs",
        "spec_is_a_frame",
        _TYPE,
        "'spec'",
        lambda: kpnn2.align_inputs(
            ["A"],
            _EDGELIST,
        ),
    ),
    _Bad(
        "align_inputs",
        "names_is_a_matrix",
        _SHAPE,
        "'names' must be one-dimensional",
        lambda: _align(np.array([["A"]])),
    ),
    _Bad(
        "align_inputs",
        "names_repeat_a_label",
        _SEMANTIC,
        "strings): A",
        lambda: _align(["A", "A"]),
    ),
    _Bad(
        "align_inputs",
        "names_miss_an_input",
        _SEMANTIC,
        "required name(s): A",
        lambda: _align(["H"]),
    ),
]

_MAP_CASES = [
    _Bad(
        "map_node_attributions",
        "attributions_is_an_array",
        _TYPE,
        "'attributions'",
        lambda: _map(attributions=np.array([[1.0], [2.0]])),
    ),
    _Bad(
        "map_node_attributions",
        "attributions_holds_an_array",
        _TYPE,
        "'attributions'",
        lambda: _map(attributions=[np.array([1.0])]),
    ),
    _Bad(
        "map_node_attributions",
        "spec_is_a_str",
        _TYPE,
        "'spec'",
        lambda: _map(spec="LayeredSpec"),
    ),
    _Bad(
        "map_node_attributions",
        "layer_is_float",
        _TYPE,
        "'layer'",
        lambda: _map(layer=1.0),
    ),
    _Bad(
        "map_node_attributions",
        "layer_is_bool",
        _TYPE,
        "'layer'",
        lambda: _map(layer=True),
    ),
    _Bad(
        "map_node_attributions",
        "dims_hold_an_int",
        _TYPE,
        "'dims'",
        lambda: _map(dims=[0, "node"]),
    ),
    _Bad(
        "map_node_attributions",
        "dims_is_a_str",
        _TYPE,
        "'dims'",
        lambda: _map(dims="node"),
    ),
    _Bad(
        "map_node_attributions",
        "dims_is_a_bool",
        _TYPE,
        "'dims'",
        lambda: _map(dims=True),
    ),
    _Bad(
        "map_node_attributions",
        "dims_is_a_set",
        _TYPE,
        "'dims' must be a sequence of strings, one per tensor axis, not a set",
        lambda: _map(dims={"node"}),
    ),
    _Bad(
        "map_node_attributions",
        "coords_not_a_mapping",
        _TYPE,
        "'coords'",
        lambda: _map(coords=[("observation", [0, 1, 2])]),
    ),
    _Bad(
        "map_node_attributions",
        "coords_value_is_none",
        _TYPE,
        "'coords['observation']'",
        lambda: _map(coords={"observation": None}),
    ),
    _Bad(
        "map_node_attributions",
        "coords_value_is_a_str",
        _TYPE,
        "'coords['observation']'",
        lambda: _map(coords={"observation": "abc"}),
    ),
    _Bad(
        "map_node_attributions",
        "coords_value_is_a_mapping",
        _TYPE,
        "not a mapping",
        lambda: _map(coords={"observation": {0: "a", 1: "b", 2: "c"}}),
    ),
    _Bad(
        "map_node_attributions",
        "axis_is_an_array",
        _TYPE,
        "'axis'",
        lambda: _map(
            layer=None,
            axis=np.array(["inputs"]),
        ),
    ),
    _Bad(
        "map_node_attributions",
        "layer_out_of_range",
        _VALUE,
        "'layer'",
        lambda: _map(layer=3),
    ),
    _Bad(
        "map_node_attributions",
        "axis_is_unknown",
        _VALUE,
        "'axis'",
        lambda: _map(
            layer=None,
            axis="outputs",
        ),
    ),
    _Bad(
        "map_node_attributions",
        "tensor_too_wide",
        _SHAPE,
        "wrong number of units",
        lambda: _map(attributions=torch.tensor([[1.0, 2.0]])),
    ),
    _Bad(
        "map_node_attributions",
        "dims_length_differs",
        _SHAPE,
        "'dims' length",
        lambda: _map(dims=["node"]),
    ),
    _Bad(
        "map_node_attributions",
        "two_axis_selectors",
        _SEMANTIC,
        "Got layer, hop_output",
        lambda: _map(hop_output=_LAYERED.hops[0]),
    ),
    _Bad(
        "map_node_attributions",
        "hop_output_from_another_spec",
        _SEMANTIC,
        "'hop_output' must match an entry of spec.hops",
        lambda: _map(
            layer=None,
            hop_output=_OTHER_HOP,
        ),
    ),
    _Bad(
        "map_node_attributions",
        "layer_on_an_adjacency_spec",
        _SEMANTIC,
        "do not apply to an AdjacencySpec",
        lambda: _map(spec=_ADJACENCY),
    ),
]

_AGGREGATE_CASES = [
    _Bad(
        "aggregate_node_attributions",
        "attributions_is_a_tensor",
        _TYPE,
        "'attributions'",
        lambda: _aggregate(attributions=torch.tensor([[1.0]])),
    ),
    _Bad(
        "aggregate_node_attributions",
        "method_is_none",
        _TYPE,
        "'method'",
        lambda: _aggregate(method=None),
    ),
    _Bad(
        "aggregate_node_attributions",
        "method_is_unknown",
        _SEMANTIC,
        "Unknown aggregation method 'no_such_method'",
        lambda: _aggregate(method="no_such_method"),
    ),
]

_BAD_CASES = [
    *_PARSE_LAYERED_CASES,
    *_PARSE_ADJACENCY_CASES,
    *_LAYERED_SPEC_CASES,
    *_ADJACENCY_SPEC_CASES,
    *_MASKED_LINEAR_CASES,
    *_PACKED_LINEAR_CASES,
    *_ATTENTION_CASES,
    *_HOP_AXIS_CASES,
    *_ALIGN_CASES,
    *_MAP_CASES,
    *_AGGREGATE_CASES,
]

_VALID_CASES = [
    _Valid(
        "parse_layered",
        "numpy_integer_widths_and_ranks",
        lambda: _layered(
            widths={"H": np.int64(2)},
            ranks={"A": np.int64(0), "H": np.int32(1), "C": np.uint8(2)},
        ),
    ),
    _Valid(
        "parse_layered",
        "lone_name_with_whitespace",
        lambda: kpnn2.parse_layered(
            pd.DataFrame(
                {
                    "source": [" A"],
                    "target": ["H"],
                }
            )
        ).node_units(" A"),
    ),
    _Valid(
        "parse_adjacency",
        "numpy_integer_widths",
        lambda: _adjacency(widths={"a": np.int32(2)}),
    ),
    _Valid(
        "parse_adjacency",
        "lone_name_with_whitespace",
        lambda: kpnn2.parse_adjacency(
            pd.DataFrame(
                {
                    "source": ["x"],
                    "target": ["y "],
                }
            )
        ).node_units("y "),
    ),
    _Valid(
        "MaskedLinear",
        "bool_mask",
        lambda: _masked(mask=torch.tensor([[True, False]])),
    ),
    _Valid(
        "MaskedLinear",
        "int_mask",
        lambda: _masked(mask=torch.tensor([[1, 0]])),
    ),
    _Valid(
        "MaskedLinear",
        "float_mask",
        lambda: _masked(mask=torch.tensor([[1.0, 0.0]])),
    ),
    _Valid(
        "MaskedLinear",
        "mask_with_an_all_zero_row",
        lambda: _masked(mask=torch.tensor([[1.0, 1.0], [0.0, 0.0]])),
    ),
    _Valid(
        "MaskedLinear",
        "bias_true",
        lambda: _masked(bias=True),
    ),
    _Valid(
        "MaskedLinear",
        "bias_false",
        lambda: _masked(bias=False),
    ),
    _Valid(
        "PackedLinear",
        "numpy_integer_sizes",
        lambda: _packed(
            out_features=np.int64(1),
            in_features=np.int32(2),
        ),
    ),
    _Valid(
        "PackedLinear",
        "numpy_index_arrays",
        lambda: _packed(
            source_index=np.array([0, 1]),
            target_index=_TARGET_INT32,
        ),
    ),
    _Valid(
        "PackedLinear",
        "numpy_integer_index_entries",
        lambda: _packed(source_index=[np.int64(0), np.uint8(1)]),
    ),
    _Valid(
        "PackedLinear",
        "int32_index_tensor",
        lambda: _packed(source_index=_INDEX_INT32),
    ),
    _Valid(
        "PackedLinear",
        "ordered_iterable_indices",
        lambda: _packed(
            source_index=range(2),
            target_index={"a": 0, "b": 0}.values(),
        ),
    ),
    _Valid(
        "PackedLinear",
        "bias_false",
        lambda: _packed(bias=False),
    ),
    _Valid(
        "PackedLinear",
        "transpose_flags_false",
        lambda: _PACKED.transpose(
            bias=False,
            tie=False,
            generator=_generator(),
        ),
    ),
    _Valid(
        "PackedMultiheadAttention",
        "numpy_integer_sizes",
        lambda: _attention(
            query_features=np.int64(1),
            key_features=np.int32(2),
            embed_dim=np.int64(4),
            num_heads=np.uint8(2),
            kdim=np.int64(4),
            chunk_size=np.int64(1),
        ),
    ),
    _Valid(
        "PackedMultiheadAttention",
        "flags_flipped",
        lambda: _attention(
            query_features=2,
            bias=False,
            batch_first=False,
            add_self_loops=True,
        ),
    ),
    _Valid(
        "PackedMultiheadAttention",
        "dropout_int_0",
        lambda: _attention(dropout=0),
    ),
    _Valid(
        "PackedMultiheadAttention",
        "dropout_int_1",
        lambda: _attention(dropout=1),
    ),
    _Valid(
        "PackedMultiheadAttention",
        "dropout_float_1",
        lambda: _attention(dropout=1.0),
    ),
    _Valid(
        "PackedMultiheadAttention",
        "forward_flags_flipped",
        lambda: _attend(
            need_weights=True,
            average_attn_weights=False,
            is_causal=False,
        ),
    ),
    _Valid(
        "map_node_attributions",
        "numpy_integer_layer",
        lambda: _map(layer=np.int64(0)),
    ),
    _Valid(
        "map_node_attributions",
        "coords_value_is_a_range_or_array",
        lambda: _map(
            attributions=torch.tensor([[[1.0]], [[2.0]], [[3.0]]]),
            dims=("observation", "step", "node"),
            coords={
                "observation": range(3),
                "step": np.array(["s0"]),
            },
        ),
    ),
    _Valid(
        "gather_hop_inputs",
        "saved_layers_share_a_multi_axis_batch",
        lambda: _gather(
            {
                0: torch.zeros(2, 3, 1),
                1: torch.zeros(2, 3, 1),
            }
        ),
    ),
]


@pytest.mark.parametrize(
    "case",
    _BAD_CASES,
    ids=[f"{case.public_name}-{case.label}" for case in _BAD_CASES],
)
def test_bad_input_raises_kpnn2_error(case):
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(case.match),
    ) as error:
        case.call()
    if case.kind in _REPORTS_RECEIVED:
        assert "Got" in str(error.value)


@pytest.mark.parametrize(
    "case",
    _VALID_CASES,
    ids=[f"{case.public_name}-{case.label}" for case in _VALID_CASES],
)
def test_valid_edge_case_does_not_raise(case):
    case.call()


def test_case_kinds_are_known():
    kinds = {
        _TYPE,
        _FLAG,
        _VALUE,
        _SHAPE,
        _SEMANTIC,
    }
    assert {case.kind for case in _BAD_CASES} <= kinds


def test_every_public_name_has_a_bad_input_row():
    public = set(kpnn2.__all__)
    covered = {case.public_name for case in _BAD_CASES}
    assert covered <= public
    assert set(_EXEMPT) <= public
    assert covered.isdisjoint(_EXEMPT)
    assert sorted(public - covered - set(_EXEMPT)) == []
