import re

import numpy as np
import pandas as pd
import pytest
import torch

from kpnn2 import (
    AdjacencySpec,
    Kpnn2Error,
    align_inputs,
    parse_adjacency,
    parse_layered,
)

_ROLE_RULE = (
    "  Expected: an edgelist that starts at the measured features, so "
    "each input node (a node with no incoming edge) is named like a "
    "feature column and no computed node is\n"
)
_MISSING_FIX = (
    "Add the missing feature columns, or give a computed node with no "
    "incoming edge its parents in the edgelist. Do not fit the width "
    "with a dense layer in front of the model: it discards the prior "
    "wiring from the features."
)
_COMPUTED_FIX = (
    "Give each measured feature its own input node, named differently "
    "from every computed node (for example STAT3_gene for the gene and "
    "STAT3 for the transcription factor). Take a column you do not "
    "feed, such as a label column named like an output node, out of "
    "the table before you take its names."
)


def _tiny_spec():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "H"],
            "target": ["H", "H", "C"],
        }
    )
    return parse_layered(edgelist)


def _cyclic_edgelist():
    """
    Inputs A and B feed a two-node cycle; H feeds C.
    """
    return pd.DataFrame(
        {
            "source": ["A", "B", "H", "K", "H"],
            "target": ["H", "K", "K", "H", "C"],
        }
    )


def _tiny_adjacency_spec():
    return parse_adjacency(_cyclic_edgelist())


def _gather_values(values, names, spec):
    col = align_inputs(
        names,
        spec,
    )
    return values[:, col]


def test_align_inputs_reorders_list_names():
    spec = _tiny_spec()
    assert spec.input_nodes == ("A", "B")
    names = ["B", "A"]
    values = np.array(
        [
            [2.0, 1.0],
            [4.0, 3.0],
        ]
    )

    col = align_inputs(
        names,
        spec,
    )

    assert col.dtype == np.int64
    assert col.tolist() == [1, 0]
    gathered = _gather_values(
        values,
        names,
        spec,
    )
    expected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    np.testing.assert_array_equal(
        gathered,
        expected,
    )


def test_align_inputs_ignores_extra_names():
    spec = _tiny_spec()
    names = ["extra", "B", "A"]
    values = np.array(
        [
            [9.0, 2.0, 1.0],
            [8.0, 4.0, 3.0],
        ]
    )

    col = align_inputs(
        names,
        spec,
    )

    assert col.tolist() == [2, 1]
    gathered = values[:, col]
    expected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    np.testing.assert_array_equal(
        gathered,
        expected,
    )


def test_align_inputs_accepts_dataframe_columns():
    spec = _tiny_spec()
    data = pd.DataFrame(
        {
            "B": [2.0, 4.0],
            "A": [1.0, 3.0],
        }
    )

    col = align_inputs(
        data.columns,
        spec,
    )

    gathered = torch.as_tensor(
        data.to_numpy()[:, col],
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(
        gathered,
        expected,
    )


def test_align_inputs_accepts_numpy_names():
    spec = _tiny_spec()
    names = np.array(["B", "A"])
    col = align_inputs(
        names,
        spec,
    )
    assert col.tolist() == [1, 0]


def test_align_inputs_matches_integer_labels_after_str():
    edgelist = pd.DataFrame(
        {
            "source": ["1", "H"],
            "target": ["H", "C"],
        }
    )
    spec = parse_layered(edgelist)
    col = align_inputs(
        [1],
        spec,
    )
    assert col.tolist() == [0]


def test_align_inputs_rejects_missing_name():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("Feature names are missing 1 of 2 input nodes."),
    ):
        align_inputs(
            ["A"],
            spec,
        )


def test_align_inputs_missing_message_counts_input_nodes():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match="^"
        + re.escape(
            "Feature names are missing 1 of 2 input nodes.\n"
            "  Got: 'B'\n"
            f"{_ROLE_RULE}"
            f"  Fix: {_MISSING_FIX}"
        )
        + "$",
    ):
        align_inputs(
            ["A", "unused"],
            spec,
        )


@pytest.mark.parametrize(
    ("names", "listed"),
    [
        pytest.param(
            ["A", "B", "H"],
            "H (hidden)",
            id="hidden",
        ),
        pytest.param(
            ["C", "A", "B"],
            "C (output)",
            id="output",
        ),
        pytest.param(
            ["H", "C", "B", "A"],
            "C (output), H (hidden)",
            id="both_sorted",
        ),
    ],
)
def test_align_inputs_rejects_names_of_computed_nodes(
    names,
    listed,
):
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "the model computes, so their columns would never be read.\n"
            f"  Got: {listed}\n"
        ),
    ) as caught:
        align_inputs(
            names,
            spec,
        )

    message = str(caught.value)
    assert "missing" not in message
    assert "STAT3_gene" in message
    assert "label column named like an output node" in message


def test_align_inputs_reports_missing_and_computed_together():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match="^"
        + re.escape(
            "Feature names are missing 1 of 2 input nodes and match 1 "
            "node(s) the model computes, so their columns would never be "
            "read.\n"
            "  Got: missing: 'B'; computed: H (hidden)\n"
            f"{_ROLE_RULE}"
            f"  Fix: {_MISSING_FIX} {_COMPUTED_FIX}"
        )
        + "$",
    ):
        align_inputs(
            ["A", "H"],
            spec,
        )


def test_align_inputs_suggests_unused_feature_names_for_missing_inputs():
    """
    H names a node, so it is no candidate for the missing B.
    """
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "  Got: missing: 'B'; computed: H (hidden)\n"
            f"{_ROLE_RULE}"
            "  Did you mean: 'b'\n"
        ),
    ):
        align_inputs(
            ["A", "b", "H", "unused"],
            spec,
        )


def test_align_inputs_suggests_for_each_missing_input():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "D"],
            "target": ["H", "H", "H"],
        }
    )
    spec = parse_layered(edgelist)
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "Feature names are missing 2 of 3 input nodes.\n"
            "  Got: 'B', 'D'\n"
            f"{_ROLE_RULE}"
            "  Did you mean: 'b' for 'B'; 'd' for 'D'\n"
        ),
    ):
        align_inputs(
            ["A", "b", "d"],
            spec,
        )


def test_align_inputs_says_to_check_identifiers_when_no_input_matches():
    """
    Ensembl IDs for a graph named by gene symbols.
    """
    edgelist = pd.DataFrame(
        {
            "source": ["TP53", "MYC"],
            "target": ["MDM2", "MDM2"],
        }
    )
    spec = parse_layered(edgelist)
    with pytest.raises(
        Kpnn2Error,
        match="^"
        + re.escape(
            "Feature names are missing 2 of 2 input nodes.\n"
            "  Got: 'MYC', 'TP53'\n"
            f"{_ROLE_RULE}"
            "  Fix: None of the input nodes is among the feature names; "
            "first check that the feature names use the same identifiers "
            f"as the edgelist. {_MISSING_FIX}"
        )
        + "$",
    ):
        align_inputs(
            ["ENSG00000141510", "ENSG00000136997"],
            spec,
        )


def test_align_inputs_suggests_names_when_the_only_input_is_missing():
    edgelist = pd.DataFrame(
        {
            "source": ["TP53"],
            "target": ["MDM2"],
        }
    )
    spec = parse_layered(edgelist)
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("  Did you mean: 'Tp53'\n"),
    ) as caught:
        align_inputs(
            ["Tp53", "ENSG00000141510"],
            spec,
        )
    assert "identifiers" not in str(caught.value)


def test_align_inputs_skips_identifier_check_without_unused_names():
    """
    Every name is a node, so the identifiers are of the same kind.
    """
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("  Got: missing: 'A', 'B'; computed: C (output)\n"),
    ) as caught:
        align_inputs(
            ["C"],
            spec,
        )
    message = str(caught.value)
    assert "identifiers" not in message
    assert "Did you mean" not in message


def test_align_inputs_rejects_computed_node_on_adjacency_spec():
    spec = _tiny_adjacency_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("  Got: K (hidden)\n"),
    ):
        align_inputs(
            ["A", "B", "K"],
            spec,
        )


def test_align_inputs_rejects_edgelist_that_starts_at_computed_layer():
    """
    TF nodes named by gene symbol, gene layer left out.
    """
    genes = ["CD3E", "IRF1", "STAT3", "ZAP70"]
    edgelist = pd.DataFrame(
        {
            "source": ["IRF1", "STAT3"],
            "target": ["ZAP70", "ZAP70"],
        }
    )
    spec = parse_layered(edgelist)
    assert spec.input_nodes == ("IRF1", "STAT3")
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("  Got: ZAP70 (output)\n"),
    ):
        align_inputs(
            genes,
            spec,
        )


def test_align_inputs_rejects_feature_merged_with_hidden_node():
    """
    Gene STAT3 and TF STAT3 share one name, so one hidden node.
    """
    genes = ["CD3E", "IL2", "STAT3"]
    edgelist = pd.DataFrame(
        {
            "source": ["IL2", "STAT3", "CD3E"],
            "target": ["STAT3", "IRF1", "IRF1"],
        }
    )
    spec = parse_layered(edgelist)
    assert spec.hidden_nodes == ("STAT3",)
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("  Got: STAT3 (hidden)\n"),
    ):
        align_inputs(
            genes,
            spec,
        )


def test_align_inputs_accepts_distinct_names_for_measured_features():
    genes = ["CD3E_gene", "IL2_gene", "STAT3_gene"]
    edgelist = pd.DataFrame(
        {
            "source": ["IL2_gene", "STAT3_gene", "CD3E_gene", "STAT3"],
            "target": ["STAT3", "IRF1", "IRF1", "IRF1"],
        }
    )
    spec = parse_layered(edgelist)
    col = align_inputs(
        genes,
        spec,
    )
    assert spec.input_nodes == ("CD3E_gene", "IL2_gene", "STAT3_gene")
    assert col.tolist() == [0, 1, 2]


def test_align_inputs_leaves_names_unchanged_when_it_raises():
    spec = _tiny_spec()
    names = ["A", "H"]
    with pytest.raises(Kpnn2Error):
        align_inputs(
            names,
            spec,
        )
    assert names == ["A", "H"]


def _tensor_reject_cases():
    return [
        pytest.param(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                dtype=torch.float32,
            ),
            id="matching_2d",
        ),
        pytest.param(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                dtype=torch.float32,
            ),
            id="wrong_width",
        ),
        pytest.param(
            torch.tensor([1.0, 2.0]),
            id="not_2d",
        ),
    ]


@pytest.mark.parametrize(
    "names",
    _tensor_reject_cases(),
)
def test_align_inputs_rejects_tensor(names):
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'names' must be a sequence of feature names, not a tensor."
        ),
    ):
        align_inputs(
            names,
            spec,
        )


def test_align_inputs_rejects_dataframe_values():
    spec = _tiny_spec()
    data = pd.DataFrame(
        {
            "A": [1.0],
            "B": [2.0],
        }
    )
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'names' must be a sequence of feature names, not a DataFrame."
        ),
    ):
        align_inputs(
            data,
            spec,
        )


class _FakeAnnData:
    var_names = pd.Index(["A", "B"])
    X = np.array([[1.0, 2.0]])
    shape = (1, 2)


def test_align_inputs_rejects_anndata_like():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'names' must be a sequence of feature names, not AnnData."
        ),
    ) as caught:
        align_inputs(
            _FakeAnnData(),
            spec,
        )
    assert "Pass its .var_names" in str(caught.value)


def test_align_inputs_rejects_2d_numpy_matrix():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match="1-dimensional",
    ):
        align_inputs(
            np.array([[1.0, 2.0]]),
            spec,
        )


def _invalid_align_cases():
    spec = _tiny_spec()
    names = ["A", "B"]
    return [
        pytest.param(
            names,
            object(),
            "LayeredSpec",
            id="non_layered_spec",
        ),
        pytest.param(
            object(),
            spec,
            "'names' must be a sequence of feature names.",
            id="unsupported_type",
        ),
        pytest.param(
            ["A", "B", "A"],
            spec,
            "duplicate",
            id="duplicate_labels",
        ),
        pytest.param(
            ["A", "B", 1, "1"],
            spec,
            "duplicate",
            id="duplicate_after_str",
        ),
        pytest.param(
            "AB",
            spec,
            "'names' must be a sequence of feature names, not a single string",
            id="string",
        ),
        pytest.param(
            b"AB",
            spec,
            "bytes",
            id="bytes",
        ),
        pytest.param(
            {"A": 0, "B": 1},
            spec,
            "'names' must be a sequence of feature names, not a mapping",
            id="mapping",
        ),
        pytest.param(
            {"A", "B"},
            spec,
            "'names' must be a sequence of feature names, not a set",
            id="set",
        ),
    ]


@pytest.mark.parametrize(
    "names, spec, match",
    _invalid_align_cases(),
)
def test_align_inputs_rejects_invalid_names_and_spec(
    names,
    spec,
    match,
):
    with pytest.raises(
        Kpnn2Error,
        match=match,
    ):
        align_inputs(
            names,
            spec,
        )


def test_align_inputs_duplicate_labels_name_a():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match="duplicate",
    ) as caught:
        align_inputs(
            ["A", "B", "A"],
            spec,
        )

    message = str(caught.value)
    assert "A" in message
    assert "converting labels to strings" in message


def test_align_inputs_duplicate_after_str_names_label_1():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match="duplicate",
    ) as caught:
        align_inputs(
            ["A", "B", 1, "1"],
            spec,
        )

    message = str(caught.value)
    assert "1" in message
    assert "converting labels to strings" in message


def test_align_inputs_duplicate_labels_sorted_comma_separated():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match="duplicate",
    ) as caught:
        align_inputs(
            ["A", "B", "A", 1, "1"],
            spec,
        )

    message = str(caught.value)
    assert "'1', 'A'" in message
    assert "converting labels to strings" in message


def test_align_inputs_rejects_bytes():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=r"feature names, not bytes\.",
    ) as caught:
        align_inputs(
            b"AB",
            spec,
        )

    message = str(caught.value)
    assert "string" not in message


def test_align_inputs_empty_names_when_no_input_nodes():
    spec = AdjacencySpec(
        nodes=(),
        node_widths=(),
        input_nodes=(),
        output_nodes=(),
        hidden_nodes=(),
        source_index=(),
        target_index=(),
        input_index=(),
        output_index=(),
    )
    col = align_inputs(
        [],
        spec,
    )
    assert col.dtype == np.int64
    assert col.tolist() == []
    unused = align_inputs(
        ["extra"],
        spec,
    )
    assert unused.dtype == np.int64
    assert unused.tolist() == []


def test_align_inputs_accepts_adjacency_spec():
    spec = _tiny_adjacency_spec()
    assert spec.input_nodes == ("A", "B")
    names = ["A", "B"]
    values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    col = align_inputs(
        names,
        spec,
    )

    assert col.dtype == np.int64
    assert col.tolist() == [0, 1]
    gathered = values[:, col]
    expected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    np.testing.assert_array_equal(
        gathered,
        expected,
    )


def test_align_inputs_reorders_names_for_adjacency_spec():
    spec = _tiny_adjacency_spec()
    names = ["extra", "B", "A"]
    values = np.array(
        [
            [9.0, 2.0, 1.0],
            [8.0, 4.0, 3.0],
        ]
    )

    col = align_inputs(
        names,
        spec,
    )

    assert col.tolist() == [2, 1]
    gathered = values[:, col]
    expected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    np.testing.assert_array_equal(
        gathered,
        expected,
    )


def test_align_inputs_adjacency_width_is_narrower_than_state():
    spec = _tiny_adjacency_spec()
    names = ["A", "B"]
    values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    col = align_inputs(
        names,
        spec,
    )
    gathered = torch.as_tensor(
        values[:, col],
        dtype=torch.float32,
    )

    n_nodes = len(spec.nodes)
    assert spec.hidden_nodes == ("H", "K")
    assert n_nodes == len(spec.nodes)
    assert col.shape == (len(spec.input_nodes),)
    assert col.shape[0] < n_nodes

    state = torch.zeros(
        gathered.shape[0],
        n_nodes,
    )
    state[:, spec.input_index] = gathered
    assert state[:, spec.input_index].tolist() == gathered.tolist()


def test_align_inputs_gives_one_column_per_input_beside_wide_nodes():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )
    spec = parse_layered(
        edgelist,
        widths={
            "H": 3,
            "C": 2,
        },
    )
    assert spec.layer_dims == (1, 3, 2)
    col = align_inputs(
        ["noise", "A"],
        spec,
    )
    assert col.tolist() == [1]
    values = np.array([[9.0, 1.0], [9.0, 2.0]])
    gathered = values[:, col]
    assert tuple(gathered.shape) == (2, spec.layer_dims[0])
    assert gathered.tolist() == [[1.0], [2.0]]


def _invalid_adjacency_align_cases():
    spec = _tiny_adjacency_spec()
    tensor = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    frame = pd.DataFrame(
        {
            "A": [1.0],
            "B": [2.0],
        }
    )
    return [
        pytest.param(
            ["A"],
            spec,
            re.escape("Feature names are missing 1 of 2 input nodes."),
            id="missing_name",
        ),
        pytest.param(
            ["A", "B", "A"],
            spec,
            "duplicate",
            id="duplicate_labels",
        ),
        pytest.param(
            tensor,
            spec,
            "feature names, not a tensor",
            id="tensor",
        ),
        pytest.param(
            frame,
            spec,
            "feature names, not a DataFrame",
            id="dataframe",
        ),
        pytest.param(
            object(),
            spec,
            "'names' must be a sequence of feature names.",
            id="unsupported_type",
        ),
    ]


@pytest.mark.parametrize(
    "names, spec, match",
    _invalid_adjacency_align_cases(),
)
def test_align_inputs_rejects_invalid_names_for_adjacency_spec(
    names,
    spec,
    match,
):
    with pytest.raises(
        Kpnn2Error,
        match=match,
    ):
        align_inputs(
            names,
            spec,
        )


def test_align_inputs_spec_error_names_both_spec_types():
    with pytest.raises(
        Kpnn2Error,
        match="LayeredSpec",
    ) as caught:
        align_inputs(
            ["A", "B"],
            object(),
        )

    message = str(caught.value)
    assert "LayeredSpec" in message
    assert "AdjacencySpec" in message


def test_align_inputs_matrix_message_reports_the_received_shape():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'names' must be a 1-dimensional sequence of feature names.\n"
            "  Got: ndarray of shape (1, 2), dtype float64\n"
            "  Fix: Pass the feature names, and apply the returned index to "
            "the matrix yourself."
        ),
    ):
        align_inputs(
            np.array([[1.0, 2.0]]),
            spec,
        )


def test_align_inputs_unsupported_type_message_reports_the_received_value():
    spec = _tiny_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'names' must be a sequence of feature names.\n  Got: 5 (int)"
        ),
    ):
        align_inputs(
            5,
            spec,
        )


@pytest.mark.parametrize(
    ("names", "message"),
    [
        pytest.param(
            5,
            "'names' must be a sequence of feature names.\n  Got: 5 (int)",
            id="int",
        ),
        pytest.param(
            "A",
            "'names' must be a sequence of feature names, not a single "
            "string.\n"
            "  Got: 'A' (str)\n"
            "  Fix: Wrap it in a list.",
            id="str",
        ),
        pytest.param(
            b"A",
            "'names' must be a sequence of feature names, not bytes.\n"
            "  Got: b'A' (bytes)",
            id="bytes",
        ),
        pytest.param(
            {"A"},
            "'names' must be a sequence of feature names, not a set: its "
            "items are matched by position.\n"
            "  Got: {'A'} (set)\n"
            "  Fix: Pass a list or tuple in the intended order.",
            id="set",
        ),
        pytest.param(
            {"A": 0},
            "'names' must be a sequence of feature names, not a mapping: its "
            "items are matched by position.\n"
            "  Got: {'A': 0} (dict)\n"
            "  Fix: Pass a list or tuple in the intended order.",
            id="mapping",
        ),
        pytest.param(
            np.array([["A"]]),
            "'names' must be a 1-dimensional sequence of feature names.\n"
            "  Got: ndarray of shape (1, 1), dtype <U1\n"
            "  Fix: Pass the feature names, and apply the returned index to "
            "the matrix yourself.",
            id="matrix",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1.0],
                    "B": [2.0],
                }
            ),
            "'names' must be a sequence of feature names, not a DataFrame.\n"
            "  Got: DataFrame of shape (1, 2)\n"
            "  Fix: Pass its feature names (for example data.columns), and "
            "apply the returned index on the matrix yourself.",
            id="dataframe",
        ),
        pytest.param(
            torch.tensor([1]),
            "'names' must be a sequence of feature names, not a tensor.\n"
            "  Got: Tensor of shape (1,), dtype torch.int64\n"
            "  Fix: Pass the feature names that label its columns, even "
            "when they already follow spec.input_nodes.",
            id="tensor",
        ),
        pytest.param(
            _FakeAnnData(),
            "'names' must be a sequence of feature names, not AnnData.\n"
            "  Got: _FakeAnnData of shape (1, 2)\n"
            "  Fix: Pass its .var_names, and apply the returned index to "
            ".X yourself.",
            id="anndata_like",
        ),
    ],
)
def test_align_inputs_rejection_messages_are_pinned(
    names,
    message,
):
    with pytest.raises(
        Kpnn2Error,
        match=f"^{re.escape(message)}$",
    ):
        align_inputs(
            names,
            _tiny_spec(),
        )


def test_align_inputs_spec_message_reports_the_received_type():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["H", "H"],
        }
    )
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'spec' must be a LayeredSpec or an AdjacencySpec.\n"
            "  Got: DataFrame of shape (2, 2)\n"
            "  Fix: Build one with parse_layered() or parse_adjacency(), or "
            "load a saved one with LayeredSpec.from_dict() or "
            "AdjacencySpec.from_dict()."
        ),
    ):
        align_inputs(
            ["A", "B"],
            edgelist,
        )


def test_align_inputs_suggests_names_that_differ_only_by_case():
    # No input node is a feature name, but each differs from one only
    # by case: a typo, not another kind of identifier.
    with pytest.raises(Kpnn2Error) as excinfo:
        align_inputs(
            ["a", "b"],
            _tiny_spec(),
        )
    message = str(excinfo.value)
    assert "\n  Got: 'A', 'B'\n" in message
    assert "\n  Did you mean: 'a' for 'A'; 'b' for 'B'\n" in message
    assert "same identifiers" not in message
