import pandas as pd
import pytest

from kpnn2 import parse_edgelist
from kpnn2.errors import Kpnn2Error


def test_parse_edgelist_ranks_a_simple_chain():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )

    spec = parse_edgelist(edgelist)

    assert spec.input_nodes == ("A",)
    assert spec.hidden_nodes == ("B",)
    assert spec.output_nodes == ("C",)
    assert spec.layer_nodes == (("A",), ("B",), ("C",))
    assert spec.layer_dims == (1, 1, 1)
    assert spec.skips == ()


def test_parse_edgelist_ranks_skip_without_pseudo_nodes():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )

    spec = parse_edgelist(edgelist)

    assert spec.input_nodes == ("A",)
    assert spec.hidden_nodes == ("H",)
    assert spec.output_nodes == ("C",)
    assert spec.layer_nodes == (("A",), ("H",), ("C",))
    assert spec.layer_dims == (1, 1, 1)


def test_parse_edgelist_allows_early_outputs():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "A", "H"],
            "target": ["H", "E", "C"],
        }
    )

    spec = parse_edgelist(edgelist)

    assert spec.input_nodes == ("A",)
    assert spec.hidden_nodes == ("H",)
    assert spec.output_nodes == ("C", "E")
    assert spec.layer_nodes == (("A",), ("E", "H"), ("C",))
    assert spec.layer_dims == (1, 2, 1)


def test_parse_edgelist_rejects_self_loops():
    edgelist = pd.DataFrame(
        {
            "source": ["A"],
            "target": ["A"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="self-loop",
    ):
        parse_edgelist(edgelist)


def test_parse_edgelist_rejects_cycles():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "C", "B"],
            "target": ["B", "C", "B", "D"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="cycle",
    ):
        parse_edgelist(edgelist)


def test_parse_edgelist_rejects_missing_input():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "A"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="input node",
    ):
        parse_edgelist(edgelist)


def test_parse_edgelist_rejects_missing_output():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "C"],
            "target": ["B", "C", "B"],
        }
    )

    with pytest.raises(
        Kpnn2Error,
        match="output node",
    ):
        parse_edgelist(edgelist)
