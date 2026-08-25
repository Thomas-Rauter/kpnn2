import pandas as pd

from kpnn2 import Skip, parse_edgelist


def test_parse_edgelist_records_skip_with_indices():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )

    spec = parse_edgelist(edgelist)

    assert spec.layer_nodes == (("A",), ("H",), ("C",))
    assert spec.skips == (
        Skip(
            source="A",
            target="C",
            source_layer=0,
            target_layer=2,
            source_index=0,
            target_index=0,
        ),
    )


def test_parse_edgelist_omits_adjacent_edges_from_skips():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )

    spec = parse_edgelist(edgelist)

    assert spec.skips == ()
