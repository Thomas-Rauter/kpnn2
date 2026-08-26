import pandas as pd
import torch

from kpnn2 import parse_layered


def test_parse_layered_masks_chain_shapes_and_dtype():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
        }
    )

    spec = parse_layered(edgelist)

    assert len(spec.masks) == 2
    assert spec.masks[0].shape == (1, 1)
    assert spec.masks[1].shape == (1, 1)
    assert spec.masks[0].dtype == torch.float32
    assert spec.masks[1].dtype == torch.float32
    ones = torch.tensor(
        [[1.0]],
        dtype=torch.float32,
    )
    assert torch.equal(
        spec.masks[0],
        ones,
    )
    assert torch.equal(
        spec.masks[1],
        ones,
    )


def test_parse_layered_masks_include_adjacent_exclude_skip():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "H", "A"],
            "target": ["H", "C", "C"],
        }
    )

    spec = parse_layered(edgelist)

    assert spec.layer_nodes == (("A",), ("H",), ("C",))
    assert len(spec.masks) == 2
    assert spec.masks[0].shape == (1, 1)
    assert spec.masks[1].shape == (1, 1)
    assert spec.masks[0].dtype == torch.float32
    assert spec.masks[1].dtype == torch.float32
    ones = torch.tensor(
        [[1.0]],
        dtype=torch.float32,
    )
    assert torch.equal(
        spec.masks[0],
        ones,
    )
    assert torch.equal(
        spec.masks[1],
        ones,
    )
    n_ones = sum(int(mask.sum().item()) for mask in spec.masks)
    assert n_ones == 2


def test_parse_layered_masks_early_output_indexing():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "A", "H"],
            "target": ["H", "E", "C"],
        }
    )

    spec = parse_layered(edgelist)

    assert spec.layer_nodes == (("A",), ("E", "H"), ("C",))
    assert spec.masks[0].shape == (2, 1)
    assert spec.masks[1].shape == (1, 2)
    assert spec.masks[0].dtype == torch.float32
    assert spec.masks[1].dtype == torch.float32
    expected_hop0 = torch.tensor(
        [[1.0], [1.0]],
        dtype=torch.float32,
    )
    expected_hop1 = torch.tensor(
        [[0.0, 1.0]],
        dtype=torch.float32,
    )
    assert torch.equal(
        spec.masks[0],
        expected_hop0,
    )
    assert torch.equal(
        spec.masks[1],
        expected_hop1,
    )
