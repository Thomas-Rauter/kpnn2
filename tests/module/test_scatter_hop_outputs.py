import pandas as pd
import pytest
import torch

from kpnn2 import (
    Kpnn2Error,
    gather_hop_inputs,
    parse_layered,
    scatter_hop_outputs,
)


def _skip_spec():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "H", "A"],
            "target": ["H", "H", "C", "C"],
        }
    )
    return parse_layered(edgelist)


def test_scatter_splits_source_layers_in_order():
    spec = _skip_spec()
    hop = spec.hops[1]
    concat = torch.tensor([[1.0, 2.0, 5.0]])
    parts = scatter_hop_outputs(
        concat,
        hop,
    )
    assert list(parts) == [0, 1]
    assert parts[0].tolist() == [[1.0, 2.0]]
    assert parts[1].tolist() == [[5.0]]


def test_scatter_returns_the_tensor_for_a_single_source():
    spec = _skip_spec()
    tensor = torch.tensor([[1.0, 2.0]])
    parts = scatter_hop_outputs(
        tensor,
        spec.hops[0],
    )
    assert list(parts) == [0]
    assert parts[0] is tensor


def test_scatter_inverts_gather_values():
    spec = _skip_spec()
    hop = spec.hops[1]
    saved = {
        0: torch.tensor([[1.0, 2.0]]),
        1: torch.tensor([[5.0]]),
    }
    gathered = gather_hop_inputs(
        saved,
        hop,
    )
    parts = scatter_hop_outputs(
        gathered,
        hop,
    )
    torch.testing.assert_close(
        parts[0],
        saved[0],
    )
    torch.testing.assert_close(
        parts[1],
        saved[1],
    )


def test_scatter_keeps_batch_dimensions():
    spec = _skip_spec()
    concat = torch.zeros(
        4,
        3,
    )
    parts = scatter_hop_outputs(
        concat,
        spec.hops[1],
    )
    assert parts[0].shape == (4, 2)
    assert parts[1].shape == (4, 1)


def test_scatter_is_differentiable():
    spec = _skip_spec()
    concat = torch.zeros(
        1,
        3,
        requires_grad=True,
    )
    parts = scatter_hop_outputs(
        concat,
        spec.hops[1],
    )
    scaled = parts[0] * torch.tensor([[1.0, 2.0]])
    loss = scaled.sum() + 3.0 * parts[1].sum()
    loss.backward()
    assert concat.grad is not None
    assert concat.grad.tolist() == [[1.0, 2.0, 3.0]]


def test_scatter_does_not_modify_values():
    spec = _skip_spec()
    concat = torch.tensor([[1.0, 2.0, 5.0]])
    before = concat.clone()
    scatter_hop_outputs(
        concat,
        spec.hops[1],
    )
    torch.testing.assert_close(
        concat,
        before,
    )


def test_scatter_splits_wide_source_layers():
    spec = parse_layered(
        pd.DataFrame(
            {
                "source": ["A", "H", "A"],
                "target": ["H", "C", "C"],
            }
        ),
        widths={"A": 2, "H": 3},
    )
    hop = spec.hops[1]
    concat = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    parts = scatter_hop_outputs(
        concat,
        hop,
    )
    assert parts[0].tolist() == [[1.0, 2.0]]
    assert parts[1].tolist() == [[3.0, 4.0, 5.0]]


def test_scatter_rejects_a_non_hop():
    with pytest.raises(
        Kpnn2Error,
        match="must be a Hop",
    ):
        scatter_hop_outputs(
            torch.zeros(1, 2),
            "hop",
        )


def test_scatter_rejects_a_non_tensor():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match="must be a torch.Tensor",
    ):
        scatter_hop_outputs(
            [[1.0, 2.0]],
            spec.hops[0],
        )


def test_scatter_rejects_a_width_mismatch():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match="wrong number of units",
    ):
        scatter_hop_outputs(
            torch.zeros(1, 4),
            spec.hops[1],
        )


def test_scatter_rejects_a_zero_dimensional_tensor():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match="0-dimensional",
    ):
        scatter_hop_outputs(
            torch.tensor(1.0),
            spec.hops[0],
        )
