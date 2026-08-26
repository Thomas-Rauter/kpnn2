import pandas as pd
import pytest
import torch

from kpnn2 import Kpnn2Error, gather_hop_inputs, parse_layered


def _skip_spec():
    edgelist = pd.DataFrame(
        {
            "source": ["A", "B", "H", "A"],
            "target": ["H", "H", "C", "C"],
        }
    )
    return parse_layered(edgelist)


def _saved(spec):
    return {
        0: torch.tensor([[1.0, 2.0]]),
        1: torch.tensor([[5.0]]),
    }


def test_gather_concatenates_source_layers_in_order():
    spec = _skip_spec()
    hop = spec.hops[1]
    assert hop.source_layers == (0, 1)

    gathered = gather_hop_inputs(
        _saved(spec),
        hop,
    )
    assert gathered.tolist() == [[1.0, 2.0, 5.0]]
    assert gathered.shape[-1] == hop.mask.shape[1]


def test_gather_returns_the_saved_tensor_for_a_single_source():
    spec = _skip_spec()
    saved = _saved(spec)
    gathered = gather_hop_inputs(
        saved,
        spec.hops[0],
    )
    assert gathered is saved[0]


def test_gather_does_not_modify_saved():
    spec = _skip_spec()
    saved = _saved(spec)
    before = {layer: tensor.clone() for layer, tensor in saved.items()}
    gather_hop_inputs(
        saved,
        spec.hops[1],
    )
    for layer, tensor in saved.items():
        torch.testing.assert_close(
            tensor,
            before[layer],
        )


def test_gather_keeps_batch_dimensions():
    spec = _skip_spec()
    saved = {
        0: torch.zeros(
            4,
            2,
        ),
        1: torch.zeros(
            4,
            1,
        ),
    }
    gathered = gather_hop_inputs(
        saved,
        spec.hops[1],
    )
    assert gathered.shape == (4, 3)


def test_gather_is_differentiable_into_every_source():
    spec = _skip_spec()
    first = torch.zeros(
        1,
        2,
        requires_grad=True,
    )
    second = torch.zeros(
        1,
        1,
        requires_grad=True,
    )
    saved = {
        0: first,
        1: second,
    }
    gathered = gather_hop_inputs(
        saved,
        spec.hops[1],
    )
    (gathered * torch.tensor([[1.0, 2.0, 3.0]])).sum().backward()

    assert first.grad is not None
    assert second.grad is not None
    assert first.grad.tolist() == [[1.0, 2.0]]
    assert second.grad.tolist() == [[3.0]]


def test_gather_rejects_a_missing_source_layer():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match="missing layer 0",
    ):
        gather_hop_inputs(
            {1: torch.zeros(1, 1)},
            spec.hops[1],
        )


def test_gather_missing_layer_message_names_the_hop():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match=r"hop into layer 2 reads layers \[0, 1\]",
    ):
        gather_hop_inputs(
            {1: torch.zeros(1, 1)},
            spec.hops[1],
        )


def test_gather_rejects_a_non_mapping_saved():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match="mapping of layer index",
    ):
        gather_hop_inputs(
            [torch.zeros(1, 2)],
            spec.hops[0],
        )


def test_gather_rejects_a_non_hop():
    with pytest.raises(
        Kpnn2Error,
        match="must be a Hop",
    ):
        gather_hop_inputs(
            {0: torch.zeros(1, 2)},
            "hop",
        )


def test_gather_rejects_a_non_tensor_entry():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match=r"saved\[0\] must be a torch.Tensor",
    ):
        gather_hop_inputs(
            {
                0: [[1.0, 2.0]],
                1: torch.zeros(1, 1),
            },
            spec.hops[1],
        )


def test_gather_rejects_a_width_mismatch():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match="wrong number of units",
    ):
        gather_hop_inputs(
            {
                0: torch.zeros(1, 3),
                1: torch.zeros(1, 1),
            },
            spec.hops[1],
        )


def test_gather_rejects_a_zero_dimensional_tensor():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match="0-dimensional",
    ):
        gather_hop_inputs(
            {0: torch.tensor(1.0)},
            spec.hops[0],
        )


def test_gather_rejects_mixed_dtypes():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match="share a dtype",
    ):
        gather_hop_inputs(
            {
                0: torch.zeros(
                    1,
                    2,
                    dtype=torch.float32,
                ),
                1: torch.zeros(
                    1,
                    1,
                    dtype=torch.float64,
                ),
            },
            spec.hops[1],
        )


def test_gather_accepts_a_shared_non_default_dtype():
    spec = _skip_spec()
    gathered = gather_hop_inputs(
        {
            0: torch.zeros(
                1,
                2,
                dtype=torch.float64,
            ),
            1: torch.zeros(
                1,
                1,
                dtype=torch.float64,
            ),
        },
        spec.hops[1],
    )
    assert gathered.dtype == torch.float64
