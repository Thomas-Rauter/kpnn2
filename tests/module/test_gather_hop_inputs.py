import re

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
    assert gathered.shape[-1] == hop.in_features


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
        match="^"
        + re.escape(
            "'saved' must hold every layer the hop into layer 2 reads.\n"
            "  Got: missing layer 0; 'saved' has keys 1\n"
            "  Expected: layers 0, 1\n"
            "  Fix: Store every layer the forward pass produces in 'saved', "
            "keyed by its depth as an int."
        )
        + "$",
    ):
        gather_hop_inputs(
            {1: torch.zeros(1, 1)},
            spec.hops[1],
        )


def test_gather_names_every_missing_layer():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("  Got: missing layers 0, 1; 'saved' has keys 2\n"),
    ):
        gather_hop_inputs(
            {2: torch.zeros(1, 1)},
            spec.hops[1],
        )


def test_gather_reports_missing_layers_before_a_bad_entry():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("  Got: missing layer 1; 'saved' has keys 0\n"),
    ):
        gather_hop_inputs(
            {0: [[1.0, 2.0]]},
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
        match=re.escape("'saved[0]' must be a torch.Tensor."),
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
        match="^"
        + re.escape(
            "'saved[0]' must have shape (..., spec.layer_dims[0]).\n"
            "  Got: shape (1, 3)\n"
            "  Expected: shape (..., 2)"
        )
        + "$",
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
        match=re.escape(
            "'saved[0]' must have shape (..., spec.layer_dims[0]).\n"
            "  Got: shape ()\n"
            "  Expected: shape (..., 2)"
        ),
    ):
        gather_hop_inputs(
            {0: torch.tensor(1.0)},
            spec.hops[0],
        )


def test_gather_rejects_mixed_dtypes():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match="^"
        + re.escape(
            "Saved layers must share a dtype.\n"
            "  Got: saved[0] is torch.float32; saved[1] is torch.float64"
        )
        + "$",
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


@pytest.mark.parametrize(
    ("second", "second_shape"),
    [
        pytest.param(
            torch.zeros(
                3,
                1,
            ),
            "(3, 1)",
            id="batch_size",
        ),
        pytest.param(
            torch.zeros(1),
            "(1,)",
            id="unbatched",
        ),
    ],
)
def test_gather_rejects_mismatched_leading_dimensions(
    second,
    second_shape,
):
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "Saved layers must share every dimension but the last.\n"
            "  Got: saved[0] has shape (4, 2); saved[1] has shape "
            f"{second_shape}"
        ),
    ):
        gather_hop_inputs(
            {
                0: torch.zeros(
                    4,
                    2,
                ),
                1: second,
            },
            spec.hops[1],
        )


def test_gather_rejects_mixed_devices():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match="^"
        + re.escape(
            "Saved layers must share a device.\n"
            "  Got: saved[0] is on cpu; saved[1] is on meta"
        )
        + "$",
    ):
        gather_hop_inputs(
            {
                0: torch.zeros(1, 2),
                1: torch.zeros(
                    1,
                    1,
                    device="meta",
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


def test_gather_concatenates_wide_source_layers():
    edgelist = pd.DataFrame(
        {
            "source": ["x", "A", "H", "A"],
            "target": ["A", "H", "C", "C"],
        }
    )
    spec = parse_layered(
        edgelist,
        widths={"A": 2, "H": 3},
    )
    hop = spec.hops[2]
    assert hop.source_layers == (1, 2)
    assert hop.source_dims == (2, 3)
    saved = {
        0: torch.tensor([[0.0]]),
        1: torch.tensor([[1.0, 2.0]]),
        2: torch.tensor([[3.0, 4.0, 5.0]]),
    }
    gathered = gather_hop_inputs(
        saved,
        hop,
    )
    assert gathered.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0]]
    assert gathered.shape[-1] == hop.in_features
    assert gathered.shape[-1] == spec.layer_dims[1] + spec.layer_dims[2]


def test_gather_non_mapping_message_reports_the_received_type():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'saved' must be a mapping of layer index to tensor.\n"
            "  Got: list of length 1"
        ),
    ) as caught:
        gather_hop_inputs(
            [torch.tensor([[123.0, 456.0]])],
            spec.hops[0],
        )
    assert "123" not in str(caught.value)


def test_gather_non_tensor_entry_message_reports_the_received_value():
    spec = _skip_spec()
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(
            "'saved[0]' must be a torch.Tensor.\n  Got: [[1.0, 2.0]] (list)"
        ),
    ):
        gather_hop_inputs(
            {
                0: [[1.0, 2.0]],
                1: torch.zeros(1, 1),
            },
            spec.hops[1],
        )


def test_gather_non_hop_message_reports_the_received_value():
    with pytest.raises(
        Kpnn2Error,
        match=re.escape("'hop' must be a Hop from spec.hops.\n  Got: 0 (int)"),
    ):
        gather_hop_inputs(
            {0: torch.zeros(1, 2)},
            0,
        )


@pytest.mark.parametrize(
    ("saved", "held"),
    [
        pytest.param(
            {
                "0": torch.zeros(1, 2),
                "1": torch.zeros(1, 1),
            },
            "'saved' has keys '0', '1'",
            id="str_keys",
        ),
        pytest.param(
            {},
            "'saved' is empty",
            id="empty",
        ),
    ],
)
def test_gather_missing_layer_message_shows_what_saved_holds(
    saved,
    held,
):
    with pytest.raises(
        Kpnn2Error,
        match=re.escape(f"  Got: missing layers 0, 1; {held}\n"),
    ):
        gather_hop_inputs(
            saved,
            _skip_spec().hops[1],
        )
