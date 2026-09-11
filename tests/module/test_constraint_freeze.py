"""
Hard freeze via constraint= and torch.where.

A gradient hook that zeroes a slot is not a freeze under
AdamW or SGD with momentum. These tests pin the hatch that
does hold: a constraint= module that replaces live-edge
values in forward.
"""

import pandas as pd
import torch
from torch import nn

from kpnn2 import MaskedLinear, PackedLinear, parse_adjacency, parse_layered


class _FreezeSlots(nn.Module):
    def __init__(
        self,
        mask,
        values,
    ):
        super().__init__()
        self.register_buffer(
            "mask",
            mask,
        )
        self.register_buffer(
            "values",
            values,
        )

    def forward(
        self,
        weight,
    ):
        return torch.where(
            self.mask,
            self.values,
            weight,
        )


def _two_parent_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "B", "H"],
            "target": ["H", "H", "C"],
        }
    )


def _cyclic_edgelist():
    return pd.DataFrame(
        {
            "source": ["x", "a", "b", "a"],
            "target": ["a", "b", "a", "y"],
        }
    )


def _freeze_packed(
    nnz,
    packed_indices,
    value,
):
    mask = torch.zeros(
        nnz,
        dtype=torch.bool,
    )
    values = torch.zeros(nnz)
    indices = list(packed_indices)
    mask[indices] = True
    values[indices] = value
    return _FreezeSlots(
        mask,
        values,
    )


def _run_steps(
    layer,
    x,
    optimizer,
    steps=20,
):
    for _ in range(steps):
        optimizer.zero_grad()
        layer(x).sum().backward()
        optimizer.step()


def test_packed_where_constraint_holds_layered_slot_under_adamw():
    torch.manual_seed(42)
    spec = parse_layered(_two_parent_edgelist())
    hop_index, packed = spec.edge_location(
        "A",
        "H",
    )
    hop = spec.hops[hop_index]
    frozen = 1.5
    layer = PackedLinear(
        hop.source_index,
        hop.target_index,
        hop.out_features,
        hop.in_features,
        bias=False,
        constraint=_freeze_packed(
            len(hop.source_index),
            packed,
            frozen,
        ),
    )
    with torch.no_grad():
        layer.weight.fill_(0.25)
    stored_before = layer.weight[packed[0]].item()
    optimizer = torch.optim.AdamW(
        layer.parameters(),
        lr=0.1,
        weight_decay=0.1,
    )
    x = torch.ones(
        8,
        hop.in_features,
    )
    _run_steps(
        layer,
        x,
        optimizer,
    )
    live = layer.constraint(layer.weight)
    torch.testing.assert_close(
        live[packed[0]],
        torch.tensor(frozen),
    )
    assert live[1].item() != 0.25
    assert layer.weight[packed[0]].item() != stored_before
    only_a = torch.tensor(
        [[1.0, 0.0]],
    )
    torch.testing.assert_close(
        layer(only_a),
        torch.tensor([[frozen]]),
    )


def test_packed_where_constraint_holds_adjacency_slot_under_sgd():
    torch.manual_seed(42)
    spec = parse_adjacency(_cyclic_edgelist())
    packed = spec.edge_location(
        "a",
        "b",
    )
    frozen = 1.5
    n = len(spec.nodes)
    layer = PackedLinear(
        spec.source_index,
        spec.target_index,
        n,
        n,
        bias=False,
        constraint=_freeze_packed(
            len(spec.source_index),
            packed,
            frozen,
        ),
    )
    with torch.no_grad():
        layer.weight.fill_(0.25)
    optimizer = torch.optim.SGD(
        layer.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.1,
    )
    x = torch.ones(
        8,
        n,
    )
    _run_steps(
        layer,
        x,
        optimizer,
    )
    live = layer.constraint(layer.weight)
    torch.testing.assert_close(
        live[packed[0]],
        torch.tensor(frozen),
    )
    unfrozen = [i for i in range(layer.nnz) if i not in packed]
    assert live[unfrozen[0]].item() != 0.25


def test_masked_where_constraint_holds_live_cell_under_adamw():
    torch.manual_seed(42)
    spec = parse_layered(_two_parent_edgelist())
    hop_index, packed = spec.edge_location(
        "A",
        "H",
    )
    hop = spec.hops[hop_index]
    frozen = 1.5
    freeze_mask = torch.zeros(
        hop.out_features,
        hop.in_features,
        dtype=torch.bool,
    )
    freeze_values = torch.zeros(
        hop.out_features,
        hop.in_features,
    )
    for index in packed:
        row = hop.target_index[index]
        column = hop.source_index[index]
        freeze_mask[row, column] = True
        freeze_values[row, column] = frozen
    layer = MaskedLinear(
        hop.to_mask(),
        bias=False,
        constraint=_FreezeSlots(
            freeze_mask,
            freeze_values,
        ),
    )
    with torch.no_grad():
        layer.parametrizations.weight.original.fill_(0.25)
    optimizer = torch.optim.AdamW(
        layer.parameters(),
        lr=0.1,
        weight_decay=0.1,
    )
    x = torch.ones(
        8,
        hop.in_features,
    )
    _run_steps(
        layer,
        x,
        optimizer,
    )
    torch.testing.assert_close(
        layer.weight[0, 0],
        torch.tensor(frozen),
    )
    assert layer.weight[0, 1].item() != 0.25
    only_a = torch.tensor(
        [[1.0, 0.0]],
    )
    torch.testing.assert_close(
        layer(only_a),
        torch.tensor([[frozen]]),
    )


def test_grad_hook_does_not_hold_packed_slot_under_adamw():
    torch.manual_seed(42)
    layer = PackedLinear(
        [0, 1],
        [0, 1],
        2,
        2,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.fill_(1.5)

    def _zero_first_slot(grad):
        out = grad.clone()
        out[0] = 0
        return out

    layer.weight.register_hook(_zero_first_slot)
    optimizer = torch.optim.AdamW(
        layer.parameters(),
        lr=0.1,
        weight_decay=0.1,
    )
    x = torch.ones(
        8,
        2,
    )
    _run_steps(
        layer,
        x,
        optimizer,
    )
    assert layer.weight[0].item() != 1.5
