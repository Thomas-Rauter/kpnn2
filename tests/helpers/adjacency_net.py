"""
Test-only nn.Module built from an AdjacencySpec.

Not part of the public kpnn2 API. One PackedLinear, applied
``n_steps`` times. Inputs are written into the state each step.
"""

import torch
import torch.nn.functional as F
from torch import nn

from kpnn2 import AdjacencySpec, PackedLinear


class AdjacencyNet(nn.Module):
    """
    Shared ``PackedLinear`` unrolled for ``n_steps``.

    2-D ``x`` is static and is rewritten every step. 3-D ``x`` is
    ``(batch, T, n_inputs)`` with ``T == n_steps``. State updates
    are out of place so autograd into ``x`` stays valid. ReLU is
    applied after every step except the last, when ``relu`` is
    True.
    """

    def __init__(
        self,
        spec: AdjacencySpec,
        n_steps: int,
        bias: bool = True,
        relu: bool = True,
    ) -> None:
        super().__init__()
        if n_steps < 1:
            raise ValueError(f"'n_steps' must be at least 1, got {n_steps}.")
        self.spec = spec
        self.n_steps = n_steps
        self.relu = relu
        n_nodes = len(spec.nodes)
        self.core = PackedLinear(
            spec.source_index,
            spec.target_index,
            n_nodes,
            n_nodes,
            bias=bias,
        )
        self.step_states: list[torch.Tensor] = []

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes = len(self.spec.nodes)
        n_inputs = len(self.spec.input_nodes)
        if x.ndim == 2:
            if x.shape[1] != n_inputs:
                raise ValueError(
                    f"Static x must have width {n_inputs}, got {x.shape[1]}."
                )
            batch = x.shape[0]
        elif x.ndim == 3:
            if x.shape[1] != self.n_steps:
                raise ValueError(
                    "Sequence length must equal n_steps "
                    f"{self.n_steps}, got {x.shape[1]}."
                )
            if x.shape[2] != n_inputs:
                raise ValueError(
                    f"Sequence x must have width {n_inputs}, got {x.shape[2]}."
                )
            batch = x.shape[0]
        else:
            raise ValueError(f"x must be 2-D or 3-D, got ndim={x.ndim}.")
        state = x.new_zeros(batch, n_nodes)
        self.step_states = []
        last = self.n_steps - 1
        input_index = list(self.spec.input_index)
        for step in range(self.n_steps):
            step_x = x if x.ndim == 2 else x[:, step, :]
            written = state.clone()
            written[:, input_index] = step_x
            state = self.core(written)
            if self.relu and step < last:
                state = F.relu(state)
            if state.requires_grad:
                state.retain_grad()
            self.step_states.append(state)
        output_index = list(self.spec.output_index)
        return state[:, output_index]


def pin_all_weights(
    module: AdjacencyNet,
    value: float = 1.0,
) -> None:
    """
    Set every packed live-edge weight to ``value``.

    Bias, if present, is zeroed. Absent edges are not parameters.
    """
    with torch.no_grad():
        module.core.weight.fill_(value)
        if module.core.bias is not None:
            module.core.bias.zero_()


def pin_edge(
    module: AdjacencyNet,
    source: str,
    target: str,
    value: float,
) -> None:
    """
    Set the packed weight of one live named edge to ``value``.

    Raises
    ------
    Kpnn2Error
        If ``source -> target`` is not a packed edge.
    """
    packed_indices = module.spec.edge_location(
        source,
        target,
    )
    with torch.no_grad():
        module.core.weight[list(packed_indices)] = value
