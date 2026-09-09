"""
Test-only nn.Module built from a LayeredSpec.

Not part of the public kpnn2 API.
"""

import torch
import torch.nn.functional as F
from torch import nn

from kpnn2 import LayeredSpec, PackedLinear, gather_hop_inputs


class LayeredNet(nn.Module):
    """
    One ``PackedLinear`` per ``spec.hops``, in depth order.

    Each hop reads every layer that feeds its target, so skip
    edges ride along inside the hop's packed indices and there
    is nothing extra to add. ReLU is applied after every hop
    except the last, when ``relu`` is True. The last hop stays
    linear.
    """

    def __init__(
        self,
        spec: LayeredSpec,
        bias: bool = True,
        relu: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.relu = relu
        layers = []
        for hop in spec.hops:
            layers.append(
                PackedLinear(
                    hop.source_index,
                    hop.target_index,
                    hop.out_features,
                    hop.in_features,
                    bias=bias,
                    identity=spec.fingerprint,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        saved = {0: x}
        hidden = x
        last_hop = len(self.layers) - 1
        for index, hop in enumerate(self.spec.hops):
            sources = gather_hop_inputs(
                saved,
                hop,
            )
            hidden = self.layers[index](sources)
            if self.relu and index < last_hop:
                hidden = F.relu(hidden)
            if hidden.requires_grad:
                hidden.retain_grad()
            saved[hop.target_layer] = hidden
        self.layer_tensors = saved
        return hidden


def pin_all_weights(
    module: LayeredNet,
    value: float = 1.0,
) -> None:
    """
    Set every live edge weight to ``value``.

    Skip edges are live packed entries of a hop, so they are
    pinned by the same line as adjacent edges.
    """
    with torch.no_grad():
        for layer in module.layers:
            layer.weight.fill_(value)


def pin_edge(
    module: LayeredNet,
    source: str,
    target: str,
    value: float,
) -> None:
    """
    Set the weight of one live named edge to ``value``.

    Adjacent and skip edges are found the same way:
    ``spec.edge_location`` returns the hop and every packed
    slot of that named-edge block. Width greater than 1 pins
    the whole block, not unit 0 alone.

    Raises
    ------
    Kpnn2Error
        If ``source -> target`` is not a live packed pair of
        any hop in ``module.spec``.
    """
    hop_index, packed_indices = module.spec.edge_location(
        source,
        target,
    )
    layer = module.layers[hop_index]
    with torch.no_grad():
        layer.weight[list(packed_indices)] = value
