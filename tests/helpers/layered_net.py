"""
Test-only nn.Module built from a LayeredSpec.

Not part of the public kpnn2 API.
"""

import torch
import torch.nn.functional as F
from torch import nn

from kpnn2 import LayeredSpec, MaskedLinear, gather_hop_inputs


class LayeredNet(nn.Module):
    """
    One ``MaskedLinear`` per ``spec.hops``, in depth order.

    Each hop reads every layer that feeds its target, so skip
    edges ride along inside the hop mask and there is nothing
    extra to add. ReLU is applied after every hop except the
    last, when ``relu`` is True. The last hop stays linear.
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
                MaskedLinear(
                    hop.mask,
                    bias=bias,
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
    Set every live edge weight and nothing else to ``value``.

    Masked-out trainable entries are set to 0. Skip edges are
    live entries of a hop mask, so they are pinned by the same
    line as adjacent edges.
    """
    with torch.no_grad():
        for layer in module.layers:
            trainable = layer.parametrizations.weight.original
            trainable.copy_(layer.mask * value)


def pin_edge(
    module: LayeredNet,
    source: str,
    target: str,
    value: float,
) -> None:
    """
    Set the weight of one live edge to ``value``.

    Adjacent and skip edges are found the same way: the hop that
    produces ``target`` names its rows, and its concatenated
    source layers name its columns.

    Raises
    ------
    ValueError
        If ``source -> target`` is not a live entry of any hop
        mask in ``module.spec``.
    """
    spec = module.spec
    with torch.no_grad():
        for index, hop in enumerate(spec.hops):
            targets = spec.layer_nodes[hop.target_layer]
            if target not in targets or source not in hop.source_nodes:
                continue
            row = targets.index(target)
            column = hop.source_nodes.index(source)
            layer = module.layers[index]
            if layer.mask[row, column].item() != 1.0:
                continue
            trainable = layer.parametrizations.weight.original
            trainable[row, column] = value
            return
    raise ValueError(f"No edge {source!r} -> {target!r} in spec.")
