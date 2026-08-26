"""
Test-only nn.Module built from a LayeredSpec.

Not part of the public kpnn2 API.
"""

import torch
import torch.nn.functional as F
from torch import nn

from kpnn2 import LayeredSpec, MaskedLinear, Skip


class LayeredNet(nn.Module):
    """
    Adjacent MaskedLinear hops plus skip residuals.

    ReLU is applied after every hop except the last, when
    ``relu`` is True. The last hop stays linear.
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
        for mask in spec.masks:
            layers.append(
                MaskedLinear(
                    mask,
                    bias=bias,
                )
            )
        self.layers = nn.ModuleList(layers)
        skip_weights = []
        for _skip in spec.skips:
            skip_weights.append(nn.Parameter(torch.zeros(1)))
        self.skip_weights = nn.ParameterList(skip_weights)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        saved = {0: x}
        hidden = x
        last_hop = len(self.layers) - 1
        for hop, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if self.relu and hop < last_hop:
                hidden = F.relu(hidden)
            target_layer = hop + 1
            hidden = _apply_skips(
                hidden=hidden,
                saved=saved,
                skips=self.spec.skips,
                skip_weights=self.skip_weights,
                target_layer=target_layer,
            )
            if hidden.requires_grad:
                hidden.retain_grad()
            saved[target_layer] = hidden
        self.layer_tensors = saved
        return hidden


def pin_all_weights(
    module: LayeredNet,
    value: float = 1.0,
) -> None:
    """
    Set live adjacent weights and skip scalars to ``value``.

    Masked-out trainable entries are set to 0.
    """
    with torch.no_grad():
        for layer in module.layers:
            trainable = layer.parametrizations.weight.original
            trainable.copy_(layer.mask * value)
        for weight in module.skip_weights:
            weight.fill_(value)


def pin_edge(
    module: LayeredNet,
    source: str,
    target: str,
    value: float,
) -> None:
    """
    Set one adjacent trainable weight entry or skip scalar.

    Raises
    ------
    ValueError
        If ``source -> target`` is not an adjacent live mask entry
        and not a skip in ``module.spec``.
    """
    spec = module.spec
    with torch.no_grad():
        for hop, layer in enumerate(module.layers):
            sources = spec.layer_nodes[hop]
            targets = spec.layer_nodes[hop + 1]
            if source not in sources or target not in targets:
                continue
            source_index = sources.index(source)
            target_index = targets.index(target)
            if layer.mask[target_index, source_index].item() != 1.0:
                continue
            trainable = layer.parametrizations.weight.original
            trainable[target_index, source_index] = value
            return
        for skip, weight in zip(
            spec.skips,
            module.skip_weights,
            strict=True,
        ):
            if skip.source == source and skip.target == target:
                weight.fill_(value)
                return
    raise ValueError(f"No edge {source!r} -> {target!r} in spec.")


def _apply_skips(
    hidden: torch.Tensor,
    saved: dict[int, torch.Tensor],
    skips: list[Skip],
    skip_weights: nn.ParameterList,
    target_layer: int,
) -> torch.Tensor:
    for skip, weight in zip(
        skips,
        skip_weights,
        strict=True,
    ):
        if skip.target_layer != target_layer:
            continue
        source = saved[skip.source_layer][
            :,
            skip.source_index,
        ]
        addition = torch.zeros_like(hidden)
        addition[:, skip.target_index] = weight * source
        hidden = hidden + addition
    return hidden
