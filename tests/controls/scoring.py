"""
Autograd scores for structural control tests.

Feature score: median (or max) absolute gradient of the attributed
output with respect to the input column.

Hidden-node score: |activation × ∂output/∂activation| at that
node's LayeredSpec layer only (not max-over-layers). Gradient×input
is zero when a dead incoming hop leaves the activation at 0, even
if downstream weights still exist. Raw |∂output/∂h| would not
match reachability-from-inputs. Output nodes are not scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from kpnn2 import LayeredSpec, align_inputs, map_node_attributions
from tests.helpers.layered_net import (
    LayeredNet,
    pin_all_weights,
    pin_edge,
)

from .scenario import StructuralScenario

DEAD_TOLERANCE = 1e-6
LIVE_FLOOR = 1e-3
N_SAMPLES = 8
SEED = 42


def independent_gaussian_features(
    spec: LayeredSpec,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Draw independent N(0, 1) columns named by ``spec.input_nodes``.

    Column order is reversed so ``align_inputs`` must reorder.
    """
    rng = np.random.default_rng(seed)
    names = list(spec.input_nodes)
    values = rng.normal(size=(n_samples, len(names)))
    frame = pd.DataFrame(
        values,
        columns=names,
    )
    return frame[list(reversed(names))]


def pin_scenario_weights(
    model: LayeredNet,
    scenario: StructuralScenario,
) -> None:
    """
    Set live edges to 1.0 and ``scenario.dead_edges`` to 0.0.
    """
    pin_all_weights(
        model,
        value=1.0,
    )
    for source, target in scenario.dead_edges:
        pin_edge(
            model,
            source,
            target,
            0.0,
        )


def feature_grad_table(
    input_grad: torch.Tensor,
    spec: LayeredSpec,
) -> pd.DataFrame:
    """
    Name input gradients with ``map_node_attributions`` at layer 0.
    """
    return map_node_attributions(
        attributions=input_grad,
        spec=spec,
        layer=0,
    ).to_pandas()


def hidden_score_table(
    model: LayeredNet,
    spec: LayeredSpec,
) -> pd.DataFrame:
    """
    Name hidden-node gradient×input scores at each node's layer.
    """
    columns: dict[str, pd.Series] = {}
    hidden = set(spec.hidden_nodes)
    n_layers = len(spec.layer_nodes)
    for layer in range(1, n_layers):
        tensor = model.layer_tensors[layer]
        grad = tensor.grad
        if grad is None:
            grad = torch.zeros_like(tensor)
        scores = tensor.detach() * grad
        mapped = map_node_attributions(
            attributions=scores,
            spec=spec,
            layer=layer,
        ).to_pandas()
        for name in spec.layer_nodes[layer]:
            if name in hidden:
                columns[name] = mapped[name]
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns)


def median_abs_scores(
    table: pd.DataFrame,
    names: tuple[str, ...],
) -> pd.Series:
    """
    Median absolute score of each named column.
    """
    return pd.Series(
        {name: float(table[name].abs().median()) for name in names},
        dtype=float,
    )


def max_abs_scores(
    table: pd.DataFrame,
    names: tuple[str, ...],
) -> pd.Series:
    """
    Max absolute score of each named column over the batch.
    """
    return pd.Series(
        {name: float(table[name].abs().max()) for name in names},
        dtype=float,
    )


def attributed_output_sum(
    output: torch.Tensor,
    spec: LayeredSpec,
    attributed_outputs: tuple[str, ...],
) -> torch.Tensor:
    """
    Sum the last-layer columns named in ``attributed_outputs``.
    """
    last_names = spec.layer_nodes[-1]
    indices = [last_names.index(name) for name in attributed_outputs]
    return output[:, indices].sum()


def align_and_enable_grad(
    features: pd.DataFrame,
    spec: LayeredSpec,
) -> torch.Tensor:
    """
    Align a named table and mark the tensor as requiring grad.
    """
    x = align_inputs(
        features,
        spec,
    )
    x.requires_grad_(True)
    return x
