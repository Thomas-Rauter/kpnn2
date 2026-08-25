"""
Train matched two-tower graphs and score them with autograd.

``LayeredNet`` uses ``bias=False`` and ``relu=False`` so a linear
label is representable. Weights are learned, not pinned.

Decoy (tower B) nodes stay fully wired to ``prediction``. The
reachability guard must pass before any rank claim: otherwise the
criterion would succeed on structural zeros.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kpnn2 import (
    GraphSpec,
    align_inputs,
    map_node_attributions,
    parse_edgelist,
)
from tests.helpers.layered_net import LayeredNet

from .graphs import (
    PREDICTION_OUTPUT,
    TwoTowerGraph,
    two_tower_feedforward,
)
from .ground_truth import assert_all_structurally_live
from .metrics import Separation, compute_separation
from .scoring import (
    align_and_enable_grad,
    attributed_output_sum,
    feature_grad_table,
)
from .simulate import (
    independent_normal_features,
    linear_logit_labels,
)

Simulator = Callable[
    [np.random.Generator, int],
    tuple[pd.DataFrame, np.ndarray],
]

MIN_VAL_ROC_AUC = 0.95
N_EPOCHS = 80
N_TRAIN = 256
N_EVAL = 128
LEARNING_RATE = 2e-2
WEIGHT_DECAY = 1e-4
N_FEATURES_PER_TOWER = 2
FEEDFORWARD_DEPTH = 2

# Linear labels: no bias, no ReLU. Documented choice for this helper.
TRAIN_BIAS = False
TRAIN_RELU = False

TRAINED_SCENARIO_IDS: tuple[str, ...] = ("linear_feedforward",)
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
MIN_SEEDS_PASSING = 4


@dataclass(frozen=True)
class TrainedRun:
    """One trained model, spec, held-out features, and val ROC-AUC."""

    model: LayeredNet
    spec: GraphSpec
    x_eval_df: pd.DataFrame
    val_roc_auc: float
    graph: TwoTowerGraph


def matched_linear_towers() -> TwoTowerGraph:
    """
    Two matched DAG towers that both terminate at ``prediction``.
    """
    return two_tower_feedforward(
        n_features_per_tower=N_FEATURES_PER_TOWER,
        depth=FEEDFORWARD_DEPTH,
        shared_output=True,
    )


def linear_coefficients(
    n_causal: int,
) -> tuple[float, ...]:
    """
    Coefficients ``1.0, 0.75, ...`` for the causal tower.
    """
    return tuple(1.0 - 0.25 * index for index in range(n_causal))


def linear_tower_simulator(
    graph: TwoTowerGraph,
) -> Simulator:
    """
    Labels depend linearly on tower-A features only.
    """
    causal = graph.tower_a_features
    coefficients = linear_coefficients(len(causal))
    feature_names = graph.tower_a_features + graph.tower_b_features

    def simulate(
        rng: np.random.Generator,
        n_samples: int,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        features = independent_normal_features(
            rng,
            n_samples,
            feature_names=feature_names,
        )
        labels = linear_logit_labels(
            features,
            causal_features=causal,
            coefficients=coefficients,
        )
        return features, labels

    return simulate


def assert_decoys_structurally_live(
    graph: TwoTowerGraph,
) -> None:
    """
    Fail if a tower-B name cannot reach ``prediction``.

    That would make trained-tier ranking a structural-zero test.
    """
    names = [
        *graph.tower_b_features,
        *graph.tower_b_nodes,
    ]
    assert_all_structurally_live(
        edgelist=graph.edgelist,
        dead_edges=(),
        attributed_outputs=(PREDICTION_OUTPUT,),
        names=names,
    )


def train_matched_linear_towers(
    *,
    seed: int,
    n_train: int = N_TRAIN,
    n_eval: int = N_EVAL,
    n_epochs: int = N_EPOCHS,
    simulate: Simulator | None = None,
) -> TrainedRun:
    """
    Train ``LayeredNet`` on linear tower-A labels; return held-out data.

    Pass ``simulate`` to override the default linear simulator (used
    for shuffled-label negative controls).
    """
    graph = matched_linear_towers()
    assert_decoys_structurally_live(graph)
    if simulate is None:
        simulate = linear_tower_simulator(graph)
    _set_seed(seed)
    rng = np.random.default_rng(seed)
    x_train_df, y_train = simulate(
        rng,
        n_train,
    )
    x_eval_df, y_eval = simulate(
        rng,
        n_eval,
    )
    spec = parse_edgelist(graph.edgelist)
    model = LayeredNet(
        spec,
        bias=TRAIN_BIAS,
        relu=TRAIN_RELU,
    )
    x_train = align_inputs(
        x_train_df,
        spec,
    )
    x_eval = align_inputs(
        x_eval_df,
        spec,
    )
    y_train_tensor = torch.as_tensor(
        y_train,
        dtype=torch.float32,
    )
    y_eval_tensor = torch.as_tensor(
        y_eval,
        dtype=torch.float32,
    )
    _fit_binary_classifier(
        model,
        x_train=x_train,
        y_train=y_train_tensor,
        n_epochs=n_epochs,
    )
    val_roc_auc = _evaluate_roc_auc(
        model,
        x_eval=x_eval,
        y_eval=y_eval_tensor,
    )
    return TrainedRun(
        model=model,
        spec=spec,
        x_eval_df=x_eval_df,
        val_roc_auc=val_roc_auc,
        graph=graph,
    )


def autograd_feature_table(
    model: LayeredNet,
    spec: GraphSpec,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-example input gradients of the attributed last-layer output.
    """
    x = align_and_enable_grad(
        features,
        spec,
    )
    output = model(x)
    attributed_output_sum(
        output,
        spec,
        (PREDICTION_OUTPUT,),
    ).backward()
    assert x.grad is not None
    return feature_grad_table(
        x.grad,
        spec,
    )


def autograd_hidden_tables(
    model: LayeredNet,
    spec: GraphSpec,
    features: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """
    Per-layer tables of |activation × ∂output / ∂activation|.

    Gradient×input goes to zero on unused paths. Raw |∂output/∂h|
    at the last hidden layer stays large because both towers keep
    outgoing weights to the shared output, so it cannot test whether
    the model ignored the decoy tower.
    """
    x = align_and_enable_grad(
        features,
        spec,
    )
    output = model(x)
    attributed_output_sum(
        output,
        spec,
        (PREDICTION_OUTPUT,),
    ).backward()
    hidden = set(spec.hidden_nodes)
    tables: dict[int, pd.DataFrame] = {}
    n_layers = len(spec.layer_nodes)
    for layer in range(1, n_layers):
        tensor = model.layer_tensors[layer]
        grad = tensor.grad
        if grad is None:
            grad = torch.zeros_like(tensor)
        mapped = map_node_attributions(
            attributions=tensor.detach() * grad,
            spec=spec,
            layer=layer,
        ).to_pandas()
        names = [name for name in spec.layer_nodes[layer] if name in hidden]
        if names:
            tables[layer] = mapped.loc[:, names]
    return tables


def trained_separations(
    run: TrainedRun,
    *,
    important_features: Sequence[str] | None = None,
    unimportant_features: Sequence[str] | None = None,
    important_nodes: Sequence[str] | None = None,
    unimportant_nodes: Sequence[str] | None = None,
) -> list[Separation]:
    """
    Feature separation plus per-layer hidden-node separations.

    Defaults to tower A important and tower B unimportant.
    """
    graph = run.graph
    if important_features is None:
        important_features = graph.tower_a_features
    if unimportant_features is None:
        unimportant_features = graph.tower_b_features
    if important_nodes is None:
        important_nodes = graph.tower_a_nodes
    if unimportant_nodes is None:
        unimportant_nodes = graph.tower_b_nodes
    feature_table = autograd_feature_table(
        run.model,
        run.spec,
        run.x_eval_df,
    )
    separations = [
        compute_separation(
            feature_table,
            label="features",
            important=important_features,
            unimportant=unimportant_features,
        )
    ]
    hidden_tables = autograd_hidden_tables(
        run.model,
        run.spec,
        run.x_eval_df,
    )
    for layer, table in hidden_tables.items():
        columns = set(table.columns)
        important = [name for name in important_nodes if name in columns]
        unimportant = [name for name in unimportant_nodes if name in columns]
        if not important or not unimportant:
            continue
        separations.append(
            compute_separation(
                table,
                label=f"layer_{layer}",
                important=important,
                unimportant=unimportant,
            )
        )
    return separations


def _fit_binary_classifier(
    model: nn.Module,
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int,
) -> None:
    """
    Full-batch AdamW with binary cross-entropy.
    """
    loader = DataLoader(
        TensorDataset(
            x_train,
            y_train,
        ),
        batch_size=len(x_train),
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _epoch in range(n_epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = loss_fn(
                logits,
                y_batch,
            )
            loss.backward()
            optimizer.step()


def _evaluate_roc_auc(
    model: nn.Module,
    *,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
) -> float:
    """
    Held-out ROC-AUC with the model in eval mode.
    """
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_eval))
    return float(
        roc_auc_score(
            y_eval.detach().cpu().numpy().reshape(-1),
            probabilities.detach().cpu().numpy().reshape(-1),
        )
    )


def _set_seed(seed: int) -> None:
    """Seed every RNG used by a training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
