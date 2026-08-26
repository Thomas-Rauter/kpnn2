"""
Integration test on a real tabular classification task.

Breast Cancer Wisconsin Diagnostic data, a sparse DAG from named
features, ordinary PyTorch training. Catches broken parse/align/
MaskedLinear behavior that still passes smaller unit tests.

No state_update / cyclic architecture.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kpnn2 import LayeredSpec, align_inputs, parse_layered
from tests.helpers.layered_net import LayeredNet

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SEED = 42
N_BASIS_OUTPUTS = 16
DROPOUT = 0.10


class _TabularClassifier(nn.Module):
    """
    LayeredSpec core plus dropout and a one-logit head.
    """

    def __init__(
        self,
        spec: LayeredSpec,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.core = LayeredNet(
            spec,
            bias=True,
            relu=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(
            spec.layer_dims[-1],
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.core(x)
        hidden = self.dropout(hidden)
        return self.head(hidden)


def test_masked_linear_dag_learns_real_tabular_task() -> None:
    """Train a MaskedLinear DAG on Breast Cancer Wisconsin."""
    metrics = _run_real_tabular_task()
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["test_f1"] > 0.85
    assert metrics["test_roc_auc"] > 0.90


def _run_real_tabular_task() -> dict[str, float]:
    """Parse, align shuffled columns, train, and evaluate."""
    _set_seed(SEED)
    x_train_df, x_test_df, y_train, y_test = _make_breast_cancer_data()
    edgelist = _make_feedforward_architecture_edgelist(list(x_train_df.columns))
    spec = parse_layered(edgelist)
    x_train_shuffled = x_train_df.loc[
        :,
        list(reversed(x_train_df.columns)),
    ]
    x_test_shuffled = x_test_df.loc[
        :,
        list(reversed(x_test_df.columns)),
    ]
    x_train = align_inputs(
        x_train_shuffled,
        spec,
    )
    x_test = align_inputs(
        x_test_shuffled,
        spec,
    )
    model = _TabularClassifier(spec)
    initial_loss, final_loss = _train_binary_classifier(
        model=model,
        x_train=x_train,
        y_train=y_train,
    )
    test_f1, test_roc_auc = _evaluate_binary_classifier(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
    }


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducible integration tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_breast_cancer_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    torch.Tensor,
    torch.Tensor,
]:
    """Load, split, scale, and return named tabular data."""
    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data.copy()
    y = dataset.target.astype(np.float32).to_numpy().reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        index=x_train.index,
        columns=x_train.columns,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        index=x_test.index,
        columns=x_test.columns,
    )
    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )
    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.float32,
    )
    return (
        x_train_scaled,
        x_test_scaled,
        y_train_tensor,
        y_test_tensor,
    )


def _feature_statistic(feature_name: str) -> str:
    """Return the statistic prefix for a breast-cancer feature name."""
    if feature_name.startswith("mean "):
        return "mean"
    if feature_name.startswith("worst "):
        return "worst"
    if feature_name.endswith(" error"):
        return "error"
    raise ValueError(f"Unexpected feature name: {feature_name!r}")


def _feature_measurement(feature_name: str) -> str:
    """Return the measurement name without statistic-specific wording."""
    if feature_name.startswith("mean "):
        return feature_name.removeprefix("mean ")
    if feature_name.startswith("worst "):
        return feature_name.removeprefix("worst ")
    if feature_name.endswith(" error"):
        return feature_name.removesuffix(" error")
    raise ValueError(f"Unexpected feature name: {feature_name!r}")


def _make_feedforward_architecture_edgelist(
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Sparse DAG: features -> statistic/measurement -> basis outputs.

    Some features also skip to basis nodes. Seed 42 for sampling.
    """
    rng = np.random.default_rng(SEED)
    basis_nodes = [
        f"classification_basis_{idx:02d}" for idx in range(N_BASIS_OUTPUTS)
    ]
    stat_nodes = {
        stat: f"statistic__{stat}"
        for stat in sorted({_feature_statistic(name) for name in feature_names})
    }
    measurement_nodes = {
        measurement: f"measurement__{measurement.replace(' ', '_')}"
        for measurement in sorted(
            {_feature_measurement(name) for name in feature_names}
        )
    }
    edges: list[tuple[str, str]] = []
    for feature_name in feature_names:
        stat = _feature_statistic(feature_name)
        measurement = _feature_measurement(feature_name)
        edges.append((feature_name, stat_nodes[stat]))
        edges.append((feature_name, measurement_nodes[measurement]))
    intermediate_nodes = list(stat_nodes.values()) + list(
        measurement_nodes.values()
    )
    for node in intermediate_nodes:
        targets = rng.choice(
            basis_nodes,
            size=4,
            replace=False,
        )
        edges.extend((node, str(target)) for target in targets)
    for feature_name in feature_names:
        if rng.random() < 0.35:
            target = rng.choice(basis_nodes)
            edges.append((feature_name, str(target)))
    edgelist = pd.DataFrame(
        edges,
        columns=["source", "target"],
    )
    return edgelist.drop_duplicates().reset_index(drop=True)


def _train_binary_classifier(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 500,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
) -> tuple[float, float]:
    """Train an ordinary PyTorch binary classifier and return losses."""
    dataset = TensorDataset(
        x_train,
        y_train,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    initial_loss = _compute_loss(
        model=model,
        x=x_train,
        y=y_train,
        loss_fn=loss_fn,
    )
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
    final_loss = _compute_loss(
        model=model,
        x=x_train,
        y=y_train,
        loss_fn=loss_fn,
    )
    return initial_loss, final_loss


def _compute_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    """Compute loss with the model in evaluation mode."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(
            logits,
            y,
        )
    model.train(was_training)
    return float(loss.detach())


def _evaluate_binary_classifier(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[float, float]:
    """Evaluate F1 and ROC-AUC on the held-out test set."""
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        probabilities = torch.sigmoid(logits)
    y_true = y_test.detach().cpu().numpy().reshape(-1)
    y_prob = probabilities.detach().cpu().numpy().reshape(-1)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    test_f1 = f1_score(
        y_true,
        y_pred,
    )
    test_roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )
    return test_f1, test_roc_auc
