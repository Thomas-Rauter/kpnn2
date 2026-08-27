"""
Train matched two-tower graphs and score them with autograd.

Scenarios share the matched DAG and independent features. They
differ in ``LayeredNet`` relu/bias, scoring policy, and DGF.

``linear_feedforward`` uses ``bias=False`` and ``relu=False`` so a
linear label is representable. ``relu_feedforward`` uses
``relu=True`` and ``bias=True`` on the same linear tower-A labels:
a user-typical KPNN has ReLU in ``forward()``, and bias is on so
this scenario is not a silent copy of the linear control. After
construction, ReLU biases are filled with ``RELU_BIAS_INIT``
(measured: the default uniform init left seeds 42 and 44 at
chance even at 320 epochs). Weights are learned, not pinned.
Epochs stay at ``N_EPOCHS`` (80); raising them did not fix those
collapsed inits, and a 640-epoch sweep lowered recovery.

``relu_product_feedforward`` uses ReLU and bias on the product of
the two tower-A features. A linear ``LayeredNet`` cannot represent
a product, so ``relu=True`` is required. Feature width stays at
``N_FEATURES_PER_TOWER`` (2) so both A features are causal.
Hidden width is ``PRODUCT_HIDDEN_WIDTH`` (8): a width-2 hidden
tower cannot represent the product (measured train AUC < 0.90
even at 320 epochs). With hidden width 8, the 0.95 gate holds
on 30/30 seeds at 80 epochs. Scoring is still features only.
Product-specific cuts (still strict; median AUC stays 1.0):
``PRODUCT_MAX_RATIO`` 0.45 and consistency 0.5, because
∂(x1 x2)/∂x1 ∝ x2 so some rows have near-zero causal grads
and decoy medians are not as small as on the linear DGF.

``train_matched_linear_towers`` defaults match the linear scenario
(``bias=False``, ``relu=False``, no bias fill, linear simulator)
for Tier 3 negative controls.

Decoy (tower B) nodes stay fully wired to ``prediction``. The
reachability guard must pass before any rank claim: otherwise the
criterion would succeed on structural zeros.
"""

from __future__ import annotations

import math
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
    LayeredSpec,
    align_inputs,
    map_node_attributions,
    parse_layered,
)
from tests.helpers.layered_net import LayeredNet

from .graphs import (
    PREDICTION_OUTPUT,
    TwoTowerGraph,
    two_tower_feedforward,
)
from .ground_truth import assert_all_structurally_live
from .metrics import (
    DEFAULT_MAX_RATIO,
    DEFAULT_MIN_PER_EXAMPLE,
    Separation,
    compute_separation,
)
from .scoring import (
    align_and_enable_grad,
    attributed_output_sum,
    feature_grad_table,
)
from .simulate import (
    independent_normal_features,
    linear_logit_labels,
    product_logit_labels,
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
# Width-2 hidden cannot represent the product. Feature count stays 2.
PRODUCT_HIDDEN_WIDTH = 8

# Linear helper / linear scenario defaults. Do not reuse these for
# relu_feedforward or relu_product_feedforward; those scenarios
# set relu and bias explicitly.
DEFAULT_TRAIN_BIAS = False
DEFAULT_TRAIN_RELU = False
# Small positive bias keeps ReLU units active at step 0.
RELU_BIAS_INIT = 0.1
# ReLU gates some examples; 0.6 is still a majority of rows.
RELU_MIN_PER_EXAMPLE_CONSISTENCY = 0.6
# Product ∂/∂x1 ∝ x2; 0.5 is still a majority of eval rows.
PRODUCT_MIN_PER_EXAMPLE_CONSISTENCY = 0.5
# Measured at hidden width 8, 80 epochs: seed 42 ratio was 0.433.
PRODUCT_MAX_RATIO = 0.45

# Pre-registered training seeds. Do not drop a seed after seeing
# it fail; re-measure the whole set if the recipe changes.
N_SEEDS = 30
SEEDS: tuple[int, ...] = tuple(range(42, 42 + N_SEEDS))
# Combined gate AND importance on SEEDS at N_EPOCHS=80.
LINEAR_MEASURED_PASSING = 22
RELU_MEASURED_PASSING = 19
PRODUCT_MEASURED_PASSING = 18


def seeds_passing_floor(
    n_passing: int,
    n_seeds: int = N_SEEDS,
    *,
    z: float = 2.5,
) -> int:
    """
    Count ~z SE below a measured pass count.

    A new RNG stream that keeps the same rate should still
    clear this bar. Do not pass a cherry-picked window.
    """
    if n_seeds <= 0:
        raise ValueError(f"'n_seeds' must be positive, got {n_seeds}.")
    if n_passing < 0 or n_passing > n_seeds:
        raise ValueError(
            f"'n_passing' must be in [0, {n_seeds}], got {n_passing}."
        )
    if n_passing == 0:
        return 0
    p = n_passing / n_seeds
    se = math.sqrt(n_seeds * p * (1.0 - p))
    return max(1, math.floor(n_passing - z * se))


@dataclass(frozen=True)
class TrainedScenario:
    """
    One trained-tier configuration: activations, bias, scoring.

    Graph is the matched two-tower DAG for every current id.
    ``product_labels`` selects the product DGF; otherwise labels
    are linear in tower A. Product uses a wider hidden layer so
    the 0.95 gate stays in force. ``n_epochs`` can differ when a
    scenario needs more fitting; ReLU currently uses the same 80
    as the linear control. ``min_seeds_passing`` is a rate floor
    on ``SEEDS``, not a 5-seed majority.
    """

    id: str
    bias: bool
    relu: bool
    min_seeds_passing: int
    n_epochs: int = N_EPOCHS
    bias_init: float | None = None
    include_hidden: bool = True
    min_per_example_consistency: float = DEFAULT_MIN_PER_EXAMPLE
    max_ratio: float = DEFAULT_MAX_RATIO
    product_labels: bool = False
    hidden_width: int | None = None
    min_val_roc_auc: float = MIN_VAL_ROC_AUC


@dataclass(frozen=True)
class TrainedRun:
    """One trained model, spec, held-out features, and val ROC-AUC."""

    model: LayeredNet
    spec: LayeredSpec
    x_eval_df: pd.DataFrame
    val_roc_auc: float
    graph: TwoTowerGraph
    important_features: tuple[str, ...]


TRAINED_SCENARIOS: tuple[TrainedScenario, ...] = (
    TrainedScenario(
        id="linear_feedforward",
        bias=DEFAULT_TRAIN_BIAS,
        relu=DEFAULT_TRAIN_RELU,
        min_seeds_passing=seeds_passing_floor(LINEAR_MEASURED_PASSING),
        n_epochs=N_EPOCHS,
    ),
    TrainedScenario(
        id="relu_feedforward",
        bias=True,
        relu=True,
        min_seeds_passing=seeds_passing_floor(RELU_MEASURED_PASSING),
        n_epochs=N_EPOCHS,
        bias_init=RELU_BIAS_INIT,
        include_hidden=False,
        min_per_example_consistency=RELU_MIN_PER_EXAMPLE_CONSISTENCY,
    ),
    TrainedScenario(
        id="relu_product_feedforward",
        bias=True,
        relu=True,
        min_seeds_passing=seeds_passing_floor(PRODUCT_MEASURED_PASSING),
        n_epochs=N_EPOCHS,
        bias_init=RELU_BIAS_INIT,
        include_hidden=False,
        min_per_example_consistency=PRODUCT_MIN_PER_EXAMPLE_CONSISTENCY,
        max_ratio=PRODUCT_MAX_RATIO,
        product_labels=True,
        hidden_width=PRODUCT_HIDDEN_WIDTH,
    ),
)
TRAINED_SCENARIO_IDS: tuple[str, ...] = tuple(
    scenario.id for scenario in TRAINED_SCENARIOS
)
_TRAINED_SCENARIOS_BY_ID: dict[str, TrainedScenario] = {
    scenario.id: scenario for scenario in TRAINED_SCENARIOS
}


def scenario_graph(
    scenario: TrainedScenario,
) -> TwoTowerGraph:
    """
    Matched towers with the scenario's hidden width.
    """
    return matched_linear_towers(
        hidden_width=scenario.hidden_width,
    )


def trained_scenario(scenario_id: str) -> TrainedScenario:
    """
    Look up a registered trained-tier scenario.

    Raises
    ------
    KeyError
        If ``scenario_id`` is not in ``TRAINED_SCENARIO_IDS``.
    """
    try:
        return _TRAINED_SCENARIOS_BY_ID[scenario_id]
    except KeyError as exc:
        known = ", ".join(TRAINED_SCENARIO_IDS)
        raise KeyError(
            f"Unknown trained scenario {scenario_id!r}. Known ids: {known}."
        ) from exc


def matched_linear_towers(
    *,
    hidden_width: int | None = None,
) -> TwoTowerGraph:
    """
    Two matched DAG towers that both terminate at ``prediction``.

    ``hidden_width`` is forwarded to ``two_tower_feedforward``.
    ``None`` keeps hidden width equal to the feature count.
    """
    return two_tower_feedforward(
        n_features_per_tower=N_FEATURES_PER_TOWER,
        depth=FEEDFORWARD_DEPTH,
        hidden_width=hidden_width,
        shared_output=True,
    )


def broken_matched_towers(
    *,
    hidden_width: int | None = None,
) -> TwoTowerGraph:
    """
    Matched towers whose A features cannot reach ``prediction``.

    Names match ``matched_linear_towers``. Tower-A last hops go to
    ``decoy_readout``; tower B still terminates at ``prediction``.
    """
    return two_tower_feedforward(
        n_features_per_tower=N_FEATURES_PER_TOWER,
        depth=FEEDFORWARD_DEPTH,
        hidden_width=hidden_width,
        shared_output=False,
        disconnected_tower="a",
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


def product_causal_features(
    graph: TwoTowerGraph,
) -> tuple[str, str]:
    """
    First two tower-A features; the product DGF uses exactly these.
    """
    causal = graph.tower_a_features[:2]
    if len(causal) != 2:
        raise ValueError(
            "Product labels need two tower-A features, got "
            f"{len(graph.tower_a_features)}."
        )
    return causal[0], causal[1]


def product_tower_simulator(
    graph: TwoTowerGraph,
) -> Simulator:
    """
    Labels depend on the product of two tower-A features only.
    """
    causal = product_causal_features(graph)
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
        labels = product_logit_labels(
            features,
            causal_features=causal,
        )
        return features, labels

    return simulate


def scenario_simulator(
    scenario: TrainedScenario,
    graph: TwoTowerGraph,
) -> Simulator:
    """
    Return the DGF simulator registered for ``scenario``.
    """
    if scenario.product_labels:
        return product_tower_simulator(graph)
    return linear_tower_simulator(graph)


def scenario_important_features(
    scenario: TrainedScenario,
    graph: TwoTowerGraph,
) -> tuple[str, ...]:
    """
    Features the scenario's DGF actually uses.

    Remaining tower-A names, if any, are omitted: they must not be
    declared important when the DGF ignores them.
    """
    if scenario.product_labels:
        return product_causal_features(graph)
    return graph.tower_a_features


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
    bias: bool = DEFAULT_TRAIN_BIAS,
    relu: bool = DEFAULT_TRAIN_RELU,
    bias_init: float | None = None,
    hidden_width: int | None = None,
    important_features: Sequence[str] | None = None,
    graph: TwoTowerGraph | None = None,
) -> TrainedRun:
    """
    Train ``LayeredNet`` on matched two-tower data; return held-out.

    Defaults (``bias=False``, ``relu=False``, no bias fill, linear
    simulator, default hidden width) match ``linear_feedforward``
    and the Tier 3 negative controls.

    Pass ``simulate`` to override the default linear simulator
    (shuffled-label negative controls or a product DGF). Pass
    ``bias`` / ``relu`` / ``n_epochs`` / ``bias_init`` /
    ``hidden_width`` to train a different ``LayeredNet``. Pass
    ``important_features`` when the DGF ignores some tower-A names.
    Pass ``graph`` to train a different TwoTowerGraph (for example
    a rewired prior). ``hidden_width`` is used only when ``graph``
    is omitted. Loss and val ROC-AUC use the last-layer column
    named ``prediction``.
    """
    if graph is None:
        graph = matched_linear_towers(
            hidden_width=hidden_width,
        )
    assert_decoys_structurally_live(graph)
    if simulate is None:
        simulate = linear_tower_simulator(graph)
    if important_features is None:
        important_features = graph.tower_a_features
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
    spec = parse_layered(graph.edgelist)
    model = LayeredNet(
        spec,
        bias=bias,
        relu=relu,
    )
    if bias_init is not None:
        _fill_biases(
            model,
            bias_init,
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
        spec=spec,
        x_train=x_train,
        y_train=y_train_tensor,
        n_epochs=n_epochs,
    )
    val_roc_auc = _evaluate_roc_auc(
        model,
        spec=spec,
        x_eval=x_eval,
        y_eval=y_eval_tensor,
    )
    return TrainedRun(
        model=model,
        spec=spec,
        x_eval_df=x_eval_df,
        val_roc_auc=val_roc_auc,
        graph=graph,
        important_features=tuple(important_features),
    )


def train_trained_scenario(
    scenario_id: str,
    *,
    seed: int,
    simulate: Simulator | None = None,
) -> TrainedRun:
    """
    Train the named registered scenario on matched towers.
    """
    scenario = trained_scenario(scenario_id)
    graph = scenario_graph(scenario)
    if simulate is None:
        simulate = scenario_simulator(
            scenario,
            graph,
        )
    return train_matched_linear_towers(
        seed=seed,
        n_epochs=scenario.n_epochs,
        simulate=simulate,
        bias=scenario.bias,
        relu=scenario.relu,
        bias_init=scenario.bias_init,
        hidden_width=scenario.hidden_width,
        important_features=scenario_important_features(
            scenario,
            graph,
        ),
    )


def autograd_feature_table(
    model: LayeredNet,
    spec: LayeredSpec,
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
    spec: LayeredSpec,
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

    Defaults to the run's DGF-important features and tower B
    unimportant. Unused tower-A names must not be declared
    important: callers store the causal set on ``TrainedRun``.
    """
    graph = run.graph
    if important_features is None:
        important_features = run.important_features
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
    spec: LayeredSpec,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int,
) -> None:
    """
    Full-batch AdamW with binary cross-entropy.

    ``shuffle`` is off: one batch is the whole train set, so
    permutation cannot change the gradient and would only burn
    the global torch RNG.
    """
    loader = DataLoader(
        TensorDataset(
            x_train,
            y_train,
        ),
        batch_size=len(x_train),
        shuffle=False,
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
            logits = _prediction_column(
                model(x_batch),
                spec,
            )
            loss = loss_fn(
                logits,
                y_batch,
            )
            loss.backward()
            optimizer.step()


def _evaluate_roc_auc(
    model: nn.Module,
    *,
    spec: LayeredSpec,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
) -> float:
    """
    Held-out ROC-AUC with the model in eval mode.
    """
    model.eval()
    with torch.no_grad():
        logits = _prediction_column(
            model(x_eval),
            spec,
        )
        probabilities = torch.sigmoid(logits)
    return float(
        roc_auc_score(
            y_eval.detach().cpu().numpy().reshape(-1),
            probabilities.detach().cpu().numpy().reshape(-1),
        )
    )


def _prediction_column(
    output: torch.Tensor,
    spec: LayeredSpec,
) -> torch.Tensor:
    """
    Last-layer logit named ``prediction``, shape ``(n, 1)``.
    """
    last_names = spec.layer_nodes[-1]
    try:
        index = last_names.index(PREDICTION_OUTPUT)
    except ValueError as exc:
        raise ValueError(
            f"{PREDICTION_OUTPUT!r} is not in the last layer {last_names}."
        ) from exc
    return output[:, index : index + 1]


def _set_seed(seed: int) -> None:
    """Seed every RNG used by a training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _fill_biases(
    model: LayeredNet,
    value: float,
) -> None:
    """
    Set every ``MaskedLinear`` bias to ``value``.

    Used by ReLU scenarios so hidden units start in the active
    ReLU regime. No-op on layers constructed without bias.
    """
    with torch.no_grad():
        for layer in model.layers:
            if layer.bias is None:
                continue
            layer.bias.fill_(value)
