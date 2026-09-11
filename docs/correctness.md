# Correctness

`kpnn2` turns a named edgelist into the wiring of a neural net.
A bug that drops a skip, leaks a missing edge, or mislabels an
attribution axis is a scientific error, not only a software
bug. The test suite is built around that.

This page is an overview of **what the tests claim**, not a
catalog of every `test_*` function. The files live on GitHub
under
[`tests/`](https://github.com/Thomas-Rauter/kpnn2/tree/main/tests).

CI runs the full pytest suite, including the slow controls, on
every change. That is a process check. The scientific argument
is the claims below.

A separate, frozen notebook repeats one published simulation
with these primitives:
[Fortelny and Bock, 2020](literature/fortelny-bock-2020.ipynb).
That page is not part of the test suite.

## Two kinds of correctness

**Technical.** The parsers, masks, packed layers, alignment,
and checkpoints do what the public contract says. A cycle is
rejected by `parse_layered`. A masked-out weight cannot affect
the output. `PackedLinear` matches `MaskedLinear` on the same
graph.

**Scientific.** The named graph is the architecture. A node
with no live path to the attributed output must not score as
important. An edge that is not in the edgelist must not
influence a prediction. A shuffled or rewired prior must not
look like a recovery of the true nodes.

The second kind is the distinctive part of the suite. It lives
in
[`tests/controls/`](https://github.com/Thomas-Rauter/kpnn2/tree/main/tests/controls),
with a few related checks in
[`tests/module/`](https://github.com/Thomas-Rauter/kpnn2/tree/main/tests/module).

## Scientific correctness

These checks use small graphs with a known live set: which
inputs and hidden nodes can affect a chosen output. The tests
do not ask whether a real biological prior is true. They ask
whether the primitives respect the prior they were given.

Two different ways an edge can be inert:

- **Absent.** The pair is not a row of the edgelist. There is
  no mask entry and no parameter.
- **Dead.** The pair is in the edgelist, so node roles stay the
  same, but tests pin that weight to 0.

### Live-path labels

<img class="correctness-icon" src="../figures/correctness/live_path.svg" alt="A live red path reaches the output; a grey branch stops short">

A name is important if and only if some path of live edges runs
from an input through that name to at least one attributed
output.

Each control graph **declares** those labels. An independent
reachability solver **derives** them from the edgelist and the
dead pins. The two must match, so a bug in the labels cannot
cancel a bug in the model.

Pinned in
[`tests/controls/ground_truth.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/ground_truth.py)
and
[`tests/controls/test_ground_truth.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_ground_truth.py).
Graphs include a disconnected decoy tower, a dead first hop, a
skip, multiple outputs, and live units that are not at tensor
index 0.

### Pinned-weight importance

<img class="correctness-icon" src="../figures/correctness/pinned_weights.svg" alt="Live edges pinned in red; a dead edge dashed and pinned at zero">

No training. Weights on live edges are pinned; dead edges stay
at 0. The net is linear so a live path has a deterministic
nonzero score.

Feature scores are `|input gradient|`. Hidden scores are
`|activation × layer gradient|` at that node's layer. Dead
names must sit near zero. Live names must sit above a floor.

Swapping the important and unimportant labels must fail: the
assertion is sensitive to the ground truth, not a tautology.

Pinned in
[`tests/controls/test_structural_importance.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_structural_importance.py).

### Trained importance

<img class="correctness-icon" src="../figures/correctness/trained_importance.svg" alt="Two matched towers into one prediction node, one red and one grey">

Two structurally matched towers, both wired to `prediction`.
Labels come from tower A only. After a learnability gate
(held-out ROC-AUC high enough that the net actually fit),
autograd scores must rank the data-generating tower above the
decoy.

The cases are a linear no-bias net, the same graph with ReLU
and bias, and a ReLU net whose labels are a product of tower-A
features. Each case uses a block of seeds and a pass-rate
floor, not a lucky window of five runs.

Pinned in
[`tests/controls/test_trained_importance.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_trained_importance.py).

### Negative controls

<img class="correctness-icon" src="../figures/correctness/negative_controls.svg" alt="A glass of water is not a drug bottle">

These break what trained importance is supposed to detect.

- **Shuffled labels.** Training on permuted `y` must not
  separate tower A from tower B, and must not fit the permuted
  task. Same idea as
  [Adebayo et al., 2018](https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps):
  a saliency method that still “recovers” structure after the
  labels are destroyed is following topology, not the data.
- **Swapped labels.** After a model that did learn, swapping
  which tower is called important must fail the trained
  criterion.

Pinned in
[`tests/controls/test_negative_controls.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_negative_controls.py).

### Rewired prior

<img class="correctness-icon" src="../figures/correctness/rewired_prior.svg" alt="A red stack plugged into a grey decoy; the prediction socket is empty">

Same feature names, same simulator (labels linear in tower A).
`G_true` gives both towers a path to `prediction`. `G_broken`
keeps the names but ends tower A at `decoy_readout`, with no
live path to the task output.

`G_true` must fit. `G_broken` must stay at chance. If a model
can learn the labels without a live path from the causal
features, the named graph is not actually constraining the
hypothesis.

Pinned in
[`tests/controls/test_rewired_prior.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_rewired_prior.py).

### Edges that are not in the graph

<img class="correctness-icon" src="../figures/correctness/absent_edge.svg" alt="A complete red path above a grey path with a missing edge marked by an X">

On two independent paths, raising a feature that feeds only a
decoy output must not change the prediction, in the forward
pass or in the input gradient.

Pinned in
[`tests/module/test_absent_edge_influence.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_absent_edge_influence.py).

Skip edges are the other direction: they **are** in the graph,
as ordinary packed pairs of the target hop, not a second
module. A skip still drives the target when the adjacent chain
is ReLU-zeroed. Zeroing only that packed weight removes only
that term. Gradient reaches the skip source directly.

Pinned in
[`tests/module/test_hop_forward.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_hop_forward.py).
[Skip edges](skip-edges.ipynb) is the design.

### Unrolled adjacency

<img class="correctness-icon" src="../figures/correctness/unrolled_adjacency.svg" alt="A grey first hop around a cycle; a red wrap reaches the output">

`parse_adjacency` graphs are a shared state vector applied `T`
times. A walk that needs three hops is dead at `T=2` and live
at `T=3` on the same edgelist. Unbounded DAG reachability is
the wrong ground truth. `T` is chosen in the test, not by
`kpnn2`.

Pinned-weight scores on a linear `PackedLinear` must match those
T-bounded labels. Swapping the `T=2` and `T=3` labels must fail.
After several extra steps, a feature that feeds only a decoy
must still not change the prediction. Dropping the self-loop
that would carry an early pulse must leave the net at chance.

Packed attention is the same absent-pair claim: a key that is
not a live edgelist source must not influence a query.

Pinned in
[`tests/controls/unroll.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/unroll.py),
[`tests/controls/test_unroll.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_unroll.py),
[`tests/controls/test_unrolled_importance.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_unrolled_importance.py),
[`tests/controls/test_unrolled_absent_edge.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_unrolled_absent_edge.py),
and
[`tests/controls/test_no_memory_prior.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_no_memory_prior.py).
Attention pairs are pinned in
[`tests/module/test_packed_attention_structure.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_attention_structure.py)
and
[`tests/module/test_packed_attention_kernel.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_attention_kernel.py).

### Name mapping

<img class="correctness-icon" src="../figures/correctness/name_mapping.svg" alt="Anonymous squares with a name tag clipped onto each column">

`kpnn2` does not import Captum. The library only attaches spec
names to a tensor axis. Tests still run Integrated Gradients
in the suite (Captum is a dev extra), then pass the tensor to
`map_node_attributions`. Dead inputs stay near zero; live
inputs stay above a floor. Hidden-layer names are checked on a
synthetic tensor whose values would fail if the axis were
permuted.

Pinned in
[`tests/controls/test_captum_mapping.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_captum_mapping.py)
and
[`tests/module/test_map_node_attributions.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_map_node_attributions.py).
[Mapping attributions](map-node-attributions.ipynb) is the
workflow.

## Technical correctness

Shorter on purpose. These pin the contract that the scientific
checks assume.

**Edgelist and layering.** `parse_layered` enforces a DAG, no
self-loops, no duplicate pairs, inferred inputs and outputs.
`parse_adjacency` allows cycles and self-loops and never
allocates an `(n, n)` tensor until `to_mask()`. Same DAG, two
layouts, two fingerprints.
[`tests/module/test_parse_layered_ranking.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_parse_layered_ranking.py),
[`tests/module/test_parse_adjacency.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_parse_adjacency.py),
[`tests/module/test_spec_serialize.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_spec_serialize.py).

**Every edge is one packed pair.** Summing the pair counts over
all hops equals the edgelist length. Each original edge is a
pair in exactly one hop, the hop of its target, skips included.
[`tests/module/test_parse_layered_hops.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_parse_layered_hops.py).

**PackedLinear.** Same graph as `MaskedLinear(spec.to_mask())`
gives the same forward values. Absent edges are not
parameters. Construction does not call `to_mask()`.
`transpose()` matches dense `W.T` on the live edges and
ties `weight` when `tie=True`. A `constraint=` module that
`torch.where`-replaces packed slots holds those live-edge
values under AdamW and SGD with momentum; a gradient hook
that zeroes `grad[i]` does not.
[`tests/module/test_packed_linear.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_linear.py),
[`tests/module/test_packed_linear_transpose.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_linear_transpose.py),
[`tests/module/test_constraint_freeze.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_constraint_freeze.py).

**Hop source axis.** `gather_hop_inputs` concatenates whole
source layers; `scatter_hop_outputs` splits that axis back.
A missing saved layer raises `Kpnn2Error` instead of
silently dropping those edges.
[`tests/module/test_gather_hop_inputs.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_gather_hop_inputs.py),
[`tests/module/test_scatter_hop_outputs.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_scatter_hop_outputs.py).

**MaskedLinear.** A zero mask entry blocks that source in the
forward pass, in `layer.weight`, and in the gradient, even
when another parametrization is stacked on `weight`.
`constraint=` (an `nn.Module`, for example `nn.Softplus`)
runs on the unconstrained tensor before that mask.
`torch.where` inside that module holds a live cell under
AdamW; a gradient hook that zeroes the slot does not.
Optimizer steps leave blocked edges dead. Degree-aware
init uses the row’s live count, not `in_features`.
`torch.compile(..., fullgraph=True)` traces without a
graph break.
[`tests/module/test_masked_linear.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_masked_linear.py),
[`tests/module/test_constraint_freeze.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_constraint_freeze.py).

**Packed attention.** Scores exist only for live
`(source, target)` pairs. An absent key does not influence a
query. Isolated queries stay zeros, not NaN. No `(n, n)` score
parameter.
[`tests/module/test_packed_attention_structure.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_attention_structure.py).

**Checkpoints.** `state_dict` carries a digest of the live mask
or packed indices, and an optional `identity` (typically
`spec.fingerprint`). Loading into a rewired layer of the same
shape raises. Loading into a same-shape rename raises when
`identity` was set. Spec interchange is `to_dict` /
`from_dict`, not pickle of the dataclass. The public import
surface is a frozen list.
[`tests/api/test_public_api.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/api/test_public_api.py).

**A real table.** Breast Cancer Wisconsin Diagnostic, a sparse
DAG from named features, ordinary PyTorch training. Catches
breakage that still passes tiny unit tests.
[`tests/integration/test_real_tabular_task.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/integration/test_real_tabular_task.py).

## What the tests do not prove

- That **your** `forward()`, loss, optimizer, or prior is
  correct. The package does not ship a model.
- That a knowledge graph from biology or another domain is a
  true mechanism. The controls use synthetic graphs with a
  known live set.
- That Captum (or any other attribution method) is a valid
  estimator. The library only names an axis.
- That trained importance on real data will recover “the true
  nodes.” The trained checks are matched-tower simulations
  with a learnability gate.
- That the package picks `n_steps` or writes your time loop.
  Unrolled-adjacency checks use tiny graphs and a fixed `T`.
- GPU or TPU numerics in CI.
  [`tests/manual/`](https://github.com/Thomas-Rauter/kpnn2/tree/main/tests/manual)
  is Colab smoke, not pytest.

## Running the suite

From a clone with the `dev` extra:

```bash
pytest
```

That includes the slow scientific controls. They are marked
`integration` and `slow`. [Installation](installation.md)
covers the development extra. CI status is on
[GitHub Actions](https://github.com/Thomas-Rauter/kpnn2/actions/workflows/ci.yml).
