# How we test

`kpnn2` builds interpretable neural networks in PyTorch from a
graph of named entities, with checks that every
[attribution](concepts.md#attribution) is reported under the right
name. This page explains those checks.

The tests guard against three mistakes. Each one leaves a model
that trains and gives attributions without an error, so it would
end up in your results. Figure 1 shows them on the graph from
Figure 2 on [Home](index.md#how-it-works):

- **An edge goes missing.** `protein_d → outcome` is a
  [skip edge](concepts.md#skip-edge): it skips the hidden layer.
  Code that connects each layer only to the next has no place for
  it and drops it.
- **An edge appears that the graph does not have.** A weight from
  `protein_a` to `protein_f` lets `protein_a` affect `protein_f`,
  although the edgelist has no such row.
- **An attribution gets the wrong name.** An attribution method
  returns a table of numbers with one column per node and no
  names. If the names go onto the columns in the wrong order, the
  attributions of `protein_e` are reported as those of
  `protein_f`.

<img class="figure-full" src="../figures/silent_mistakes.svg" alt="Three copies of the Home graph. (a) The dashed skip edge from protein_d to outcome is grey and crossed out. (b) A thick red edge from protein_a to protein_f, which the edgelist does not have. (c) Two attribution tables with the same values: under the right names protein_e is dark and protein_f pale, and with the names swapped protein_f is dark.">

**Figure 1.** Three mistakes on the graph from Home, none of which
raises an error. In (c) the model relies on `protein_e`, but with
the names swapped the report says `protein_f`.

Each of these changes the conclusion, not only the code, so it is
a scientific error and not only a software bug. The test suite is
built around that.
[Three mistakes in hand-written code](#three-mistakes-in-hand-written-code)
shows how each one comes about.

This page is an overview of **what the tests claim**, not a
catalog of every `test_*` function. It has three parts: the
scientific claims, which ask whether the
[named graph](concepts.md#graph) really constrains the model; the
technical claims underneath them; and an explicit list of what
none of it proves. The files live on
GitHub under
[`tests/`](https://github.com/Thomas-Rauter/kpnn2/tree/main/tests).

CI runs the full pytest suite, including the slow controls, on
every change. That is a process check; the scientific argument
is the claims below.

A separate, frozen notebook repeats one published simulation
with these primitives:
[Fortelny and Bock, 2020](literature/fortelny-bock-2020.ipynb).
That page is not part of the test suite.

## Three mistakes in hand-written code

Each mistake takes a few lines of plain PyTorch, and each runs
without an error. The examples build the network of Figure 1 by
hand. `edgelist` holds its seven rows, and `torch`, `nn` (from
`torch`), and `pandas` as `pd` are imported.

### An edge goes missing

A common way to build a sparsely connected network is one 0/1
mask per pair of neighboring layers. This code writes the layers
by hand and fills each mask from the edgelist:

```python
layers = [
    ["protein_a", "protein_b", "protein_c", "protein_d"],
    ["protein_e", "protein_f"],
    ["outcome"],
]
masks = []
for sources, targets in zip(
    layers[:-1],
    layers[1:],
):
    mask = torch.zeros(
        len(targets),
        len(sources),
    )
    for source, target in zip(
        edgelist["source"],
        edgelist["target"],
    ):
        if source in sources and target in targets:
            mask[targets.index(target), sources.index(source)] = 1
    masks.append(mask)
```

`protein_d` is in the first layer and `outcome` in the third, so
the row `protein_d → outcome` matches neither pair of layers. The
two masks hold six of the seven edges, and nothing reports the
seventh.

`parse_layered()` puts each edge into the [hop](concepts.md#hop)
of its target, so `protein_d → outcome` sits in the hop into
`outcome`, next to the edges from `protein_e` and `protein_f`. The
tests check that the hops together hold every row of the edgelist
and that a skip edge drives its target on its own: see
[Edges that are not in the graph](#edges-that-are-not-in-the-graph)
and [Technical correctness](#technical-correctness).

### An edge appears

To use the masks, this code zeroes the absent edges of each
`nn.Linear` once, before training:

```python
hidden = nn.Linear(
    4,
    2,
)
with torch.no_grad():
    hidden.weight *= masks[0]
```

The weight of `protein_a → protein_f` starts at zero, but its
gradient is usually not zero, so the first optimizer step moves
it. After training, `protein_a` affects `protein_f` through an
edge the edgelist does not have, and nothing in the training run
shows it.

`MaskedLinear` applies its mask in every forward pass, so a
blocked weight stays at zero under SGD with momentum, Adam, and
AdamW. `PackedLinear` has no weight for an absent edge at all.
[Edges that are not in the graph](#edges-that-are-not-in-the-graph)
and the `MaskedLinear` checks under
[Technical correctness](#technical-correctness) pin both.

### An attribution gets the wrong name

This version takes the hidden layer from the edgelist instead of
writing it by hand: each node in the order it first appears as a
target. The masks are built in that order, as in the first
example:

```python
targets = edgelist["target"].unique()
hidden_nodes = [node for node in targets if node != "outcome"]
```

After training, `attributions` holds what an attribution method
returned for the hidden layer: one row per sample and one column
per hidden [unit](concepts.md#unit), in the order of
`hidden_nodes`. The code names
the columns from a sorted list:

```python
table = pd.DataFrame(
    attributions.numpy(),
    columns=sorted(hidden_nodes),
)
```

On the edgelist of Figure 1 both orders are `protein_e`,
`protein_f`, so the names are right. If a later release of the
edgelist lists the rows of `protein_f` first, `hidden_nodes`
becomes `protein_f`, `protein_e`, and the sorted names stay as
they were. The attributions of `protein_e` then appear under
`protein_f`. Code that gave the right names when it was written
gives the wrong ones, and nothing fails.

The parsers sort the nodes of each layer alphabetically, so the
row order of the edgelist does not move a unit, and
`map_node_attributions()` reads the names from the same spec the
layers were built from. [Name mapping](#name-mapping) checks that
the names land on the right columns.

## Two kinds of correctness

**Technical.** The [parsers](concepts.md#parser), masks,
[packed](concepts.md#packed-indices) layers, alignment, and
checkpoints do what the public contract says. A cycle is
rejected by `parse_layered`. A masked-out weight cannot affect
the output. `PackedLinear` matches `MaskedLinear` on the same
graph.

**Scientific.** The named graph is the architecture. A
[node](concepts.md#node) with no live path to the attributed
output must not score as important. An edge that is not in the
edgelist must not influence a prediction. A shuffled or rewired prior must not
look like a recovery of the true nodes.

Scientific correctness is the distinctive part of the suite. It
lives in
[`tests/controls/`](https://github.com/Thomas-Rauter/kpnn2/tree/main/tests/controls),
with a few related checks in
[`tests/module/`](https://github.com/Thomas-Rauter/kpnn2/tree/main/tests/module).

## Scientific correctness

These checks use small graphs with a known live set: which
inputs and hidden nodes can affect a chosen output. The tests
do not ask whether a real biological prior is true. They ask
whether the primitives respect the prior they were given.

A control graph needs edges that carry no signal, and an edge
can be inert in two ways:

- **Absent.** The pair is not a row of the edgelist. There is
  no mask entry and no parameter.
- **Dead.** The pair is a row of the edgelist, so node roles
  stay the same, but tests pin that weight to 0.

### Live-path labels

<img class="correctness-icon" src="../figures/correctness/live_path.svg" alt="A live red path reaches the output; a grey branch stops short">

Every later check needs to know which names should score as
important, so the definition comes first. A name is important
if and only if some path of [live edges](concepts.md#live-edge),
edges the graph actually has, runs from an input through that
name to at least one attributed output.

Each control graph **declares** those labels by hand. An
independent reachability solver **derives** them from the
edgelist and the dead pins. The two must match, so a mistake in
the hand-written labels cannot cancel a mistake in the model.

Pinned in
[`tests/controls/ground_truth.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/ground_truth.py)
and
[`tests/controls/test_ground_truth.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_ground_truth.py).
Graphs include a disconnected decoy tower, a dead first
[hop](concepts.md#hop) — everything arriving at one layer — a
skip, multiple outputs, and live units that are not at tensor
index 0.

### Pinned-weight importance

<img class="correctness-icon" src="../figures/correctness/pinned_weights.svg" alt="Live edges pinned in red; a dead edge dashed and pinned at zero">

The first check takes training out of the picture, so that
nothing but the wiring can explain a score. Weights on live
edges are pinned, dead edges stay at 0, and the net is linear,
so every live path carries a deterministic nonzero score.

Feature scores are `|input gradient|`. Hidden scores are
`|activation × layer gradient|` at that node's layer. Dead
names must sit near zero. Live names must sit above a floor.

Swapping the important and unimportant labels must fail: the
assertion is sensitive to the ground truth, not a tautology.

Pinned in
[`tests/controls/test_structural_importance.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_structural_importance.py).

### Trained importance

<img class="correctness-icon" src="../figures/correctness/trained_importance.svg" alt="Two matched towers into one prediction node, one red and one grey">

The next check lets training set the weights. Two structurally
matched towers are both wired to `prediction`, but only tower A
generates the labels. After a learnability gate (held-out
ROC-AUC high enough that the net actually fit), autograd scores
must rank the data-generating tower above the decoy.

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
- **Swapped labels.** Take a model that did pass the
  learnability gate, then swap which tower is called important:
  the trained criterion must fail.

Pinned in
[`tests/controls/test_negative_controls.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_negative_controls.py).

### Rewired prior

<img class="correctness-icon" src="../figures/correctness/rewired_prior.svg" alt="A red stack plugged into a grey decoy; the prediction socket is empty">

A prior only constrains a model if breaking the prior breaks
the fit. Both graphs here use the same feature names and the
same simulator, whose labels are linear in tower A. `G_true`
gives both towers a path to `prediction`. `G_broken` keeps the
names but ends tower A at `decoy_readout`, with no live path to
the task output.

`G_true` must fit. `G_broken` must stay at chance. If a model
can learn the labels without a live path from the causal
features, the named graph is not actually constraining the
hypothesis.

Pinned in
[`tests/controls/test_rewired_prior.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_rewired_prior.py).

### Edges that are not in the graph

<img class="correctness-icon" src="../figures/correctness/absent_edge.svg" alt="A complete red path above a grey path with a missing edge marked by an X">

An absent edge has no weight to pin, so the claim is about the
prediction itself. In a graph of two disjoint paths, raising a
feature that feeds only a decoy output must not change the
prediction, either in the forward pass or in the input
gradient.

Pinned in
[`tests/module/test_absent_edge_influence.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_absent_edge_influence.py).

Skip edges make the opposite claim: they **are** in the graph,
as ordinary packed pairs of the target hop, not a second
module. A skip therefore still drives the target when the
adjacent chain is ReLU-zeroed. Zeroing only that packed weight
removes only that term. Gradient reaches the skip source
directly.

Pinned in
[`tests/module/test_hop_forward.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_hop_forward.py).
[Edges that skip layers](skip-edges.ipynb) is the design.

### Unrolled adjacency

<img class="correctness-icon" src="../figures/correctness/unrolled_adjacency.svg" alt="A grey first pass around a cycle; a red wrap reaches the output">

A `parse_adjacency` graph has no layers. The user applies one
shared module `T` times to the
[state vector](concepts.md#state-vector) — a single vector over
all nodes — so how far a signal travels depends on `T`. The step
count therefore enters the ground truth. A path three edges long
is dead at `T=2` and live at `T=3` on the same edgelist, which
makes unbounded DAG reachability the wrong labels here. `T` is
chosen in the test, not by `kpnn2`.

Pinned-weight scores on a linear `PackedLinear` must match those
T-bounded labels. Swapping the `T=2` and `T=3` labels must fail.
After several extra steps, a feature that feeds only a decoy
must still not change the prediction. Dropping the self-loop
that would carry an early pulse must leave the net at chance.

Packed attention is the same absent-pair claim in another
module: a key that is not a live edgelist source must not
influence a query.

Pinned in
[`tests/controls/unroll.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/unroll.py),
[`tests/controls/test_unroll.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_unroll.py),
[`tests/controls/test_unrolled_importance.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_unrolled_importance.py),
[`tests/controls/test_unrolled_absent_edge.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_unrolled_absent_edge.py),
and
[`tests/controls/test_no_memory_prior.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_no_memory_prior.py).
Attention pairs are pinned in
[`tests/module/test_packed_attention_structure.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_attention_structure.py),
[`tests/module/test_packed_attention_kernel.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_attention_kernel.py),
and
[`tests/module/test_packed_attention_chunk_size.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_attention_chunk_size.py).

### Name mapping

<img class="correctness-icon" src="../figures/correctness/name_mapping.svg" alt="Anonymous squares with a name tag clipped onto each column">

An attribution score is only interpretable if it carries the
right name, and attaching [spec](concepts.md#spec) names — the
frozen structure a parser returns — to a tensor axis is all
the library does here: `kpnn2` does not import Captum. The
tests still run Integrated Gradients in the suite (Captum is a
dev extra), then pass the resulting tensor to
`map_node_attributions`. Dead inputs stay near zero; live
inputs stay above a floor. Hidden-layer names are checked on a
synthetic tensor whose values would fail if the axis were
permuted.

Captum `LayerConductance` also runs on a hop whose input and
output are both two units wide, so the width check cannot say
which side a tensor came from. With a hidden node cut off from
the output, its zero score must land under that node's name via
`hop_output=`. With an input that feeds nothing, its zero score
must land under the input's name via `hop_input=` (Captum's
`attribute_to_layer_input=True`). `axis="inputs"` names an
input-width tensor (input units on an `AdjacencySpec`, the
same result as `layer=0` on a `LayeredSpec`) and is never
inferred from the width.

Pinned in
[`tests/controls/test_captum_mapping.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/controls/test_captum_mapping.py)
and
[`tests/module/test_map_node_attributions.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_map_node_attributions.py).
[Attributions by node name](map-node-attributions.ipynb) is the
workflow.

## Technical correctness

Shorter on purpose: these checks pin the contract that every
scientific claim above rests on.

**Edgelist and layering.** `parse_layered` enforces a DAG, no
self-loops, no duplicate pairs, inferred inputs and outputs.
`parse_adjacency` allows cycles and self-loops and never
allocates an `(n, n)` tensor until `to_mask()`. Same DAG, two
layouts, two fingerprints.
[`tests/module/test_parse_layered_ranking.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_parse_layered_ranking.py),
[`tests/module/test_parse_adjacency.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_parse_adjacency.py),
[`tests/module/test_spec_serialize.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_spec_serialize.py).

**Named edges pack as unit pairs.** Each named edge belongs to
exactly one hop, the hop of its target, skips included. At
width 1, summing the packed pair counts over all hops equals
the edgelist length. A wider node expands that edge into a
`k_source * k_target` block, and the sum counts those pairs.
Width on an input node raises `Kpnn2Error` in both parsers and
in `from_dict`.
[`tests/module/test_parse_layered_hops.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_parse_layered_hops.py),
[`tests/module/test_parse_layered_widths.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_parse_layered_widths.py).

**PackedLinear.** On the same graph, `PackedLinear` and
`MaskedLinear(spec.to_mask())` produce the same forward values.
Absent edges are not parameters. Construction does not call
`to_mask()`.
`transpose()` matches dense `W.T` on the live edges and
ties `weight` when `tie=True`. A `constraint=` module that
`torch.where`-replaces packed slots holds those live-edge
values under AdamW and SGD with momentum; a gradient hook
that zeroes `grad[i]` does not. A keep-mask buffer in that
module is the in-training prune: a zeroed slot contributes
nothing, Adam state and `index_digest` stay, the buffer
reloads into a layer rebuilt from the same spec, and a tied
transpose sees the zero. `optimizer.load_state_dict` across
a reparse is accepted when the parameter shapes match, which
a packed layer still has when `nnz` is unchanged. `forward`
disables `torch.autocast` and casts `x` to the parameter
dtype.
[`tests/module/test_packed_linear.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_linear.py),
[`tests/module/test_packed_linear_transpose.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_linear_transpose.py),
[`tests/module/test_constraint_freeze.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_constraint_freeze.py),
[`tests/module/test_keep_mask_prune.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_keep_mask_prune.py).

**Hop source axis.** `gather_hop_inputs` concatenates whole
source layers; `scatter_hop_outputs` splits that axis back.
A missing saved layer raises `Kpnn2Error` instead of
silently dropping those edges.
[`tests/module/test_gather_hop_inputs.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_gather_hop_inputs.py),
[`tests/module/test_scatter_hop_outputs.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_scatter_hop_outputs.py).

**Alignment.** `align_inputs` maps a sequence of feature names
to a 1-D `int64` column index. It does not take the matrix or
return a data tensor. A DataFrame, tensor, string, bytes,
mapping, set, AnnData-like object, or a matrix (`ndim != 1`)
raises `Kpnn2Error`. So does a missing input name, and a
feature named like a hidden or output node; one case leaves
the gene layer out of an edgelist whose nodes carry gene
symbols. The index has one position per input
node on both layouts, because the parsers reject an input
wider than 1. On an `AdjacencySpec` it still needs scatter
into the state vector.
[`tests/module/test_align_inputs.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_align_inputs.py).

**MaskedLinear.** A zero mask entry blocks that source in the
forward pass, in `effective_weight()`, and in the gradient,
even when a parametrization is registered on `weight`.
`constraint=` (an `nn.Module`, for example `nn.Softplus`)
runs on the unconstrained tensor before that mask.
`torch.where` inside that module holds a live cell under
AdamW; a gradient hook that zeroes the slot does not.
Blocked entries of `weight` start at 0 and stay 0 under SGD
with momentum, Adam, and AdamW. `weight` is a plain parameter,
as on `PackedLinear` and `nn.Linear`, so `.weight` param-group
filters, pickling, and `torch.nn.utils.prune` behave the same,
and a kpnn2 0.1 checkpoint still loads. Degree-aware
init uses the row’s live count, not `in_features`.
`forward` disables `torch.autocast` and casts `x` to the
parameter dtype. `torch.compile(..., fullgraph=True)`
traces without a graph break.
[`tests/module/test_masked_linear.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_masked_linear.py),
[`tests/module/test_constraint_freeze.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_constraint_freeze.py).

**Packed attention.** Scores exist only for live
`(source, target)` pairs. An absent key does not influence a
query. Isolated queries stay zeros, not NaN. No `(n, n)` score
parameter. `chunk_size=None` gathers all live pairs at once; a
positive `chunk_size` matches that mix and its gradients.
`forward` disables `torch.autocast` and casts query, key, and
value to the parameter dtype.
[`tests/module/test_packed_attention_structure.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_attention_structure.py),
[`tests/module/test_packed_attention_chunk_size.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/module/test_packed_attention_chunk_size.py).

**Checkpoints.** `state_dict` carries a digest of the live mask
or packed indices, and an optional `identity` (typically
`spec.fingerprint`). Loading into a rewired layer of the same
shape raises. Loading into a same-shape rename raises when
`identity` was set. Spec interchange is `to_dict` /
`from_dict`, not pickle of the dataclass. The public import
surface is a frozen list.
[`tests/api/test_public_api.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/api/test_public_api.py).

**Bad input.** We call every public entry point with bad
input, and each call must raise `Kpnn2Error` with a message
that names the argument or the offending node.
[`tests/api/test_input_validation.py`](https://github.com/Thomas-Rauter/kpnn2/blob/main/tests/api/test_input_validation.py).

**A real table.** One integration test trains on the Breast
Cancer Wisconsin Diagnostic data through a sparse DAG built
from the named features, with ordinary PyTorch training. It
catches breakage that tiny unit tests still pass.
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
- That GPU or TPU numerics are covered in CI.
  [`tests/manual/`](https://github.com/Thomas-Rauter/kpnn2/tree/main/tests/manual)
  is Colab smoke, not pytest.

## Running the suite

From a clone with the `dev` extra:

```bash
pytest
```

That includes the slow scientific controls, which are marked
`integration` and `slow`. [Installation](installation.md)
covers the development extra. CI status is on
[GitHub Actions](https://github.com/Thomas-Rauter/kpnn2/actions/workflows/ci.yml).
