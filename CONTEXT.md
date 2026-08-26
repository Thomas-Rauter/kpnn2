# kpnn2 — AI context document

This file is **AI-first documentation** for assistants working in this
repository or explaining the package to users. It is more detailed and
operational than `README.md`.

**Release:** `0.1.0` as package `kpnn2` (`import kpnn2`).
This document is the implementation contract.

Do not reintroduce a graph compiler or a ready-made model object.
Do not rename the distribution, import, or `src/` package away
from `kpnn2`.

---

## One-sentence summary

Build sparsely connected PyTorch neural networks from a named
edgelist, using native nn.Module layers.

---

## What this package does

1. **Parse:** `parse_layered()` reads a pandas DataFrame with columns
   `source` and `target` only, validates a layered DAG, and returns a
   `LayeredSpec`. `parse_adjacency()` reads the same table and
   returns an `AdjacencySpec` instead: one square mask over every
   node, cycles and self-loops allowed. The user picks the layout;
   a DAG is valid input to both.
2. **Specify:** `LayeredSpec` holds named nodes by layer, adjacent-hop
   mask tensors, and skip-edge metadata. `AdjacencySpec` holds all
   node names, one square mask, and the input/output positions in
   it. Neither constructs an `nn.Module`.
3. **Build:** The user writes a PyTorch `nn.Module` using
   `MaskedLinear(spec.masks[i])` for adjacent hops,
   `SkipAdd(spec)` for skip indexing, and their own activations,
   norms, loops, and heads. On an `AdjacencySpec` the state update
   is `MaskedLinear(spec.mask)` and the recurrence is the user's
   `forward()`.
4. **Align:** `align_inputs()` maps a named DataFrame onto
   `spec.input_nodes` as a `float32` tensor. Pre-ordered tensors
   go straight to the model.
5. **Train:** The user owns loss, optimizer, and the training loop.
6. **Map attributions:** `map_node_attributions()` labels a tensor
   at one LayeredSpec layer as an `xarray.DataArray` (`node` names
   from `layer_nodes[layer]`). Captum is not a library dependency;
   the user runs Captum (or any other method) themselves. `xarray`
   is a core dependency used only here.

### Primary use cases

- Sparse neural nets whose connectivity is a named edgelist.
- Knowledge-primed neural networks (KPNNs): prior knowledge defines
  which edges exist; weights and `forward()` are user PyTorch.
- Any domain where nodes have stable string names (genes, pathways,
  sensors, etc.).

### What kpnn2 is NOT

- **Not a compiler.** It does not return a ready-made model or choose
  a backend.
- **Not a GNN library.** No message passing, batched variable graphs,
  or edge-feature convolutions.
- **Not a trainer.** No losses, optimizers, or training loops.
- **Not Captum.** No Captum import anywhere in the library. Attribution
  mapping is name alignment only.
- **Not AnnData (v1).** No `anndata` support in `align_inputs`.
- **Not a time machine.** `parse_adjacency` accepts cycles and
  self-loops, but nothing here unrolls time, picks a step count, or
  re-injects inputs between steps. Recurrence is user `forward()`
  over `AdjacencySpec.mask`. `parse_layered` stays DAG-only and
  still raises `Kpnn2Error` on a cycle.
- **Not pseudo-node expansion.** Skip edges are metadata plus
  `SkipAdd` (or a residual add in user code), not dummy neurons
  in masks.

---

## Package philosophy

**Primitives, not a compiled container.** `kpnn2` owns edgelist
parsing, mask tensors, skip indexing, named I/O alignment, and
attribution column names. The user owns `nn.Module.forward()`,
call order, nonlinearities, and training.

Division of labor:

| Layer | Owner |
|-------|--------|
| Edgelist → `LayeredSpec` (ranks, masks, skips) | kpnn2 |
| Edgelist → `AdjacencySpec` (nodes, one square mask) | kpnn2 |
| `MaskedLinear` (frozen mask) | kpnn2 |
| `SkipAdd` (skip indexing onto a layer tensor) | kpnn2 |
| `forward()`, activations, norms, heads, call order | User (PyTorch) |
| Training and evaluation | User (PyTorch) |
| Captum / other attribution algorithms | User |
| Tensor → named `xarray.DataArray` | kpnn2 |

---

## Public API (only these names)

Exported from `kpnn2` (`src/kpnn2/__init__.py`):

| Symbol | Role |
|--------|------|
| `parse_layered` | Edgelist DataFrame → `LayeredSpec` (DAG only) |
| `parse_adjacency` | Edgelist DataFrame → `AdjacencySpec` (cycles allowed) |
| `LayeredSpec` | Frozen structural dataclass (masks, layers, skips) |
| `Skip` | One skip-edge record (see below); exported because `spec.skips` uses it |
| `AdjacencySpec` | Frozen structural dataclass (nodes, one square mask) |
| `MaskedLinear` | `nn.Module`: masked linear layer |
| `SkipAdd` | `nn.Module`: inject skip sources into a layer pre-activation |
| `align_inputs` | Named DataFrame → `float32` input tensor |
| `map_node_attributions` | Layer tensor → labeled `xarray.DataArray` |
| `Kpnn2Error` | User-facing error type |
| `__version__` | Package version string |

`__all__` contains exactly these names (including `Skip`,
`SkipAdd`, and `__version__`), in the order of the table above.
`tests/api/test_public_api.py` compares it as an ordered list. No
other public symbols.

Do **not** export or implement: `compile_graph`, `customize_model`,
`interpret_model`, `align_features_to_input_nodes`, `edge_weights`,
`CompileArtifact`, backends, `ConstrainedMaskedLinear`.

There are **two parsers and two specs, never a dispatcher**. Do not
add `parse(..., layout=...)`, do not choose a parser by inspecting
the graph for cycles, and do not represent an `AdjacencySpec` as a
one-layer `LayeredSpec`. A DAG is valid input to both parsers; the
layout is the user's choice.

---

## Edgelist format

```python
edgelist = pd.DataFrame(
    {
        "source": ["feature_a", "feature_b", "hidden"],
        "target": ["hidden", "hidden", "output_1"],
    }
)
```

**Required columns:** `source`, `target`. Each row is one directed
edge in the direction of computation (source feeds target).

**No other columns are read.** Extra columns, if present, are ignored
and must not change parsing. There is no `initial_weight` or
`constraint` support.

Node names are stored as strings. Non-string values in `source` /
`target` are converted with `str(...)`.

---

## Graph rules (v1)

`parse_layered()` enforces:

| Rule | On violation |
|------|----------------|
| Input is a `pandas.DataFrame` | `Kpnn2Error` |
| Columns `source` and `target` exist | `Kpnn2Error` |
| No missing values in `source` or `target` | `Kpnn2Error` |
| No empty-string node names | `Kpnn2Error` |
| At least one edge | `Kpnn2Error` |
| No duplicate `(source, target)` pairs | `Kpnn2Error` naming the pair(s), sorted |
| No self-loops (`source == target`) | `Kpnn2Error` naming the node(s), sorted |
| Graph is a DAG (no cycles) | `Kpnn2Error` naming unranked leftover nodes |
| At least one input (in-degree 0) | `Kpnn2Error` |
| At least one output (out-degree 0) | `Kpnn2Error` |

`parse_adjacency()` enforces **every rule in that table except the
two structural ones**: self-loops are allowed and cycles are
allowed. Everything else (DataFrame, columns, missing values,
empty names, at least one edge, duplicate pairs, at least one
input, at least one output) is identical, and identically worded,
because both parsers call the same validation helpers. Self-loops
are the only edgelist rule the two parsers disagree on.

**Node roles (inferred, not user-declared):**

- **Input nodes:** in-degree 0. Sorted alphabetically.
  Stored in `spec.input_nodes`. These are also the tensor column
  order for `align_inputs` / `MaskedLinear` on hop 0.
- **Output nodes:** out-degree 0. Sorted alphabetically.
  Stored in `spec.output_nodes`. Early outputs (terminals whose
  depth is not the maximum depth) **are allowed**. They remain in
  their own layer; they are not moved, padded, or rejected.
- **Hidden nodes:** every other named node (not input, not output).
  Sorted alphabetically. Stored in `spec.hidden_nodes`.

**Layering (Kahn / longest-path from inputs):**

- Process nodes in topological order. A node becomes ready when all
  parents are assigned a depth.
- `depth(input) = 0`.
- `depth(node) = 1 + max(depth(parent) for parent in parents)`.
  Equivalently: the depth assigned when in-degree hits 0 in a Kahn
  sweep that increments depth each frontier.
- `layer_nodes[d]` is the alphabetically sorted list of nodes with
  `depth == d`.
- Layer 0 is the first layer (all depth-0 nodes, i.e. all inputs).
  If a graph somehow had a depth-0 non-input, that would violate
  in-degree 0 ⇔ input; do not invent extra depth-0 nodes.

A cycle is detected when `len(depths) != len(nodes)` after the
sweep. The `Kpnn2Error` message still says the edgelist has a
cycle and that only DAGs are supported, and lists every unranked
name (`nodes` minus keys of `depths`), sorted alphabetically and
comma-separated. That leftover set may include nodes downstream
of a cycle, not only vertices on a directed cycle. Do not run a
separate cycle-extraction algorithm.

Duplicate `(source, target)` pairs raise `Kpnn2Error` with the
count of extra rows after the first of each pair
(`DataFrame.duplicated().sum()`) and the unique duplicated pairs
as `{source} -> {target}`, sorted lexicographically by
`(source, target)` and comma-separated. Self-loops raise
`Kpnn2Error` with the row count (`source == target`) and unique
node names, sorted alphabetically and comma-separated.

Isolated nodes cannot appear: the node set is the union of `source`
and `target` values only.

---

## `LayeredSpec` fields

`LayeredSpec` is a frozen dataclass. It holds structure only: no
`nn.Module`, no parameters, no execution plan object. Sequences
are tuples. Do not reassign fields. Mask tensors reject in-place
writes, `out=` writes into the mask, and numpy aliases of stored
storage (`Kpnn2Error` for torch writes). `copy.deepcopy` of a
`LayeredSpec` succeeds; copied masks stay frozen float32 and do
not share storage with the original.

| Field | Type | Meaning |
|-------|------|---------|
| `input_nodes` | `tuple[str, ...]` | In-degree 0 nodes, alphabetical. Tensor column order. |
| `output_nodes` | `tuple[str, ...]` | Out-degree 0 nodes, alphabetical. |
| `hidden_nodes` | `tuple[str, ...]` | Neither input nor output, alphabetical. |
| `layer_nodes` | `tuple[tuple[str, ...], ...]` | `layer_nodes[i]` = names at depth `i`, alphabetical. Index 0 is the first layer. |
| `layer_dims` | `tuple[int, ...]` | `layer_dims[i] == len(layer_nodes[i])`. |
| `masks` | `tuple[torch.Tensor, ...]` | Adjacent-hop masks (see below). |
| `skips` | `tuple[Skip, ...]` | Skip edges with depth gap `> 1` (see below). |

### Masks

- `len(masks) == len(layer_nodes) - 1`.
- `masks[i]` is the hop from layer `i` to layer `i+1`.
- Dtype: `torch.float32`.
- Shape: `(n_{i+1}, n_i)` = `(layer_dims[i+1], layer_dims[i])`.
  This matches `nn.Linear.weight` layout `(out_features, in_features)`.
- `masks[i][target_index, source_index] = 1.0` iff there is an
  original edgelist edge from `layer_nodes[i][source_index]` to
  `layer_nodes[i+1][target_index]` **and** that edge has depth gap
  exactly 1.
- All other entries are `0.0`.
- **Skip edges (gap > 1) do not appear in any mask.**
- In-place writes (`fill_`, `copy_`, item assignment) and
  `out=` into a stored mask raise `Kpnn2Error`. `numpy()` does
  not yield a writable view of the stored storage.
- `copy.deepcopy` succeeds. Copied masks keep the same values,
  stay frozen float32, and do not share storage with the
  original.
- `MaskedLinear(spec.masks[i])` stores an independent frozen
  copy. A write (or failed write) on one side does not change
  the other.

### `Skip` records

Each skip is a frozen dataclass `Skip` with:

| Field | Type | Meaning |
|-------|------|---------|
| `source` | `str` | Source node name |
| `target` | `str` | Target node name |
| `source_layer` | `int` | Depth of `source` |
| `target_layer` | `int` | Depth of `target`; `target_layer - source_layer > 1` |
| `source_index` | `int` | Index of `source` in `layer_nodes[source_layer]` |
| `target_index` | `int` | Index of `target` in `layer_nodes[target_layer]` |

Every original edgelist edge with depth gap `> 1` appears once in
`skips`. Adjacent edges (gap `== 1`) appear only in `masks`, never
in `skips`.

---

## Skip connections (no pseudo nodes)

Masks only encode **adjacent** hops. A skip `A → H2` that jumps
one or more layers is **not** inserted as a dummy channel and is
**not** written into any mask.

A skip means the source is an extra parent of the target.
`SkipAdd` injects `w * saved_source` into the target
pre-activation **after** that hop's `MaskedLinear` and **before**
ReLU / BatchNorm / dropout. It does not undo ReLU, does not send
the source through the adjacent weight matrix, and does not
modify saved source tensors. There is no skip bias; unit bias
stays on `MaskedLinear`.

`kpnn2` owns skip indexing (`spec.skips` plus `SkipAdd`). The
user owns call order and nonlinearities. Hand-written residual
adds remain valid.

```python
self.skips = k2.SkipAdd(spec)      # once; one scalar per skip

saved = {0: x}
h = self.lin0(x)                   # adjacent hop 0→1
h = self.skips(h, saved, 1)        # no-op if nothing targets 1
h = torch.relu(h)
saved[1] = h
c = self.lin1(h)                   # adjacent hop 1→2
c = self.skips(c, saved, 2)        # A → C into C's pre-activation
```

Construct `SkipAdd` once. Call it after every hop; it is a
no-op when nothing targets that layer. Empty `spec.skips` is
identity. `MaskedLinear` must not contain skip edges. There is
no identity overwrite and no compiler-generated node names.

---

## `parse_adjacency` and `AdjacencySpec`

```text
parse_adjacency(edgelist) -> AdjacencySpec
```

The second layout. Every node goes into one state vector, sorted
alphabetically, and every edge goes into one square mask. Nothing
is ranked, so **cycles and self-loops are allowed**. This is the
layout for recurrent networks; the recurrence itself is user
`forward()` code.

`parse_adjacency` must not instantiate an `nn.Module`, unroll
time, choose a step count, or re-inject inputs between steps.

### `AdjacencySpec` fields

Frozen dataclass, same rules as `LayeredSpec`: no reassignment,
sequences are tuples, the mask rejects in-place writes and `out=`
writes (`Kpnn2Error`), `numpy()` is not a writable view of stored
storage, and `copy.deepcopy` succeeds with a frozen float32 copy
that does not share storage.

| Field | Type | Meaning |
|-------|------|---------|
| `nodes` | `tuple[str, ...]` | Every node name, alphabetical. Row and column order of `mask`, and the unit order of the state vector. |
| `input_nodes` | `tuple[str, ...]` | In-degree 0 nodes, alphabetical. Column order for `align_inputs`. |
| `output_nodes` | `tuple[str, ...]` | Out-degree 0 nodes, alphabetical. |
| `hidden_nodes` | `tuple[str, ...]` | Neither input nor output, alphabetical. |
| `mask` | `torch.Tensor` | Square connectivity (see below). Singular, not a tuple. |
| `input_index` | `tuple[int, ...]` | Position of each `input_nodes` name in `nodes`. |
| `output_index` | `tuple[int, ...]` | Position of each `output_nodes` name in `nodes`. |

There is no `layer_nodes`, no `layer_dims`, no `masks` tuple, and
no `skips`. Skips are a depth concept and depth does not exist
here. `SkipAdd` does not accept an `AdjacencySpec`.

### The square mask

- Dtype `torch.float32`, shape `(n, n)` with `n == len(nodes)`.
- `mask[target_index, source_index] = 1.0` for **every** original
  edgelist edge; all other entries `0.0`. Same
  `(out_features, in_features)` convention as the hop masks.
- Self-loops land on the diagonal.
- Frozen exactly like `LayeredSpec.masks`.
  `MaskedLinear(spec.mask)` stores an independent frozen copy.

### Two consequences

1. **Input width is not mask width.** `align_inputs` returns
   `len(spec.input_nodes)` columns, the state vector is `n` wide.
   The inputs are scattered into the state vector via
   `spec.input_index`. In the layered case an aligned tensor
   feeds `masks[0]` directly; here it does not.
2. **Input rows are structurally zero.** Input nodes have
   in-degree 0, so their rows of `mask` are all zeros and
   `fan_in == 0`. Under the degree-aware init of `MaskedLinear`
   those rows stay at 0 forever. Writing the inputs into the
   state vector each step is therefore required, not cosmetic.

```python
spec = k2.parse_adjacency(edgelist)
core = k2.MaskedLinear(spec.mask)     # one square hop

x = k2.align_inputs(df, spec)         # width len(input_nodes)
state = torch.zeros(x.shape[0], len(spec.nodes))
state[:, spec.input_index] = x        # required, see above
state = torch.relu(core(state))       # one step; loop as needed
logits = state[:, spec.output_index]
```

---

## `MaskedLinear`

Drop-in sparse linear layer. Same job as `torch.nn.Linear`
(call as `layer(x)`); not a subclass. Not a full model.

```text
MaskedLinear(mask, bias=True)
```

- `mask`: `torch.Tensor`, shape `(out_features, in_features)`.
  Inferred `in_features` / `out_features` from `mask.shape`.
  Do not take separate size arguments.
- Register `mask` as a **non-persistent buffer** (not a
  parameter, not in `state_dict`), `float32`. LayeredSpec masks
  stay float32. After `Module.half()`, `.to(dtype=torch.bfloat16)`,
  or `.double()`, `layer.mask.dtype` is still `float32`. In-place
  writes (`fill_`, `copy_`, item assignment, `out=` into the
  mask) and replacement (`layer.mask = ...` or
  `register_buffer("mask", ...)`) raise `Kpnn2Error`. `numpy()`
  does not yield a writable view of the stored storage.
  `MaskedLinear(spec.masks[i])` stores an independent frozen
  copy. Rebuild from the edgelist / `LayeredSpec` to change
  wiring. The stored mask does not follow the module floating
  dtype. `copy.deepcopy` of a `MaskedLinear` (and of a user
  `nn.Module` that holds `MaskedLinear` layers and a
  `LayeredSpec`) succeeds. Parameters on the copy are distinct
  objects. Copied masks stay frozen float32.
- Trainable `raw_weight`: same shape as `mask`.
- Optional `bias`: shape `(out_features,)`. If `bias=False`, no bias
  parameter.
- Forward multiplies in the parameter dtype:
  `effective = raw_weight * mask.to(dtype=raw_weight.dtype,
  device=raw_weight.device)`, then
  `Y = F.linear(X, effective, bias)`.
  Equivalently `Y = X @ (W ⊙ M).T + b` with `M` cast to `W`'s
  dtype/device. This is why `.half()` / bfloat16 / `.double()`
  work like `nn.Linear`.
- **Degree-aware init:** for each output row `j`,
  `fan_in = int(mask[j].sum())` (count of 1s in that row).
  Use that `fan_in` for kaiming/uniform scale of that row (and for
  bias, use a documented rule: e.g. bias bound from that row's
  `fan_in`, or from mean fan_in; prefer **per-row fan_in** for
  weights). If `fan_in == 0`, leave that row at 0 and use bias
  bound 0.
- Do **not** use full `in_features` as `fan_in`.
- No edge constraints, no `initial_weight`, no softplus.
- Masked-out weights still exist as parameters but are multiplied
  by 0 in the forward pass.

Typical construction: `MaskedLinear(spec.masks[i])`.

---

## `SkipAdd`

Skip counterpart to `MaskedLinear`. Indexes `spec.skips`; does
not change masks.

```text
SkipAdd(spec)
```

- One learnable scalar per `spec.skips`, initialized to `0`.
- No skip bias.
- `forward(hidden, saved, target_layer)` returns `hidden` plus
  every skip whose `target_layer` matches.
- `saved` maps layer index → tensor with width
  `len(spec.layer_nodes[layer])`. Entries are only read.
- Empty `spec.skips` is identity.
- Construct once; call after every hop; no-op if nothing
  targets that layer.
- The add uses `hidden.dtype` / `hidden.device` (skip weight
  and source are cast like `MaskedLinear` casts the mask).
- `copy.deepcopy` succeeds. Parameters on the copy are
  distinct. The stored `LayeredSpec` stays frozen (do not
  mutate `spec.skips` or masks).
- Public failures: `Kpnn2Error` (bad spec, invalid
  `target_layer`, missing saved layer, width mismatch).

---

## `align_inputs(data, spec)`

`spec` is a `LayeredSpec` **or** an `AdjacencySpec`. Only
`spec.input_nodes` is read, so the DataFrame rules below are
identical for both. Anything else raises `Kpnn2Error`.

Returns `torch.float32` tensor of shape
`(n_samples, len(spec.input_nodes))`.

**Width differs by layout.** For a `LayeredSpec` that width is
`layer_dims[0]`, so the tensor feeds `MaskedLinear(spec.masks[0])`
directly. For an `AdjacencySpec` it is **not** the mask width:
`mask` is `(n, n)` over every node, while the aligned tensor is
only `len(input_nodes)` wide. Scatter it into the `n`-wide state
vector via `spec.input_index` before calling
`MaskedLinear(spec.mask)`:

```python
x = k2.align_inputs(df, spec)
state = torch.zeros(x.shape[0], len(spec.nodes))
state[:, spec.input_index] = x
```

**`pandas.DataFrame`:**

- Required columns: `spec.input_nodes` (any order).
- Match column labels after converting them to strings (so integer
  column names can match string node ids).
- Extra columns are ignored.
- Missing required columns: `Kpnn2Error`.
- Duplicate column names (including after string conversion):
  `Kpnn2Error`. The message names the unique duplicated labels
  after `str(...)`, sorted, comma-separated.
- Required columns must be numeric; non-numeric: `Kpnn2Error`.
- Reorder columns to `spec.input_nodes`.

**`torch.Tensor`:**

- Illegal. Raise `Kpnn2Error`. The message must say that `data`
  is a tensor (or that a tensor is not accepted) and that a
  pandas DataFrame is required.
- Do not check width / ndim as a substitute for alignment.
- Do not return a cast tensor.
- Pre-ordered tensors go **straight to the model**. Users who
  need alignment pass a DataFrame.

**Not supported in v1:** AnnData, numpy arrays, dicts of columns.

---

## `map_node_attributions(attributions, spec, layer=None, *, dims=None, coords=None)`

Unopinionated name mapping. No Captum import. Returns
`xarray.DataArray`. Does not aggregate.

`spec` is a `LayeredSpec` **or** an `AdjacencySpec`, and `layer` is
optional. The four combinations are exhaustive:

| Spec | `layer` | Result |
|------|---------|--------|
| `LayeredSpec` | `int` | Names from `layer_nodes[layer]`; scalar `layer` coordinate attached |
| `LayeredSpec` | omitted | `Kpnn2Error`: `layer` is required |
| `AdjacencySpec` | omitted | Names from `spec.nodes`; **no** `layer` coordinate |
| `AdjacencySpec` | `int` | `Kpnn2Error`: `layer` does not apply |

An `AdjacencySpec` has no depths, so there is no layer index to
report and none is invented. Do not fabricate `layer=0` for it.

- `attributions`: `torch.Tensor`, or a non-empty tuple/list of
  equal-shaped tensors (stacked on a new `step` axis).
- `layer`: `int` index into `spec.layer_nodes` (0-based), stored as
  scalar coordinate `layer`. `LayeredSpec` only.
- The `node` axis length must equal the number of named units:
  `len(spec.layer_nodes[layer])` for a `LayeredSpec`,
  `len(spec.nodes)` for an `AdjacencySpec`. That axis gets those
  names as its coordinate, in order.
- Default dims: 1-D → `(node,)`; 2-D → `(observation, node)`; a
  stacked sequence of 2-D tensors → `(step, observation, node)`.
  Rank 3+ (except that stacked default) requires `dims=` containing
  `node` exactly once.
- `coords`: optional labels for axes other than `node` and `layer`.
- Values: detached CPU copy of the tensor. No abs/sum/mean.
- Long table: `da.to_dataframe(name="score").reset_index()`.
  Wide 2-D table: `da.to_pandas()`.
- Invalid `spec`, `layer`, shape, `dims`, or `coords`: `Kpnn2Error`.

The user obtains `attributions` however they like (Captum
LayerConductance, IntegratedGradients, custom grads, etc.). This
function only attaches spec names to the `node` axis. Pass
`layer=i+1` for `MaskedLinear(spec.masks[i])`. Do not name-map
BatchNorm or other unnamed modules.

For a recurrent net on an `AdjacencySpec` there is no layer to
index; the natural extra axis is `step`. Pass one tensor per time
step as a sequence and they stack onto `(step, observation, node)`.

---

## Internal unit layout

`src/kpnn2/_layout.py` owns every mapping from a node name to a
position on a tensor axis. Nothing else in `src/` computes a
column index by hand.

- A node owns a **contiguous slice** of units (`NodeSlot`), not a
  single column. `DEFAULT_NODE_WIDTH` is `1`, so every slice has
  width 1, `slot.start` is the node's column index, and every
  shape documented above is unchanged.
- A `Layout` places the nodes of one axis in order without gaps.
  `layout.n_units` is that axis length. `layer_dims[i]` and the
  square mask size come from `n_units`, not from `len(names)`.
- Masks are written with `fill_block`, which sets the whole
  `(target.width, source.width)` block of an edge. At width 1
  that is one entry per edge.
- `Skip.source_index` and `Skip.target_index` store a **block
  start**. `SkipAdd` turns them back into slots with
  `Layout.slot_at` and indexes with `slot.units`.
- `align_inputs` passes its ordered columns through
  `expand_columns`, a no-op at width 1.
- `map_node_attributions` takes the node-axis length from
  `layout.n_units` and the coordinate from `layout.unit_names()`.

This is a **hedge, not a feature.** There is no public node width
and no way for a user to request one; do not add either unless a
later prompt asks. The point is that adding node width later
means handing `build_layout` real widths, instead of rewriting
index arithmetic in five modules at once. Keep it that way: new
code asks a `Layout` for a slot instead of using `list.index()`
or `enumerate` positions.

---

## Errors

All user-facing failures from the public API raise `Kpnn2Error`.
Do not leak raw `ValueError` / `KeyError` for contract violations
at the public boundary (internal helpers may use them if wrapped).

---

## Typical workflow

```python
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

import kpnn2 as k2

edgelist = pd.DataFrame(
    {
        "source": ["A", "H"],
        "target": ["H", "C"],
    }
)
# add a skip with another row A -> C when needed

spec = k2.parse_layered(edgelist)

class Net(nn.Module):
    def __init__(self, spec: k2.LayeredSpec):
        super().__init__()
        self.lin0 = k2.MaskedLinear(spec.masks[0])
        self.lin1 = k2.MaskedLinear(spec.masks[1])
        self.skips = k2.SkipAdd(spec)
        self.spec = spec

    def forward(self, x):
        saved = {0: x}
        h = F.relu(
            self.skips(
                self.lin0(x),
                saved,
                target_layer=1,
            )
        )
        saved[1] = h
        c = self.skips(
            self.lin1(h),
            saved,
            target_layer=2,
        )
        return c

model = Net(spec)
x_df = pd.DataFrame({"A": [0.1, 0.2]})
x = k2.align_inputs(x_df, spec)
y = model(x)

# optional: user ran some attribution method themselves
da = k2.map_node_attributions(
    attributions=y.detach(),
    spec=spec,
    layer=len(spec.layer_nodes) - 1,
)
```

`SkipAdd` indexes each matching skip onto the correct unit.
Call it after `MaskedLinear` and before the hop's nonlinearity
(last hop may stay linear).

---

## Typical `nn.Module` shape

There is no graph compiler and no ready-made model. Write ordinary
PyTorch:

1. `spec = k2.parse_layered(edgelist)`
2. `self.layer_i = k2.MaskedLinear(spec.masks[i])` for each hop
3. `self.skips = k2.SkipAdd(spec)` once
4. Put ReLU / BatchNorm / Dropout in `forward()` yourself.
   Call `SkipAdd` after each hop's `MaskedLinear` and before
   that hop's nonlinearity.
5. `x = k2.align_inputs(df, spec)`
6. Run Captum (or another method) yourself; then
   `map_node_attributions(...)`

Do not add a compiled core or mutate connectivity after parse.
`copy.deepcopy` of this module shape succeeds. Copied masks
stay frozen float32.

The Python distribution and import name are **`kpnn2`**.
Do not rename them.

---

## Repository layout

```
src/kpnn2/
  __init__.py                 # public exports only
  _parse.py                   # parse_layered
  _parse_adjacency.py         # parse_adjacency
  _spec.py                    # LayeredSpec, Skip
  _adjacency_spec.py          # AdjacencySpec
  _masked_linear.py           # MaskedLinear
  _skip_add.py                # SkipAdd
  _align.py                   # align_inputs
  _attributions.py            # map_node_attributions
  _errors.py                  # Kpnn2Error
  _frozen_mask.py             # read-only connectivity tensors
  _layout.py                  # node name -> units on an axis

tests/
  api/                        # public import surface
  module/                     # unit tests per primitive

CONTEXT.md                    # this file
README.md
docs/
  fig_gen/                    # SVG generators; write to figures/
  figures/
```

`src/kpnn2/__init__.py` is the **only** public import path. Users
write `import kpnn2` or `from kpnn2 import MaskedLinear`, never
`from kpnn2._masked_linear import MaskedLinear`.

Every implementation module carries a leading underscore and is
private. Private modules may be renamed, split, merged, or promoted
to subpackages (for example `_parse.py` → `_parse/`) without a
breaking change, as long as `kpnn2.__all__` and the documented
signatures stay identical. Do not add a second public path for a
symbol that `__init__.py` already exports.

`docs/api.md` therefore points mkdocstrings at the façade paths
(`::: kpnn2.MaskedLinear`), not at module paths. Tests may import
private modules directly to reach internal helpers.

---

## Dependencies

**Core:** `torch`, `pandas`, `numpy`, `xarray`, Python `>=3.10`.
On Python 3.11+, `xarray>=2026.4,<2026.8`. On Python 3.10,
`xarray>=2024.11,<2025.7` (last xarray line that supports 3.10).

**Not required:** captum, anndata.

Randomness in tests: `random.seed(42)`, `numpy.random.seed(42)`,
`torch.manual_seed(42)` whenever RNG is used.

Code style: lines `<= 80` characters. Function definitions and
calls with 2+ arguments put each argument on its own line and the
closing `)` on its own line. Format Python with
`python -m ruff` from the `dev` extra (exact pin in
`pyproject.toml`). Do not use a global `ruff` on `PATH`; it can
disagree with CI.

---

## Guidance for AI assistants

- CONTEXT.md is the contract. If a later prompt disagrees, this
  file wins after it exists.
- Do not rename the package, import, or `src/kpnn2/` directory.
- Do not reintroduce compilers, backends, pseudo nodes, Captum
  adapters, edge constraints, or AnnData in v1.
- Do not add a high-level `LayeredNet` / convenience model unless
  a later prompt explicitly asks.
- `parse_layered` and `parse_adjacency` must not instantiate
  `nn.Module`.
- Keep the two parsers separate: no `layout=` flag, no dispatch
  on whether the graph has a cycle, and no `AdjacencySpec`
  faked as a one-layer `LayeredSpec`.
- `MaskedLinear` must not store other layers' activations or
  implement skip routing. `SkipAdd` owns skip indexing; the
  user owns call order and nonlinearities.
- One graph node is one unit in v1 (no public node width). Keep
  index arithmetic in `_layout.py`: build a `Layout`, ask it for
  slots, and write masks with `fill_block`. See "Internal unit
  layout".
- Public failures: `Kpnn2Error` only.
- After Python edits, run `python -m ruff format .` from the
  `dev` extra. Do not use a global `ruff` on `PATH`.
