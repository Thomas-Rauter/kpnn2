# kpnn2 — implementation contract

This file is the **product contract**: purpose, architecture,
locked decisions, public API, and primitive specs. It is more
detailed than `README.md`.

How to work in this repository lives in `AGENTS.md`. Read
that file first. Open this file when the task needs the
contract (see `AGENTS.md`).

Do not reintroduce a graph compiler or a ready-made model object.
Do not rename the distribution, import, or `src/` package away
from `kpnn2`.
Do not pin a package version number in this file; it lives in
`pyproject.toml` and `kpnn2.__version__`.

---

## One-sentence summary

Turn a named edgelist into sparsely connected PyTorch layers
you assemble yourself.

---

## Locked contrasts

"Sparse" means two different things. Do not collapse them.

| Axis | Meaning | Owner | Lock |
|------|---------|--------|------|
| Graph / network | Which edges exist | kpnn2 | Always dense compute |
| Data / X | Feature-matrix storage | User | Host may stay sparse; minibatches are dense |

**Graph / network.** Connectivity is sparse in the edgelist.
Hop packed indices and `AdjacencySpec` store O(edges).
`MaskedLinear` stays dense float32 tensors times `F.linear`.
`PackedLinear` is a 1-D dense `weight` of length `nnz` plus
`index_add` on ordinary dense tensors.
`PackedMultiheadAttention` scores only live edgelist pairs.
Not `torch.sparse`, not COO/CSR storage, not sparse mm.
Sparse-tensor acceleration is **not planned**, now or later.
Do not add it for biology, RAM, or a convenience layer.

**Data / X.** Host feature storage, row-block slicing, and
device copies are the caller's (a training loop, or a
convenience package built on these primitives). kpnn2 has
no minibatcher and no device policy. If the host matrix is
sparse, it stays sparse for the whole call. Each step: take
the next row block → densify **that block** → move that
dense tensor to the device → dense forward. If X is already
a dense DataFrame, host stays dense; only the batch is
copied to the device. Never: the full feature matrix on
GPU/TPU, sparse minibatches, sparse kernels, or a silent
full densify of the host matrix.

**Module boundary.** `MaskedLinear`, `PackedLinear`,
`PackedMultiheadAttention`, `gather_hop_inputs`,
`scatter_hop_outputs`, and `map_node_attributions` take
and return ordinary dense `torch.Tensor`s. Captum is not
in this package; when the caller runs it, that call stays
dense (no sparse IG here, and none to add).

**`align_inputs` is the feature-axis index.** It maps a sequence
of feature names onto `spec.input_nodes` as a 1-D `int64`
index. It does not take the matrix, copy sample rows, or
densify. Apply the index on the caller's storage (`X[:, col]`
for numpy, scipy CSR/CSC, AnnData `.X` in those formats, or
a tensor; `df.to_numpy()[:, col]` for a DataFrame). COO-style
sparse layouts need a format that supports integer column
indexing. Width greater than 1 repeats indices and copies
those columns on CSR/CSC. Pre-ordered dense tensors already
skip `align_inputs`. A caller with sparse host X applies the
index on the sparse layout, then densifies only each row
block.

Do not add AnnData, scipy sparse, a densifying `align_inputs`,
minibatching, or device-copy helpers to this package unless a
later prompt asks.

---

## What this package does

1. **Parse:** `parse_layered()` reads a pandas DataFrame with columns
   `source` and `target` only, validates a layered DAG, and returns a
   `LayeredSpec`. Optional keyword-only `widths=` gives a named node
   several units (DCell-style); omitted or `None` is width 1.
   Optional keyword-only `ranks=` assigns compact depths so official
   ontology levels (P-NET / Reactome) need not be longest-path hops;
   omitted or `None` is longest-path, bit-identical to today's
   default. `parse_adjacency()` reads the same table and
   returns an `AdjacencySpec` instead: packed source/target indices
   over every node, cycles and self-loops allowed. It never
   allocates an `(n, n)` tensor and has no `widths=` or `ranks=`
   argument. The user picks the layout; a DAG is valid input to both.
2. **Specify:** `LayeredSpec` holds named nodes by layer, per-node
   widths, and one `Hop` per layer after the first. A hop's packed
   indices are in **unit** space: a named edge `A→B` expands into
   a `(k_B × k_A)` block of live pairs, skips included. There is no
   stored mask; `Hop.to_mask()` allocates. `AdjacencySpec` holds
   all node names, packed edge indices, and the input/output
   positions in the state vector. Neither constructs an
   `nn.Module`. Packed adjacency indices remain one pair per
   named edge.
3. **Build:** The user writes a PyTorch `nn.Module` using one
   `PackedLinear(hop.source_index, hop.target_index,
   hop.out_features, hop.in_features)` per hop,
   `gather_hop_inputs(saved, hop)` to assemble that hop's input,
   and their own activations, norms, loops, and heads.
   `PackedLinear.transpose()` is the tied decoder map (same
   packed slots, shared or copied `weight`, untied bias).
   `scatter_hop_outputs` splits a transposed hop's concatenated
   output back onto source layers.
   `MaskedLinear(hop.to_mask())` densifies and remains valid for
   small graphs. On an `AdjacencySpec` the large-n path is
   `PackedLinear(spec.source_index, spec.target_index, n, n)`
   with `n = len(spec.nodes)`; that never allocates `(n, n)`.
   The same packed indices can feed
   `PackedMultiheadAttention`. `MaskedLinear(spec.to_mask())`
   densifies and remains valid for small graphs. The
   update (loop, attention, head) is the user's `forward()`.
4. **Align:** `align_inputs()` maps feature names onto
   `spec.input_nodes` as a 1-D `int64` column index. Apply
   it on the host matrix (`X[:, col]`). Pre-ordered dense
   tensors go straight to the model. Sparse host matrices
   stay sparse until each row block is densified; see
   **Locked contrasts**.
5. **Train:** The user owns loss, optimizer, and the training loop.
6. **Map attributions:** `map_node_attributions()` labels a tensor
   at one LayeredSpec layer (`layer=`, or `hop_output=` for what
   a hop's module returns), or the concatenated source axis of
   one hop (`hop_input=`), as an `xarray.DataArray`. The hop
   side is always stated. Captum is not a library
   dependency; the user runs Captum (or any other method)
   themselves. `xarray` is a core dependency used for that
   mapping and for `aggregate_node_attributions()`.
7. **Aggregate attributions (optional):**
   `aggregate_node_attributions()` folds named scores with a
   registered method. The default `rauter_mangano_2026` is
   registered and not yet implemented; calling it raises
   `Kpnn2Error`. `list_aggregation_methods()` lists the
   registry. The mapper still does not aggregate.

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
- **Not a ready-made Transformer.** `PackedMultiheadAttention`
  is a contraction primitive on live edgelist pairs. It is
  not an encoder, not a Transformer stack, and not a reason
  to add `parse_attention`. Do not reuse a hop rectangle as a
  square attention matrix.
- **Not a ready-made autoencoder.** `PackedLinear.transpose`
  and `scatter_hop_outputs` are primitives. The user owns
  `forward()`, activations, and whether weights are tied.
  Do not add a `TiedAutoencoder` class or a reverse parser.
- **Not a trainer.** No losses, optimizers, or training loops.
- **Not a per-edge policy DSL.** Mixed sign constraints
  (activation vs inhibition) and frozen live-edge values
  occur in real KPNNs (for example LEMBAS / OmniPath MOA).
  kpnn2 does not own them and does not block them. They are
  user convenience on top of the spec, not a missing
  primitive: keep extra columns on the caller's DataFrame,
  parse `source`/`target` only, locate slots with
  `edge_location`, then apply ordinary PyTorch. A hard
  freeze is a `constraint=` module that `torch.where`-
  replaces those slots in forward. That is what holds the
  live-edge value: AdamW's decoupled weight decay and SGD
  with momentum still update the stored `nn.Parameter`,
  but the forward map overwrites the slot every step. A
  loss barrier is a soft prior, not a hold. A gradient
  hook that zeroes the slot is not a freeze: those
  optimizers still move it, so a "frozen" prior silently
  drifts. `constraint=` stays one `nn.Module` over the
  whole weight tensor. `PackedLinear.weight` is one
  `nn.Parameter`; per-slot `requires_grad=False` is not a
  PyTorch operation. First-class `sign` / `freeze` /
  `initial_weight` columns would change parse,
  `to_edgelist()`, fingerprints, and spec fields for a
  workflow the caller can already run. Do not add those
  columns, a sign tensor on the spec, or a second
  constrained layer class. See **Package philosophy**.
- **Not a mutable graph.** Topology is frozen after parse.
  `LayeredSpec` and `AdjacencySpec` are frozen dataclasses
  with no edge add/remove methods. `PackedLinear` index
  buffers have no setter. Assigning `MaskedLinear.mask`
  can prune in place, but then `mask_digest` will not
  match a model rebuilt from the original spec. In-place
  ParsVNN / self-pruning BINN pruning and PathExpSurv edge
  growth are not a supported public contract. Those
  workflows are rare special cases. First-class support
  would re-pack hops on both layouts, resize packed
  `weight`, remap optimizer slots, and rewrite
  fingerprints and digests on every edge change. That
  would make the code much more complex and harder to
  maintain. Callers who need a different prior drop or
  add DataFrame rows, parse again, and copy surviving
  tensors **by name** (see **Reparse hatch**). Do not add
  `add_edge` / `remove_edge`, PackedLinear index setters,
  a transfer helper, or an in-place prune / grow API.
- **Not a data-residency layer.** No minibatcher, no device
  policy, no "keep X sparse" helper, and no rule that moves
  the full feature matrix to GPU. Host-sparse storage (for
  example AnnData `.X` as scipy CSR) is valid in the
  **caller's** loop; this package does not implement that
  path. See **Locked contrasts**.
- **Not Captum.** No Captum import anywhere in the library. Attribution
  mapping is name alignment only.
- **Not AnnData (v1).** No `anndata` type in `align_inputs`.
  Pass `adata.var_names` (or any name sequence); apply the
  index on `.X` in caller code. Do not add AnnData here.
- **Not a time machine.** `parse_adjacency` accepts cycles and
  self-loops, but nothing here unrolls time, picks a step count, or
  re-injects inputs between steps. The loop is user `forward()`
  over `PackedLinear(...)` or `MaskedLinear(spec.to_mask())`.
  Do not add `MaskedRNN` / `MaskedGRU` / `MaskedLSTM`.
  `parse_layered` stays DAG-only and still raises `Kpnn2Error`
  on a cycle.
- **Not pseudo-node expansion.** A skip edge is a packed pair of
  its target's hop, never a dummy neuron and never an extra
  channel inserted into an intermediate layer.
- **Not sparse-tensor accelerated.** This is the **graph**
  axis, not a ban on sparse feature matrices in the caller's
  host RAM. Connectivity is sparse in the graph.
  `AdjacencySpec` and each `Hop` store O(edges) index tuples;
  `MaskedLinear` stays dense (`parametrize` + `F.linear`).
  `PackedLinear` is a 1-D dense weight of length `nnz` plus
  `index_add` on ordinary dense tensors; it is not
  `torch.sparse`, COO/CSR, or sparse matmul.
  `PackedMultiheadAttention` scores only live pairs; it is
  not `torch.sparse` and never allocates `(n, n)`. Sparse
  tensor formats and sparse mm are **not planned**. Do not
  fold packed into `MaskedLinear`. Do not fold attention
  into `PackedLinear` or `MaskedLinear`. `PackedLinear` is
  the large-n path on hops and on an `AdjacencySpec`.
  `MaskedLinear(hop.to_mask())` / `MaskedLinear(spec.to_mask())`
  is the dense GEMM hatch when the rectangle fits.
  Correctness, ease of maintenance, and
  explainability of the code outrank memory and speed.
  Host-sparse scipy CSR in a caller loop does not violate
  this bullet.

---

## Package philosophy

**Primitives, not a compiled container.** `kpnn2` owns edgelist
parsing, packed hop and adjacency indices, hop input assembly,
hop-output split, packed transpose, named I/O alignment, and
attribution column names. The user owns `nn.Module.forward()`,
call order, nonlinearities, and training. There is no
ready-made autoencoder class. Do not add a high-level
`LayeredNet` / convenience model unless a later prompt
explicitly asks.

**Per-edge signs and frozen values stay with the caller.**
kpnn2 owns which edges exist and where their packed slots
are. It does not own mixed per-edge signs, freeze flags, or
initial numerical values. Those appear in published KPNNs
and are not blocked: the user keeps them on their own table
and implements them with standard PyTorch together with
`constraint=` and `edge_location`. A hard freeze is
`torch.where` inside that `constraint=` module. A loss
barrier is a soft prior. Do not treat a gradient hook as a
freeze. Shipping that as parse columns or spec fields would
be convenience, not a necessity, and would make the public
contract heavier. Do not add it. See **What kpnn2 is NOT**.

**Topology is frozen after parse.** Specs are parse snapshots,
not a live graph. Training-time prune and grow (ParsVNN,
self-pruning BINN, PathExpSurv) stay in the caller's loop:
edit the edgelist, parse again, copy by name (see
**Reparse hatch**). Do not add mutation APIs to make those
papers first-class. They are rare relative to a fixed prior,
and supporting them in kpnn2 would make the code much more
complex and harder to maintain. See **What kpnn2 is NOT**.

**Two sparsity axes.** Graph connectivity is kpnn2's: always
dense compute, sparse only as "which edges exist." Feature
matrix X is the user's: sparse host storage is allowed in
**their** code; tensors that enter `forward()` are always
dense. See **Locked contrasts**. Do not add `torch.sparse`
kernels to "support sparse X." Pass feature names to
`align_inputs` and apply the index on host X; do not densify
sparse X to feed that function.

**Correctness over speed.** `MaskedLinear` stays dense float32
tensors times `F.linear`, not `torch.sparse` layouts.
`PackedLinear` is a 1-D weight plus `index_add`
(ordinary dense tensors, length `nnz`); it never uses
`torch.sparse` / COO / CSR / sparse mm and never densifies
inside the module. `Hop` and `AdjacencySpec` store O(edges);
`to_mask()` is the allocating dense escape hatch on both.
`PackedLinear` is the large-n path. `MaskedLinear` is the
dense GEMM hatch when the rectangle fits. Do not fold packed
into `MaskedLinear`. `PackedMultiheadAttention` is the same
kind of packed primitive for attention; it must not be
folded into `PackedLinear` or `MaskedLinear`.
Sparse-tensor acceleration is **not planned**, now or later.
This package values correctness, ease of maintenance, and
explainability of the code more than memory and speed
performance.

Division of labor:

| Layer | Owner |
|-------|--------|
| Edgelist → `LayeredSpec` (ranks, packed hops, skip metadata) | kpnn2 |
| Edgelist → `AdjacencySpec` (nodes, packed edge indices) | kpnn2 |
| `MaskedLinear` (fixed mask, dense GEMM) | kpnn2 |
| `PackedLinear` (1-D weight per live edge, `index_add`) | kpnn2 |
| `PackedMultiheadAttention` (packed MHA on live pairs) | kpnn2 |
| `gather_hop_inputs` (source axis of one hop) | kpnn2 |
| `scatter_hop_outputs` (split that axis onto source layers) | kpnn2 |
| `PackedLinear.transpose` (tied packed `W.T`) | kpnn2 |
| Named node → unit slice (`node_units` / `hop_units`) | kpnn2 |
| Feature names → column index (`align_inputs`) | kpnn2 |
| Apply that index on host X (`X[:, col]`) | User |
| `forward()`, activations, norms, heads, call order | User (PyTorch) |
| Encoder stack, FFN, residuals | User (PyTorch) |
| Training and evaluation | User (PyTorch) |
| Training-time prune / grow of the prior | User (reparse hatch) |
| Host feature layout (dense table vs sparse AnnData `.X`) | User |
| Minibatch slice → densify that block → device copy | User |
| Captum / other attribution algorithms | User |
| Tensor → named `xarray.DataArray` | kpnn2 |
| Named scores → per-node aggregate | kpnn2 |
| Sparse-tensor kernels (`torch.sparse`, sparse mm) | Not planned |

---

## Public API (only these names)

Exported from `kpnn2` (`src/kpnn2/__init__.py`):

| Symbol | Role |
|--------|------|
| `parse_layered` | Edgelist DataFrame → `LayeredSpec` (DAG only; optional `widths=`, `ranks=`) |
| `parse_adjacency` | Edgelist DataFrame → `AdjacencySpec` (packed layout; cycles allowed) |
| `LayeredSpec` | Frozen structural dataclass (layers, packed hops, skip metadata) |
| `Hop` | One layer's incoming packed edges; exported because `spec.hops` uses it |
| `Skip` | One skip-edge record (see below); exported because `spec.skips` uses it |
| `AdjacencySpec` | Frozen structural dataclass (nodes, packed edge indices; no stored square) |
| `MaskedLinear` | `nn.Module`: masked linear layer |
| `PackedLinear` | `nn.Module`: one trainable scalar per live edge |
| `PackedMultiheadAttention` | `nn.Module`: packed multi-head attention on live edgelist pairs |
| `gather_hop_inputs` | Saved layer tensors + `Hop` → that hop's input tensor |
| `scatter_hop_outputs` | Concatenated hop axis → per-source-layer tensors |
| `align_inputs` | Feature names → `int64` column index |
| `map_node_attributions` | Layer tensor → labeled `xarray.DataArray` |
| `aggregate_node_attributions` | Named scores → per-node `xarray.Dataset` (method registry) |
| `list_aggregation_methods` | Registry table of aggregation methods |
| `Kpnn2Error` | User-facing error type |
| `__version__` | Package version string |

`__all__` contains exactly these names (including `Hop`, `Skip`,
and `__version__`), in the order of the table above.
`tests/api/test_public_api.py` compares it as an ordered list. No
other public symbols.

Do **not** export or implement: `compile_graph`, `customize_model`,
`interpret_model`, `align_features_to_input_nodes`, `edge_weights`,
`CompileArtifact`, backends, `ConstrainedMaskedLinear`, `SkipAdd`,
`SparseMaskedLinear`.

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
and must not change parsing. There is no `initial_weight` column,
no `sign` / `freeze` column, and no edgelist `constraint`
column. A uniform per-entry map belongs on `MaskedLinear` /
`PackedLinear` as `constraint=` (one `nn.Module` over the
whole tensor). Mixed signs and frozen live-edge values are
the caller's: keep those columns on the DataFrame, parse
connectivity only, locate slots with `edge_location`, and
`torch.where`-replace frozen slots inside `constraint=`.
A loss barrier is a soft prior; a gradient hook is not a
freeze. First-class support is convenience, not a
necessity; do not add those columns.

A spec returns this two-column form from `to_edgelist()`. Rows
are sorted lexicographically by `(source, target)`, not the
original parse input order. Extra columns from the DataFrame
that was parsed are not reproduced.

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
  Stored in `spec.input_nodes`. These are also the feature-axis
  order `align_inputs` indexes into, and the column order of
  `MaskedLinear` on hop 0.
- **Output nodes:** out-degree 0. Sorted alphabetically.
  Stored in `spec.output_nodes`. Early outputs (terminals whose
  depth is not the maximum depth) **are allowed**. They remain in
  their own layer; they are not moved, padded, or rejected.
- **Hidden nodes:** every other named node (not input, not output).
  Sorted alphabetically. Stored in `spec.hidden_nodes`.

**Layering (Kahn / longest-path from inputs, or `ranks=`):**

- Default (`ranks=None` or omitted): process nodes in topological
  order. A node becomes ready when all parents are assigned a
  depth.
- `depth(input) = 0`.
- `depth(node) = 1 + max(depth(parent) for parent in parents)`.
  Equivalently: the depth assigned when in-degree hits 0 in a Kahn
  sweep that increments depth each frontier.
- `layer_nodes[d]` is the alphabetically sorted list of nodes with
  `depth == d`.
- Layer 0 is the first layer (all depth-0 nodes, i.e. all inputs).
  If a graph somehow had a depth-0 non-input, that would violate
  in-degree 0 ⇔ input; do not invent extra depth-0 nodes.

Optional `parse_layered(..., ranks=)` replaces longest-path with
user depths. Keys are matched after `str(...)`, same as `widths`.
Every graph node must be present; unknown names raise
`Kpnn2Error`; unique names sorted, comma-separated. Values are
non-negative ints; reject `bool`, negatives, and non-ints.
Let `used` be the sorted unique rank values. Node layer index is
`used.index(rank[node])`, so layers are `0 .. L-1` with **no
empty layers**. User numbers need not be 0-based or consecutive
(0/10/10/20 becomes layers 0, 1, 2). All in-degree-0 nodes share
the minimum user rank, and no non-input sits at that minimum;
after compacting they are exactly `layer_nodes[0]`. Every named
edge is strictly forward: compacted(source) < compacted(target).
Same-rank edges are illegal. A node at layer `d>0` may have only
skip parents (no parent at `d-1`). Dummy depth-padding nodes are
not an acceptable workaround; skips must not become
pseudo-nodes.

`ranks=None` is bit-identical to longest-path: same `layer_nodes`,
hops, `to_dict` keys, fingerprint. `widths=` still applies after
ranking. Do **not** add `ranks=` to `parse_adjacency`.

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

### `parse_layered(..., widths=, ranks=)`

```python
parse_layered(
    edgelist: pd.DataFrame,
    *,
    widths: Mapping[str, int] | None = None,
    ranks: Mapping[str, int] | None = None,
) -> LayeredSpec
```

`widths=None` or omitted: every node is 1. Missing names in the
mapping default to 1. Unknown names (after `str(...)`) raise
`Kpnn2Error`, unique names sorted, comma-separated. Values must
be positive ints. Reject `bool` (`bool` is an `int`). Reject 0
and negatives. `Kpnn2Error`.

`ranks=None` or omitted: longest-path, unchanged. When provided,
every graph node must have a non-negative int; see **Layering**.

Do **not** add `widths=` or `ranks=` to `parse_adjacency`.

---

## `LayeredSpec` fields

`LayeredSpec` is a frozen dataclass. It holds structure only: no
`nn.Module`, no parameters, no execution plan object. Sequences
are tuples. Do not reassign fields. There are no edge
add/remove methods; a different prior is a new parse
(see **What kpnn2 is NOT**). There is no stored mask
tensor and no densifying `mask` property on a hop.
`copy.deepcopy` of a `LayeredSpec` succeeds and copies the
index tuples; two `Hop.to_mask()` results do not share storage.

| Field | Type | Meaning |
|-------|------|---------|
| `input_nodes` | `tuple[str, ...]` | In-degree 0 nodes, alphabetical. Tensor column order. |
| `output_nodes` | `tuple[str, ...]` | Out-degree 0 nodes, alphabetical. |
| `hidden_nodes` | `tuple[str, ...]` | Neither input nor output, alphabetical. |
| `layer_nodes` | `tuple[tuple[str, ...], ...]` | `layer_nodes[i]` = names at depth `i`, alphabetical. Index 0 is the first layer. One name per node. |
| `layer_dims` | `tuple[int, ...]` | Unit count of depth `i`: `layer_dims[i] == sum(layer_widths[i])`. Equals `len(layer_nodes[i])` only when every node at that depth has width 1. |
| `layer_widths` | `tuple[tuple[int, ...], ...]` | `layer_widths[i][j]` is the width of `layer_nodes[i][j]`. Default parse is 1 for every node. |
| `hops` | `tuple[Hop, ...]` | One incoming packed hop per layer after the first (see below). |
| `skips` | `tuple[Skip, ...]` | Skip edges with depth gap `> 1`, as metadata (see below). |

### `Hop` records

One hop per layer after the first. A hop is exactly what one
`PackedLinear` (or `MaskedLinear` after `to_mask()`) computes,
and its packed indices hold **every** parent of its target
layer. There is no second mechanism for edges that span layers.

| Field | Type | Meaning |
|-------|------|---------|
| `target_layer` | `int` | Depth this hop produces; `>= 1` |
| `source_layers` | `tuple[int, ...]` | Depths it reads, ascending, all `< target_layer` |
| `source_dims` | `tuple[int, ...]` | Units per entry of `source_layers`; sums to `in_features` |
| `source_nodes` | `tuple[str, ...]` | Node names of the concatenated source axis, **one name per node**, not per unit. When any source node has `k>1`, `len(source_nodes) != hop.in_features`. |
| `target_dim` | `int` | Units in the target layer; equal to `out_features` |
| `source_index` | `tuple[int, ...]` | Concat-column of each live **unit** pair (see below) |
| `target_index` | `tuple[int, ...]` | Target-layer row of each live **unit** pair |

`Hop.column_offsets` is a derived property: the first source
column of each entry of `source_layers`. `in_features` is
`sum(source_dims)`. `out_features` is `target_dim`.
`LayeredSpec.hop_units` locates one named node on that
concatenated axis; `column_offsets` only walks source layers.

- `len(hops) == len(layer_nodes) - 1` and
  `hops[i].target_layer == i + 1`.
- `source_layers` lists only layers that really feed the target.
  Under longest-path ranking, `target_layer - 1` is always one of
  them, because that ranking gives every node a parent one layer
  down. With `ranks=`, a hop may omit the previous layer when a
  node has only skip parents. `gather_hop_inputs` follows
  `source_layers`; it does not assume adjacency.
- `hops[0].source_layers == (0,)` always: after compact ranking,
  layer 1 can only have layer-0 parents. So a row block gathered
  with the `align_inputs` index feeds `hops[0]` directly, with
  no gathering.
- A graph with no skip edges gives every hop a single source
  layer, and then `hops[i]` is the plain adjacent hop from
  layer `i` to layer `i+1`.
- `source_index` and `target_index` address **units**. Named
  edges are canonical lexicographic by `(source name, target
  name)`; within one named edge `A→B` every unit pair of the
  `(k_B × k_A)` block is emitted, target-unit outer, source-unit
  inner, matching `fill_block`. At width 1 that is one pair per
  named edge, same order as before. A dense rectangle would have
  `1.0` at `[target_index[i], source_index[i]]`.
- **Every named edgelist edge belongs to exactly one hop**, the
  one of its target layer. Packed `nnz` for a hop is the sum over
  those named edges of `(k_source * k_target)`. Applying a hop
  applies all of its target's parents at once.
- Because a hop carries every parent, the per-row degree
  `PackedLinear` and `MaskedLinear` initialize from is the
  unit's real fan-in, skips included.
- `parse_layered` must not allocate an `(out, in)` tensor. There
  is no `mask` field and no densifying `mask` property.
  `Hop.to_mask()` is the allocating dense escape hatch: a fresh
  `(out_features, in_features)` float32 rectangle on every call.
  Mutating it does not change the hop.
- `copy.deepcopy` succeeds and copies the index tuples.
- `PackedLinear(hop.source_index, hop.target_index,
  hop.out_features, hop.in_features)` never densifies.
  `MaskedLinear(hop.to_mask())` stores an independent copy.

### `Skip` records

Each skip is a frozen dataclass `Skip`. It is **metadata only**:
the edge itself is already a block of packed unit pairs in
`hops[target_layer - 1]`, exactly like an adjacent edge.
Read `skips` to report or inspect which prior-knowledge edges
span layers; nothing in a forward pass needs it.

| Field | Type | Meaning |
|-------|------|---------|
| `source` | `str` | Source node name |
| `target` | `str` | Target node name |
| `source_layer` | `int` | Depth of `source` |
| `target_layer` | `int` | Depth of `target`; `target_layer - source_layer > 1` |
| `source_in_layer` | `int` | First unit of `source` in its layer (block start). At width 1 this equals the node's index in `layer_nodes[source_layer]`. |
| `target_in_layer` | `int` | First unit of `target` in its layer (block start). At width 1 this equals the node's index in `layer_nodes[target_layer]`. |

Every original edgelist edge with depth gap `> 1` appears once in
`skips` (one metadata record per named edge, not per unit pair).
Adjacent edges (gap `== 1`) never appear in `skips`.
Membership in `skips` changes nothing about how the edge is
computed. `LayeredSpec.edge_location(source, target)` returns
the packed slots of that named edge. Locating the skip among
packed indices means every unit pair in that block is live,
not a single `(column, row)` pair.

### `LayeredSpec.to_edgelist()`

```python
layered_spec.to_edgelist() -> pandas.DataFrame
```

Returns the original edges as a two-column table. Columns are
exactly `source` then `target`; no other columns are present.
Rows are the canonical sorted pairs: lexicographic order by
`(source, target)`, one row per original edge, names as
strings. This is **not** the original parse input order, and
extra columns from the pre-parse DataFrame are not
reproduced.

`parse_layered(spec.to_edgelist())` without `widths=` or `ranks=`
reconstructs the same `input_nodes`, `output_nodes`,
`hidden_nodes`, `layer_nodes`, `layer_dims`, hop `source_layers`
/ `source_dims` / `source_nodes`, and packed hop indices **when
every node has width 1 and ranking is longest-path**. Widths and
ranks live on the spec dict, not the edgelist:
`parse_layered(spec.to_edgelist())` is not the same spec if the
original had `k>1` or user ranks that differ from longest-path.
Round-trip those graphs with `LayeredSpec.from_dict` or
`parse_layered(..., widths=, ranks=)`.
Skip *tuple* order may
follow the sorted edgelist rather than the original parse
input; the skip *set* of `(source, target, source_layer,
target_layer, source_in_layer, target_in_layer)` matches.

### `LayeredSpec.edge_location()`

```python
layered_spec.edge_location(source, target) -> tuple[int, tuple[int, ...]]
```

Returns `(hop_index, packed_indices)`. `hop_index` is `i` such
that the named edge is in `spec.hops[i]`. `packed_indices`
indexes `hops[i].source_index` / `target_index` and the
corresponding `PackedLinear.weight`.

`source` and `target` are matched after `str(...)`, same as
parse. At width 1 the index tuple has length 1. With
`widths=`, a named edge `A→B` is the full `(k_B × k_A)`
block of live unit pairs, target-unit outer, source-unit
inner, in the order already stored. This is identity into
those packed slots, not a constraint DSL. Mixed signs and
frozen values belong in user PyTorch that indexes these
slots (`torch.where` inside `constraint=` to hold a
value), not on the spec.

Missing pair, empty names, or a name that is not a node:
`Kpnn2Error`. The message names the pair as
`{source} -> {target}`.

To copy weights onto a new prior, look up the same named
edge on **both** specs. See **Reparse hatch**.

### `LayeredSpec.node_units()` / `hop_units()`

```python
layered_spec.node_units(name) -> tuple[int, slice]
layered_spec.hop_units(hop, name) -> slice
```

`node_units` returns `(layer, units)` for one named node.
`layer` is the depth in `layer_nodes`. `units` is a
contiguous slice of that layer's last axis (`saved[layer]`),
length `k` of that node. Always a slice, including at
width 1. Index as `saved[layer][..., units]`.

`hop_units` returns the same node's slice on the concatenated
source axis of `hop` (`gather_hop_inputs` output /
hop-module input). `hop` must compare equal to one entry of
`spec.hops`. A graph node that is not a source of that hop
raises `Kpnn2Error`. Index as `sources[..., units]`.
`Hop.column_offsets` still locates a whole source **layer**
on that axis; this locates one named node.

`name` is matched after `str(...)`, same as parse. Empty
name or a name that is not a node: `Kpnn2Error`. Do **not**
export `Layout`. Do **not** add this lookup on
`AdjacencySpec` (every node is one unit of `spec.nodes`).

These are identity into the unit axis, not a dropout module
and not a head helper. After a reparse, copy bias with
`node_units` only when width and layer still match; see
**Reparse hatch**.

---

## Skip connections (no pseudo nodes, no second mechanism)

A skip `A → H2` that jumps one or more layers is the source
being an extra parent of the target. It is **not** a dummy
channel, **not** a separate module, and **not** a learnable
scalar added after the fact. It is a block of packed unit pairs
of `hops[target_layer - 1]`, and its weights are ordinary
entries of that layer's `PackedLinear` (or `MaskedLinear`).

The consequences are the point of this design:

- A hop that has skip parents reads more than one layer, so its
  input is those layers concatenated. `gather_hop_inputs` builds
  that tensor and raises `Kpnn2Error` if a needed layer is
  missing, so a forgotten activation is an error rather than a
  quietly dropped edge.
- The unit's fan-in for the degree-aware init counts skip
  parents, because they are packed pairs of the same hop.
- The skip weight is a full weight, not a tied scalar, and it
  starts from the same degree-aware draw as every other edge.
- There is no skip bias; unit bias stays on `MaskedLinear`.
- Nothing undoes ReLU and nothing modifies saved tensors. The
  source enters through the target's own weight matrix.

`kpnn2` owns the packed hop layout and the gather. The user owns
call order and nonlinearities.

```python
saved = {0: x}
for index, hop in enumerate(spec.hops):
    sources = kpnn2.gather_hop_inputs(saved, hop)   # concat, checked
    h = self.hops[index](sources)                # PackedLinear
    if hop.target_layer < len(spec.layer_nodes) - 1:
        h = torch.relu(h)
    saved[hop.target_layer] = h
```

Store every layer you produce in `saved`; a later hop may read
it. A hand-written residual add stays valid PyTorch, but it is
no longer needed to express a skip edge.

---

## `parse_adjacency` and `AdjacencySpec`

```text
parse_adjacency(edgelist) -> AdjacencySpec
```

The second layout. Every node goes into one state vector, sorted
alphabetically, and every edge goes into packed source/target
index tuples. Nothing is ranked, so **cycles and self-loops are
allowed**. This is the general packed form, not a cyclic-only
parser: the same spec is what a shared-state loop, a time-series
cell, and packed attention use. The update itself is user
`forward()` code.

`parse_adjacency` must not instantiate an `nn.Module`, unroll
time, choose a step count, or re-inject inputs between steps.
It must not allocate an `(n, n)` tensor.

### `AdjacencySpec` fields

Frozen dataclass, same rules as `LayeredSpec`: no reassignment,
sequences are tuples, no edge add/remove methods. There is no
stored mask tensor and no densifying `mask` property.
`copy.deepcopy` succeeds and copies the index tuples; two
`to_mask()` results do not share storage.

| Field | Type | Meaning |
|-------|------|---------|
| `nodes` | `tuple[str, ...]` | Every node name, alphabetical. Unit order of the state vector, and the row/column order of `to_mask()`. |
| `input_nodes` | `tuple[str, ...]` | In-degree 0 nodes, alphabetical. Column order for `align_inputs`. |
| `output_nodes` | `tuple[str, ...]` | Out-degree 0 nodes, alphabetical. |
| `hidden_nodes` | `tuple[str, ...]` | Neither input nor output, alphabetical. |
| `source_index` | `tuple[int, ...]` | For each original edge, the column in `nodes` (the source). |
| `target_index` | `tuple[int, ...]` | For each original edge, the row in `nodes` (the target). |
| `input_index` | `tuple[int, ...]` | Position of each `input_nodes` name in `nodes`. |
| `output_index` | `tuple[int, ...]` | Position of each `output_nodes` name in `nodes`. |

`source_index` and `target_index` have the same length as the
edge count. They include cycles and self-loops. Order is
canonical: lexicographic by `(source name, target name)`,
identical to `to_edgelist()` row order. A dense square would
still have `1.0` at `[target_index[i], source_index[i]]`.

There is no `mask` field, no `layer_nodes`, no `layer_dims`, no
`hops` tuple, and no `skips`. Skips are a depth concept and
depth does not exist here: in this layout every edge, however
far it would span, is already a packed index pair.
`gather_hop_inputs` does not accept an `AdjacencySpec`.

### `AdjacencySpec.to_edgelist()`

```python
adjacency_spec.to_edgelist() -> pandas.DataFrame
```

Same two-column canonical table as
`LayeredSpec.to_edgelist()`: columns exactly `source` then
`target`, rows sorted lexicographically by `(source,
target)`, including cycle edges and self-loops. Extra
columns from the pre-parse DataFrame are not reproduced.

`parse_adjacency(spec.to_edgelist())` reconstructs the same
`nodes`, `input_nodes`, `output_nodes`, `hidden_nodes`,
`source_index`, `target_index`, `input_index`, and
`output_index`.

### `AdjacencySpec.edge_location()`

```python
adjacency_spec.edge_location(source, target) -> tuple[int, ...]
```

Packed indices into `spec.source_index` / `spec.target_index`
(and into `PackedLinear.weight` built from those arrays).
Adjacency stays one pair per named edge; there are no
widths. Order agrees with `to_edgelist()` rows.

`source` and `target` are matched after `str(...)`, same as
parse. Missing pair, empty names, or a name that is not a
node: `Kpnn2Error`. The message names the pair as
`{source} -> {target}`. This is identity into packed slots,
not a constraint DSL. Mixed signs and frozen values belong
in user PyTorch that indexes these slots (`torch.where`
inside `constraint=` to hold a value), not on the spec.

To copy weights onto a new prior, look up the same named
edge on **both** specs. See **Reparse hatch**.

### `to_mask()` (allocating dense escape hatch)

```python
adjacency_spec.to_mask() -> torch.Tensor
```

The spec does **not** store a square. `to_mask()` is the only
way to build the old dense mask so
`MaskedLinear(spec.to_mask())` still works. The large-n path
is `PackedLinear` on the packed indices and never allocates
`(n, n)`.

- Every call allocates a new dense `float32` tensor of shape
  `(n, n)` with `n` from the node layout (`len(nodes)` at
  width 1).
- The tensor starts at zeros; then `1.0` is written at each
  `[target_index[i], source_index[i]]`.
- Mutating the result does not change the spec or the next
  `to_mask()` call.
- Self-loops land on the diagonal.
- `MaskedLinear(spec.to_mask())` stores an independent copy.

Do not add a `mask` field or a `@property mask` that densifies
on access: that would silently allocate.

### Two consequences

1. **Input width is not state width.** `align_inputs` returns
   `len(spec.input_nodes)` positions, the state vector is `n`
   wide. Gathered input columns are scattered into the state
   vector via `spec.input_index`. In the layered case an
   aligned row block feeds `hops[0]` directly; here it does
   not.
2. **Input rows are structurally zero.** Input nodes have
   in-degree 0, so they have no packed incoming edges, their
   rows of `to_mask()` are all zeros, and `fan_in == 0`.
   Under the degree-aware init of `PackedLinear` and
   `MaskedLinear` those units stay at 0 forever. Writing
   the inputs into the state vector each step is therefore
   required, not cosmetic.

```python
spec = kpnn2.parse_adjacency(edgelist)
n = len(spec.nodes)
core = kpnn2.PackedLinear(
    spec.source_index,
    spec.target_index,
    n,
    n,
)
# MaskedLinear(spec.to_mask()) densifies; valid for small graphs

col = kpnn2.align_inputs(df.columns, spec)
x = torch.as_tensor(
    df.to_numpy()[:, col],
    dtype=torch.float32,
)
# x is len(input_nodes) wide
state = torch.zeros(x.shape[0], n)
state[:, spec.input_index] = x        # required, see above
state = torch.relu(core(state))       # one step; loop as needed
logits = state[:, spec.output_index]
```

---

## Reparse hatch

Topology stays frozen after parse. To prune or grow the
prior, edit the edgelist DataFrame, call `parse_layered` or
`parse_adjacency` again, build **new** `PackedLinear` or
`MaskedLinear` modules from the new spec, and copy surviving
tensors **by name**. Do not add `add_edge` / `remove_edge`,
a transfer helper, or an in-place prune / grow API.

**Name-to-name copy.** For each named edge that exists on
both specs, look up packed slots on the **old** spec and on
the **new** spec (`LayeredSpec.edge_location` /
`AdjacencySpec.edge_location`) and copy those scalars onto
the new layer. A surviving edge can sit on a different hop
with a different packed index. Do **not** `Tensor.copy_` a
whole `weight`, and do **not** `load_state_dict` across
different priors. `PackedLinear.weight` is 1-D of length
`nnz`. Dropping one named edge and adding another leaves
`nnz` unchanged, so slot `i` is a different named edge.
`index_digest` / `mask_digest` hash numeric indices (or the
mask) and shapes, not names. A rename that leaves the packed
index pattern unchanged loads silently unless the new layers
were built with `identity=spec.fingerprint`. Always pass
that on the new `PackedLinear` / `MaskedLinear`. On
`MaskedLinear`, copy the named live cells of
`parametrizations.weight.original`, not the effective
`weight` property and not the whole rectangle.

**Reparse rebuilds the blueprint.** Parse is not "the same
graph minus a row." Both parsers recompute `input_nodes` /
`output_nodes` (in-degree / out-degree 0). `parse_layered`
also recomputes longest-path depths (unless `ranks=`), hop
membership, concat source axes, and `skips`. Adding an
incoming edge to a former input removes it from
`input_nodes`, and dropping an input's last edge removes the
input, so `align_inputs` indices change: recompute them on
the new spec. A stale index of a different width raises in
`PackedLinear.forward`; one of the same width (an input
swapped for another) cannot be caught. Pass the
same `widths=` / `ranks=` as the original parse, or the new
spec will not match.

**Bias is not a packed slot.** Bias is shape
`(out_features,)`, one value per output unit, not per edge.
Copy it by named node → unit slice:
`LayeredSpec.node_units` indexes the hop's output axis (the
target layer of `hops[layer - 1]`). Copy only when the node
still exists, its width is unchanged, and it still lives on
the same layer. On an `AdjacencySpec` there is no
`node_units`; index `spec.nodes` (the state-vector / bias
axis) by name. Do not copy bias by packed edge index. New
nodes keep the degree-aware init.

**Optimizer state is not copied.** Adam moments (`exp_avg`,
`exp_avg_sq`) are keyed by `Parameter` identity on the old
module. Construct a new optimizer on the new parameters, or
accept that those moments are lost. Do not add an
optimizer-remap helper.

---

## Spec interchange (`to_dict`, `from_dict`, `fingerprint`)

A DataFrame of edges does not record which parser produced the
spec: a DAG is valid for both layouts. Checkpoints use a
JSON-safe tagged dict, not pickle or `torch.save` of the
dataclass.

```python
payload = spec.to_dict()
spec = LayeredSpec.from_dict(payload)  # or AdjacencySpec
digest = spec.fingerprint
```

`to_dict()` returns a **new** dict with these keys:

```python
{
    "kpnn2_spec": 1,
    "layout": "layered",
    "edges": [["A", "H"], ["H", "C"]],
}
```

When every layered node has width 1, emit exactly those three
keys unless compacted layers differ from longest-path. Do
**not** emit `"widths"`. Then
`parse_layered(edgelist).fingerprint` is unchanged from width-1
longest-path graphs. When any node has width other than 1, also
emit:

```python
"widths": {"H": 3, "C": 2}
```

JSON object, node name → int. Include only nodes whose width is
not 1 (missing means 1). `json.dumps(..., sort_keys=True)` makes
this fingerprint-stable. Never emit `"widths"` for an
`AdjacencySpec`.

When compacted `layer_nodes` equal longest-path on the same
edges, do **not** emit `"ranks"` (a `ranks=` that happens to
match default does not change fingerprints). When they differ,
emit:

```python
"ranks": {"GeneA": 0, "PathLeaf": 1, "PathRoot": 1}
```

JSON object, **every** node, value = compacted layer index
(0-based, not the raw user numbers). `json.dumps(sort_keys=True)`
keeps fingerprints stable. Never emit `"ranks"` for an
`AdjacencySpec`.

| Key | Value |
|-----|--------|
| `kpnn2_spec` | Integer `1` (schema version). |
| `layout` | `"layered"` for `LayeredSpec`, `"adjacency"` for `AdjacencySpec`. Must not be omitted. |
| `edges` | List of `[source, target]` lists (JSON-safe, not tuples), same order as `to_edgelist()` rows / `canonical_edges`. One row per **named** edge, not per unit pair. |
| `widths` | Optional. Layered only. Node name → int for nodes whose width is not 1. |
| `ranks` | Optional. Layered only. Node name → compacted 0-based layer index for **every** node. Present only when that layering differs from longest-path. |

Unknown extra keys on an otherwise valid payload are ignored
(forward compatible). A stray `"widths"` or `"ranks"` key on an
adjacency payload is ignored like any extra key.

`LayeredSpec.from_dict(payload)` calls `parse_layered` on a
DataFrame built from `payload["edges"]`, passing
`payload["widths"]` when present and `payload["ranks"]` when
present. Absent or empty `"widths"` is all 1. Absent `"ranks"`
is longest-path. Invalid type or values raise `Kpnn2Error`.
`AdjacencySpec.from_dict` calls `parse_adjacency`. Hops and
masks are not hand-rebuilt. A layout mismatch (an adjacency
dict into `LayeredSpec.from_dict`, or the reverse) raises
`Kpnn2Error` naming the mismatch.

`from_dict` also raises `Kpnn2Error` when: `payload` is not a
dict; `kpnn2_spec` is missing or not `1`; `layout` is missing
or not `"layered"` / `"adjacency"`; `edges` is missing, is not
a sequence of pairs, or a pair is not two nonempty
string-convertible names; layered `"widths"` is present and
not a mapping of positive ints; or layered `"ranks"` is
present and not a mapping of non-negative ints covering every
node.

`fingerprint` is a property: the SHA-256 hex digest (64
lowercase hex characters) of
`json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"),
ensure_ascii=False).encode("utf-8")`. Do not use Python
`hash()`. `parse_layered(edgelist).fingerprint` equals
`parse_layered(spec.to_edgelist()).fingerprint` when every
width is 1 and ranking is longest-path. The same DAG parsed
layered vs adjacency yields different fingerprints because
`layout` differs. Adding, removing, or renaming a node, changing
an edge, changing a layered width, or changing compacted ranks
relative to longest-path changes the fingerprint.

These three names are methods / a property on the spec
classes. They are not package-level exports. There is no
module-level `spec_from_dict`. Pickle and `torch.save` of the
dataclass are **not** the supported interchange.

---

## `MaskedLinear`

Drop-in sparse linear layer. Same job as `torch.nn.Linear`
(call as `layer(x)`); not a subclass. Not a full model.

```text
MaskedLinear(mask, bias=True, *, identity=None, constraint=None, generator=None)
```

- `mask`: `torch.Tensor`, shape `(out_features, in_features)`.
  Inferred `in_features` / `out_features` from `mask.shape`.
  Do not take separate size arguments.
- Optional `bias`: shape `(out_features,)`. If `bias=False`,
  no bias parameter.
- Optional `identity`: opaque `str`, typically
  `spec.fingerprint`. Stored in `state_dict` next to
  `mask_digest` as UTF-8 `uint8` bytes. `load_state_dict`
  raises `Kpnn2Error` when a present identity does not
  match. A missing identity is not an error, even with
  `strict=True`. `None` means this layer does not claim
  an identity.
- Optional `constraint`: `nn.Module` or `None`. A per-entry
  map on the unconstrained weight, applied **before** the
  mask. `nn.Softplus()` is the textbook non-negative edge
  reparametrization. Must return a tensor of the same shape
  as the unconstrained weight. `reset_parameters` writes
  that unconstrained tensor; it does not invert this map.
  `PackedLinear` takes the same argument. Do not add a
  second class (`ConstrainedMaskedLinear`) and do not
  read a `constraint` column from the edgelist. Mixed
  per-edge signs and frozen slots go inside this module
  (a sign buffer, or `torch.where` for a hard freeze).
  That holds the effective `layer.weight` even when AdamW
  or SGD with momentum updates the stored tensor. A loss
  barrier is a soft prior. A gradient hook that zeroes a
  slot is not a freeze. That is user PyTorch, not a
  missing primitive.
- Optional `generator`: `torch.Generator` or `None`. Init
  draws from that generator. `None` (the default) uses the
  process default generator, bit-identical to omitting the
  argument. Not stored on the module. `reset_parameters`
  takes the same argument. Do not add `seed=`.
- The unconstrained tensor is stored through
  **`torch.nn.utils.parametrize.register_parametrization`** on
  the parameter named `weight`. That is the blessed PyTorch
  mechanism for "the effective weight is a function of a stored
  parameter". Do not replace it with a hand-rolled second
  parameter name. Connectivity is **not** an inner
  composable factor: `register_parametrization` appends, and
  a later map that does not preserve zeros (for example
  `softplus`) would resurrect blocked edges if the mask ran
  first. The parametrization list therefore multiplies by
  the mask **last**, so `layer.weight` and the forward pass
  stay masked no matter what else is stacked. Do not shadow
  `weight` with a plain property to achieve this.
- `layer.weight` is therefore the **effective masked weight**
  (constructor `constraint` if any, then any later maps,
  then the mask), recomputed on access. It is not an
  `nn.Parameter`: in-place writes to it are discarded.
  Assigning (`layer.weight = w`, under `torch.no_grad()`)
  copies `w` into the trainable tensor; the mask (and
  `constraint`, if set) are applied on read.
- The trainable tensor is
  `layer.parametrizations.weight.original`, same shape as
  `mask`. Masked-out entries may be nonzero there and never
  reach the output. There is no `raw_weight`.
  `model.parameters()` includes that tensor. A param-group
  filter that uses `"weight" in name` matches it;
  `name.endswith(".weight")` does not. Do not describe this
  as "the usual `weight` name."
- `layer.constraint` is the constructor module, or `None`.
- `state_dict` keys are `parametrizations.weight.original`,
  optional `bias`, `mask_digest`, and `identity` when the
  constructor was given one. `mask` stays out: it remains a
  non-persistent float32 buffer. `mask_digest` is a 1-D CPU
  `torch.uint8` tensor of length 32, the SHA-256 of the live
  mask's float32 C-contiguous bytes at save time, not a
  registered persistent buffer. `identity` is a 1-D CPU
  `torch.uint8` tensor of the UTF-8 bytes of the constructor
  string, also not a registered buffer. `load_state_dict`
  raises `Kpnn2Error` when a present digest or identity does
  not match the live layer, and does not load the weights. A
  missing digest or identity is not an error, even with
  `strict=True`. The digest catches same-shape rewiring. A
  rename that leaves the 0/1 pattern unchanged is caught by
  `identity` when callers pass `spec.fingerprint`. Do not
  `load_state_dict` or `Tensor.copy_` a weight across a
  reparse; see **Reparse hatch**. `repr` reports
  `ParametrizedMaskedLinear` (PyTorch swaps in a
  subclass to install the `weight` property);
  `isinstance(layer, MaskedLinear)` stays `True`, and
  `extra_repr` reports `in_features`, `out_features`, `bias`
  as `nn.Linear` does. Pickling the module object raises, as
  for any parametrized module; `copy.deepcopy` and
  `state_dict` work. Do not call `remove_parametrizations` on
  `weight`: that drops the mask.
- Register `mask` as a **non-persistent buffer** (not a
  parameter, not in `state_dict`), `float32`, and a **plain
  `torch.Tensor`**. It lives on the parametrization module, so
  its `named_buffers` key is `parametrizations.weight.0.mask`,
  and `layer.mask` is a property onto it that also accepts
  assignment. After
  `Module.half()`, `.to(dtype=torch.bfloat16)`, or `.double()`,
  `layer.mask.dtype` is still `float32`: the stored mask does
  not follow the module floating dtype.
  `MaskedLinear(spec.hops[i].to_mask())` stores an independent
  copy, so later writes to that allocated tensor do not reach
  the layer. Rebuild from the edgelist / `LayeredSpec` to
  change wiring.
  Nothing blocks a write to `layer.mask`; it is documented
  read-only, like any PyTorch buffer. `copy.deepcopy` of a
  `MaskedLinear` (and of a user `nn.Module` that holds
  `MaskedLinear` layers and a `LayeredSpec`) succeeds.
  Parameters on the copy are distinct objects. Copied masks
  stay float32.
- Forward multiplies in the parameter dtype:
  `weight` is the effective tensor (constraint, then later
  maps, then `mask.to(dtype=original.dtype,
  device=original.device)`), then
  `Y = F.linear(X, weight, bias)`.
  Equivalently `Y = X @ (C(W) ⊙ M).T + b` with `C` the
  constructor constraint (identity when omitted) and `M`
  cast to `W`'s dtype/device. This is why `.half()` /
  bfloat16 / `.double()` work like `nn.Linear`.
  `torch.autocast` is unsupported: `forward` disables it
  and casts `x` to the parameter dtype. Skip-edge
  `gather_hop_inputs` then still sees one dtype. In the
  common float32 case `mask.to` returns the buffer
  itself, so forward allocates nothing extra for the
  mask. The multiply is **dense** on purpose.
  `x` is an ordinary dense activation tensor, not a sparse
  host feature matrix. Sparse-tensor acceleration is not
  planned; see **Locked contrasts**.
- The forward path holds no tensor subclass, so
  `torch.compile(layer, fullgraph=True)` traces it without a
  graph break, parametrization included. Keep it that way.
  Do not add a `__torch_function__` override or any other
  write guard. A previous `FrozenMask` subclass broke
  `fullgraph=True` and cloned the whole mask on every
  forward pass. `tests/module/test_masked_linear.py` keeps
  a `fullgraph` regression test.
- **Degree-aware init:** for each output row `j`,
  `fan_in = int(mask[j].sum())` (count of 1s in that row).
  Use that `fan_in` for kaiming/uniform scale of that row (and for
  bias, use a documented rule: e.g. bias bound from that row's
  `fan_in`, or from mean fan_in; prefer **per-row fan_in** for
  weights). If `fan_in == 0`, leave that row at 0 and use bias
  bound 0. `reset_parameters` writes into
  `parametrizations.weight.original`, row by row, in that
  order. Optional `generator` isolates those draws from
  other torch RNG consumers; `None` keeps the default
  stream. A different draw count shifts the torch RNG stream
  and which trained-tier seeds pass; those controls now test
  a pass rate over 30 seeds, not a 5-seed window.
- Do **not** use full `in_features` as `fan_in`.
- No edgelist `initial_weight` column. `constraint=` is the
  supported per-entry map; do not stack `softplus` after the
  mask yourself. Mixed signs and frozen live-edge values
  stay in the caller's `constraint=` module (`torch.where`
  for a hard freeze); see **Package philosophy**.
- Masked-out weights still exist as parameters but are multiplied
  by 0 in the effective weight and the forward pass.

Typical construction:

```python
MaskedLinear(
    spec.hops[i].to_mask(),
    identity=spec.fingerprint,
)
```

---

## `PackedLinear`

Packed-edge linear layer. Same job as `torch.nn.Linear` (call
as `layer(x)`); not a subclass. Not a full model. One
trainable scalar per live edge; not `torch.sparse`; forward
is `index_add` on ordinary dense tensors.

```text
PackedLinear(
    source_index,
    target_index,
    out_features,
    in_features,
    bias=True,
    *,
    identity=None,
    constraint=None,
    generator=None,
)
```

- `source_index`, `target_index`: 1-D integer tensor or
  sequence of int, length `nnz >= 1`, copied to int64
  buffers. `0 <= source_index < in_features` and
  `0 <= target_index < out_features`. Duplicate
  `(source, target)` pairs, empty indices, bad types /
  ndim, or length mismatch raise `Kpnn2Error`.
- `out_features`, `in_features`: positive ints.
- Optional `bias`: shape `(out_features,)`. If
  `bias=False`, no bias parameter.
- Optional `identity`: opaque `str`, typically
  `spec.fingerprint`. Stored in `state_dict` next to
  `index_digest` as UTF-8 `uint8` bytes. `load_state_dict`
  raises `Kpnn2Error` when a present identity does not
  match. A missing identity is not an error, even with
  `strict=True`. `None` means this layer does not claim
  an identity.
- Optional `constraint`: `nn.Module` or `None`. A per-entry
  map on the packed `weight`, applied in `forward`.
  `nn.Softplus()` is the textbook non-negative edge
  reparametrization. Must return a tensor of shape `(nnz,)`.
  There are no absent edges here, so this map cannot
  resurrect a blocked cell. `reset_parameters` writes the
  unconstrained packed tensor; it does not invert this map.
  `MaskedLinear` takes the same argument. `layer.constraint`
  is that module, or `None`. Still no `parametrize`. Mixed
  per-edge signs and frozen slots are user PyTorch on this
  tensor, not parse columns and not per-slot
  `requires_grad=False`. A hard freeze is `torch.where`
  inside this module: forward uses the replaced value, so
  AdamW and SGD with momentum cannot move the live edge.
  The stored unconstrained slot may still drift; read
  `constraint(weight)`, or write the constants back after
  `optimizer.step()` if a checkpoint must match. A loss
  barrier is a soft prior. A gradient hook that zeroes the
  slot is not a freeze.
- Optional `generator`: `torch.Generator` or `None`. Init
  draws from that generator. `None` (the default) uses the
  process default generator, bit-identical to omitting the
  argument. Not stored on the module. `reset_parameters`
  takes the same argument. `transpose` forwards it to the
  inner constructor. Do not add `seed=`.
- `weight` is an `nn.Parameter` of shape `(nnz,)`. No
  `parametrize`. No dense `(out, in)` `layer.weight`.
  The parameter name is `weight`. When `constraint` is set,
  this tensor is unconstrained; `forward` uses
  `constraint(weight)`.
- Index buffers `source_index` and `target_index` are
  persistent so the module round-trips. They stay integer
  after `.half()` / bfloat16 / `.double()`; `weight` and
  `bias` follow the module floating dtype like `nn.Linear`.
  `torch.autocast` is unsupported: `forward` disables it
  and casts `x` to the parameter dtype so AMP cannot mix
  Half into `index_add`. Cast the module with
  `.to(dtype=...)` (or `.half()` / `.double()`) instead.
- Forward, `x` shape `(..., in_features)`:

  ```text
  live = constraint(weight) if constraint else weight
  x = x.to(dtype=live.dtype)
  contrib = x[..., source_index] * live
  y = zeros(..., out_features)  # same batch dims, dtype, device
  y.index_add_(-1, target_index, contrib)
  if bias is not None:
      y = y + bias
  ```

  The recipe runs with `torch.autocast` disabled.

  Before the recipe, `forward` raises `Kpnn2Error` unless `x`
  is a tensor with `x.shape[-1] == in_features` (0-d
  included), as `nn.Linear` / `MaskedLinear` (through
  `F.linear`) and `PackedMultiheadAttention` do. The gather
  reads columns by position, so without the check a wider
  tensor would be accepted and silently read: for example a
  stale `align_inputs` index after a reparse that removed an
  input. Do not drop this check. A stale index of the **same**
  width cannot be detected here; the reparse hatch says to
  recompute `align_inputs`.

  Never allocate `(out, in)`. Never scatter into a dense
  `(out, in)` matrix in forward. Never import
  `torch.sparse`. Never take a dense mask or an
  `AdjacencySpec`. `x` is an ordinary dense activation
  tensor, not a sparse host feature matrix.
- **Degree-aware init:** `fan_in` for output row `j` is
  the number of packed edges with `target_index == j`
  (`bincount`, `minlength=out_features`). Each live edge
  into row `j` (and `bias[j]`, if present) is drawn from
  `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`. `fan_in == 0`: no
  weights in that row; `bias[j]` stays 0. Input nodes on
  an `AdjacencySpec` have in-degree 0, so they have no
  packed incoming edges; do not invent identity
  connections. The user still writes inputs into the
  state vector each step. Optional `generator` isolates
  those draws from other torch RNG consumers; `None`
  keeps the default stream.
- `extra_repr` reports `in_features`, `out_features`,
  `nnz`, and `bias`.
- `state_dict` keys are `weight`, optional `bias`,
  `source_index`, `target_index`, `index_digest`, and
  `identity` when the constructor was given one.
  `index_digest` is a 1-D CPU `torch.uint8` tensor of
  length 32, the SHA-256 of the int64 C-contiguous bytes
  of `source_index`, then `target_index`, plus
  `out_features` and `in_features` as fixed-width
  integers so a reshape cannot collide. It is not a
  registered persistent buffer. `identity` is a 1-D CPU
  `torch.uint8` tensor of the UTF-8 bytes of the
  constructor string, also not a registered buffer.
  `load_state_dict` raises `Kpnn2Error` when a present
  digest or identity does not match, and does not load
  the weights. A missing digest or identity is not an
  error, even with `strict=True`. The digest catches
  same-shape rewiring. A rename that leaves the packed
  index pattern unchanged is caught by `identity` when
  callers pass `spec.fingerprint`. Do not `load_state_dict`
  or `Tensor.copy_` a `weight` across a reparse; see
  **Reparse hatch**. `copy.deepcopy` works.
- The forward path holds no tensor subclass and no sparse
  layout, so `torch.compile(layer, fullgraph=True)` traces
  it without a graph break. Keep it that way.

Typical construction:

```python
spec = kpnn2.parse_adjacency(edgelist)
core = kpnn2.PackedLinear(
    spec.source_index,
    spec.target_index,
    len(spec.nodes),
    len(spec.nodes),
    identity=spec.fingerprint,
)
```

`MaskedLinear(spec.to_mask())` remains valid for small
graphs. The same constructor on a hop is
`PackedLinear(hop.source_index, hop.target_index,
hop.out_features, hop.in_features)`. `PackedLinear` does
not add model capacity relative to `MaskedLinear`: dead
edges already did not affect training. `MaskedLinear` is
the better kernel when the dense rectangle fits (GEMM).

Do not name this `SparseMaskedLinear`, `SparseLinear`, or
`PackedMaskedLinear`.

### `PackedLinear.transpose()`

```text
layer.transpose(bias=True, *, tie=True, identity=None, generator=None)
```

The tied-autoencoder helper. Returns a new `PackedLinear`
that applies the same live edges backwards: `source_index`
and `target_index` swapped, `in_features` and
`out_features` swapped, same `nnz`. Packed slot `i` is
still the same edge; the 1-D `weight` is not permuted.

- `tie=True` (default): the returned layer's `weight` is
  this layer's `weight` `nn.Parameter`. Gradients from both
  forwards accumulate there.
- `tie=False`: copy the current values into a new
  Parameter.
- Bias is never tied. Default `bias=True` allocates a new
  bias of shape `(in_features,)` (the original input
  width), degree-aware init on the transposed fan-in.
  `bias=False` means no bias. This layer's bias is unused.
- `constraint` on the result is a deepcopy of this layer's
  constraint, or `None`. Do not pass the same `nn.Module`
  into two `PackedLinear` constructors: PyTorch would
  steal the submodule from the first parent.
- `identity` defaults to `None`; this layer's identity is
  not copied. The index digest differs because buffers and
  sizes are swapped, so an encoder `state_dict` will not
  load into the transpose.
- Optional `generator`: forwarded to the inner
  `PackedLinear` constructor. `None` keeps today's global
  init. A full init still runs, then `weight` is replaced;
  discarded weight draws still advance that generator.
- Device and floating dtype follow this layer's `weight`.
  Index buffers stay integer.

Do **not** parse a reversed edgelist and assign
`dec.weight = enc.weight`. Packed order is lexicographic
by `(source name, target name)` of each spec, so those
slots do not line up. Do **not** add a packed-slot
permutation helper or a reverse parser; this method is the
tying path.

On a layered hop, the transposed output is
`hop.in_features` wide. Split it with `scatter_hop_outputs`
when the hop reads several source layers; add those pieces
into the caller's `saved` dict, because two reversed hops
may write the same earlier layer. An `AdjacencySpec` map
is already `n`-wide; no gather or scatter.

`MaskedLinear` has no packed slots: use
`F.linear(h, layer.weight.T, dec_bias)`. Do not add
`MaskedLinear.transpose`. Do not add a `TiedAutoencoder`
class; the user owns `forward()`.

---

## `PackedMultiheadAttention`

Packed multi-head attention on live edgelist pairs. Same
job as `torch.nn.MultiheadAttention` (call as
`layer(query, key, value)`); not a subclass. Not a
Transformer block and not a full model. Scores exist only
for live `(source, target)` pairs: query = target, key /
value = source. Forward never allocates an `(n, n)` or
`(L, S)` score matrix and does not import `torch.sparse`.
This module does not take an `AdjacencySpec`; pass packed
indices (typically from `parse_adjacency`).

```text
PackedMultiheadAttention(
    source_index,
    target_index,
    query_features,
    key_features,
    embed_dim,
    num_heads,
    dropout=0.0,
    bias=True,
    kdim=None,
    vdim=None,
    batch_first=True,
    add_self_loops=False,
    *,
    identity=None,
    generator=None,
    chunk_size=None,
)
```

- `source_index`, `target_index`: 1-D integer tensor or
  sequence of int, length `nnz >= 1`, copied to int64
  buffers. `0 <= source_index < key_features` and
  `0 <= target_index < query_features`. Duplicate
  `(source, target)` pairs, empty indices, bad types /
  ndim, or length mismatch raise `Kpnn2Error`.
- `query_features`, `key_features`: positive ints.
  Sequence lengths of `query` and of `key` / `value`.
- `embed_dim`: positive int, divisible by `num_heads`.
- `num_heads`: positive int.
- `dropout`: float `>= 0` on packed attention weights.
  Integer `0` is accepted. `bool` and negatives raise
  `Kpnn2Error`.
- Optional `bias`: on the four `nn.Linear` projections.
- `kdim`, `vdim`: must be `None` or equal to
  `embed_dim`. Other values raise `Kpnn2Error`.
- `batch_first`: default `True` (kpnn2 sample-major).
  That differs from `nn.MultiheadAttention`, whose
  default is sequence-major. Batched tensors are
  `(..., seq, embed_dim)` when True;
  `(seq, batch, embed_dim)` when False. Unbatched 2-D
  `(seq, embed)` ignores this flag.
- `add_self_loops`: if `True`, OR missing `(i, i)`
  pairs into the module buffers when
  `query_features == key_features`. Caller index
  objects are not mutated. Existing self-loops are
  kept, not duplicated. If
  `query_features != key_features`, raise
  `Kpnn2Error`.
- Optional `identity`: opaque `str`, typically
  `spec.fingerprint`. Stored in `state_dict` next to
  `index_digest` as UTF-8 `uint8` bytes. `load_state_dict`
  raises `Kpnn2Error` when a present identity does not
  match. A missing identity is not an error, even with
  `strict=True`. `None` means this layer does not claim
  an identity.
- Optional `generator`: Xavier-uniform on the four
  projections uses that `torch.Generator`. `None` keeps
  today's path: `nn.Linear` kaiming-init then Xavier on
  the global stream, bit-identical to omitting the
  argument. When set, those `nn.Linear` constructors must
  not advance the global stream. Not stored on the
  module. `reset_parameters` takes the same argument.
  Do not add `seed=`.
- Optional `chunk_size`: `None` (default) gathers all
  live pairs at once. A positive int gathers chunked
  live pairs: softmax stays over every live key of a
  query, but `q`/`k`/`v` gathers are slices of that
  many edges. Backward rematerializes those gathers so
  training does not save `(..., nnz, heads, head_dim)`
  pair tensors. Packed softmax `(..., nnz, heads)` is
  still stored. `bool` and other non-`int` values, `0`,
  and negatives raise `Kpnn2Error`. Not stored in
  `state_dict` or `index_digest`; a checkpoint loads
  into a layer with a different `chunk_size`.
- Projections are four separate
  `embed_dim → embed_dim` `nn.Linear`s (`q_proj`,
  `k_proj`, `v_proj`, `out_proj`), not a fused
  `in_proj_weight`.
- Index buffers `source_index` and `target_index` are
  persistent so the module round-trips. They stay
  integer after `.half()` / bfloat16 / `.double()`.
  `torch.autocast` is unsupported, as on `PackedLinear`.
  `forward` disables it and casts query, key, and value
  to the parameter dtype. Cast the module with
  `.to(dtype=...)` instead.

```text
forward(
    query,
    key,
    value,
    key_padding_mask=None,
    need_weights=False,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False,
) -> (output, None) or (output, packed_weights)
```

- Always returns a 2-tuple. `need_weights` defaults to
  `False` (MHA defaults `True`); then the second entry is
  `None`. If `need_weights` is `True`, the second entry is
  packed per-edge softmax aligned with `source_index` /
  `target_index`, **not** MHA's dense `(L, S)` map.
  `average_attn_weights=True` (default) averages heads:
  shape `(..., nnz)`. `False` keeps heads:
  `(..., nnz, num_heads)`. Batch layout follows the
  output, including `batch_first`. Does not allocate
  `(L, S)`. Packed length is the module `nnz`, including
  pairs OR-ed by `add_self_loops`; zip with
  `source_index` / `target_index`, not `to_edgelist()`,
  when that flag added pairs.
- `query`, `key`, and `value` are required; `key` is
  not defaulted to `query`.
- `attn_mask` must be `None` (the edgelist is the
  structural mask). `is_causal` must be `False`. Both
  raise `Kpnn2Error` otherwise.
- `key_padding_mask` is `None` or a packed boolean
  mask: `True` means ignore that key. Shape `(S,)`
  unbatched or `(N, S)` batched. Applied in packed
  space; does not allocate `(n, n)`. Copied onto
  the scores' device; stays boolean. Float padding
  masks raise `Kpnn2Error`.
- Isolated queries (no packed keys, and none added by
  self-loops) stay zeros after the mix, then still go
  through `out_proj`. After `key_padding_mask`, a
  query with no remaining keys also stays zeros.
  There is no NaN softmax.
- Never allocate `(n, n)` or `(L, S)`. Never import
  `torch.sparse`. `chunk_size` does not change that:
  it only slices live-pair gathers. Never take a dense
  mask or an `AdjacencySpec`. `query` / `key` / `value`
  are ordinary dense activation tensors.
- `extra_repr` reports `query_features`,
  `key_features`, `embed_dim`, `num_heads`, `nnz`,
  `dropout`, `batch_first`, `add_self_loops`, and
  `chunk_size`.
- `state_dict` includes `index_digest`, a 1-D CPU
  `uint8` tensor of length 32: the SHA-256 of the
  int64 C-contiguous bytes of `source_index`, then
  `target_index`, plus `query_features`,
  `key_features`, `embed_dim`, and `num_heads` as
  fixed-width integers so a reshape or a different
  head split cannot collide. It is not a registered
  persistent buffer. When the constructor was given
  `identity`, `state_dict` also includes that string
  as UTF-8 `uint8` bytes. `load_state_dict` raises
  `Kpnn2Error` when a present digest or identity does
  not match, and does not load the weights. A missing
  digest or identity is not an error, even with
  `strict=True`. Same pattern as `PackedLinear`.
  `copy.deepcopy` works.

Typical construction:

```python
spec = kpnn2.parse_adjacency(edgelist)
n = len(spec.nodes)
attn = kpnn2.PackedMultiheadAttention(
    spec.source_index,
    spec.target_index,
    n,
    n,
    embed_dim,
    num_heads,
    identity=spec.fingerprint,
)
```

Do not name this a Transformer. Do not add
`parse_attention`. Do not reuse a hop rectangle as a square
attention matrix. Do not fold this layer into
`PackedLinear` or `MaskedLinear`.

---

## `gather_hop_inputs(saved, hop)`

The source axis of one hop. Call it in `forward()` just before
`PackedLinear` (or `MaskedLinear(hop.to_mask())`); it sits
between hops. Together with `scatter_hop_outputs` (the inverse
split) this is what the layered layout needs beyond the linear
primitive, and it holds no parameters. It does not inject
values into the previous layer and does not pick skip nodes
by name: it concatenates **whole** source layers. Unused skip
columns stay in that tensor; `PackedLinear` never reads them.

```text
gather_hop_inputs(saved, hop) -> torch.Tensor
```

- `saved` maps layer index → that layer's activation, width
  `layer_dims[i]`. Only `hop.source_layers` are read, and they
  are not modified. Extra keys are ignored.
- `hop` is a `Hop` from `spec.hops`.
- Returns the source layers concatenated on the last axis in
  `hop.source_layers` order, shape `(..., hop.in_features)`,
  ready for `PackedLinear` or `MaskedLinear(hop.to_mask())`.
- A hop with one source layer returns that saved tensor itself,
  without a copy (plain adjacent, or skip-only parents from one
  depth). A hop with several source layers concatenates them in
  `source_layers` order; that need not include
  `target_layer - 1` when `ranks=` left a node with only skip
  parents.
- Store every produced layer in `saved`. A missing source
  **layer** (not a missing named node) raises `Kpnn2Error`
  instead of silently dropping those edges. The message names
  the layer and the hop that wanted it.
- Differentiable into every source: `torch.cat` passes gradient
  back to each part.
- All parts must share dtype and device; mismatches raise rather
  than promote silently.
- Public failures: `Kpnn2Error` (not a mapping, not a `Hop`,
  missing layer, non-tensor entry, wrong unit count, mixed
  dtype or device).

There is **no** module here on purpose. Anything with parameters
would reintroduce a second place for edge weights to live.

---

## `scatter_hop_outputs(tensor, hop)`

The inverse of `gather_hop_inputs` for that concatenated
axis. A tied decoder (`PackedLinear.transpose` on the hop)
emits `hop.in_features` columns; this splits them back
onto `hop.source_layers`. It holds no parameters. It does
not take `saved` and does not add into it.

```text
scatter_hop_outputs(tensor, hop) -> dict[int, Tensor]
```

- `tensor` is the concatenated source axis, last dimension
  `hop.in_features`. Typically the output of
  `PackedLinear.transpose()` on this hop.
- `hop` is a `Hop` from `spec.hops`.
- Returns `{layer: piece}` for each entry of
  `hop.source_layers`, in that order. `piece` has last
  dimension `hop.source_dims[i]`.
- A hop with one source layer returns `{source_layers[0]:
  tensor}` itself, without a copy. Several source layers
  are `torch.split` views on the last axis.
- The caller adds those pieces into their decoder `saved`
  dict. Two reversed hops may write the same earlier layer
  (a skip and an adjacent reverse), so add, do not
  overwrite.
- Last dimension must be `hop.in_features`. A missing
  match, a non-tensor, a 0-dimensional tensor, or a non-
  `Hop` raises `Kpnn2Error`.
- Differentiable into `tensor`.
- An `AdjacencySpec` has no hops and is not accepted.

There is **no** module here on purpose, same as gather.
Do not add a packed-slot permutation API; `transpose`
keeps slot `i` as the same edge.

---

## `align_inputs(names, spec)`

`spec` is a `LayeredSpec` **or** an `AdjacencySpec`. Only
`spec.input_nodes` is read (and, for a `LayeredSpec`,
`layer_widths[0]`). Anything else raises `Kpnn2Error`.

Returns a 1-D `numpy.ndarray` of dtype `int64`, length
`width`. It is a column index into the caller's feature
axis, not a data tensor. This function does not take the
matrix, copy sample rows, or densify. Apply the index on
the caller's storage. It is not a minibatch API, not a
device-copy helper, and not a densifying path. See
**Locked contrasts**.

**Width differs by layout.** For a `LayeredSpec` that length is
`layer_dims[0]` (the unit width of layer 0), not
`len(input_nodes)` when an input node is wider than 1.
`hops[0]` reads layer 0 alone, so a row block indexed with
this result feeds `PackedLinear` on `hops[0]` (or
`MaskedLinear(spec.hops[0].to_mask())`) directly with no
gathering. The name list still has one label per input
**node**; a width greater than 1 repeats that node's index
across its units. For an `AdjacencySpec` the length is **not**
the state width: `to_mask()` is `(n, n)` over every node,
while the index is only `len(input_nodes)` long. Scatter the
gathered columns into the `n`-wide state vector via
`spec.input_index` before calling `MaskedLinear(spec.to_mask())`:

```python
col = kpnn2.align_inputs(df.columns, spec)
x = torch.as_tensor(
    df.to_numpy()[:, col],
    dtype=torch.float32,
)
state = torch.zeros(x.shape[0], len(spec.nodes))
state[:, spec.input_index] = x
```

**Names:**

- Required labels: `spec.input_nodes` (any order).
- Match labels after converting them to strings (so an integer
  label `1` can match string node id `"1"`).
- Extra names are ignored.
- Missing required names: `Kpnn2Error`.
- Duplicate labels (including after string conversion):
  `Kpnn2Error`. The message names the unique duplicated labels
  after `str(...)`, sorted, comma-separated.
- Accepted: a 1-D sequence of labels (`df.columns`,
  `adata.var_names`, a numpy array, a list). The annotated
  type is `object` so a DataFrame or tensor is rejected at
  runtime, not treated as names.

**Rejected:**

- `pandas.DataFrame`: `Kpnn2Error`. The message must say that
  `names` is a DataFrame and that feature names are required
  (for example `data.columns`).
- `torch.Tensor`: `Kpnn2Error`. The message must say that
  `names` is a tensor and that the feature names that label
  that axis are required. Pre-ordered dense tensors go
  **straight to the model**.
- A string, bytes, mapping, set, AnnData-like object
  (`var_names` and `X`), or a matrix (`ndim != 1`):
  `Kpnn2Error`. The string message must say that `names` is
  a string; the bytes message must say that `names` is
  bytes.
- Do not check value dtypes. There are no values.
- Do not return a tensor of data.

**Not a kpnn2 matrix type:** AnnData, numpy matrices, dicts of
columns, scipy sparse matrices. Pass the labels that sit on
that axis and apply the index on the matrix (`X[:, col]`).
scipy CSR/CSC (and AnnData `.X` in those formats) stay sparse
until the caller densifies one row block. COO-style layouts
cannot be indexed that way; convert first. Width greater
than 1 repeats indices and copies those columns on CSR/CSC.
Passing `adata.to_df()` (or any full densify) and then a
DataFrame into `align_inputs` is rejected at the DataFrame
check.

---

## `map_node_attributions(attributions, spec, layer=None, *, hop_input=None, hop_output=None, dims=None, coords=None)`

Unopinionated name mapping. No Captum import. Returns
`xarray.DataArray`. Does not aggregate.

`spec` is a `LayeredSpec` **or** an `AdjacencySpec`. `layer` stays
positional; `hop_input` and `hop_output` are keyword-only. A
`LayeredSpec` takes **exactly one** of the three; an
`AdjacencySpec` takes none:

| Spec | Given | Result |
|------|-------|--------|
| `LayeredSpec` | `layer=int` | Names from `unit_names` of that layer (a wide node's name repeats); scalar `layer` coordinate attached |
| `LayeredSpec` | `hop_output=Hop` | Same as `layer=hop.target_layer`: the layer that hop's module **writes**, length `hop.out_features`, scalar `layer` coordinate attached |
| `LayeredSpec` | `hop_input=Hop` | Concatenated **source-unit** names of that hop (what its module **reads**), length `hop.in_features`. **No** `layer` coordinate (this axis is not one depth). |
| `LayeredSpec` | none | `Kpnn2Error`: need one of `layer`, `hop_input`, `hop_output` |
| `LayeredSpec` | two or three | `Kpnn2Error`: pass exactly one; the message names what was given |
| `AdjacencySpec` | none | Names from `spec.nodes`; **no** `layer` coordinate |
| `AdjacencySpec` | any | `Kpnn2Error`: `layer`, `hop_input`, and `hop_output` do not apply |

**The hop side is always stated, never inferred.** There is no
side-less `hop=` keyword. A hop's input and output often have the
same width (for example `A, B -> H1, H2`), so the node-axis
length check cannot tell them apart; a side-less keyword let
Captum layer outputs (Captum's default side) be labelled with the
hop's input names without an error. `hop_output=spec.hops[i]`
lets the caller name a Captum result on `model.hops[i]` with the
same `i` as the module, with no `i -> i+1` translation; it
resolves to `hop.target_layer`, so it is not a second notion of
target layer. Do not add a side-less `hop=` back, and do not
infer the side from the tensor width.

An `AdjacencySpec` has no depths, so there is no layer index to
report and none is invented. Do not fabricate `layer=0` for it.
`hop_input` / `hop_output` is a `Hop` that must compare equal to
one entry of `spec.hops` (frozen dataclass equality). Do not also
take an int hop index.

Names on the hop-input axis are **unit** names, not
`hop.source_nodes`. `source_nodes` is one name per node;
`in_features` is units. Repeat a wide node's name `k` times,
same as `Layout.unit_names()` / layer-mode mapping. Build that
list from the spec, not from `hop.source_nodes`:

```python
concat_layouts(
    [
        build_layout(
            spec.layer_nodes[i],
            spec.layer_widths[i],
        )
        for i in hop.source_layers
    ]
).unit_names()
```

- `attributions`: `torch.Tensor`, or a non-empty tuple/list of
  equal-shaped tensors (stacked on a new `step` axis).
- `layer`: `int` index into `spec.layer_nodes` (0-based), stored as
  scalar coordinate `layer`. `LayeredSpec` only, mutually
  exclusive with `hop_input` and `hop_output`.
- `hop_output`: a `Hop` equal to one of `spec.hops`. Labels the
  layer that hop's module returns, `hop.target_layer`, length
  `hop.out_features`, with the scalar `layer` coordinate; the
  result is identical to `layer=hop.target_layer`. Captum layer
  methods (`LayerConductance` and the rest) score this side by
  default. `LayeredSpec` only, mutually exclusive with `layer`
  and `hop_input`.
- `hop_input`: a `Hop` equal to one of `spec.hops`. Labels the
  concatenated source axis of that hop (`gather_hop_inputs`
  output / hop-module input; Captum with
  `attribute_to_layer_input=True`), length `hop.in_features`.
  No `layer` coordinate. `LayeredSpec` only, mutually exclusive
  with `layer` and `hop_output`.
- The `node` axis length must equal the number of named units:
  `spec.layer_dims[layer]` when `layer` is given (names from
  `layout.unit_names()`, so a wide node's name repeats `k`
  times; do not aggregate), `hop.out_features` for
  `hop_output`, `hop.in_features` for `hop_input` (same
  unit-name rule on the concatenated source layers),
  `len(spec.nodes)` for an `AdjacencySpec`. That axis gets
  those names as its coordinate, in order.
- Default dims: 1-D → `(node,)`; 2-D → `(observation, node)`; a
  stacked sequence of 2-D tensors → `(step, observation, node)`.
  Rank 3+ (except that stacked default) requires `dims=` containing
  `node` exactly once.
- `coords`: optional labels for axes other than `node` and `layer`.
- Values: detached CPU copy of the tensor. No abs/sum/mean.
  `bfloat16` is stored as float32 because NumPy has no
  bfloat16 dtype; those values are unchanged. Other dtypes
  are kept.
- Long table: `da.to_dataframe(name="score").reset_index()`.
  Wide 2-D table: `da.to_pandas()`.
- Invalid `spec`, `layer`, `hop_input`, `hop_output`, shape,
  `dims`, or `coords`: `Kpnn2Error`.

Fold those named scores with `aggregate_node_attributions`
(does not change this function).

The user obtains `attributions` however they like (Captum
LayerConductance, IntegratedGradients, custom grads, etc.). This
function only attaches spec names to the `node` axis. The input
is already dense scores, not a host feature matrix. For the
output of the module on `spec.hops[i]` pass
`hop_output=spec.hops[i]` (equivalently
`layer=spec.hops[i].target_layer`, that is `i+1`). For scores on
that hop's concatenated source axis pass
`hop_input=spec.hops[i]`. Do not name-map BatchNorm or other
unnamed modules.

For a cyclic net on an `AdjacencySpec` there is no layer to
index; the natural extra axis is `step`. Pass one tensor per
unrolled step as a sequence and they stack onto
`(step, observation, node)`.

---

## `aggregate_node_attributions(attributions, labels=None, *, method="rauter_mangano_2026", **method_kwargs)`

Dispatcher only. Looks up `method`, emits status warnings,
calls the registered function with
`(attributions, labels, **method_kwargs)`, stamps `method`,
`method_params`, and `kpnn2_version` on the result. All
three are strings so `.to_netcdf` keeps them:
`method_params` is JSON of the bound parameters with
defaults, minus the first two (data) parameters; numpy
values are unwrapped, anything else not JSON-encodable is
stored as its `repr`. Does not
know about binary classes or seeds. Public failures:
`Kpnn2Error`. Unknown names list callable methods (not
`removed`).

`list_aggregation_methods()` returns a pandas table of every
registry row: name, status, description, references,
`added_in`, `deprecated_in`, `removed_in`, replacement.

Statuses: `recommended` (default), `supported`,
`experimental` (`UserWarning`: results may change),
`deprecated` (`FutureWarning`, not `DeprecationWarning`),
`removed` (entry stays, call raises, names the replacement).

Implementation lives in private package
`src/kpnn2/_aggregation/`. Methods are modules under
`_aggregation/_methods/`. The decorator
`register_aggregation_method` is not public. Do not export
`rauter_mangano_2026`. Adding a method: one function with
that shared signature, decorate it, import the module from
`_methods/__init__.py`, put the full contract on that
function's docstring, add
`docs/reference/aggregation/{name}.md`, and add a row to
the Methods table on
`docs/reference/aggregate_node_attributions.md`. No
dispatcher code or dispatcher docstring edits. Method
pages are not public names and do not appear in the
Callables table of `docs/reference/api.md`. See `AGENTS.md`.

Do not attach labels inside `map_node_attributions`. Do not
guess that `step` or a Captum `class` dim is `seed`.

---

## `rauter_mangano_2026` (default aggregation method)

Registered under this name. Not a public import; pass
`method="rauter_mangano_2026"`. The private function's
docstring is the user-facing contract (rendered at
`docs/reference/aggregation/rauter_mangano_2026.md`).

The method is not yet implemented. Calling it raises
`Kpnn2Error` with that message. Arguments are ignored.
It stays the default, with status `recommended`, so the
name is reserved for the method that will replace this
placeholder. `list_aggregation_methods()` still lists it.

When the method is implemented, state the formulas that
the code computes on that private docstring. Do not add
ranking advice or other claims beyond those formulas.

---

## Internal unit layout

`src/kpnn2/_layout.py` owns every mapping from a node name to a
position on a tensor axis. Nothing else in `src/` computes a
column index by hand.

**Layered width is public.** `parse_layered(..., widths=)` sets
per-node unit counts on a `LayeredSpec`. `layer_widths` stores
them next to `layer_nodes`. Default `k=1` leaves every existing
edgelist, fingerprint, hop index tuple, and tutorial numerically
identical.

**Adjacency width is not.** `parse_adjacency` has no `widths=`
argument. Packed adjacency indices remain one pair per named
edge / block start. Do not implement adjacency width.

- A node owns a **contiguous slice** of units (`NodeSlot`).
  Default width is `DEFAULT_NODE_WIDTH == 1`. With `widths=`, a
  node may own `k>1` units.
- A `Layout` places the nodes of one axis in order without gaps.
  `layout.n_units` is that axis length. `layer_dims[i]` and the
  hop `to_mask()` axis length come from `n_units`, not from
  `len(names)`.
- A hop's column axis is `concat_layouts` of its source layers'
  layouts, so a source node's block on that axis is its own
  block shifted by the widths in front of it. `source_dims` and
  `Hop.column_offsets` come from the same widths.
  `Hop.source_nodes` stays one name per node.
- `Hop.source_index` / `target_index` store **every unit pair**
  of each named edge entering that hop (`iter_block_pairs`,
  target-unit outer, source-unit inner). `Hop.to_mask()` writes
  `1.0` at each `[target_index[i], source_index[i]]`, matching
  `fill_block`. Reconstruct named edges with
  `Layout.slot_containing(unit)`, never by indexing
  `source_nodes` with a unit index.
  `LayeredSpec.edge_location` and `AdjacencySpec.edge_location`
  are that lookup for callers: a named edge to packed weight
  slots. They are identity, not constraints. Callers who
  need mixed signs or frozen values index these slots in
  their own `constraint=` module (`torch.where` to hold a
  value); do not store that policy on the spec.
  `LayeredSpec.node_units` and `LayeredSpec.hop_units` are
  the matching lookup for a named node: a contiguous unit
  slice on a layer tensor or a hop source axis. Do not
  export `Layout`.
- `AdjacencySpec.source_index` / `target_index` store
  `layout.start_of` (the block start) for each original edge.
  `to_mask()` writes `1.0` at each
  `[target_index[i], source_index[i]]`.
- `Skip.source_in_layer` and `Skip.target_in_layer` store a
  **block start** inside their own layer. One `Skip` per named
  edge.
- `align_inputs` builds a `LayeredSpec` layout from
  `input_nodes` plus `layer_widths[0]` and repeats a node's
  column index across that node's units. An `AdjacencySpec`
  does not expand widths and does not call `build_layout`.
- `map_node_attributions` on a `LayeredSpec` uses
  `build_layout(layer_nodes[layer], layer_widths[layer])` when
  `layer=` is given, and on `hop.target_layer` when
  `hop_output=` is given. With `hop_input=`, it concatenates
  `build_layout` of each `hop.source_layers` entry via
  `concat_layouts` and labels with `unit_names()` (not
  `hop.source_nodes`). The node axis length is `n_units`; the
  coordinate is `layout.unit_names()`. An `AdjacencySpec`
  still uses `build_layout(names)` with no widths.

New code asks a `Layout` for a slot instead of using
`list.index()` or `enumerate` positions.

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

import kpnn2

edgelist = pd.DataFrame(
    {
        "source": ["A", "H"],
        "target": ["H", "C"],
    }
)
# add a skip with another row A -> C when needed

spec = kpnn2.parse_layered(edgelist)

class Net(nn.Module):
    def __init__(self, spec: kpnn2.LayeredSpec):
        super().__init__()
        self.spec = spec
        self.hops = nn.ModuleList(
            [
                kpnn2.PackedLinear(
                    hop.source_index,
                    hop.target_index,
                    hop.out_features,
                    hop.in_features,
                    identity=spec.fingerprint,
                )
                for hop in spec.hops
            ]
        )

    def forward(self, x):
        saved = {0: x}
        last = len(self.hops) - 1
        hidden = x
        for index, hop in enumerate(self.spec.hops):
            sources = kpnn2.gather_hop_inputs(saved, hop)
            hidden = self.hops[index](sources)
            if index < last:
                hidden = F.relu(hidden)
            saved[hop.target_layer] = hidden
        return hidden

model = Net(spec)
x_df = pd.DataFrame({"A": [0.1, 0.2]})
col = kpnn2.align_inputs(x_df.columns, spec)
x = torch.as_tensor(
    x_df.to_numpy()[:, col],
    dtype=torch.float32,
)
y = model(x)

# optional: user ran some attribution method themselves
da = kpnn2.map_node_attributions(
    attributions=y.detach(),
    spec=spec,
    layer=len(spec.layer_nodes) - 1,
)
# optional: fold observations (and seeds) to one score
# per node. The default method rauter_mangano_2026 is
# not yet implemented and raises Kpnn2Error.
```

Every edge, including `A → C` when that row is present, is
already a packed unit-pair block of some hop. The loop applies
each hop once, so nothing has to be remembered per skip edge.

That snippet applies `align_inputs` to feature names, then
gathers a dense table. A caller who already has a sparse host
matrix (for example AnnData `.X`) passes `adata.var_names`,
applies `X[:, col]` on the sparse layout, densifies only each
row block, moves that dense tensor, and calls `model`. Tensors
at the module boundary are always dense. See **Locked
contrasts**.

### Checkpoints

A `MaskedLinear` `state_dict` is not self-describing: it cannot
reconstruct node names or layout. Pickling a `MaskedLinear`
module raises, because the mask is a parametrization; save
`state_dict`, not the module. `torch.save(spec)` pickles the
dataclass and will break when spec fields move. `to_dict()` is
the interchange. Alphabetical unit identity is unchanged after
`from_dict`. The connectivity mask stays out of `state_dict`.
`mask_digest` / `index_digest` only check that the rebuilt
layer's 0/1 pattern or packed indices match training; they do
not restore names. Pass `identity=spec.fingerprint` when
constructing layers so `state_dict` carries that digest next
to the connectivity digest. `load_state_dict` raises
`Kpnn2Error` when a present identity does not match: a rename
that preserves alphabetical position (for example
`('g1','g2')` → `('g1b','g2')`) is then a loud failure
instead of silent mis-attribution. A missing identity is not
an error (old checkpoints). There is no `layout=` parser
flag: choose `LayeredSpec.from_dict` or
`AdjacencySpec.from_dict` from `blob["spec"]["layout"]`.

```python
payload = {
    "spec": spec.to_dict(),
    "state_dict": model.state_dict(),
}
torch.save(payload, path)

blob = torch.load(path, weights_only=False)
spec = kpnn2.LayeredSpec.from_dict(blob["spec"])
# or AdjacencySpec.from_dict when blob["spec"]["layout"]
# is "adjacency"
model = Net(spec)
model.load_state_dict(blob["state_dict"])
```

---

## Typical `nn.Module` shape

There is no graph compiler and no ready-made model. Write ordinary
PyTorch:

1. `spec = kpnn2.parse_layered(edgelist)`
2. `kpnn2.PackedLinear(hop.source_index, hop.target_index,
   hop.out_features, hop.in_features,
   identity=spec.fingerprint)` for each `hop` in `spec.hops`.
   `MaskedLinear(hop.to_mask())` is the dense hatch.
3. In `forward()`, keep a `saved` dict of layer index → tensor,
   and feed each hop `kpnn2.gather_hop_inputs(saved, hop)`.
   A tied decoder is `layer.transpose()` plus
   `scatter_hop_outputs` on skip hops; add the pieces into
   the decoder `saved` dict. There is no autoencoder class.
4. Put ReLU / BatchNorm / Dropout in `forward()` yourself, after
   the hop that produced the tensor. Store the value you want
   later hops to read.
5. `col = kpnn2.align_inputs(names, spec)` then `X[:, col]`
   on the host matrix. Pre-ordered dense tensors skip this.
   Sparse host X is the caller's loop (apply the index, then
   row block → densify → device).
6. Run Captum (or another method) yourself; then
   `map_node_attributions(...)`
7. Optionally `aggregate_node_attributions(...)` to fold
   observations (and optional seeds) with a registered method.
8. Save `spec.to_dict()` next to `state_dict`. Rebuild from
   `from_dict`, then `load_state_dict`. Weights alone cannot
   reconstruct names or layout. Pass
   `identity=spec.fingerprint` on each connectivity module
   so a rename cannot load silently.

Do not add a compiled core or mutate connectivity after parse.
See **What kpnn2 is NOT** (mutable graph). `copy.deepcopy` of
this module shape succeeds.

The Python distribution and import name are **`kpnn2`**.
Do not rename them.

---

## Repository layout

```
src/kpnn2/
  __init__.py                 # public exports only
  _parse.py                   # parse_layered
  _parse_adjacency.py         # parse_adjacency
  _spec.py                    # LayeredSpec, Hop, Skip
  _adjacency_spec.py          # AdjacencySpec
  _masked_linear.py           # MaskedLinear
  _packed_linear.py           # PackedLinear
  _constraint.py              # constraint= validation for both linears
  _generator.py               # generator= validation; isolate Linear init
  _packed_multihead_attention.py  # PackedMultiheadAttention
  _gather.py                  # gather_hop_inputs, scatter_hop_outputs
  _align.py                   # align_inputs
  _attributions.py            # map_node_attributions
  _aggregation/               # aggregate_node_attributions
    _registry.py              # method decorator and table
    _dispatch.py              # dispatcher
    _bind.py                  # labels onto observation
    _methods/                 # one module per method
  _errors.py                  # Kpnn2Error
  _mask_tensor.py             # float32 connectivity copies
  _layout.py                  # node name -> units on an axis
  _identity.py                # opaque checkpoint identity
  _serialize.py               # private spec edges, dicts, fingerprints

tests/
  api/                        # public import surface
  module/                     # unit tests per primitive
  controls/                   # live-path, importance, unrolled-adjacency
  integration/                # real tabular task
  manual/                     # Colab GPU/TPU smoke; not pytest

dev/
  docs_notebooks.py           # tutorials in CI; literature opt-in
  docs_readme_figures.py      # MkDocs hook; local README figures

AGENTS.md                     # how to work; read first
CONTEXT.md                    # this file (product contract)
README.md
CHANGELOG.md                  # notable API / core only; see below
docs/
  reference/                  # api.md + one page per public name
    aggregation/              # one page per method; not public names
  supported.md                # architecture-family support table
  packed_linear.md            # PackedLinear; tutorials use it on hops
  how_we_test.md              # test overview; scientific + technical
  literature/                 # frozen paper notebooks; not CI-executed
  literature/fortelny-bock-2020.ipynb
  feedforward-example.ipynb   # feedforward DAG; PackedLinear on hops
  cyclic-graph-example.ipynb  # cyclic graph; AdjacencySpec
  time-series-example.ipynb   # sequence x_t; AdjacencySpec
  transformer-example.ipynb   # PackedMultiheadAttention walkthrough
  fig_gen/                    # figure generators; write to figures/
  fig_gen/correctness/        # scientific-correctness icons
  figures/
  figures/correctness/
```

Generated local/CI outputs go under gitignored `out/`:
MkDocs `out/site`, coverage `out/htmlcov` and
`out/coverage.xml`, wheels `out/dist`. Do not write `site/`,
`htmlcov/`, or `dist/` at the repo root. `python -m build`
needs `-o out/dist`; there is no pyproject key for that
outdir. `__pycache__/` stays next to `.py` files.

`src/kpnn2/__init__.py` is the **only** public import path. Users
write `import kpnn2` or `from kpnn2 import MaskedLinear`, never
`from kpnn2._masked_linear import MaskedLinear`. Examples and
docs use `import kpnn2` and qualify names as `kpnn2.parse_layered`,
not `import kpnn2 as k2`.

Every implementation module carries a leading underscore and is
private. Private modules may be renamed, split, merged, or promoted
to subpackages (for example `_parse.py` → `_parse/`) without a
breaking change, as long as `kpnn2.__all__` and the documented
signatures stay identical. Do not add a second public path for a
symbol that `__init__.py` already exports.

`docs/reference/` therefore points mkdocstrings at the façade paths
(`::: kpnn2.MaskedLinear`), not at module paths. Each public name
has its own page; `docs/reference/api.md` is the grouped index.
Aggregation method pages under `docs/reference/aggregation/`
render the private method function; they are not public names.
Tests may import private modules directly to reach internal helpers.

---

## Dependencies

**Core:** `torch`, `pandas`, `numpy`, `xarray`, Python `>=3.10`.
`xarray>=2024.11`. No CalVer upper bound; xarray's own
`Requires-Python` selects a 3.10-capable wheel on 3.10
(last line `2025.6`). Do not add a sliding ceiling unless
a tested release breaks `DataArray` construction.

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

## Changelog

`CHANGELOG.md` is for **important API and core changes**
only, written concisely. It is not a diary of the
repository.

**Do add a line** when callers must know: a new public
name, a removed or renamed public name, a breaking
signature or behavior change, or a core-dependency /
contract change. Keep the bullet short (the name and
what it is). A new public primitive is the bar;
`PackedMultiheadAttention` is an example, not a
version pin.

**Do not add a line** for documentation, examples,
notebooks, site nav, wording, spelling, architecture
tables, tests, CI, refactors with no user-visible
behavior change, or any other small edit. A new docs
page is not a changelog event. Clarifying that sequence
models use `MaskedLinear` / `PackedLinear` and a user
time loop is not a changelog event.

If the change is not notable by that bar, **leave
`CHANGELOG.md` untouched**. Do not pad Unreleased. Do
not write "also updated docs." Public-API lockstep still
updates `__init__.py`, this file, `docs/reference/`, and
`tests/api/test_public_api.py` when the export set or a
public signature moves; that lockstep does **not** by
itself justify a changelog line.

---

## Docs terminology (Concepts, glosses, abbreviations)

Three homes, one question each. Do not add a fourth. A first-use
gloss is not a home: it is a pointer at one of these, deliberately
too short to define anything.

| Home | Answers | Examples |
|------|---------|----------|
| `docs/concepts.md` | What is the idea? | hop, spec, live edge |
| `docs/includes/abbreviations.md` | What do the letters mean? | GEMM, GNN |
| Docstring → `docs/reference/` | What is the exact contract? | `Hop.source_layers` |

**Concepts page.** Ordered by the pipeline (graph → edgelist →
node → DAG → parser → spec → layer → hop → skip edge → packed
indices → live edge → state vector → attribution → KPNN), never
alphabetically: the terms are a dependency chain, not a set. One
`##` per term so every entry is a link target. It carries the
concept; the docstring carries the contract.

**First use per page gets a short gloss plus a link to the
anchor; every later mention on that page is bare.** Pages are
meant to be read start to finish. Do not gloss or link the same
term twice on one page.

`README.md` is the exception to first-use: its overview
paragraphs are deliberately pre-terminology, so its links start
after the **Concepts** pointer that follows the edgelist and
graph definitions, not at a term's literal first mention.

**A gloss is a handhold, not a definition** — just enough that a
reader does not stall ("an edge the graph actually has", "one
unit per node", "everything entering one layer"). It must never
restate the entry.

**A gloss must not sit inside a list, split a fixed phrase, or
hang off a negated noun.** "the parsers, masks, packed layers — two parallel index
lists — alignment, and checkpoints" breaks the list it interrupts,
"one packed hop — everything entering one layer — per layer"
splits the phrase "one hop per layer" the reader is still holding,
and "not hop rectangles, one packed block per layer" glosses the
thing the sentence is rejecting. Where no natural slot exists,
link without glossing and let the entry carry it; where a slot
exists, prefer em-dashes over commas so the aside cannot be read
as another list item.

**Zero-edit test (governs glosses and abbr strings alike):**
changing a definition on the Concepts page must not make any
other file wrong. If an edit elsewhere would become *wrong*
rather than merely terse, it was a definition in disguise and is
too long.

**How much gloss depends on the term:**

- **Coinages** a PyTorch user cannot guess — hop, live edge,
  packed indices, skip edge, spec, state vector — get gloss and
  link.
- **Borrowed words that are narrower here** — graph, node,
  layer, attribution, parser — get the link; gloss only where
  the local meaning differs from the obvious one (e.g. layer,
  because depth is longest-path).

**Abbreviations.** `abbr` text is the **expansion only** when the
term has a Concepts entry (DAG, KPNN). One orienting clause is
allowed only when it has **no** entry (GNN, GEMM), because that
clause is then the whole explanation the reader gets. Do not add
an abbreviation whose expansion tells a reader of this package
nothing (`NN`, `RAM`).

**A term may have both, but never stack them on one word.** Link
the spelled-out phrase and let the acronym carry the tooltip:

```markdown
a [directed acyclic graph](concepts.md#dag) (DAG)
```

Not `[DAG](concepts.md#dag)`, which renders `<a><abbr>` with two
competing affordances. For these terms the spelled-out phrase is
the gloss; add no further aside. Where the acronym appears bare and
spelling it out would be clumsy (`parse_layered()` is DAG-only),
add no link on that page and let the tooltip stand alone.

**An unlinked entry is not a dead entry.** The page is read
start to finish, so the chain must stay complete even where
nothing points in. `edgelist` is introduced on the landing page
because it is that central; the entry still carries it, so that is
a preview, not a fourth home, and it needs no inbound link.

**Link form by source:** `concepts.md#anchor` from a `docs/*.md`
page, `../concepts/#anchor` from a `docs/*.ipynb` notebook,
`../../concepts/#anchor` from `docs/literature/*.ipynb`, and
`docs/concepts.md#anchor` from `README.md` (include-markdown
rewrites it). Do not link from docstrings into narrative docs.

**`abbr` reaches `.md` pages only.** mkdocs-jupyter renders
notebooks outside the markdown pipeline, so acronyms in
notebooks get no tooltip; that is accepted. Do not post-process
rendered notebook HTML to inject them — it would match inside
code cells and outputs. Notebooks still follow the gloss-and-link
rule.

**These rules are enforced, not aspirational.**
`validation.links.anchors: warn` in `mkdocs.yml` makes
`mkdocs build --strict` fail on a link to a missing anchor — but
only for `.md` pages, because mkdocs never sees notebook links.
`test_notebook_concepts_links_resolve` in
`tests/module/test_docs_notebooks.py` covers the notebooks.
Renaming a Concepts heading therefore breaks CI rather than
silently 404-ing eleven pages; rename the anchor and its inbound
links together.

Build config, both traps found the hard way: `pymdownx.snippets`
needs `base_path: [docs]` or `auto_append` silently renders
nothing, and `docs/includes/` needs `exclude_docs` or the
snippet is published as an empty page and lands in `sitemap.xml`
(`not_in_nav` only silences the nav warning). Both keys, plus
`validation`, need mkdocs >= 1.6, declared in the `docs` extra.

**"Graph" never means** PyTorch's autograd graph, nor a graph as
*data* in the GNN sense. It is the prior wiring that becomes the
architecture. The word stops at the parser: what comes out is a
**spec**, a chosen layout of that graph.

---

## Agent process

Process, commands, and conventions live in `AGENTS.md`.
Read that file first. This file is the product contract.
Do not paste either file into replies or into other rules.
Product locks stay in the sections above; do not copy them
into `AGENTS.md`.
