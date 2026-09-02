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
Hop masks and `MaskedLinear` stay dense float32 tensors times
`F.linear`. `PackedLinear` is a 1-D dense `weight` of length
`nnz` plus `index_add` on ordinary dense tensors. Not
`torch.sparse`, not COO/CSR storage, not sparse mm.
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
`gather_hop_inputs`, and `map_node_attributions` take and
return ordinary dense `torch.Tensor`s. Captum is not in
this package; when the caller runs it, that call stays
dense (no sparse IG here, and none to add).

**`align_inputs` is the dense-table path.** It maps a named
pandas DataFrame onto `spec.input_nodes` as a dense
`float32` CPU tensor of **all rows**. It is not a minibatch
helper and not the sparse-host path. Do not convert sparse
AnnData (or a scipy sparse matrix) to a DataFrame and pass
it through `align_inputs`: that densifies the whole matrix.
A caller with sparse host X column-aligns on the sparse
layout themselves, densifies only each row block, then
feeds a dense tensor to the model. Pre-ordered dense
tensors already skip `align_inputs`.

Do not add AnnData, scipy sparse, a sparse-preserving
`align_inputs`, minibatching, or device-copy helpers to
this package unless a later prompt asks.

---

## What this package does

1. **Parse:** `parse_layered()` reads a pandas DataFrame with columns
   `source` and `target` only, validates a layered DAG, and returns a
   `LayeredSpec`. `parse_adjacency()` reads the same table and
   returns an `AdjacencySpec` instead: packed source/target indices
   over every node, cycles and self-loops allowed. It never
   allocates an `(n, n)` tensor. The user picks the layout; a DAG
   is valid input to both.
2. **Specify:** `LayeredSpec` holds named nodes by layer and one
   `Hop` per layer after the first. A hop's mask carries **every**
   edge entering its layer, skips included. `AdjacencySpec` holds
   all node names, packed edge indices, and the input/output
   positions in the state vector. Neither constructs an
   `nn.Module`.
3. **Build:** The user writes a PyTorch `nn.Module` using one
   `MaskedLinear(spec.hops[i].mask)` per hop,
   `gather_hop_inputs(saved, hop)` to assemble that hop's input,
   and their own activations, norms, loops, and heads. On an
   `AdjacencySpec` the large-n path is
   `PackedLinear(spec.source_index, spec.target_index, n, n)`
   with `n = len(spec.nodes)`; that never allocates `(n, n)`.
   `MaskedLinear(spec.to_mask())` densifies and remains valid
   for small graphs. The recurrence is the user's `forward()`.
4. **Align:** `align_inputs()` maps a named DataFrame onto
   `spec.input_nodes` as a dense `float32` CPU tensor of all
   rows. Pre-ordered dense tensors go straight to the model.
   Sparse host matrices are not a kpnn2 input type; see
   **Locked contrasts**.
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
- **Not a data-residency layer.** No minibatcher, no device
  policy, no "keep X sparse" helper, and no rule that moves
  the full feature matrix to GPU. Host-sparse storage (for
  example AnnData `.X` as scipy CSR) is valid in the
  **caller's** loop; this package does not implement that
  path. See **Locked contrasts**.
- **Not Captum.** No Captum import anywhere in the library. Attribution
  mapping is name alignment only.
- **Not AnnData (v1).** No `anndata` support in `align_inputs`.
  Callers may keep sparse AnnData in their own code; do not
  add AnnData here to enable that.
- **Not a time machine.** `parse_adjacency` accepts cycles and
  self-loops, but nothing here unrolls time, picks a step count, or
  re-injects inputs between steps. Recurrence is user `forward()`
  over `PackedLinear(...)` or `MaskedLinear(spec.to_mask())`.
  `parse_layered` stays DAG-only and still raises `Kpnn2Error`
  on a cycle.
- **Not pseudo-node expansion.** A skip edge is a column of its
  target's hop mask, never a dummy neuron and never an extra
  channel inserted into an intermediate layer.
- **Not sparse-tensor accelerated.** This is the **graph**
  axis, not a ban on sparse feature matrices in the caller's
  host RAM. Connectivity is sparse in the graph.
  `AdjacencySpec` stores O(edges) index tuples; hop masks and
  `MaskedLinear` stay dense (`parametrize` + `F.linear`).
  `PackedLinear` is a 1-D dense weight of length `nnz` plus
  `index_add` on ordinary dense tensors; it is not
  `torch.sparse`, COO/CSR, or sparse matmul. Sparse tensor
  formats and sparse mm are **not planned**. Do not fold packed
  into `MaskedLinear`. `PackedLinear` is for RAM when
  `n_nodes` is large; it is not a better default. Correctness,
  ease of maintenance, and explainability of the code outrank
  memory and speed. Host-sparse scipy CSR in a caller loop
  does not violate this bullet.

---

## Package philosophy

**Primitives, not a compiled container.** `kpnn2` owns edgelist
parsing, mask tensors, hop input assembly, named I/O alignment,
and attribution column names. The user owns `nn.Module.forward()`,
call order, nonlinearities, and training.

**Two sparsity axes.** Graph connectivity is kpnn2's: always
dense compute, sparse only as "which edges exist." Feature
matrix X is the user's: sparse host storage is allowed in
**their** code; tensors that enter `forward()` are always
dense. See **Locked contrasts**. Do not add `torch.sparse`
kernels to "support sparse X." Do not route sparse X through
`align_inputs`.

**Correctness over speed.** Hop masks and `MaskedLinear` stay
dense float32 tensors times `F.linear`, not `torch.sparse`
layouts. `PackedLinear` is a 1-D weight plus `index_add`
(ordinary dense tensors, length `nnz`); it never uses
`torch.sparse` / COO / CSR / sparse mm and never densifies
inside the module. `MaskedLinear` stays the dense GEMM
default. `PackedLinear` is for RAM when `n_nodes` is large;
it is not a better default, and must not be folded into
`MaskedLinear`. `AdjacencySpec` itself is O(edges);
`to_mask()` is the allocating dense escape hatch.
Sparse-tensor acceleration is **not planned**, now or later.
This package values correctness, ease of maintenance, and
explainability of the code more than memory and speed
performance.

Division of labor:

| Layer | Owner |
|-------|--------|
| Edgelist → `LayeredSpec` (ranks, hop masks, skip metadata) | kpnn2 |
| Edgelist → `AdjacencySpec` (nodes, packed edge indices) | kpnn2 |
| `MaskedLinear` (fixed mask, dense GEMM) | kpnn2 |
| `PackedLinear` (1-D weight per live edge, `index_add`) | kpnn2 |
| `gather_hop_inputs` (source axis of one hop) | kpnn2 |
| Named DataFrame → dense CPU tensor (`align_inputs`) | kpnn2 |
| `forward()`, activations, norms, heads, call order | User (PyTorch) |
| Training and evaluation | User (PyTorch) |
| Host feature layout (dense table vs sparse AnnData `.X`) | User |
| Minibatch slice → densify that block → device copy | User |
| Captum / other attribution algorithms | User |
| Tensor → named `xarray.DataArray` | kpnn2 |
| Sparse-tensor kernels (`torch.sparse`, sparse mm) | Not planned |

---

## Public API (only these names)

Exported from `kpnn2` (`src/kpnn2/__init__.py`):

| Symbol | Role |
|--------|------|
| `parse_layered` | Edgelist DataFrame → `LayeredSpec` (DAG only) |
| `parse_adjacency` | Edgelist DataFrame → `AdjacencySpec` (cycles allowed) |
| `LayeredSpec` | Frozen structural dataclass (layers, hops, skip metadata) |
| `Hop` | One layer's incoming mask (see below); exported because `spec.hops` uses it |
| `Skip` | One skip-edge record (see below); exported because `spec.skips` uses it |
| `AdjacencySpec` | Frozen structural dataclass (nodes, packed edge indices; no stored square) |
| `MaskedLinear` | `nn.Module`: masked linear layer |
| `PackedLinear` | `nn.Module`: one trainable scalar per live edge |
| `gather_hop_inputs` | Saved layer tensors + `Hop` → that hop's input tensor |
| `align_inputs` | Named DataFrame → `float32` input tensor |
| `map_node_attributions` | Layer tensor → labeled `xarray.DataArray` |
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
and must not change parsing. There is no `initial_weight` or
`constraint` support.

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
are tuples. Do not reassign fields. Mask tensors are **plain
`torch.Tensor`** and are not write-protected: document them as
read-only, do not enforce it. `copy.deepcopy` of a `LayeredSpec`
succeeds; copied masks stay float32 and do not share storage
with the original.

| Field | Type | Meaning |
|-------|------|---------|
| `input_nodes` | `tuple[str, ...]` | In-degree 0 nodes, alphabetical. Tensor column order. |
| `output_nodes` | `tuple[str, ...]` | Out-degree 0 nodes, alphabetical. |
| `hidden_nodes` | `tuple[str, ...]` | Neither input nor output, alphabetical. |
| `layer_nodes` | `tuple[tuple[str, ...], ...]` | `layer_nodes[i]` = names at depth `i`, alphabetical. Index 0 is the first layer. |
| `layer_dims` | `tuple[int, ...]` | `layer_dims[i] == len(layer_nodes[i])`. |
| `hops` | `tuple[Hop, ...]` | One incoming mask per layer after the first (see below). |
| `skips` | `tuple[Skip, ...]` | Skip edges with depth gap `> 1`, as metadata (see below). |

### `Hop` records

One hop per layer after the first. A hop is exactly what one
`MaskedLinear` computes, and its mask holds **every** parent of
its target layer. There is no second mechanism for edges that
span layers.

| Field | Type | Meaning |
|-------|------|---------|
| `target_layer` | `int` | Depth this hop produces; `>= 1` |
| `source_layers` | `tuple[int, ...]` | Depths it reads, ascending, all `< target_layer` |
| `source_dims` | `tuple[int, ...]` | Units per entry of `source_layers`; sums to `mask.shape[1]` |
| `source_nodes` | `tuple[str, ...]` | Node names of the mask columns, source layers concatenated |
| `mask` | `torch.Tensor` | Connectivity (see below) |

`Hop.column_offsets` is a derived property: the first mask column
of each entry of `source_layers`.

- `len(hops) == len(layer_nodes) - 1` and
  `hops[i].target_layer == i + 1`.
- `source_layers` lists only layers that really feed the target.
  `target_layer - 1` is always one of them, because longest-path
  ranking gives every node a parent one layer down.
- `hops[0].source_layers == (0,)` always: layer 1 can only have
  layer-0 parents. So an `align_inputs` tensor feeds `hops[0]`
  directly, with no gathering.
- A graph with no skip edges gives every hop a single source
  layer, and then `hops[i].mask` is the plain adjacent-hop mask
  from layer `i` to layer `i+1`.
- Dtype: `torch.float32`.
- Shape: `(layer_dims[target_layer], sum(source_dims))`. This
  matches `nn.Linear.weight` layout `(out_features, in_features)`.
- Rows are named by `layer_nodes[target_layer]`, columns by
  `source_nodes`. An entry is `1.0` iff there is an original
  edgelist edge from the node naming that column to the node
  naming that row, and `0.0` otherwise.
- **Every edgelist edge is a one in exactly one hop mask**, the
  one of its target layer. Summing all the ones over all hops
  gives the edge count. This invariant is what makes an edge
  impossible to drop silently: applying a hop applies all of its
  target's parents at once.
- Because a hop mask carries every parent, the per-row degree
  `MaskedLinear` initializes from is the unit's real fan-in,
  skips included.
- Ordinary `torch.Tensor`, never a subclass. Writes are not
  blocked; treat the tensors as read-only and rebuild from the
  edgelist to change wiring.
- `copy.deepcopy` succeeds. Copied masks keep the same values,
  stay float32, and do not share storage with the original.
- `MaskedLinear(spec.hops[i].mask)` stores an independent copy, so
  a write on one side does not change the other.

### `Skip` records

Each skip is a frozen dataclass `Skip`. It is **metadata only**:
the edge itself is already a one in
`hops[target_layer - 1].mask`, exactly like an adjacent edge.
Read `skips` to report or inspect which prior-knowledge edges
span layers; nothing in a forward pass needs it.

| Field | Type | Meaning |
|-------|------|---------|
| `source` | `str` | Source node name |
| `target` | `str` | Target node name |
| `source_layer` | `int` | Depth of `source` |
| `target_layer` | `int` | Depth of `target`; `target_layer - source_layer > 1` |
| `source_index` | `int` | Index of `source` in `layer_nodes[source_layer]` |
| `target_index` | `int` | Index of `target` in `layer_nodes[target_layer]` |

Every original edgelist edge with depth gap `> 1` appears once in
`skips`. Adjacent edges (gap `== 1`) never appear in `skips`.
Membership in `skips` changes nothing about how the edge is
computed.

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

`parse_layered(spec.to_edgelist())` reconstructs the same
`input_nodes`, `output_nodes`, `hidden_nodes`, `layer_nodes`,
`layer_dims`, hop `source_layers` / `source_dims` /
`source_nodes`, and hop masks (`torch.equal` on each
`hop.mask`). Skip *tuple* order may follow the sorted
edgelist rather than the original parse input; the skip
*set* of `(source, target, source_layer, target_layer,
source_index, target_index)` matches.

---

## Skip connections (no pseudo nodes, no second mechanism)

A skip `A → H2` that jumps one or more layers is the source
being an extra parent of the target. It is **not** a dummy
channel, **not** a separate module, and **not** a learnable
scalar added after the fact. It is a column of
`hops[target_layer - 1].mask`, and its weight is an ordinary
entry of that layer's `MaskedLinear`.

The consequences are the point of this design:

- A hop that has skip parents reads more than one layer, so its
  input is those layers concatenated. `gather_hop_inputs` builds
  that tensor and raises `Kpnn2Error` if a needed layer is
  missing, so a forgotten activation is an error rather than a
  quietly dropped edge.
- The unit's fan-in for the degree-aware init counts skip
  parents, because they are in the same mask row.
- The skip weight is a full weight, not a tied scalar, and it
  starts from the same degree-aware draw as every other edge.
- There is no skip bias; unit bias stays on `MaskedLinear`.
- Nothing undoes ReLU and nothing modifies saved tensors. The
  source enters through the target's own weight matrix.

`kpnn2` owns the mask layout and the gather. The user owns call
order and nonlinearities.

```python
saved = {0: x}
for index, hop in enumerate(spec.hops):
    sources = kpnn2.gather_hop_inputs(saved, hop)   # concat, checked
    h = self.hops[index](sources)                # MaskedLinear
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
allowed**. This is the layout for recurrent networks; the
recurrence itself is user `forward()` code.

`parse_adjacency` must not instantiate an `nn.Module`, unroll
time, choose a step count, or re-inject inputs between steps.
It must not allocate an `(n, n)` tensor.

### `AdjacencySpec` fields

Frozen dataclass, same rules as `LayeredSpec`: no reassignment,
sequences are tuples. There is no stored mask tensor and no
densifying `mask` property. `copy.deepcopy` succeeds and copies
the index tuples; two `to_mask()` results do not share storage.

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
   `len(spec.input_nodes)` columns, the state vector is `n` wide.
   The inputs are scattered into the state vector via
   `spec.input_index`. In the layered case an aligned tensor
   feeds `hops[0].mask` directly; here it does not.
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

x = kpnn2.align_inputs(df, spec)         # width len(input_nodes)
state = torch.zeros(x.shape[0], n)
state[:, spec.input_index] = x        # required, see above
state = torch.relu(core(state))       # one step; loop as needed
logits = state[:, spec.output_index]
```

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

| Key | Value |
|-----|--------|
| `kpnn2_spec` | Integer `1` (schema version). |
| `layout` | `"layered"` for `LayeredSpec`, `"adjacency"` for `AdjacencySpec`. Must not be omitted. |
| `edges` | List of `[source, target]` lists (JSON-safe, not tuples), same order as `to_edgelist()` rows / `canonical_edges`. |

Unknown extra keys on an otherwise valid payload are ignored
(forward compatible). `to_dict()` does not emit extra keys.

`LayeredSpec.from_dict(payload)` calls `parse_layered` on a
DataFrame built from `payload["edges"]`.
`AdjacencySpec.from_dict` calls `parse_adjacency`. Hops and
masks are not hand-rebuilt. A layout mismatch (an adjacency
dict into `LayeredSpec.from_dict`, or the reverse) raises
`Kpnn2Error` naming the mismatch.

`from_dict` also raises `Kpnn2Error` when: `payload` is not a
dict; `kpnn2_spec` is missing or not `1`; `layout` is missing
or not `"layered"` / `"adjacency"`; `edges` is missing, is not
a sequence of pairs, or a pair is not two nonempty
string-convertible names.

`fingerprint` is a property: the SHA-256 hex digest (64
lowercase hex characters) of
`json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"),
ensure_ascii=False).encode("utf-8")`. Do not use Python
`hash()`. `parse_layered(edgelist).fingerprint` equals
`parse_layered(spec.to_edgelist()).fingerprint`. The same DAG
parsed layered vs adjacency yields different fingerprints
because `layout` differs. Adding, removing, or renaming a
node, or changing an edge, changes the fingerprint.

These three names are methods / a property on the spec
classes. They are not package-level exports. There is no
module-level `spec_from_dict`. Pickle and `torch.save` of the
dataclass are **not** the supported interchange.

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
- The mask is applied through
  **`torch.nn.utils.parametrize.register_parametrization`** on
  the parameter named `weight`. That is the blessed PyTorch
  mechanism for "the effective weight is a function of a stored
  parameter". Do not replace it with a hand-rolled second
  parameter name.
- `layer.weight` is therefore the **effective masked weight**,
  recomputed on access. It is not an `nn.Parameter`: in-place
  writes to it are discarded. Assigning
  (`layer.weight = w`, under `torch.no_grad()`) copies `w` into
  the trainable tensor; the mask then hides the entries it
  zeroes.
- The trainable tensor is
  `layer.parametrizations.weight.original`, same shape as
  `mask`. Masked-out entries may be nonzero there and never
  reach the output. There is no `raw_weight`.
  `model.parameters()` includes that tensor. A param-group
  filter that uses `"weight" in name` matches it;
  `name.endswith(".weight")` does not. Do not describe this
  as "the usual `weight` name."
- `state_dict` keys are `parametrizations.weight.original`,
  optional `bias`, and `mask_digest`. `mask` stays out: it
  remains a non-persistent float32 buffer. `mask_digest` is a
  1-D CPU `torch.uint8` tensor of length 32, the SHA-256 of
  the live mask's float32 C-contiguous bytes at save time, not
  a registered persistent buffer. `load_state_dict` raises
  `Kpnn2Error` when a present digest does not match the live
  mask, and does not load the weights. A missing digest is not
  an error, even with `strict=True`. This catches same-shape
  rewiring, not a rename that leaves the 0/1 pattern unchanged
  (that is `spec.fingerprint`). `repr` reports
  `ParametrizedMaskedLinear` (PyTorch swaps in a subclass to
  install the `weight` property); `isinstance(layer,
  MaskedLinear)` stays `True`, and `extra_repr` reports
  `in_features`, `out_features`, `bias` as `nn.Linear` does.
  Pickling the module object raises, as for any parametrized
  module; `copy.deepcopy` and `state_dict` work. Do not call
  `remove_parametrizations` on `weight`: that drops the mask.
- Register `mask` as a **non-persistent buffer** (not a
  parameter, not in `state_dict`), `float32`, and a **plain
  `torch.Tensor`**. It lives on the parametrization module, so
  its `named_buffers` key is `parametrizations.weight.0.mask`,
  and `layer.mask` is a property onto it that also accepts
  assignment. LayeredSpec masks stay float32. After
  `Module.half()`, `.to(dtype=torch.bfloat16)`, or `.double()`,
  `layer.mask.dtype` is still `float32`: the stored mask does
  not follow the module floating dtype.
  `MaskedLinear(spec.hops[i].mask)` stores an independent copy,
  so later writes to that mask do not reach the layer.
  Rebuild from the edgelist / `LayeredSpec` to change wiring.
  Nothing blocks a write to `layer.mask`; it is documented
  read-only, like any PyTorch buffer. `copy.deepcopy` of a
  `MaskedLinear` (and of a user `nn.Module` that holds
  `MaskedLinear` layers and a `LayeredSpec`) succeeds.
  Parameters on the copy are distinct objects. Copied masks
  stay float32.
- Optional `bias`: shape `(out_features,)`. If `bias=False`, no bias
  parameter.
- Forward multiplies in the parameter dtype:
  `weight = original * mask.to(dtype=original.dtype,
  device=original.device)`, then
  `Y = F.linear(X, weight, bias)`.
  Equivalently `Y = X @ (W ⊙ M).T + b` with `M` cast to `W`'s
  dtype/device. This is why `.half()` / bfloat16 / `.double()`
  work like `nn.Linear`. In the common float32 case that `.to`
  returns the buffer itself, so forward allocates nothing extra
  for the mask. The multiply is **dense** on purpose.
  `x` is an ordinary dense activation tensor, not a sparse
  host feature matrix. Sparse-tensor acceleration is not
  planned; see **Locked contrasts**.
- The forward path holds no tensor subclass, so
  `torch.compile(layer, fullgraph=True)` traces it without a
  graph break, parametrization included. Keep it that way.
- **Degree-aware init:** for each output row `j`,
  `fan_in = int(mask[j].sum())` (count of 1s in that row).
  Use that `fan_in` for kaiming/uniform scale of that row (and for
  bias, use a documented rule: e.g. bias bound from that row's
  `fan_in`, or from mean fan_in; prefer **per-row fan_in** for
  weights). If `fan_in == 0`, leave that row at 0 and use bias
  bound 0. `reset_parameters` writes into
  `parametrizations.weight.original`, row by row, in that
  order. A different draw count shifts the torch RNG stream
  and which trained-tier seeds pass; those controls now test
  a pass rate over 30 seeds, not a 5-seed window.
- Do **not** use full `in_features` as `fan_in`.
- No edge constraints, no `initial_weight`, no softplus.
- Masked-out weights still exist as parameters but are multiplied
  by 0 in the forward pass.

Typical construction: `MaskedLinear(spec.hops[i].mask)`.

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
- `weight` is an `nn.Parameter` of shape `(nnz,)`. No
  `parametrize`. No dense `(out, in)` `layer.weight`.
  The parameter name is `weight`.
- Index buffers `source_index` and `target_index` are
  persistent so the module round-trips. They stay integer
  after `.half()` / bfloat16 / `.double()`; `weight` and
  `bias` follow the module floating dtype like `nn.Linear`.
- Forward, `x` shape `(..., in_features)`:

  ```text
  contrib = x[..., source_index] * weight
  y = zeros(..., out_features)  # same batch dims, dtype, device
  y.index_add_(-1, target_index, contrib)
  if bias is not None:
      y = y + bias
  ```

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
  state vector each step.
- `extra_repr` reports `in_features`, `out_features`,
  `nnz`, and `bias`.
- `state_dict` keys are `weight`, optional `bias`,
  `source_index`, `target_index`, and `index_digest`.
  `index_digest` is a 1-D CPU `torch.uint8` tensor of
  length 32, the SHA-256 of the int64 C-contiguous bytes
  of `source_index`, then `target_index`, plus
  `out_features` and `in_features` as fixed-width
  integers so a reshape cannot collide. It is not a
  registered persistent buffer. `load_state_dict` raises
  `Kpnn2Error` when a present digest does not match, and
  does not load the weights. A missing digest is not an
  error, even with `strict=True`. `copy.deepcopy` works.
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
)
```

`MaskedLinear(spec.to_mask())` remains valid for small
graphs. Do not change `LayeredSpec` or hop masks to use
this layer. `PackedLinear` does not add model capacity
relative to `MaskedLinear`: dead edges already did not
affect training. `MaskedLinear` remains better for usual
hops (dense GEMM, `(out, in)` weight).

Do not name this `SparseMaskedLinear`, `SparseLinear`, or
`PackedMaskedLinear`.

---

## `gather_hop_inputs(saved, hop)`

The source axis of one hop. Call it in `forward()` just before
`MaskedLinear(hop.mask)`; it sits between hops. This is the only
thing the layered layout needs beyond `MaskedLinear`, and it
holds no parameters. It does not inject values into the previous
layer and does not pick skip nodes by name: it concatenates
**whole** source layers. The hop mask zeros columns that are not
edges.

```text
gather_hop_inputs(saved, hop) -> torch.Tensor
```

- `saved` maps layer index → that layer's activation, width
  `layer_dims[i]`. Only `hop.source_layers` are read, and they
  are not modified. Extra keys are ignored.
- `hop` is a `Hop` from `spec.hops`.
- Returns the source layers concatenated on the last axis in
  `hop.source_layers` order, shape `(..., hop.mask.shape[1])`,
  ready for `MaskedLinear(hop.mask)`.
- A hop with one source layer (adjacent, no skips) returns that
  saved tensor itself, without a copy. A hop with skips
  concatenates the previous layer plus older layers beside it.
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

## `align_inputs(data, spec)`

`spec` is a `LayeredSpec` **or** an `AdjacencySpec`. Only
`spec.input_nodes` is read, so the DataFrame rules below are
identical for both. Anything else raises `Kpnn2Error`.

Returns `torch.float32` tensor of shape
`(n_samples, len(spec.input_nodes))`. The tensor is dense and
lives on CPU. This function materializes **every row** of the
DataFrame. It is not a minibatch API, not a device-copy
helper, and not the sparse-host path. See **Locked
contrasts**.

**Width differs by layout.** For a `LayeredSpec` that width is
`layer_dims[0]`, and `hops[0]` reads layer 0 alone, so the tensor
feeds `MaskedLinear(spec.hops[0].mask)` directly with no
gathering. For an `AdjacencySpec` it is **not** the state width:
`to_mask()` is `(n, n)` over every node, while the aligned tensor
is only `len(input_nodes)` wide. Scatter it into the `n`-wide
state vector via `spec.input_index` before calling
`MaskedLinear(spec.to_mask())`:

```python
x = kpnn2.align_inputs(df, spec)
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
- Pre-ordered dense tensors go **straight to the model**. Users who
  need alignment pass a DataFrame.

**Not supported in v1:** AnnData, numpy arrays, dicts of columns,
scipy sparse matrices. Do not add them here so a caller can
keep host X sparse. That caller column-aligns on the sparse
layout themselves and densifies only each row block before
the model. Passing `adata.to_df()` (or any full densify)
into `align_inputs` is exactly the silent full densify
**Locked contrasts** forbids.

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
function only attaches spec names to the `node` axis. The input
is already dense scores, not a host feature matrix. For the
output of `MaskedLinear(spec.hops[i].mask)` pass
`layer=spec.hops[i].target_layer`, that is `i+1`. Do not
name-map BatchNorm or other unnamed modules.

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
  `to_mask()` axis length come from `n_units`, not from
  `len(names)`.
- A hop's column axis is `concat_layouts` of its source layers'
  layouts, so a source node's block on that axis is its own
  block shifted by the widths in front of it. `source_dims` and
  `Hop.column_offsets` come from the same widths.
- Hop masks are written with `fill_block`, which sets the whole
  `(target.width, source.width)` block of an edge. At width 1
  that is one entry per edge. Adjacent and skip edges go through
  the same call, so both block-expand.
- `AdjacencySpec.source_index` / `target_index` store
  `layout.start_of` (the block start) for each original edge.
  `to_mask()` writes `1.0` at each
  `[target_index[i], source_index[i]]`.
- `Skip.source_index` and `Skip.target_index` store a **block
  start** inside their own layer.
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
            [kpnn2.MaskedLinear(hop.mask) for hop in spec.hops]
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
x = kpnn2.align_inputs(x_df, spec)
y = model(x)

# optional: user ran some attribution method themselves
da = kpnn2.map_node_attributions(
    attributions=y.detach(),
    spec=spec,
    layer=len(spec.layer_nodes) - 1,
)
```

Every edge, including `A → C` when that row is present, is
already inside a hop mask. The loop applies each hop once, so
nothing has to be remembered per skip edge.

That snippet is the dense-DataFrame path: `align_inputs`
materializes the whole table on CPU. A caller who already has
a sparse host matrix (for example AnnData `.X`) must not
densify it through `align_inputs` or DataFrame conversion.
They column-align on the sparse layout, densify only each
row block, move that dense tensor, and call `model`. Tensors
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
`mask_digest` only checks that the rebuilt layer's mask matches
training; it does not restore names. There is no `layout=`
parser flag: choose `LayeredSpec.from_dict` or
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
2. `kpnn2.MaskedLinear(hop.mask)` for each `hop` in `spec.hops`
3. In `forward()`, keep a `saved` dict of layer index → tensor,
   and feed each hop `kpnn2.gather_hop_inputs(saved, hop)`
4. Put ReLU / BatchNorm / Dropout in `forward()` yourself, after
   the hop that produced the tensor. Store the value you want
   later hops to read.
5. `x = kpnn2.align_inputs(df, spec)` when X is a named
   DataFrame. Pre-ordered dense tensors skip this. Sparse
   host X is the caller's loop (row block → densify →
   device); do not send it through `align_inputs`.
6. Run Captum (or another method) yourself; then
   `map_node_attributions(...)`
7. Save `spec.to_dict()` next to `state_dict`. Rebuild from
   `from_dict`, then `load_state_dict`. Weights alone cannot
   reconstruct names or layout.

Do not add a compiled core or mutate connectivity after parse.
`copy.deepcopy` of this module shape succeeds. Copied masks
stay float32.

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
  _gather.py                  # gather_hop_inputs
  _align.py                   # align_inputs
  _attributions.py            # map_node_attributions
  _errors.py                  # Kpnn2Error
  _mask_tensor.py             # float32 connectivity copies
  _layout.py                  # node name -> units on an axis
  _serialize.py               # private spec edges, dicts, fingerprints

tests/
  api/                        # public import surface
  module/                     # unit tests per primitive
  manual/                     # Colab GPU/TPU smoke; not pytest

scripts/
  docs_notebooks.py           # execute/repair docs notebooks

CONTEXT.md                    # this file
README.md
docs/
  packed_linear.md            # PackedLinear; tutorials stay MaskedLinear
  fig_gen/                    # figure generators; write to figures/
  figures/
```

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
- Two sparsity axes (see **Locked contrasts**). Graph
  connectivity is always dense compute in this package.
  Feature-matrix storage is the caller's. Do not collapse
  "sparse X" into `torch.sparse` kernels, and do not treat
  host-sparse scipy CSR in a caller loop as a violation of
  the graph rule.
- Do **not** add sparse-tensor acceleration (`torch.sparse`,
  COO/CSR storage, sparse mm). `MaskedLinear` stays dense
  float32 tensors times `F.linear`. `PackedLinear` is a 1-D
  dense `weight` of length `nnz` plus `index_add` on
  ordinary dense tensors; it must not densify to `(out, in)`
  inside the module, must not take a dense mask, and must
  not import `torch.sparse`. Do not fold packed into
  `MaskedLinear`. Do not replace `MaskedLinear` as the
  dense default for usual hops. `PackedLinear` is for RAM
  when `n_nodes` is large; it is not a better default. Do
  not add `parse(..., sparse=)`. Sparse-tensor formats are
  not a v1 deferral: they are not planned. Correctness,
  ease of maintenance, and explainability of the code
  outrank memory and speed.
- Do not add AnnData, scipy sparse, a sparse-preserving
  `align_inputs`, a minibatcher, or device-copy helpers so
  this package can "support sparse X." `align_inputs` stays
  a full named-DataFrame densifier on CPU. Sparse host X
  must not go through `align_inputs` (including via
  `adata.to_df()`). Callers densify only each row block and
  pass a dense tensor to the model. Do not put the full
  feature matrix on GPU inside this package; this package
  never moves X to a device.
- Do not add sparse minibatches or sparse Captum / IG.
  Module inputs stay ordinary dense tensors.
- `parse_adjacency` must not allocate an `(n, n)` tensor. Do
  not add a densifying `mask` property on `AdjacencySpec`.
  Materialize the square only through `to_mask()`. Still no
  `layout=` parser flag.
- Masks are plain `torch.Tensor`. Do not add a tensor subclass,
  a `__torch_function__` override, or any other write guard.
  A previous `FrozenMask` subclass broke
  `torch.compile(fullgraph=True)` and cloned the whole mask on
  every forward pass. Document masks as read-only instead;
  PyTorch does not write-protect buffers either.
  `tests/module/test_masked_linear.py` keeps a `fullgraph`
  regression test.
- `MaskedLinear` masks its weight with
  `torch.nn.utils.parametrize`. The public attribute is
  `weight` (effective, masked); the trainable tensor is
  `parametrizations.weight.original`. `named_parameters` /
  `state_dict` keys use that path, not `weight`. Do not
  claim suffix-`.weight` filters or "the usual weight
  name" see it. Do not reintroduce a second name such as
  `raw_weight`, do not shadow `weight` with a plain
  property, and do not make `MaskedLinear` a subclass of
  `nn.Linear` (its `__init__` signature and its own
  `reset_parameters` would fight ours).
- Do not add a high-level `LayeredNet` / convenience model unless
  a later prompt explicitly asks.
- `parse_layered` and `parse_adjacency` must not instantiate
  `nn.Module`.
- Keep the two parsers separate: no `layout=` flag, no dispatch
  on whether the graph has a cycle, and no `AdjacencySpec`
  faked as a one-layer `LayeredSpec`.
- `MaskedLinear` must not store other layers' activations. The
  user owns the `saved` dict, call order, and nonlinearities.
- Do **not** reintroduce a skip module, a per-skip parameter, or
  any second place where an edge weight can live. A skip edge is
  a column of its target's hop mask; that is what makes the edge
  count, the fan-in, and the "no silently dropped edge"
  guarantee hold. `SkipAdd` existed until 0.1.0 and was removed
  for exactly these reasons: it could be forgotten at a call
  site without any error, its per-edge scalar left skip parents
  out of the degree-aware init, it allocated one batch-sized
  temporary per skip edge, and its single scalar could not
  generalize to node width.
- One graph node is one unit in v1 (no public node width). Keep
  index arithmetic in `_layout.py`: build a `Layout`, ask it for
  slots, and write masks with `fill_block`. See "Internal unit
  layout".
- Public failures: `Kpnn2Error` only.
- After Python edits, run `python -m ruff format .` from the
  `dev` extra. Do not use a global `ruff` on `PATH`.
- Docs, README, and doctests use `import kpnn2` and
  `kpnn2.parse_layered(...)` (same for the other public names).
  Do not introduce `import kpnn2 as k2`.
- Docs tutorials (getting-started, skip-edges,
  map-node-attributions, layered vs adjacency, recurrent
  example) stay on `MaskedLinear`. `PackedLinear` has its
  own page (`docs/packed_linear.md`). Do not sprinkle
  `PackedLinear` through getting-started.
- Docs notebooks must be valid nbformat v4. Stream outputs need
  `name` (`stdout` / `stderr`); editors often drop it and
  mkdocs-jupyter then fails. Execute with
  `python scripts/docs_notebooks.py` (venv kernel, not
  `ipykernel install --user --name python3`). `mkdocs serve`
  repairs missing stream names on pre-build.
- `tests/manual/` is Colab GPU/TPU smoke, not pytest and not
  docs. Do not execute it in CI or with
  `scripts/docs_notebooks.py`. Open from GitHub via
  `dev/colab.txt`. Install from TestPyPI with `--no-deps`.
