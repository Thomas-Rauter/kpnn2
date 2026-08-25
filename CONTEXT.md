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

1. **Parse:** `parse_edgelist()` reads a pandas DataFrame with columns
   `source` and `target` only, validates a layered DAG, and returns a
   `GraphSpec`.
2. **Specify:** `GraphSpec` holds named nodes by layer, adjacent-hop
   mask tensors, and skip-edge metadata. It does not construct an
   `nn.Module`.
3. **Build:** The user writes a PyTorch `nn.Module` using
   `MaskedLinear(spec.masks[i])` for adjacent hops and their own
   activations, norms, loops, and heads.
4. **Align:** `align_inputs()` maps a named DataFrame onto
   `spec.input_nodes` as a `float32` tensor. Pre-ordered tensors
   go straight to the model.
5. **Train:** The user owns loss, optimizer, and the training loop.
6. **Map attributions:** `map_node_attributions()` labels a tensor
   at one GraphSpec layer as an `xarray.DataArray` (`node` names
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
- **Not cyclic (v1).** Cycles raise `Kpnn2Error`. Recurrence is
  user `forward()` on a future adjacency layout, not a v1 parser mode.
- **Not pseudo-node expansion.** Skip edges are metadata plus a
  residual add in user code, not dummy neurons in masks.

---

## Package philosophy

**Primitives, not a compiled container.** `kpnn2` owns edgelist
parsing, mask tensors, named I/O alignment, and attribution column
names. The user owns `nn.Module.forward()`, nonlinearities, skip
residuals, and training.

Division of labor:

| Layer | Owner |
|-------|--------|
| Edgelist → `GraphSpec` (ranks, masks, skips) | kpnn2 |
| `MaskedLinear` (frozen mask) | kpnn2 |
| `forward()`, activations, norms, heads | User (PyTorch) |
| Training and evaluation | User (PyTorch) |
| Captum / other attribution algorithms | User |
| Tensor → named `xarray.DataArray` | kpnn2 |

---

## Public API (only these names)

Exported from `kpnn2` (`src/kpnn2/__init__.py`):

| Symbol | Role |
|--------|------|
| `parse_edgelist` | Edgelist DataFrame → `GraphSpec` |
| `GraphSpec` | Frozen structural dataclass (masks, layers, skips) |
| `Skip` | One skip-edge record (see below); exported because `spec.skips` uses it |
| `MaskedLinear` | `nn.Module`: masked linear layer |
| `align_inputs` | Named DataFrame → `float32` input tensor |
| `map_node_attributions` | Layer tensor → labeled `xarray.DataArray` |
| `Kpnn2Error` | User-facing error type |
| `__version__` | Package version string |

`__all__` contains exactly these names (including `Skip` and
`__version__`). No other public symbols.

Do **not** export or implement: `compile_graph`, `customize_model`,
`interpret_model`, `align_features_to_input_nodes`, `edge_weights`,
`CompileArtifact`, backends, `ConstrainedMaskedLinear`.

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

`parse_edgelist()` enforces:

| Rule | On violation |
|------|----------------|
| Input is a `pandas.DataFrame` | `Kpnn2Error` |
| Columns `source` and `target` exist | `Kpnn2Error` |
| No missing values in `source` or `target` | `Kpnn2Error` |
| No empty-string node names | `Kpnn2Error` |
| At least one edge | `Kpnn2Error` |
| No duplicate `(source, target)` pairs | `Kpnn2Error` |
| No self-loops (`source == target`) | `Kpnn2Error` |
| Graph is a DAG (no cycles) | `Kpnn2Error` naming unranked leftover nodes |
| At least one input (in-degree 0) | `Kpnn2Error` |
| At least one output (out-degree 0) | `Kpnn2Error` |

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

Isolated nodes cannot appear: the node set is the union of `source`
and `target` values only.

---

## `GraphSpec` fields

`GraphSpec` is a frozen dataclass. It holds structure only: no
`nn.Module`, no parameters, no execution plan object. Sequences
are tuples. Do not reassign fields. Mask tensors reject in-place
writes.

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

Masks only encode **adjacent** hops. A skip `A → C` that jumps one
or more layers is **not** inserted as a dummy channel.

The user keeps the source activation in a Python variable and adds
it when the target layer is computed, using a learnable scalar (or
any user module):

```python
# spec.skips tells you source_layer, target_layer, indices
a = x                              # layer 0, node A
h = torch.relu(self.lin0(a))       # adjacent hop 0→1
c = self.lin1(h)                   # adjacent hop 1→2
c = c + self.w_skip * a            # skip A → C
```

`MaskedLinear` layers must not contain skip edges. There is no
identity overwrite and no compiler-generated node names.

If several skips exist, keep a dict of saved layer tensors (or
named slices) and apply each `Skip` when its `target_layer` is
built: add `w * saved[source]` into the target index.

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
  parameter, not in `state_dict`), `float32`. In-place writes
  and replacement raise `Kpnn2Error`. Rebuild from the
  edgelist / `GraphSpec` to change wiring.
- Trainable `raw_weight`: same shape as `mask`.
- Optional `bias`: shape `(out_features,)`. If `bias=False`, no bias
  parameter.
- Forward: `Y = F.linear(X, raw_weight * mask, bias)`.
  Equivalently `Y = X @ (W ⊙ M).T + b`.
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

## `align_inputs(data, spec)`

Returns `torch.float32` tensor of shape
`(n_samples, len(spec.input_nodes))`.

**`pandas.DataFrame`:**

- Required columns: `spec.input_nodes` (any order).
- Match column labels after converting them to strings (so integer
  column names can match string node ids).
- Extra columns are ignored.
- Missing required columns: `Kpnn2Error`.
- Duplicate column names (including after string conversion):
  `Kpnn2Error`.
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

## `map_node_attributions(attributions, spec, layer, *, dims=None, coords=None)`

Unopinionated name mapping. No Captum import. Returns
`xarray.DataArray`. Does not aggregate.

- `attributions`: `torch.Tensor`, or a non-empty tuple/list of
  equal-shaped tensors (stacked on a new `step` axis).
- `layer`: `int`, index into `spec.layer_nodes` (0-based). Stored
  as scalar coordinate `layer`.
- The `node` axis length must equal `len(spec.layer_nodes[layer])`.
  That axis gets coordinate `spec.layer_nodes[layer]` in order.
- Default dims: 1-D → `(node,)`; 2-D → `(observation, node)`; a
  stacked sequence of 2-D tensors → `(step, observation, node)`.
  Rank 3+ (except that stacked default) requires `dims=` containing
  `node` exactly once.
- `coords`: optional labels for axes other than `node` and `layer`.
- Values: detached CPU copy of the tensor. No abs/sum/mean.
- Long table: `da.to_dataframe(name="score").reset_index()`.
  Wide 2-D table: `da.to_pandas()`.
- Invalid `layer`, shape, `dims`, or `coords`: `Kpnn2Error`.

The user obtains `attributions` however they like (Captum
LayerConductance, IntegratedGradients, custom grads, etc.). This
function only attaches GraphSpec names to the `node` axis. Pass
`layer=i+1` for `MaskedLinear(spec.masks[i])`. Do not name-map
BatchNorm or other unnamed modules.

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

spec = k2.parse_edgelist(edgelist)

class Net(nn.Module):
    def __init__(self, spec: k2.GraphSpec):
        super().__init__()
        self.lin0 = k2.MaskedLinear(spec.masks[0])
        self.lin1 = k2.MaskedLinear(spec.masks[1])
        self.w_skip = nn.Parameter(torch.zeros(1))
        self.spec = spec

    def forward(self, x):
        h = F.relu(self.lin0(x))
        c = self.lin1(h)
        for skip in self.spec.skips:
            # user indexes saved activations by skip.source_layer
            c = c + self.w_skip * x
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

The skip loop above is illustrative. Real code should add
`w * saved_source` into `layer_nodes[target_layer][target_index]`
using the `Skip` fields, not blindly add the full input vector.

---

## Typical `nn.Module` shape

There is no graph compiler and no ready-made model. Write ordinary
PyTorch:

1. `spec = k2.parse_edgelist(edgelist)`
2. `self.layer_i = k2.MaskedLinear(spec.masks[i])` for each hop
3. Put ReLU / BatchNorm / Dropout in `forward()` yourself
4. Implement skips as residuals using `spec.skips`
5. `x = k2.align_inputs(df, spec)`
6. Run Captum (or another method) yourself; then
   `map_node_attributions(...)`

Do not add a compiled core or mutate connectivity after parse.

The Python distribution and import name are **`kpnn2`**.
Do not rename them.

---

## Repository layout

```
src/kpnn2/
  __init__.py                 # public exports only
  parse_edgelist.py           # parse_edgelist
  graph_spec.py               # GraphSpec, Skip
  masked_linear.py            # MaskedLinear
  align_inputs.py
  map_node_attributions.py
  errors.py                   # Kpnn2Error
  _frozen_mask.py             # read-only connectivity tensors

tests/
  api/                        # public import surface
  module/                     # unit tests per primitive

CONTEXT.md                    # this file
README.md
docs/
```

Implementation may split private helpers, but public import paths
stay as above.

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
- `parse_edgelist` must not instantiate `nn.Module`.
- `MaskedLinear` must not store other layers' activations or
  implement skip routing.
- One graph node is one scalar in v1 (no node width).
- Public failures: `Kpnn2Error` only.
- After Python edits, run `python -m ruff format .` from the
  `dev` extra. Do not use a global `ruff` on `PATH`.
