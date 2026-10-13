# Proposal: Add `AdjacencySpec.unit_index(names)`

| Field | Value |
|-------|-------|
| Status | `done` |
| Decision | 2026-10-12: accepted and done. Slice 2 landed first and uses `node_units` instead of `unit_index`; the name-list checks moved to `_validate.as_name_list` in their own commit. Open questions: `unit_index`, repeats allowed, `()` for empty, no `LayeredSpec` form. |
| Date | 2026-10-11 |
| Based on | `runs/2026-10-11_fc5b67c-dirty/report.md` |
| kpnn2 commit | `fc5b67c` dirty (paper-fit files and `AGENTS.md` only; nothing under `src/`) |
| Kind | add |
| Locks touched | none |
| Agent | Claude Code, Claude Opus 5.5 (`claude-opus-5-5`) |

## Problem

Code on an `AdjacencySpec` reads and writes the state units of
nodes the caller names (N46 readouts, N82 per-sample reads, N45
data on non-input nodes) through an index it builds by hand,
and that index is right only while every listed node has
width 1.

kpnn2 gives this index for the roles it infers (`input_index`,
`output_index`) and `node_units(name)` for one name. Papers need
it for roles they name: LEMBAS's TFs (some not sinks) and drug
targets, DeepSignalingFlow's omics on every gene and drug nodes
per sample, GSNN's hidden blocks. The sketches write
`[spec.node_units(t).start for t in names]` or a list over
`spec.nodes`. At width k, `.start` is a valid index for the
first of k units, so nothing raises. kpnn2's own reparse example
(`docs/packed_linear.md`) copies adjacency bias over
`enumerate(spec.nodes)`, the same idiom.

## Evidence

| Need | Paper | Run mapping | Glue today | Card |
|------|-------|-------------|------------|------|
| N46 | `lembas-2022` | glue | TF readout and drug-target write through `node_units(t).start` (sketch lines 32–33, 39–40; 4 GLUE) | Needs N46, N45; Glue "Align data columns to nodes; input scatter and output gather by name" (~45) |
| N46 | `self-pruning-2025` | glue | TF readout through `node_units(t).start` (41–42; 2 GLUE) | Needs N46; Glue "Inputs and outputs matched to nodes by name" (~15) |
| N82 | `deepsignalingflow-2024` | glue | Drug nodes per sample: a dict of `node_units(d).start`, then a gather (70–71, 76; 3 GLUE) | Needs N82; Glue "Per-sample drug-node lookup in the batched graph" (~15); Fragile spots: `drug_index` holds global node numbers |
| N45 | `deepsignalingflow-2024` | glue | Omics reindexed onto `spec.nodes`; pair flags as lists over `spec.nodes` (58, 64–65; 3 GLUE) | Needs N45; Glue "Build per-sample node feature rows" (~80) |
| N64 (watch list) | `gsnn-2024` | glue | Hidden-block index as a `torch.cat` of `node_units` slices (32–33; 2 of 8 GLUE) | Needs N64; Glue "Hidden unit → node groups; node values ↔ edge values" (~35) |

`self-pruning-2025` reimplements LEMBAS (its card's Other code
row), so it counts once with `lembas-2022`. That leaves two
supporting papers. N64 joins as a watch-list need and would stay
glue. The run treats N27, N46, N55, N82, and N45 on non-input
nodes as one pattern (Gaps; Caveats). Its layered members,
`kpnn-2020` (N27) and `genomic-prediction-2025` (N55), are not
served here; see Open question 3.

## Proposal

- Public surface: one method on `AdjacencySpec`.

  ```python
  AdjacencySpec.unit_index(self, names: object) -> tuple[int, ...]
  ```

  First docstring line: "Return every state unit of the named
  nodes, in order."
- Behavior:
  - `names` is a 1-D sequence of node names: a list, a tuple, a
    1-D numpy array, or a `pandas.Index`. Each name is matched
    after `str(...)`, as in `node_units`.
  - Returns every unit of each name, in `names` order, units of
    one node ascending, as Python `int`s: the rule that builds
    `input_index` and `output_index`, so
    `spec.unit_index(spec.input_nodes) == spec.input_index`
    (same for outputs). At width 1 position `i` belongs to
    `names[i]`; a node of width k gives k entries.
  - A repeated name repeats its units (Open question 2);
    `index_copy` through repeats is the caller's concern,
    `index_add` sums. An empty `names` returns `()` (Open
    question 4).
  - Used as `input_index` is:
    `torch.as_tensor(spec.unit_index(names), dtype=torch.int64)`
    as a buffer, then `state[..., index]`, `index_copy`, or
    `index_add`.
  - It takes no tensor or data, never ignores an unknown name,
    and is not on `LayeredSpec`. It is identity into the unit
    axis, not a head helper (CONTEXT **`LayeredSpec.node_units()`
    / `hop_units()`**). Stored fields, `to_dict()`, and
    `fingerprint` do not change.
- Errors, all `Kpnn2Error`:

  | Caller mistake | Message |
  |----------------|---------|
  | `names` is a `str` | 'names' is a string; pass a sequence of node names. |
  | `names` is `bytes` | 'names' is bytes; pass a sequence of node names. |
  | `names` is a tensor | 'names' is a tensor; pass a sequence of node names. |
  | A set or mapping | `reject_unordered(names, "names", "a sequence of node names")`, the existing message |
  | `ndim != 1` (a DataFrame, a 2-D array) | 'names' must be one-dimensional. Got <received>. |
  | Not iterable | Unsupported names type. Expected a sequence of node names. Got <received>. |
  | An empty string among the names | Node name is empty. (as `node_units`) |
  | Names that are not nodes | Unknown node name(s): Q, Z. Valid names are in spec.nodes. (each unknown name once, sorted, comma-separated) |

- Where it lives:
  - `src/kpnn2/_adjacency_spec.py`: the method, with doctests at
    width 1 and with `widths={"b": 3}`.
  - `src/kpnn2/_layout.py`, which owns every name-to-position
    mapping (CONTEXT **Internal unit layout**):
    `Layout.units_of(names) -> tuple[int, ...]`, moved from
    `_parse_adjacency._units_of` and used there for
    `input_index` and `output_index`; and
    `resolve_node_names(names, known_names, listed_in)`, the
    many-name form of `resolve_node_name`.
  - `src/kpnn2/_validate.py`: the generic checks of
    `_align._labels_from_names` (string, bytes, set or mapping,
    `ndim != 1`, not iterable), with the noun as an argument.
    `align_inputs` keeps its tensor, DataFrame, and AnnData
    checks first, and every message byte for byte.

## Why in kpnn2

CONTEXT **Package philosophy** gives kpnn2 "named I/O
alignment", and its Division of labor gives kpnn2 "Named node →
unit slice (`node_units` / `hop_units`)". The method extends
that lookup from one name to many, as `input_index` and
`output_index` already do for the roles kpnn2 infers. The glue
it replaces is a fragile spot by both tests. It depends on
kpnn2's unit layout: the sketches assume one unit per node. It
fails silently: `node_units(t).start` stays a valid index at any
width, so a wide node is read or written through its first unit
alone. CONTEXT **Internal unit layout** forbids hand-computed
column indices inside `src/`. This gives callers the same
protection.

## Before and after

### `lembas-2022`

```python
# before
self.register_buffer("out", torch.as_tensor(  # GLUE: TFs by name,
    [spec.node_units(t).start for t in TF]))  # GLUE: sinks or not
self.register_buffer("tgt", torch.as_tensor(  # GLUE: drug targets
    [spec.node_units(t).start for t in dt.layer_nodes[1]]))  # GLUE
# after
self.register_buffer("out", torch.as_tensor(
    spec.unit_index(TF)))  # PROPOSED
self.register_buffer("tgt", torch.as_tensor(
    spec.unit_index(dt.layer_nodes[1])))  # PROPOSED
```

`self-pruning-2025` lines 41–42 change the same way, and
`gsnn-2024` lines 32–33 become
`hid = torch.as_tensor(spec.unit_index([f"{f}_h" for f in fn]))`.

### `deepsignalingflow-2024`

```python
# before
om = torch.as_tensor(omics.reindex(spec.nodes, fill_value=0).to_numpy().T,
                     dtype=torch.float32)  # GLUE: N45 data onto state nodes
def flags(a, b):
    either = [(v in tg[a]) | (v in tg[b]) for v in spec.nodes]  # GLUE
    both = [(v in tg[a]) & (v in tg[b]) for v in spec.nodes]  # GLUE
    return torch.tensor([either, both]).float()
pos = {d: spec.node_units(d).start for d in tg}  # GLUE: N82 drug nodes
ia, ib = (torch.tensor([pos[p[i]] for p in pairs]) for i in (0, 1))  # GLUE
# after
genes = omics.index.intersection(spec.nodes)  # rows without a node drop
om = torch.zeros(2, n).index_copy(-1, torch.as_tensor(
    spec.unit_index(genes)), torch.as_tensor(  # PROPOSED
        omics.loc[genes].to_numpy().T, dtype=torch.float32))
def flags(a, b):
    f = torch.zeros(2, n)
    f[0, list(spec.unit_index(sorted(tg[a] | tg[b])))] = 1  # PROPOSED
    f[1, list(spec.unit_index(sorted(tg[a] & tg[b])))] = 1  # PROPOSED
    return f
ia, ib = (torch.as_tensor(spec.unit_index(  # PROPOSED
    [p[i] for p in pairs])) for i in (0, 1))  # one unit per drug node
```

## Predicted effect on the next run

| Paper | Need | Now | After | `# GLUE` lines |
|-------|------|-----|-------|----------------|
| `lembas-2022` | N46 | glue | covered | 9 → 5 |
| `self-pruning-2025` | N46 | glue | covered | 11 → 9 |
| `deepsignalingflow-2024` | N82 | glue | covered | 12 → 6 (both rows) |
| `deepsignalingflow-2024` | N45 | glue | covered; the `intersection` line may stay tagged | see above (7 if it stays) |
| `gsnn-2024` | N64 | glue | glue; the unit-type masks and the two passes stay | 8 → 6 |
| `kpnn-2020` | N27 | glue | glue; layered, not served | 10 |
| `genomic-prediction-2025` | N55 | glue | glue; layered, not served | 1 |

- Verdicts: `deepsignalingflow-2024` moves from P to R, so the
  totals go from 16 R / 6 P to 17 R / 5 P. Its P comes from N45
  and N82, named nodes outside the inferred roles. What stays is
  N81, a degree division in `forward` that the run counts as R
  for `pinnet-2023`, and score analysis (N25, N33, N72).
  `lembas-2022` and `self-pruning-2025` stay P (N47), `gsnn-2024`
  stays P (N64), and `kpnn-2020` stays P (N27, N29).
- Gaps: N46 leaves the Gaps table. The pattern shrinks to N27
  (`kpnn-2020`) and N55 (`genomic-prediction-2025`), one paper
  each (watch list).
- Orphans: none of the run's orphans gain users.
  `AdjacencySpec.node_units()` loses its four sketch users,
  because each called it only to build this index; see Cost and
  risks.

## Cost and risks

- Public surface: one method (`AdjacencySpec.unit_index`) with
  one positional parameter (`names`). Nothing in `__all__`, no
  keyword argument, nothing removed.
- Internal code: the method and a many-name resolver (about 40
  lines with docstrings), one helper moved to `_layout.py`, and
  one validation helper shared with `align_inputs`.
- Migration: none; additive. Numbers: none change.
  `input_index`, `output_index`, packed indices, and
  fingerprints stay identical, as existing tests pin.
- Risks:
  - Two lookups for one concept on `AdjacencySpec`:
    `node_units` (one name, a slice view) and `unit_index`
    (many names, a tuple). The next run may list
    `AdjacencySpec.node_units()` as an orphan; the time-series
    tutorial still uses it for a self-loop slot.
  - The `align_inputs` refactor must keep every message, which
    `tests/api/test_input_validation.py` pins.
  - Per-node data written into a wide node still needs one
    column per unit; the method gives positions, not a repeat.
    No supporting paper has a wide adjacency node.
  - Breadth is thin: two papers. The broader gaps (N18, N33, the
    N24 family) failed **Why in kpnn2**.

## Slices

1. **Add `AdjacencySpec.unit_index(names)`.**
   - Code: `_layout.py` (`Layout.units_of`,
     `resolve_node_names`), `_validate.py` (shared name-list
     checks), `_align.py` (calls them), `_parse_adjacency.py`
     (uses `units_of`), `_adjacency_spec.py` (the method).
   - Tests: new `tests/module/test_unit_index.py`: the
     `input_index` and `output_index` equalities, ordinals at
     width 1, blocks with `widths=`, a cross-check against the
     unit names `map_node_attributions` gives a `state_dim`
     tensor, repeats, `()`, integer names, `int` items, and an
     unchanged caller sequence. Rows under `AdjacencySpec` in
     `tests/api/test_input_validation.py`.
   - Lockstep: yes. `CONTEXT.md` **`AdjacencySpec` fields**
     (next to `node_units`), the Division of labor row,
     **Internal unit layout**, and the **Errors** list of
     arguments matched by position; the member list of
     `docs/reference/AdjacencySpec.md`. `__init__.py` and
     `tests/api/test_public_api.py` do not change.
   - Changelog: yes, one line under Added, as `node_units()`
     had. Numbers changed: no.
2. **Copy adjacency bias by `unit_index` in the reparse hatch.**
   - `docs/packed_linear.md`, `AdjacencySpec` example: size the
     layers with `state_dim`, not `len(spec.nodes)`, and copy
     bias with one `unit_index` call per spec over the
     surviving nodes whose width is unchanged, replacing the
     `enumerate(spec.nodes)` loop. Read
     `dev/docs_writing_guidelines.txt` first.
   - `CONTEXT.md` **Reparse hatch**: replace "On an
     `AdjacencySpec` there is no `node_units`; index
     `spec.nodes` ... by name" with the same rule.
   - Lockstep: no. Changelog: no. Numbers changed: no.

## Open questions

1. **Name.** `unit_index` mirrors `input_index` and
   `output_index`; alternatives are `units_of` and `node_index`.
   Recommend `unit_index`.
2. **Repeated names.** Allow them (N82 reads one drug node for
   several samples) or reject them as `align_inputs` does, so
   the caller deduplicates and maps per sample. Recommend allow,
   documented.
3. **A `LayeredSpec` counterpart.** Its pattern members avoid
   the fragile idiom: `kpnn-2020` concatenates `node_units`
   slices (right at any width), and `genomic-prediction-2025`
   reindexes on `layer_nodes[2]` (loud at width > 1). A layered
   form also needs a layer rule, and `kpnn-2020`'s roots span
   layers. Recommend not now; revisit if a run shows `.start` on
   a `LayeredSpec`.
4. **Empty `names`.** Return `()` (two drugs with no shared
   target give an empty DeepSignalingFlow flag set) or raise.
   `torch.as_tensor(())` is float32 and fails as an index with a
   torch error. Recommend `()`, with the docstring showing
   `dtype=torch.int64`.

## Considered and not chosen

| Candidate | Supporting papers | Why not now |
|-----------|-------------------|-------------|
| N18 plain folds over samples as registered aggregation methods | 8: `kpnn-2020`, `pnet-2021`, `mulgonet-2026`, `pathhdnn-2025`, `deepkegg-2024`, `celltics-2023`, `sigprimednet-2023`, `deepgami-2023` | Fails **Why in kpnn2**. Each fold is one xarray reduction on the mapper's `DataArray` (1–2 GLUE lines): sum (3 papers), mean (3), mean of absolute values (2), over all samples, one class, or each class. No card lists a fragile spot in it, and all eight papers have width-1 nodes. CONTEXT reserves the default for `rauter_mangano_2026`. Its internals (`bind_labels`, the `seed` dim, the repeated-name check) already prepare these folds, and this run is design evidence for that method. |
| N33 Combine scores across repeats | 5: `kpnn-2020`, `lembas-2022` (with `self-pruning-2025`), `deepsignalingflow-2024`, `deepkegg-2024`, `reliable-interpretability-2023` | Fails **Why in kpnn2**. Named runs join in one `xr.concat` or `pd.DataFrame` line. The positional fold, a DeepSignalingFlow fragile spot, is already gone. The run filter (KPNN) and N102 are analysis. |
| N24 family: per-node counts (N24, N67, N81, N99, N103, N104) | 6: `pnet-2021` (with `reliable-interpretability-2023`), `pathhdnn-2025`, `dlbcl-survival-2024`, `gsnn-2024`, `deepsignalingflow-2024`, `pinnet-2023` | Fails **Why in kpnn2**. Degree by name is 1–2 `value_counts` lines on `to_edgelist()`. The per-unit scale of N81 and N99 already exists per slot: `init_bound()` is `1/sqrt(fan_in)`, usable in `constraint=`. Reachability counts and betweenness (N24 in `pathhdnn-2025`, N103) are graph algorithms outside the Division of labor. **2026-10-13: overridden by the maintainer** and done as `node_table()` on both specs. The error-prone parts count as fragile spots: `bincount` on packed indices counts unit pairs, not parents; set sizes need reachability; counts keyed by name must be aligned to unit order. Betweenness stays out. |
| N25 and N97: scores joined to edges | 3: `pnet-2021`, `pathhdnn-2025`, `deepsignalingflow-2024`; plus `pideel-2023` (watch list) | Fails **Why in kpnn2**: one `merge` on `to_edgelist()` by name, 2–3 lines. |
| N72 Node scores from scored edges | 2: `deepsignalingflow-2024`, `gsnn-2024` | Fails **Why in kpnn2**: a pandas groupby. Centrality needs networkx, a new dependency. |
| N47 Keep a recurrent matrix contractive | 1: `lembas-2022` (with `self-pruning-2025`) | Fails the evidence rule once shared code counts once. The penalty is training (**What kpnn2 is NOT**, Not a trainer). |
| N34 Structure-only baseline | 2: `kpnn-2020`, `reliable-interpretability-2023` | **Locked contrasts**: control data lives on the data axis, which is the user's. |
| Remove `PackedMultiheadAttention`, `PackedLinear.transpose()`, `scatter_hop_outputs` | none | "No application in this sample" alone. They serve CONTEXT use cases (attention on live pairs; tied decoders). The papers that would test them are not in the sample: `sctransformer-2026` is excluded (no code), and `vega-2021`, `expimap-2023`, `ontovae-2023`, and `binn-2023` are pending in `papers.csv`. |

## Earlier proposals

None. `proposals/` held only `_template.md`.

## Caveats

- Freshness gate: passed. The run covers all 22 included papers,
  and `git diff fc5b67c -- src/` is empty, in the work tree too.
  The run directory ends in `-dirty`: its uncommitted files are
  paper-fit files and `AGENTS.md`. kpnn2 is 0.2.0.
- Every supporting paper has width-1 adjacency nodes, so the
  fragile spot does not change its published numbers. It bites
  once `widths=` is used on an `AdjacencySpec`, as `gsnn-2024`
  does.
- The run's per-paper text gives `lembas-2022` 2 GLUE lines for
  N46; the sketch tags 4 (TF readout and drug targets). This
  proposal uses the sketch.
- The `deepsignalingflow-2024` verdict move assumes the next
  assessor keeps this run's rule that a degree division in
  `forward` is R (`pinnet-2023`).
- CONTEXT **Reparse hatch** says an `AdjacencySpec` has no
  `node_units`, but it has one (CHANGELOG: "`edge_location()` and
  `node_units()` on both specs"). Slice 2 corrects that.
- N18 is the broadest gap and was dropped on **Why in kpnn2**,
  not on breadth. If the maintainer rules that the Division of
  labor row "Named scores → per-node aggregate" is reason
  enough, N18 outranks this proposal.
- No paper code was opened; cards and sketches were the only
  sources.
