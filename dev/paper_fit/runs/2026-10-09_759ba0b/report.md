# Paper-fit run 2026-10-09

| Field | Value |
|-------|-------|
| kpnn2 version | 0.2.0 |
| kpnn2 commit | `759ba0b` (clean tree) |
| Scope | All included papers |
| Cards | 6: `drugcell-2020`, `kpnn-2020`, `lembas-2022`, `mulgonet-2026`, `pathhdnn-2025`, `pnet-2021` |
| Previous run | None |
| Agent | Claude Code, Claude Opus 5.5 (`claude-opus-5-5`) |

## Summary

Verdicts are 4 R (`drugcell-2020`, `mulgonet-2026`, `pathhdnn-2025`,
`pnet-2021`), 2 P (`kpnn-2020`, `lembas-2022`), and 0 N, and all six
sketches run. Given a paper's final edgelist, kpnn2 builds its whole
network: the edge mechanism, the layers (skip edges and wide nodes
included), the inputs by name, and the names on the scores. The one
gap shared by more than two papers is N18 (four papers): every sketch
folds scores over samples in xarray, because the only aggregation
method raises "not yet implemented". The two P verdicts come from
reading named nodes as outputs (N27, N46), plus KPNN's per-node
dropout and LEMBAS's spectral rescale. `ranks=` joins the orphans: no
paper needs it once its edgelist is final.

## Verdicts

| Paper | Verdict | Sketch | kpnn2 lines | Glue lines | Needs left to glue or missing |
|-------|---------|--------|-------------|------------|-------------------------------|
| `drugcell-2020` | R | pass | 16 | 0 | None |
| `kpnn-2020` | P | pass | 20 | 10 | N18, N27, N29, N33, N34 |
| `lembas-2022` | P | pass | 19 | 9 | N33, N46, N47 |
| `mulgonet-2026` | R | pass | 15 | 3 | N18 |
| `pathhdnn-2025` | R | pass | 15 | 5 | N18, N24, N25 |
| `pnet-2021` | R | pass | 16 | 8 | N18, N24, N25 |

How the verdicts were drawn:

- **R:** what is left is score analysis (folding, normalizing, joining
  to edges) or ordinary training code.
- **P:** a need that maps the graph into the model is left to glue.
  That means reading named nodes as outputs, or model structure
  derived from the graph.
- **Out of scope (pre-edgelist):** a need that only decides which
  rows the edgelist holds. The maintainer ruled on 2026-10-09 that
  everything before the final edgelist is no concern of kpnn2, since
  the edgelist carries no history. `CONTEXT.md` implies this in **One-sentence
  summary** and **What this package does** (step 1), but does not say
  it outright (see Caveats).

Counting rules:

- **kpnn2 lines:** physical lines that call a `kpnn2.` name, or read a
  public attribute or method of a spec, hop, or kpnn2 layer.
- **Glue lines:** lines tagged `# GLUE`. A line is tagged when it
  serves a need mapped `glue` or `missing`.
- **Not tagged:** ordinary model code, and code for needs mapped `out
  of scope`.
- A line can count in both columns. Imports, `def` lines, comments,
  and `print` diagnostics are not counted.

Each sketch starts from its paper's final edgelist, written as a
literal table. In an earlier version of this run, three sketches
derived that table and passed `ranks=`. The rewritten sketches parse
to specs with identical fingerprints (checked).

## Per paper

### `drugcell-2020`

- Sketch: `sketches/drugcell-2020.py`, pass.
  - 17 nodes. Terms have 6 units each.
  - Genes are annotated at the bottom, middle, and top terms.
  - `Tb1` has two parents.
- Needs:
  - N01 covered: `parse_layered`. Term rows and gene rows both become
    edges. The `type` column is ignored.
  - N06 covered: gene rows are edges, and `align_inputs` matches them
    by name. A gene column with no annotation (`g11`) is ignored.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N11 out of scope: **Typical `nn.Module` shape**, step 4. A
    per-term `BatchNorm1d(k)` equals one `BatchNorm1d` over the
    layer's units, because BatchNorm normalizes each unit on its own.
  - N12 covered: `PackedLinear` on 6×6 term blocks and 6×1 gene
    blocks.
  - N26 covered: longest-path layering, skip pairs in hops, and
    `gather_hop_inputs`. A term's longest-path depth is DrugCell's
    height plus 1. Genes at any height are skip sources.
  - N31 covered: `map_node_attributions(hop_output=)` on the post-BN
    activations.
  - N38 covered: `parse_layered(widths=)`.
  - N39 covered: `PackedLinear` stores no absent entries, so the init
    mask and the per-step gradient mask go away.
  - N40 out of scope: **Package philosophy** (Division of labor:
    heads). `LayeredSpec.node_units` finds each term's 6 units.
  - N41 out of scope: **Package philosophy** (Division of labor: the
    user writes `forward()`).
- Author glue:
  - Read the ontology, split annotations, map gene names (~43):
    removed.
  - Check genes below every term, one root, one component (~38):
    left. These are properties of the edgelist, so pre-edgelist.
    - kpnn2 still refuses a term with no gene below: it becomes an
      input node, and `align_inputs` then raises (checked).
    - The sketch keeps a 2-line one-root check as ordinary model
      code, because its head reads one root.
  - Masked gene-selection layer per term (~10): removed (see Notes).
  - Order terms; build per-term layers and heads (~43): shrunk to one
    `PackedLinear` per layer plus a `ModuleDict` of heads.
  - Concatenate child outputs and gene values (~28): removed by
    `gather_hop_inputs`.
  - One 0/1 mask per term (~18): removed.
  - Mask initial weights and every gradient (~13): removed.
  - Write per-term activations (~4): removed by
    `map_node_attributions`.
- Notes:
  - The code's per-annotation gene layer is not reproduced. It holds
    one weight and one bias per annotation, with no nonlinearity, so
    it is a linear reparameterization of the direct 6×1 block. The
    function class is the paper's; training dynamics differ.
  - The fragile spots tied to masks disappear: the lookup by name
    prefix, masking only in `train_model`, and the set-ordered rows.
  - The ×0.1 initialization is not reproduced (user code).

### `kpnn-2020`

- Sketch: `sketches/kpnn-2020.py`, pass.
  - 21 edges.
  - Two labeled roots at different depths: `CD8` at layer 3, `CD4` at
    layer 4.
  - One skip edge, `g8_gene → R1`.
  - One data column (`g10_gene`) that is not in the graph.
- Needs:
  - N01 covered: `parse_layered`. It rejects duplicates and cycles,
    as `KPNN_Function.py:508` does.
  - N04 out of scope (pre-edgelist). If an unmeasured gene stayed in
    the table, it would become an input node and `align_inputs` would
    raise.
  - N06 covered: TF→gene rows are edges. `align_inputs` matches by
    name and ignores extra columns.
  - N10 covered.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N12 covered: `PackedLinear`.
  - N15 out of scope: **Package philosophy** (Division of labor:
    attribution algorithms).
  - N17 covered: `map_node_attributions(hop_output=)`.
  - N18 glue: `.mean("observation")` in xarray (2 lines).
  - N26 covered: longest-path layering, skip pairs, and
    `gather_hop_inputs`.
  - N27 glue: the roots are `output_nodes`, but they sit in different
    layers. The logits are gathered with `LayeredSpec.node_units` and
    `torch.cat` (2 lines). There is no output-side counterpart to
    `align_inputs`. Dropping unlabeled roots is pre-edgelist.
  - N28 out of scope (pre-edgelist).
  - N29 missing: siblings counted with a pandas merge on the final
    edgelist, a keep vector per layer, and a Bernoulli mask in
    `forward` (6 lines).
  - N30 out of scope: **Package philosophy** (Division of labor). The
    sketch uses Captum's layer gradient, the ε→0 limit of the central
    difference.
  - N31 covered: `map_node_attributions` on the saved activations.
  - N32 covered: `to_edgelist`, `edge_location`, and
    `effective_weight` (2 lines).
  - N33 glue: not sketched. It would be `xr.concat` over runs, a
    filter on test error, and scaling.
  - N34 missing: simulated control inputs. Not sketched.
  - N35 covered for `--shuffleGenes`: `align_inputs` on a permuted
    name list (1 line). The paper's edge swaps are pre-edgelist, and
    `parse_layered` would reject a swap that adds a cycle.
- Author glue:
  - Match data rows to leaves (~15): removed by `align_inputs`.
  - Match label columns to roots (~10): shrunk to 2 lines.
  - Prune unlabeled roots and unmatched leaves (~15): left,
    pre-edgelist.
  - Topological order with cycle check (~20): removed.
  - Child index lists and weight blocks (~45): removed by the hops.
  - One op group per node (~25): removed.
  - Graph-dependent dropout (~15): left, 6 lines. The regex parent
    lookup (a fragile spot) is gone.
  - Activation perturbation hook (~6): removed by using Captum's
    gradient. It is left if the exact ±ε score is wanted.
  - Finite-difference node scores by name (~30): shrunk to about 6
    lines.
  - Edge weights and activations by name (~30): shrunk to 2 lines.
    `edge_location` replaces the implied stacking order (a fragile
    spot).
  - Control inputs; input shuffle (~40): the shuffle is 1 line; the
    simulation is left.
  - Collect, filter, scale across replicates (~40, R): left.
- Notes:
  - Logits are read from pre-activations. The sigmoid runs on whole
    layers, which is harmless because output units feed nothing.
  - A layer of only outputs (`CD4`) has an unused activation, so
    Captum cannot hook it. The score loop skips it.
  - `gather_hop_inputs` caught a float64 tensor that the dropout glue
    introduced.

### `lembas-2022`

- Sketch: `sketches/lembas-2022.py`, pass.
  - A signed, cyclic edge list with 11 nodes.
  - Two loops: `K1 → K3 → K1` and `K2 → T1 → K2`.
  - A ligand `L3` with no data column.
  - Two drugs projecting onto kinases (the viability variant).
- Needs:
  - N06 covered: the drug→target pairs go through `parse_layered` as
    one hop and a bias-free `PackedLinear`. Placing them on the state
    with `AdjacencySpec.node_units` takes 2 GLUE lines.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads). Not sketched.
  - N10 covered.
  - N12 covered: `PackedLinear` on `AdjacencySpec` indices. Slots are
    named by `edge_location` and `to_edgelist`, so the CSR order
    fragile spot is gone.
  - N17 covered: `map_node_attributions(dims=, coords=)` on the
    knockout effects.
  - N28 out of scope (pre-edgelist).
  - N31 covered: `forward` returns the full state, and
    `map_node_attributions` names it.
  - N32 covered: `to_edgelist().assign(weight=effective_weight())`.
  - N33 glue: not sketched. It would be a median over fold models.
  - N42 covered: `parse_adjacency` keeps the cycles. The sign column
    stays on the caller's table, and `edge_location` places it
    (3 lines). Duplicate rows raise instead of summing silently (a
    fragile spot).
  - N43 out of scope: **What kpnn2 is NOT** (Not a time machine). The
    loop is 4 lines (see Notes).
  - N44 out of scope: **Package philosophy** (per-edge signs stay
    with the caller). The sign-aware init and the L1 penalty take
    3 lines.
  - N45 covered: `align_inputs` and `input_index`. A ligand with no
    data column needs `reindex(fill_value=0)` (1 GLUE line), because
    `align_inputs` rejects missing names.
  - N46 glue: `T1` feeds `K2`, so it is not in `output_nodes`. The
    TFs are read with `AdjacencySpec.node_units` (2 GLUE lines).
  - N47 glue: the init rescale rebuilds a dense matrix from the packed
    slots and calls `eigvals` (4 GLUE lines). `to_mask()` gives the
    pattern, not the weights. The training penalty is a loss and is
    not sketched.
  - N48 out of scope: **What kpnn2 is NOT** (Not a trainer). Not
    sketched.
  - N49 out of scope: **Package philosophy** (Division of labor).
    `forward` takes an `offset`.
  - N50 out of scope: the same `offset` mechanism. Not sketched.
  - N51 covered: `edge_location` per edge, and `node_units` for
    biases.
  - N52 out of scope (pre-edgelist).
  - N53 covered: the drug→target `PackedLinear` stores no absent
    entries.
- Author glue:
  - Clean OmniPath edges (~70): left, pre-edgelist.
  - Reachability prune; pass-through removal (~65): left,
    pre-edgelist.
  - Edge table to indices, sign masks, node order (~35): removed. The
    signs take 3 lines with `edge_location`.
  - Sparse recurrent forward and adjoint backward (~75): the forward
    is removed (`PackedLinear` plus a 4-line loop). The adjoint is
    dropped for unrolled autograd.
  - Spectral radius and its gradient (~110): the init is shrunk to 4
    lines; the penalty is left.
  - Sign-aware init and sign violations (~30): shrunk to 3 lines.
  - Align columns, input scatter, output gather (~45): shrunk to
    `align_inputs` and `input_index`, plus 2 gather lines and 1
    zero-fill line.
  - Save and load parameters by edge name (~70): shrunk to 1 line to
    save and a 2-line loop to load.
  - Knockout and sensitivity batches (~40): shrunk to an `offset`
    argument and 2 lines.
  - Drug-target mask and per-batch re-masking (~30): removed.
- Notes:
  - Inputs are added to the pre-activation at every step,
    `h ← f(A·h + x + b)`, as LEMBAS does. The CONTEXT example writes
    inputs into the state instead, which is a different update.
  - Backprop through the unrolled loop is not LEMBAS's adjoint
    gradient at the steady state, and it stores every step.
    `PackedLinear.transpose(bias=False)` would apply the Aᵀ product
    the adjoint iterates, but no sketch tries it.
  - The zero-filled column made the host matrix float64.
    `PackedLinear` casts its input, but the user's `index_add` does
    not. The sketch builds `x` as float32.

### `mulgonet-2026`

- Sketch: `sketches/mulgonet-2026.py`, pass.
  - Two final edgelists, BP and MF, with 3 levels (the paper uses 5).
  - 4 omics per gene, 26 nodes in total.
  - `B111` has two parents.
- Needs:
  - N01 covered: `parse_layered`.
  - N02 covered: `parse_layered`'s default layering. The table joins
    only adjacent levels, so the longest path equals the depth from
    the root. This was checked on both branches against `ranks=`:
    same layers, same fingerprint. The depth cut is pre-edgelist.
  - N03 out of scope (pre-edgelist).
  - N04 out of scope (pre-edgelist).
  - N05 out of scope (pre-edgelist).
  - N06 covered: annotation rows are edges, and `align_inputs`
    ignores unannotated features.
  - N07 out of scope (pre-edgelist): `g1_amp → B111` is an ordinary
    row.
  - N08 covered: one `parse_layered` and one `align_inputs` per
    namespace on the same columns.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N10 covered.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N12 covered: `PackedLinear`.
  - N13 out of scope: **What kpnn2 is NOT** (Not a trainer). It is 1
    line with `effective_weight()`.
  - N14 out of scope: **Package philosophy** (Division of labor).
    Captum IG runs on the tuple of branch inputs. Each branch is named
    with `axis="inputs"`, and the two are summed by name (2 GLUE
    lines).
  - N15 out of scope: `LayerIntegratedGradients`, named with
    `hop_output=`.
  - N16 out of scope: Captum IG with 20 steps. The repository's
    variant is not reproduced.
  - N17 covered: `map_node_attributions`, plus the merge above.
  - N18 glue: a sum over recurrence-positive samples in xarray.
- Author glue:
  - Read GO edges per namespace; find roots (~16): left,
    pre-edgelist.
  - Assign depth layers (~15): removed. `parse_layered` places the
    terms; the depths used to decide the cut are pre-edgelist.
  - Remove terms without a child (~16): left, pre-edgelist.
  - Term×term 0/1 matrix per layer pair (~19): removed.
  - Load annotations; drop terms over 200 genes (~29): left,
    pre-edgelist.
  - Cut the bottom layer to annotated terms (~7): left, pre-edgelist.
  - Keep omics features that have a term (~17): removed.
    `align_inputs` ignores the rest.
  - Gene×term masks per omics (~55): removed. The per-omics rows are
    edges.
  - Concatenate omics inputs in mask order (~6): removed.
  - Edge-vector layer scattered into a dense kernel (~86): removed by
    `PackedLinear`.
  - Wire five layers per branch (~48): shrunk to a 15-line `Branch`
    module.
  - Per-layer attribution loop and sum (~77): shrunk to about 8
    lines.
  - Label scores; write per layer (~45): removed by
    `map_node_attributions`.
- Notes:
  - MULGONET keeps bias-only terms after its cut. In a kpnn2 table,
    such a term has no incoming edge, so it would be an input. Leaving
    it out keeps the function class, because its constant output is
    absorbed by the parent's bias.
  - Alignment by position is replaced by names. A BP-only gene simply
    has no MF column.

### `pathhdnn-2025`

- Sketch: `sketches/pathhdnn-2025.py`, pass.
  - A final edgelist with 3 levels (the paper uses 4).
  - One copy node, `R12_copy1`.
  - Propagated features at `R111`.
  - `g4_amp` is a model input that is missing from the data.
- Needs:
  - N01 covered: `parse_layered`.
  - N02 covered: the default layering, as in `mulgonet` (checked).
  - N03 out of scope (pre-edgelist).
  - N06 covered: membership rows are edges, matched by
    `align_inputs`. A model input missing from the data needs
    `reindex(fill_value=0)` (1 GLUE line).
  - N07 out of scope (pre-edgelist).
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N10 covered.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N14 out of scope: **Package philosophy** (Division of labor).
    Captum `LayerDeepLiftShap(attribute_to_layer_input=True)`.
  - N15 out of scope: as N14.
  - N17 covered: `map_node_attributions(hop_input=)`, and `layer=`
    for the output layer's input.
  - N18 glue: the mean of |SHAP| in xarray.
  - N19 out of scope (pre-edgelist): a copy node is an ordinary named
    node in the table.
  - N20 out of scope (pre-edgelist).
  - N21 out of scope (pre-edgelist).
  - N22 covered: `MaskedLinear(hop.to_mask())`.
  - N23 out of scope: Captum DeepLiftShap.
  - N24 missing: not sketched; the script turns it off.
  - N25 glue: `to_edgelist()` merged with the scores (2 lines).
- Author glue:
  - Per-type feature names and membership tables (R, ~60): left,
    pre-edgelist.
  - Read the hierarchy, add a root, cut to data pathways (~45): left,
    pre-edgelist.
  - Propagate memberships; attach features (~50): left, pre-edgelist.
  - Layer by depth, pad branches, build sorted 0/1 matrices (~80):
    depth and padding left (pre-edgelist); the matrices are removed.
  - Apply masks via pruning; record unit names (~44): removed.
  - Align data rows to the feature list (~24): shrunk to
    `align_inputs` plus 1 zero-fill line.
  - Per-layer SHAP into a named edge table (~100): shrunk to about 8
    lines.
  - Importance graph, normalization, subgraphs (~190): mostly left.
- Notes:
  - Init after pruning is fixed: `MaskedLinear`'s degree-aware init
    reaches the live entries.
  - kpnn2 requires distinct names, so copies cannot share an id (a
    fragile spot).
  - Masked entries are exactly 0 and get no gradient, so weight decay
    has nothing to move.

### `pnet-2021`

- Sketch: `sketches/pnet-2021.py`, pass.
  - A final edgelist of 28 nodes, with 3 pathway levels (the paper
    uses 5).
  - 3 inputs per gene.
  - Copy nodes `P21_copy1` and `P112_copy1`.
  - One gene with no pathway.
- Needs:
  - N01 covered: `parse_layered`.
  - N02 covered: the default layering, as in `mulgonet` (checked).
  - N03 out of scope (pre-edgelist).
  - N06 covered: gene-set rows are edges.
  - N07 out of scope (pre-edgelist). Hop 0 is then P-NET's `Diagonal`
    as a `PackedLinear` with three edges per gene, matched by name
    instead of column position.
  - N10 covered.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N12 covered: `PackedLinear`, the same parameterization as
    `SparseTF`.
  - N13 out of scope: **What kpnn2 is NOT** (Not a trainer).
  - N14 out of scope: **Package philosophy** (Division of labor).
    Captum `DeepLift`, named with `axis="inputs"`.
  - N15 out of scope: `LayerDeepLift`, named with `hop_output=`.
  - N17 covered: `map_node_attributions`.
  - N18 glue: a sum in xarray.
  - N19 out of scope (pre-edgelist).
  - N24 missing: the degree from the edgelist and the mean + 5 SD
    rule (3 GLUE lines).
  - N25 glue: `to_edgelist()` merged with the scores (3 GLUE lines).
  - N31 covered: `map_node_attributions`.
  - N32 covered: as in the `kpnn` sketch.
  - N36 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N37 out of scope: Captum DeepLift from a zero baseline.
- Author glue:
  - Read Reactome and GMT, filter, add a root (~75): left,
    pre-edgelist.
  - Cut at depth, pad copies, child dicts, genes at the bottom (~75):
    depth and copies left (pre-edgelist); the child dicts are removed.
  - 0/1 map per layer, align genes, chain names (~50): removed.
  - Gene layer by column blocks; edge-vector layer (~135): removed.
  - Assemble layers, heads, dropout, names (~155): shrunk to a
    25-line module.
  - Order data columns by gene, then type (~50): removed by
    `align_inputs`.
  - Label scores, activations, and edge weights (~170): shrunk.
  - Degree from trained weights; adjustment (~80): left, 3 lines.
  - Hidden-layer references in DeepExplain (~8): removed, by Captum.
  - Sankey (~90): the edge table shrinks to 3 lines; the plot is
    left.
- Notes:
  - Copies need distinct names. A copy named like its original would
    be a self-loop, which the parser rejects.
  - P-NET keeps bias-only pathways. See the `mulgonet` note.
  - Reloading by position (a fragile spot) is caught by
    `identity=spec.fingerprint` and `index_digest`.

## Gaps

Needs that are `missing` or `glue` in two or more papers.

| Need | Papers | What an abstraction would do | Lock touched |
|------|--------|------------------------------|--------------|
| N18 Aggregate scores over a sample subset | `kpnn-2020`, `pnet-2021`, `mulgonet-2026`, `pathhdnn-2025` (glue) | Plain registered folds over `observation` (sum, mean, mean of absolute value), optionally limited to one label. Each paper folds differently, and none matches a method kpnn2 has today. | None. `AGENTS.md` **How to add an aggregation method**. |
| N33 Combine scores across training repeats | `kpnn-2020`, `lembas-2022` (glue; not sketched) | A registry method that folds a run dimension (median or mean). The caller names that dimension. | None. CONTEXT forbids guessing that `step` is `seed`. |
| N24 Normalize node scores by connectivity | `pnet-2021` (sketched), `pathhdnn-2025` (not sketched) (missing) | A per-node degree, or an up- plus downstream count, from a spec as a named DataArray to divide by. | None. |
| N25 Scores as a graph | `pnet-2021`, `pathhdnn-2025` (glue) | Join named node scores to `to_edgelist()`. It costs 2–3 lines per sketch. | None. |

Watch list, one paper only:

| Need | Paper | Note |
|------|-------|------|
| N27 Output units are named graph roots | `kpnn-2020` | The same pattern as N46: read named nodes as outputs when they are not in one layer or are not all sinks. Both sketches use a `node_units` comprehension. |
| N29 Dropout rate per node from the graph | `kpnn-2020` | 6 GLUE lines, the largest block left in the sample. |
| N34 Structure-only baseline from control inputs | `kpnn-2020` | Data simulation. Not sketched. |
| N46 Read outputs from named graph nodes | `lembas-2022` | See N27. |
| N47 Keep a recurrent edge matrix contractive | `lembas-2022` | A dense rebuild from packed slots. `to_mask()` gives the pattern only. |

These needs are not gaps, because they only decide which rows the
edgelist holds. They are out of scope (pre-edgelist):

| Need | Papers |
|------|--------|
| N03 Keep only adjacent-layer edges | `mulgonet-2026`, `pathhdnn-2025`, `pnet-2021` |
| N04 Prune nodes with no child below | `kpnn-2020`, `mulgonet-2026` |
| N05 Filter nodes by gene-set size | `mulgonet-2026` |
| N07 One input per entity and data type | `mulgonet-2026`, `pathhdnn-2025`, `pnet-2021` |
| N19 Pad short branches with copy nodes | `pathhdnn-2025`, `pnet-2021` |
| N20 Keep only nodes above the data | `pathhdnn-2025` |
| N21 Propagate memberships up the hierarchy | `pathhdnn-2025` |
| N28 Prune nodes that cannot reach an output | `kpnn-2020`, `lembas-2022` |
| N52 Prune nodes no input can reach | `lembas-2022` |

Three more are partly pre-edgelist:

- N02: the depth cut. The layers themselves are covered.
- N27: dropping unlabeled roots.
- N35: the edge-swap control.

## Orphans

Every name in `kpnn2.__all__`, and each documented keyword argument
and public method of those names.

| kpnn2 name or option | Used by | Need served, or "no application in this sample" |
|----------------------|---------|-------------------------------------------------|
| `parse_layered` | drugcell, kpnn, mulgonet, pathhdnn, pnet, lembas (drug hop) | N01, N02, N26 |
| `parse_layered(widths=)` | drugcell | N38 |
| `parse_layered(ranks=)` | none | No application in this sample. Each paper's final edgelist joins only adjacent levels, so the default longest path already gives its levels (checked on pnet, pathhdnn, and both mulgonet branches). It serves a table that keeps level-skipping edges while pinning official levels. |
| `parse_adjacency` | lembas | N42 |
| `parse_adjacency(widths=)` | none | N38 on a cyclic graph. No paper here combines them. |
| `LayeredSpec` | all layered sketches | N01 |
| `LayeredSpec.to_edgelist()` | kpnn, pathhdnn, pnet | N25, N32 |
| `LayeredSpec.edge_location()` | kpnn | N32 |
| `LayeredSpec.node_units()` | drugcell, kpnn | N27, N40 |
| `LayeredSpec.hop_units()` | none | No application in this sample. DrugCell's RLIPP would use it, but RLIPP is paper-only. |
| `LayeredSpec.to_dict()` / `from_dict()` | none | N51 ("reload a trained one"). P-NET's reload by position is a fragile spot. |
| `LayeredSpec.fingerprint` | all six (`identity=`) | Checkpoint identity |
| `Hop` (`in_features`, `out_features`) | all layered sketches, lembas (drug hop) | N12 |
| `Hop.to_mask()` | pathhdnn | N22 |
| `Hop.column_offsets` | none | No application in this sample |
| `Skip` | none (kpnn prints `len(spec.skips)` only) | Metadata. Skip edges themselves serve N26 in drugcell and kpnn. |
| `AdjacencySpec` (`state_dim`, `input_index`) | lembas | N42, N45 |
| `AdjacencySpec.node_units()` | lembas | N06 (drug targets), N46 |
| `AdjacencySpec.to_edgelist()` | lembas | N32 |
| `AdjacencySpec.edge_location()` | lembas | N44, N51 |
| `AdjacencySpec.to_mask()` | none | No application in this sample. N47 needs the weights, not the pattern. |
| `AdjacencySpec.to_dict()` / `from_dict()` | none | N51 |
| `AdjacencySpec.fingerprint` | lembas | Checkpoint identity |
| `AdjacencySpec.output_index` (field) | none | N46 in principle. LEMBAS's `T1` is not a sink. |
| `MaskedLinear` (`identity=`) | pathhdnn | N22 |
| `MaskedLinear(bias=)` | default only | N10 |
| `MaskedLinear(constraint=)`, `(generator=)` | none | See the `PackedLinear` rows |
| `MaskedLinear.effective_weight()`, `init_bound()`, `reset_parameters()` | none | N13 or N32 on a dense layer. Not exercised. |
| `PackedLinear` (`identity=`) | drugcell, kpnn, lembas, mulgonet, pnet | N12 |
| `PackedLinear(bias=False)` | lembas (drug→target) | N06, N53 |
| `PackedLinear(constraint=)` | none | Nearest is N44, but LEMBAS's sign prior is soft (a loss). No hard sign or freeze in this sample. |
| `PackedLinear(generator=)` | none | N33 (independent init per repeat). Not sketched. |
| `PackedLinear.effective_weight()` | kpnn, lembas, mulgonet | N13, N32, N44 |
| `PackedLinear.init_bound()` | none | No application in this sample |
| `PackedLinear.reset_parameters()` | none | N33 (re-initialize between repeats) |
| `PackedLinear.transpose()` | none | N43: with `bias=False` it applies the Aᵀ product of LEMBAS's adjoint backward. Not sketched. |
| `PackedMultiheadAttention` and all its arguments | none | No application in this sample. `sctransformer-2026` is excluded as no-code. |
| `gather_hop_inputs` | drugcell, kpnn, mulgonet, pathhdnn, pnet | N26 |
| `scatter_hop_outputs` | none | No application in this sample (no decoder) |
| `align_inputs` | all six | N06, N35, N45 |
| `map_node_attributions` | all six | N17, N31 |
| `map_node_attributions(layer=)` | pathhdnn | N23 |
| `map_node_attributions(hop_output=)` | drugcell, kpnn, mulgonet, pnet | N15, N31 |
| `map_node_attributions(hop_input=)` | pathhdnn | N23 |
| `map_node_attributions(axis="inputs")` | mulgonet, pnet | N14 |
| `map_node_attributions(dims=, coords=)` | lembas | N49 |
| `map_node_attributions` on a tensor sequence (`step` axis) | none | No application in this sample. LEMBAS uses only the steady state. |
| `aggregate_node_attributions` (`method=`, `labels=`, kwargs) | none | N18 (4 papers) and N33 (2 papers). Unusable today, because its only method raises. |
| `list_aggregation_methods` | none | As above |
| `Kpnn2Error` | no sketch catches it | It fired twice while the sketches were written: a dtype check (kpnn), and a geneless term (DrugCell probe). |
| `__version__` | none | Metadata |

## Friction

Glue that repeats in three or more sketches:

- **Folding observations** (kpnn, pnet, mulgonet, pathhdnn). A
  `.sum("observation")` or `.mean("observation")` follows the
  `map_node_attributions` calls, because `aggregate_node_attributions`
  cannot fold yet.
- **Boilerplate, not glue** (all five layered sketches). The 3-line
  `PackedLinear(...)` comprehension and the `saved` /
  `gather_hop_inputs` loop repeat, as **Typical `nn.Module` shape**
  prescribes. A model class is locked out (**Package philosophy**).

kpnn2 calls the sketches had to work around:

- **Missing names in `align_inputs`.** It rejects them, but PathHDNN
  and LEMBAS expect zeros for model inputs missing from the data. Both
  sketches call `reindex(..., fill_value=0)` first. The fill needs the
  matrix, which `align_inputs` does not take (**Locked contrasts**).
- **Named readouts outside `input_nodes`.** KPNN's roots in several
  layers, and LEMBAS's TFs and drug targets, are indexed with a
  `node_units(...)` comprehension. There is no output-side
  counterpart to `align_inputs`.
- **Two branches on one input** (mulgonet). Input scores come back per
  spec and are summed by name.
- **One node table from per-layer maps** (kpnn). It needs
  `xr.concat(..., coords="different", compat="equals")`.

## Changes since the previous run

First run.

## Caveats

- **Revised in the same session.** The first version of this report
  had different verdicts (1 R, 5 P). It classified needs that only
  shape the edgelist as gaps, and three sketches derived their table
  and passed `ranks=`. After the maintainer's ruling:
  - those needs are out of scope (pre-edgelist);
  - every sketch starts from its final edgelist, with the same spec
    fingerprints as before (checked);
  - verdicts, gaps, orphans, and friction were redone.
- **The ruling is not in `CONTEXT.md`.** The contract implies it but
  does not state it, so a future run could repeat the first
  classification. A row in **Package philosophy**'s Division of labor
  table, or a rule in `assess_prompt.md`, would fix that.
- **Paper code:** none was opened. The cards were the only source.
- **Cards and catalog:** no card looks wrong. Two observations:
  - `README.md`'s Pilot table lists `sctransformer-2026`, which
    `papers.csv` marks excluded.
  - N27 and N46 describe one pattern from two sides.
- **Simplifications:**
  - Toy depths of 3 (the papers use 4 or 5), with 17–28 nodes per
    graph.
  - Training is one backward pass, so scores come from untrained
    weights.
  - KPNN uses Captum's gradient instead of ±ε.
  - MULGONET uses proper IG.
  - DrugCell drops the per-annotation gene layer.
  - LEMBAS backpropagates through the unrolled loop.
- **Not sketched:** N24 (pathhdnn), N33 (kpnn, lembas), N34, N48, N50,
  and the training-time spectral penalty.
- **Sketch size:** 35–64 code lines each, not counting imports, `def`
  lines, comments, prints, or blank lines. The sketches pack
  arguments onto shared lines, against `AGENTS.md`'s
  one-argument-per-line rule. Ruff skips `tests/paper_fit/`.
- **Line counts** are physical lines; read them as orders of
  magnitude.
- **Environment:** Captum 0.9.0 and torch 2.13.0. No sketch imports
  networkx any more.
