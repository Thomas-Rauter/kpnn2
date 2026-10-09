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

Verdicts: 1 R (`drugcell-2020`), 5 P, 0 N. All six sketches run.
kpnn2 replaces every paper's edge mechanism: the edge-vector scatter,
the per-node op groups, the dense masks, and the sparse recurrent
matrix. What is left is graph shaping before the parse and folding
scores after it. The strongest gaps:

- N18, folding scores over samples (4 papers). Every sketch does it
  in xarray, because the only aggregation method is not implemented.
- The depth-from-root recipe, N02, N03, and N07 (3 papers). It is
  the largest glue block in the sample.

Orphans with no user in this sample: `PackedMultiheadAttention`,
`scatter_hop_outputs`, `PackedLinear.transpose`, `constraint=`,
`generator=`, `hop_units`, `column_offsets`, `to_dict` / `from_dict`,
and `widths=` on `parse_adjacency`. First run, so there is nothing to
compare.

## Verdicts

| Paper | Verdict | Sketch | kpnn2 lines | Glue lines | Needs left to glue or missing |
|-------|---------|--------|-------------|------------|-------------------------------|
| `drugcell-2020` | R | pass | 16 | 2 | None |
| `kpnn-2020` | P | pass | 20 | 16 | N04, N18, N27, N28, N29, N33, N34 |
| `lembas-2022` | P | pass | 19 | 13 | N28, N33, N46, N47, N52 |
| `mulgonet-2026` | P | pass | 15 | 17 | N02, N03, N04, N05, N07, N18 |
| `pathhdnn-2025` | P | pass | 15 | 20 | N02, N03, N07, N18, N19, N20, N21, N24, N25 |
| `pnet-2021` | P | pass | 16 | 23 | N02, N03, N07, N18, N19, N24, N25 |

Counting rules:

- **kpnn2 lines:** physical lines that call a `kpnn2.` name or read a
  public attribute or method of a spec, hop, or kpnn2 layer.
- **Glue lines:** lines tagged `# GLUE`. A line is tagged when it
  serves a card need mapped `glue` or `missing`. Every physical line
  of such a statement is tagged.
- **Not tagged:** ordinary model code, and code for needs mapped `out
  of scope`. That covers heads, norms, dropout, the steady-state loop,
  losses, and the Captum calls.
- A line can count in both columns.
- Not counted at all: imports, `def` lines, comments, and `print`
  diagnostics.

## Per paper

### `drugcell-2020`

- Sketch: `sketches/drugcell-2020.py`, pass. 17 nodes. Terms have 6
  units. Genes are annotated at the bottom, middle, and top terms.
  `Tb1` has two parents.
- Needs:
  - N01 covered: `parse_layered`. Term rows and gene rows both become
    edges after renaming `child`/`parent` to `source`/`target`. The
    `type` column is ignored.
  - N06 covered: gene rows are edges, and `align_inputs` matches
    names. An unannotated gene column (`g11`) is ignored.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N11 out of scope: **Typical `nn.Module` shape**, step 4. A
    per-term `BatchNorm1d(k)` is the same as one `BatchNorm1d` over
    the layer's units, because BatchNorm normalizes each unit on its
    own.
  - N12 covered: `PackedLinear` on 6×6 term blocks and 6×1 gene
    blocks.
  - N26 covered: `parse_layered` (longest path), skip pairs in hops,
    and `gather_hop_inputs`. A term's longest-path depth is DrugCell's
    height plus 1. Genes at any height are skip sources
    (`source_layers` `(0, 1)` and `(0, 2)`).
  - N31 covered: `map_node_attributions(hop_output=)` on the post-BN
    activations. A node's name repeats once per unit.
  - N38 covered: `parse_layered(widths={term: 6})`.
  - N39 covered: `PackedLinear` stores no weight for an absent edge,
    so the init mask and the per-step gradient mask go away.
    `MaskedLinear` would also hold absent entries at exactly 0.
  - N40 out of scope: **Package philosophy** (Division of labor:
    heads). `LayeredSpec.node_units` finds each term's 6 units for
    its head.
  - N41 out of scope: **Package philosophy** (Division of labor: the
    user writes `forward()`).
- Author glue:
  - Read ontology, split annotations, map gene names (~43): removed.
    One rename line remains.
  - Check genes below every term, one root, one component (~38):
    shrunk. A term with no gene below becomes an input node, and
    `align_inputs` then raises (checked in a probe). The one-root
    check is 2 GLUE lines. kpnn2 does not check for one component;
    the sketch does not either.
  - Masked gene-selection layer per term (~10): removed (see Notes).
  - Order terms by peeling leaves; build per-term layers and heads
    (~43): shrunk to one `PackedLinear` per layer plus a `ModuleDict`
    of heads.
  - Concatenate child outputs and gene values in forward (~28):
    removed by `gather_hop_inputs`.
  - One 0/1 mask per term (~18): removed.
  - Mask initial weights and every gradient (~13): removed.
  - Write per-term activations (~4): removed by
    `map_node_attributions`.
- Notes:
  - The code's per-annotation gene layer is not reproduced. It holds
    one weight and one bias per annotation, with no nonlinearity. That
    is a linear reparameterization of the direct 6×1 gene→term block,
    and the term's own bias absorbs the extra bias. So the function
    class is the one the paper describes, but training dynamics
    differ.
  - Three fragile spots disappear because there are no masks: the
    lookup by name prefix, masking only inside `train_model`, and the
    set-ordered gene rows.
  - Initialization ×0.1 is user code and is not reproduced.

### `kpnn-2020`

- Sketch: `sketches/kpnn-2020.py`, pass. 26 edges before pruning. It
  has two labeled roots at different depths (`CD8` at layer 3, `CD4`
  at layer 4), one unlabeled root `XYZ` with a dead branch below it,
  one unmeasured gene, and one skip edge `g8_gene → R1`.
- Needs:
  - N01 covered: `parse_layered`. It rejects duplicates and cycles,
    as `KPNN_Function.py:508` does.
  - N04 missing: a pandas loop (6 GLUE lines, shared with N28). If it
    is skipped, an unmatched gene leaf becomes an input node, and
    `align_inputs` raises.
  - N06 covered: TF→gene rows are ordinary edges. `align_inputs`
    matches by name and ignores extra data columns. The `_gene`
    suffix is a one-line rename.
  - N10 covered: default width 1 and the `PackedLinear` bias.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N12 covered: `PackedLinear`.
  - N15 out of scope: **Package philosophy** (Division of labor:
    attribution algorithms). Captum runs on each hop's sigmoid.
  - N17 covered: `map_node_attributions(hop_output=)`.
  - N18 glue: `.mean("observation")` in xarray (2 lines).
  - N26 covered: longest-path layering with skip pairs and
    `gather_hop_inputs`. KPNN's leaf-peeling order fixes the same
    dependencies.
  - N27 glue: the roots are `output_nodes`, but they sit in different
    layers. Their logits are gathered with `LayeredSpec.node_units`
    and `torch.cat` (2 lines). kpnn2 has no output-side counterpart
    to `align_inputs`.
  - N28 missing: the same loop as N04. It removes `XYZ`, then `S3`,
    then `TF4`, until stable.
  - N29 missing: siblings are counted with a pandas merge, then a
    per-unit keep vector per layer drives a Bernoulli mask in
    `forward` (6 GLUE lines).
  - N30 out of scope: **Package philosophy** (Division of labor). The
    sketch uses Captum's layer gradient
    (`LayerGradientXActivation(multiply_by_inputs=False)`). That is
    the ε→0 limit of the paper's central difference. An exact ±ε
    score needs a forward hook.
  - N31 covered: `map_node_attributions` on the saved activations, as
    the DrugCell sketch shows. Per-class means are xarray.
  - N32 covered: `to_edgelist`, `edge_location`, and
    `effective_weight` (2 lines). Biases could be read with
    `node_units` (not sketched).
  - N33 glue: not sketched. It would be `xr.concat` over runs, a
    filter on test error, and scaling.
  - N34 missing: the simulated control inputs are data generation.
    Not sketched.
  - N35 covered, for the coded `--shuffleGenes` control:
    `align_inputs` on a permuted name list (1 line). The paper's
    edge-swap control is not in the repository. It would be pandas,
    and `parse_layered` would reject a swap that adds a cycle.
- Author glue:
  - Match data rows to graph leaves (~15): removed by `align_inputs`.
  - Match label columns to roots (~10): shrunk to 2 lines with
    `node_units`.
  - Prune unlabeled roots and unmatched leaves (~15): left, 6 lines.
  - Topological order with cycle check (~20): removed by
    `parse_layered`.
  - Child index lists and weight blocks per node (~45): removed by
    the hops.
  - One op group per node (~25): removed. One `PackedLinear` per
    layer plus `gather_hop_inputs` replaces them.
  - Graph-dependent dropout per node (~15): left, 6 lines. The
    regex-based parent lookup (a fragile spot) is gone.
  - Activation perturbation hook (~6): removed in the sketch by
    switching to Captum's gradient. It is left if the exact ±ε score
    is wanted.
  - Finite-difference node scores by name (~30): shrunk to about 6
    lines of Captum plus `map_node_attributions`.
  - Edge weights and activations by name (~30): shrunk to 2 lines.
    `edge_location` ties each slot to its edge by name. That replaces
    the implied stacking order, which was a fragile spot.
  - Control inputs; input shuffle (~40): the shuffle is shrunk to 1
    line. The control simulation is left.
  - Collect, filter, and scale scores across replicates (~40, R):
    left.
- Notes:
  - KPNN puts no sigmoid on its outputs. The sketch reads logits from
    pre-activations and applies the sigmoid to whole layers. That is
    harmless, because output units feed nothing.
  - A layer that holds only outputs (`CD4`) therefore has an unused
    activation. Captum cannot attribute to it, so the score loop
    skips that layer.
  - `gather_hop_inputs` caught a float64 keep-probability tensor that
    the dropout glue introduced.
  - The control-mode fragile spot (extra roots after pruning) cannot
    occur. The sketch prunes the final edge list before parsing it.

### `lembas-2022`

- Sketch: `sketches/lembas-2022.py`, pass. The signed, cyclic network
  has 11 nodes after pruning. It has the loops `K1 → K3 → K1` and
  `K2 → T1 → K2`, and a ligand `L3` with no data column. Two drugs
  project onto kinases (the viability variant).
- Needs:
  - N06 covered (viability variant). The drug→target pairs go through
    `parse_layered` as one hop and a bias-free `PackedLinear`. They
    land on the state through `AdjacencySpec.node_units` (2 GLUE
    lines).
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads). Not sketched.
  - N10 covered.
  - N12 covered: `PackedLinear` on `AdjacencySpec` indices. This
    removes the CSR `.data` order fragile spot. Slots are named by
    `edge_location` and `to_edgelist`.
  - N17 covered: `map_node_attributions` with
    `dims=("node", "tf")` and `coords=` on the knockout effects.
  - N28 missing: a networkx reachability prune (4 GLUE lines, shared
    with N52). It removes `K5`.
  - N31 covered: `forward` returns the full state, and
    `map_node_attributions(state, spec)` names it.
  - N32 covered: `to_edgelist().assign(weight=effective_weight())`.
    The packed order equals the `to_edgelist` order at width 1.
  - N33 glue: not sketched. It would be a median over fold models in
    xarray.
  - N42 covered: `parse_adjacency` keeps the cycles. The sign column
    stays on the caller's table, and `edge_location` puts it on the
    packed slots (3 lines). Duplicate rows raise, where LEMBAS sums
    them silently (a fragile spot).
  - N43 out of scope: **What kpnn2 is NOT** (Not a time machine). The
    loop is 4 lines in `forward`. See Notes for the gradient.
  - N44 out of scope: **Package philosophy** (per-edge signs stay
    with the caller). The sign-aware init and the L1 penalty on
    `effective_weight()` take 3 lines.
  - N45 covered: `align_inputs` and `input_index`. The scale is user
    code. A graph ligand with no data column needs
    `reindex(fill_value=0)` (1 GLUE line), because `align_inputs`
    rejects missing names.
  - N46 glue: `T1` feeds `K2`, so it is not in `output_nodes`. The
    TFs are read with `AdjacencySpec.node_units` (2 GLUE lines).
  - N47 glue: the init rescale needs a dense matrix rebuilt from
    packed slots, then `eigvals` (4 GLUE lines). `to_mask()` gives
    the pattern, not the weights. The training penalty is a loss
    (**What kpnn2 is NOT**: Not a trainer) and is not sketched.
  - N48 out of scope: **What kpnn2 is NOT** (Not a trainer). Not
    sketched.
  - N49 out of scope: **Package philosophy** (Division of labor).
    `forward` takes an `offset` on the pre-activation, and a −3
    identity batch gives every knockout at once.
  - N50 out of scope: the same `offset` mechanism. Not sketched.
  - N51 covered: `edge_location` writes values per edge (shown for
    signs), and biases go by `node_units`.
  - N52 missing: the same prune as N28. It removes `K9`.
  - N53 covered: the drug→target `PackedLinear` stores no absent
    entries. That replaces the per-batch re-zeroing and its last
    stale update.
- Author glue:
  - Clean OmniPath edges (~70): left. kpnn2 only rejects duplicates
    and bad names.
  - Reachability prune and pass-through removal (~65): the prune is
    left at 4 lines. Pass-through removal is not sketched.
  - Edge table to indices, sign masks, node order (~35): removed by
    `parse_adjacency`. Signs shrink to 3 lines with `edge_location`.
  - Sparse recurrent forward and adjoint backward (~75): the forward
    is removed (`PackedLinear` plus a 4-line loop). The adjoint is
    dropped in favor of unrolled autograd (see Notes).
  - Spectral radius and its gradient (~110): the init is shrunk to 4
    lines. The penalty and the sparse eigensolver gradient are left.
  - Sign-aware init and sign violations (~30): shrunk to 3 lines.
  - Align columns, input scatter, output gather (~45): shrunk.
    `align_inputs` and `input_index` remain, plus 2 gather lines and
    1 zero-fill line.
  - Save and load parameters by edge name (~70): shrunk. Saving is 1
    line; loading is the 2-line `edge_location` loop.
  - Knockout and sensitivity batches (~40): shrunk to an `offset`
    argument and 2 lines.
  - Drug-target mask and per-batch re-masking (~30): removed.
- Notes:
  - Inputs are added to the pre-activation at every step,
    `h ← f(A·h + x + b)`, as LEMBAS does. The CONTEXT example instead
    writes inputs into the state, which is a different update.
  - Backprop through the unrolled loop is not LEMBAS's gradient.
    LEMBAS uses an adjoint fixed point at the steady state; the
    unrolled loop backpropagates through every step and stores all
    of them. Matching LEMBAS would need a custom autograd function
    again. This sits under the "time machine" lock.
  - The zero-filled column made the host matrix float64.
    `PackedLinear` casts its input by itself, but the user's
    `index_add` does not. The sketch builds `x` as float32.

### `mulgonet-2026`

- Sketch: `sketches/mulgonet-2026.py`, pass. BP and MF graphs,
  3 levels (the paper uses 5), 4 omics, 26 nodes in total. The toy
  exercises four drops:
  - `B112` has too many genes (N05), so it loses its annotations.
  - `B211` is not annotated.
  - `B21` and `B2` lose all their children and are pruned.
  - `B1111` is deeper than the cut.
- Needs:
  - N01 covered: `parse_layered`. The namespace filter is one pandas
    line.
  - N02 glue: `ranks=` takes the depths. `roots[0]` and the BFS from
    it are networkx (5 GLUE lines including the rank map).
  - N03 missing: a 2-line filter.
  - N04 missing: a loop of 3 GLUE lines. It runs after the bottom
    cut, so it also prunes the terms MULGONET keeps as bias-only
    units (see Notes).
  - N05 missing: 1 pandas line.
  - N06 covered: annotation rows are edges, and `align_inputs`
    ignores features that no term annotates.
  - N07 glue: a cross join with the omics names (3 GLUE lines).
  - N08 covered: one `parse_layered` per namespace and one
    `align_inputs` per spec on the same columns. A BP-only gene has
    no MF column at all, where MULGONET uses an all-zero mask row.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N10 covered.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N12 covered: `PackedLinear`.
  - N13 out of scope: **What kpnn2 is NOT** (Not a trainer). It is
    1 line with `effective_weight()` of each branch's first hop.
  - N14 out of scope: **Package philosophy** (Division of labor).
    Captum IG runs on the tuple of branch inputs. Each branch is
    named with `axis="inputs"`, and the two are summed by name
    (2 GLUE lines).
  - N15 out of scope: `LayerIntegratedGradients` per activation,
    named with `hop_output=`.
  - N16 out of scope: Captum IG with 20 steps, as the paper states.
    The repository's variant, which takes the gradient at the real
    input at every step, is not reproduced.
  - N17 covered: `map_node_attributions`, plus the branch merge
    above.
  - N18 glue: a sum over recurrence-positive samples in xarray.
- Author glue:
  - Read GO edges per namespace; find roots (~16): shrunk to 2 lines.
  - Assign depth layers (~15): shrunk to the BFS plus `ranks=`.
  - Remove terms without a child (~16): left, 3 lines.
  - Term×term 0/1 matrix per layer pair (~19): removed.
  - Load annotations; drop terms over 200 genes (~29): the filter is
    left (1 line).
  - Cut the bottom layer to annotated terms (~7): removed. The prune
    loop covers it.
  - Keep omics features that have a term (~17): removed.
    `align_inputs` ignores the rest.
  - Gene×term masks per omics (~55): shrunk to the 3-line cross join.
  - Concatenate omics inputs in mask order (~6): removed.
    `align_inputs` works by name.
  - Edge-vector layer scattered into a dense kernel (~86): removed by
    `PackedLinear`.
  - Wire five layers per branch (~48): shrunk to a 15-line `Branch`
    module.
  - Per-layer attribution loop and sum (~77): shrunk to about 8
    lines.
  - Label scores; write per layer (~45): removed by
    `map_node_attributions`.
- Notes:
  - kpnn2 would infer a bias-only term as an input, and `ranks=`
    rejects it (checked). Pruning such terms does not shrink the
    function class: a constant unit's contribution is absorbed by its
    parent's bias. Only the dropout noise on that unit differs.
  - Alignment by position, the fragile spot behind every mask, is
    replaced by names.
  - The `roots[0]` fragile spot is kept on purpose, for fidelity.

### `pathhdnn-2025`

- Sketch: `sketches/pathhdnn-2025.py`, pass. 3 levels (the paper
  uses 4). The toy exercises five cases:
  - Branches without data (`R3`, `R212`) are removed.
  - `R1111`, below the cut, passes its features up to `R111`.
  - The short leaf `R12` gets one copy.
  - `g5_mut`, annotated only above the bottom layer, is not an input.
  - `g4_amp` is a model feature that is missing from the data.
- Needs:
  - N01 covered: `parse_layered`.
  - N02 glue: `ranks=`, plus a root and BFS in networkx.
  - N03 missing: a 2-line filter.
  - N06 covered: membership rows are edges, and `align_inputs`
    matches them. A feature missing from the data needs
    `reindex(fill_value=0)` (1 GLUE line).
  - N07 glue: the membership table arrives with per-type names, as
    `DataProcessing.R` builds it. That expansion stays user code.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N10 covered.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N14 out of scope: **Package philosophy** (Division of labor).
    Captum `LayerDeepLiftShap(attribute_to_layer_input=True)` runs on
    each masked layer and on the output layer.
  - N15 out of scope: as N14.
  - N17 covered: `map_node_attributions(hop_input=)`, and `layer=`
    for the output layer's input.
  - N18 glue: the mean of |SHAP| in xarray.
  - N19 missing (lock): the copy chain is networkx (see the `pnet`
    entry).
  - N20 missing: a networkx ancestor subset (2 GLUE lines).
  - N21 missing: networkx descendants (1 GLUE line).
  - N22 covered: `MaskedLinear(hop.to_mask())` (see Notes).
  - N23 out of scope: Captum DeepLiftShap with the training samples
    as the background.
  - N24 missing: not sketched; the script turns it off. The log2 of
    the up- plus downstream count would be one networkx line.
  - N25 glue: `to_edgelist()` merged with the named scores (2 lines).
- Author glue:
  - Per-type feature names and membership tables (R, ~60): left,
    outside the sketch.
  - Read the hierarchy, add a root, cut to annotated pathways (~45):
    shrunk to 5 lines.
  - Propagate memberships and attach features (~50): shrunk to 2
    lines.
  - Layer by depth, pad branches, build sorted 0/1 matrices (~80):
    shrunk to 9 lines. The matrices are removed.
  - Apply masks via pruning; record unit names (~44): removed by
    `MaskedLinear` and the spec's names.
  - Align data rows to the feature list (~24): shrunk to
    `align_inputs` plus the 1 zero-fill line.
  - Per-layer SHAP into a named edge table (~100): shrunk to about 8
    lines.
  - Importance graph, normalization, subgraphs (~190): mostly left.
    `to_edgelist` supplies the edge table.
- Notes:
  - Three fragile spots are fixed:
    - Init after pruning: `MaskedLinear`'s degree-aware init reaches
      the live entries, where PathHDNN's later Xavier call misses the
      masked layers.
    - Copies sharing an id: kpnn2 requires distinct names.
    - Depth-L nodes with a kept child take no features: every node at
      the cut now takes its descendants' features.
  - Masked entries are exactly 0 and get no gradient, so weight decay
    has nothing to move.

### `pnet-2021`

- Sketch: `sketches/pnet-2021.py`, pass. 28 nodes. It has 3 pathway
  levels (the paper uses 5), 3 inputs per gene, a same-layer edge
  `P11 → P112`, two short leaves that get copies, and one gene with no
  pathway.
- Needs:
  - N01 covered: `parse_layered`. networkx is kept for N02.
  - N02 glue: `ranks=` takes the depths. The virtual root, the BFS
    with a cutoff, and the rank map are networkx (7 GLUE lines).
  - N03 missing: a 2-line filter. kpnn2 would keep a layer-skipping
    edge as a skip, and it rejects a same-layer edge under `ranks=`
    (checked).
  - N06 covered: gene-set rows are edges.
  - N07 glue: the name expansion is 1 line. After it, the gene layer
    is hop 0, a `PackedLinear` with 3 edges per gene. That is P-NET's
    `Diagonal`, matched by name instead of column position.
  - N10 covered.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N12 covered: `PackedLinear`. `SparseTF` uses the same
    parameterization, one vector entry per edge.
  - N13 out of scope: **What kpnn2 is NOT** (Not a trainer). See the
    `mulgonet` entry.
  - N14 out of scope: **Package philosophy** (Division of labor).
    Captum `DeepLift`, named with `axis="inputs"`.
  - N15 out of scope: `LayerDeepLift` per activation, named with
    `hop_output=`.
  - N17 covered: `map_node_attributions`.
  - N18 glue: a sum in xarray.
  - N19 missing: a 5-line copy chain. This touches a lock: kpnn2's
    answer is a skip edge, not a copy chain (see Gaps and Notes).
  - N24 missing: the degree from the edge list and the mean + 5 SD
    rule (3 GLUE lines). With `PackedLinear` every edge is live, so
    the edge-list degree equals P-NET's nonzero-weight degree.
  - N25 glue: `to_edgelist()` merged with the scores (3 GLUE lines).
    Drawing is left.
  - N31 covered: `map_node_attributions` on the activations.
  - N32 covered: as in the `kpnn` sketch.
  - N36 out of scope: **Package philosophy** (Division of labor:
    heads). Each hop output gets a `Linear` and a sigmoid, with a
    weighted BCE.
  - N37 out of scope: Captum DeepLift from a zero baseline. Captum
    computes the hidden-layer references that P-NET's vendored
    DeepExplain had to patch in.
- Author glue:
  - Read Reactome and GMT, filter to human, add a root (~75): shrunk.
    The root and BFS are 3 lines; file reading is outside the sketch.
  - Cut at depth, pad copies, child dicts, genes at the bottom (~75):
    shrunk to about 10 lines. The child dicts are removed.
  - 0/1 map per layer, align genes, chain names (~50): removed.
  - Gene layer by column blocks; edge-vector layer (~135): removed by
    `PackedLinear`.
  - Assemble layers, heads, dropout, names (~155): shrunk to a
    25-line module. Names come from the spec.
  - Order data columns by gene, then type (~50): removed.
    `align_inputs` accepts any column order.
  - Label scores, activations, and edge weights (~170): shrunk.
  - Degree from trained weights; adjustment (~80): left, 3 lines.
  - Hidden-layer references in DeepExplain (~8): removed, by Captum
    rather than by kpnn2.
  - Sankey (~90): the edge table shrinks to 3 lines; the plot is
    left.
- Notes:
  - The copy chain could not use P-NET's shared names. A copy named
    like its original would be a self-loop, which the parser rejects.
    So the copies need distinct names.
  - A short leaf can sit at its official depth with only skip parents
    (checked). That is kpnn2's alternative to a copy chain. It is a
    different model: there is no tanh and no bias per copy.
  - Real Reactome would need a prune pass that P-NET does not do.
    P-NET keeps pathways with no data gene, which kpnn2 would infer
    as inputs and `ranks=` would reject. The toy avoids such
    pathways. The `mulgonet` note explains why pruning them changes
    nothing.
  - Reloading by position (a fragile spot) is caught by
    `identity=spec.fingerprint` and `index_digest`.

## Gaps

Needs that are `missing` or `glue` in two or more papers.

| Need | Papers | What an abstraction would do | Lock touched |
|------|--------|------------------------------|--------------|
| N18 Aggregate scores over a sample subset | `kpnn-2020`, `pnet-2021`, `mulgonet-2026`, `pathhdnn-2025` (glue) | Registered plain folds over `observation`: sum, mean, and mean of absolute value, optionally limited to one label. Today the only method, `rauter_mangano_2026`, raises "not yet implemented" (checked), so every sketch folds in xarray. | None. `AGENTS.md` **How to add an aggregation method** needs no dispatcher change. |
| N02 Layer nodes by depth from a root | `pnet-2021`, `mulgonet-2026`, `pathhdnn-2025` (glue) | Shortest-path depth from a virtual root over the parentless nodes, a cut at L levels, and the mapping into `ranks=`. The same 5–7 lines appear in all three sketches. | **Graph rules**, Layering: `ranks=` is the documented hatch and depths are the caller's. A ranking mode or a helper would grow the public API. |
| N03 Keep only edges between adjacent layers | `pnet-2021`, `mulgonet-2026`, `pathhdnn-2025` (missing) | Given ranks, drop same-rank and layer-skipping edges before the parse. | **Skip connections**: kpnn2 keeps a layer-skipping edge as a skip on purpose, and `ranks=` makes same-rank edges illegal. A pre-parse table helper would touch neither. |
| N07 One input per entity and data type | `pnet-2021`, `mulgonet-2026`, `pathhdnn-2025` (glue) | Repeat an entity's edges once per data-type suffix. It costs 1–3 lines of pandas per sketch. | **`align_inputs`**: a wide input node repeats one column, so `widths=` cannot carry different data types. |
| N04 Prune nodes with no child below | `kpnn-2020`, `mulgonet-2026` (missing) | Drop nodes with no path from a declared feature set, until stable. kpnn2 refuses such nodes but does not prune them: they become inputs that `align_inputs` rejects, or `ranks=` raises. | **Graph rules**: node roles are inferred from degree, not declared. A pruning helper needs the feature and label names declared. |
| N28 Prune nodes that cannot reach an output | `kpnn-2020`, `lembas-2022` (missing) | Drop nodes with no path to a declared output set. This is the same family as N04, N20, and N52, which together appear in 4 of the 6 papers. | As N04. |
| N19 Pad short branches with copy nodes | `pnet-2021`, `pathhdnn-2025` (missing) | Build the copy chain. | **What kpnn2 is NOT** (Not pseudo-node expansion), and `parse_layered(..., ranks=)` ("Dummy depth-padding nodes are not an acceptable workaround"). kpnn2's answer is a skip edge. Stop there. |
| N24 Normalize node scores by connectivity | `pnet-2021`, `pathhdnn-2025` (missing) | A per-node degree, or an up- plus downstream count, from a spec as a named DataArray to divide by. It could be a keyword argument of an aggregation method. | None. |
| N25 Scores as a graph | `pnet-2021`, `pathhdnn-2025` (glue) | Join named node scores to `to_edgelist()` with layers. It costs 2–3 lines per sketch. | None. |
| N33 Combine scores across training repeats | `kpnn-2020`, `lembas-2022` (glue; not sketched) | A registry method that folds a run dimension (median or mean) after the caller filters the runs. | None. CONTEXT says the dispatcher does not guess that `step` is `seed`, so the caller names the run dimension. |

Watch list, one paper only:

| Need | Paper | Note |
|------|-------|------|
| N05 Filter nodes by gene-set size | `mulgonet-2026` | 1 pandas line. |
| N20 Keep only nodes above the data | `pathhdnn-2025` | Reachability family (N04, N28, N52). 2 networkx lines. |
| N21 Propagate memberships up the hierarchy | `pathhdnn-2025` | 1 networkx line. |
| N27 Output units are named graph roots | `kpnn-2020` | Same pattern as N46: read named nodes as outputs when they are not one layer or not all sinks. Both sketches use a `node_units` comprehension. Two papers under two IDs. |
| N29 Dropout rate per node from the graph | `kpnn-2020` | 6 GLUE lines. The largest single-paper glue block. |
| N34 Structure-only baseline from control inputs | `kpnn-2020` | Data generation. Not sketched. |
| N46 Read outputs from named graph nodes | `lembas-2022` | See N27. |
| N47 Keep a recurrent edge matrix contractive | `lembas-2022` | A dense rebuild from packed slots for `eigvals`. `to_mask()` gives the pattern, not the weights. |
| N52 Prune nodes no input can reach | `lembas-2022` | Reachability family. |

## Orphans

Every name in `kpnn2.__all__`, and each documented keyword argument
and public method of those names.

| kpnn2 name or option | Used by | Need served, or "no application in this sample" |
|----------------------|---------|-------------------------------------------------|
| `parse_layered` | drugcell, kpnn, mulgonet, pathhdnn, pnet, lembas (drug hop) | N01, N26 |
| `parse_layered(widths=)` | drugcell | N38 |
| `parse_layered(ranks=)` | mulgonet, pathhdnn, pnet | N02 |
| `parse_adjacency` | lembas | N42 |
| `parse_adjacency(widths=)` | none | N38 on a cyclic graph. No paper here combines N38 with N42. |
| `LayeredSpec` | all layered sketches | N01 |
| `LayeredSpec.to_edgelist()` | kpnn, pathhdnn, pnet | N25, N32 |
| `LayeredSpec.edge_location()` | kpnn | N32 |
| `LayeredSpec.node_units()` | drugcell, kpnn | N27, N40 |
| `LayeredSpec.hop_units()` | none | No application in this sample. DrugCell's RLIPP would read child units per term, but it is paper-only and not a need. |
| `LayeredSpec.to_dict()` / `from_dict()` | none | N51 ("reload a trained one"). P-NET reloads by position (a fragile spot). |
| `LayeredSpec.fingerprint` | all six (`identity=`) | Checkpoint identity (P-NET reload fragile spot) |
| `Hop` (`in_features`, `out_features`) | all layered sketches, lembas (drug hop) | N12 |
| `Hop.to_mask()` | pathhdnn | N22 |
| `Hop.column_offsets` | none | No application in this sample |
| `Skip` | none (kpnn prints `len(spec.skips)` only) | Metadata. Skip edges themselves serve N26 through the hops in drugcell and kpnn. |
| `AdjacencySpec` (`state_dim`, `input_index`) | lembas | N42, N45 |
| `AdjacencySpec.node_units()` | lembas | N06 (drug targets), N46 |
| `AdjacencySpec.to_edgelist()` | lembas | N32 |
| `AdjacencySpec.edge_location()` | lembas | N44, N51 |
| `AdjacencySpec.to_mask()` | none | No application in this sample. N47 needs the weights, not the pattern. |
| `AdjacencySpec.to_dict()` / `from_dict()` | none | N51 |
| `AdjacencySpec.fingerprint` | lembas | Checkpoint identity |
| `AdjacencySpec.output_index` (field) | none | N46 in principle. LEMBAS's TF `T1` is not a sink, so the sketch reads TFs by name. |
| `MaskedLinear` (`identity=`) | pathhdnn | N22 |
| `MaskedLinear(bias=)` | default only | N10 |
| `MaskedLinear(constraint=)`, `(generator=)` | none | See the `PackedLinear` rows |
| `MaskedLinear.effective_weight()`, `init_bound()`, `reset_parameters()` | none | N13 or N32 on a dense layer. Not exercised. |
| `PackedLinear` (`identity=`) | drugcell, kpnn, lembas, mulgonet, pnet | N12 |
| `PackedLinear(bias=False)` | lembas (drug→target) | N06, N53 |
| `PackedLinear(constraint=)` | none | Nearest is N44, but LEMBAS's sign prior is soft (a loss term). No paper here holds a hard sign or a freeze. |
| `PackedLinear(generator=)` | none | N33 (independent init per repeat). No sketch trains repeats. |
| `PackedLinear.effective_weight()` | kpnn, lembas, mulgonet | N13, N32, N44 |
| `PackedLinear.init_bound()` | none | No application in this sample. LEMBAS replaces the init. |
| `PackedLinear.reset_parameters()` | none | N33 (re-initialize between repeats; PathHDNN's unused `explain_average`) |
| `PackedLinear.transpose()` | none | No application in this sample (no decoder) |
| `PackedMultiheadAttention` and all its arguments (`dropout`, `bias`, `kdim`, `vdim`, `batch_first`, `add_self_loops`, `identity`, `generator`, `chunk_size`; forward `key_padding_mask`, `need_weights`, `average_attn_weights`) | none | No application in this sample. The one attention paper, `sctransformer-2026`, is excluded as no-code. |
| `gather_hop_inputs` | drugcell, kpnn, mulgonet, pathhdnn, pnet | N26 |
| `scatter_hop_outputs` | none | No application in this sample (no decoder) |
| `align_inputs` | all six | N06, N35, N45 |
| `map_node_attributions` | all six | N17, N31 |
| `map_node_attributions(layer=)` | pathhdnn | N23 at the output layer's input |
| `map_node_attributions(hop_output=)` | drugcell, kpnn, mulgonet, pnet | N15, N31 |
| `map_node_attributions(hop_input=)` | pathhdnn | N23 at the inputs of the masked layers |
| `map_node_attributions(axis="inputs")` | mulgonet, pnet | N14 |
| `map_node_attributions(dims=, coords=)` | lembas | N49 (node × TF effects) |
| `map_node_attributions` on a tensor sequence (`step` axis) | none | No application in this sample. LEMBAS uses only the steady state. |
| `aggregate_node_attributions` (`method=`, `labels=`, method kwargs) | none | N18 (4 papers) and N33 (2 papers). Unusable today, because its only method raises. |
| `list_aggregation_methods` | none | As above |
| `Kpnn2Error` | no sketch catches it | It fired twice while the sketches were written: the `gather_hop_inputs` dtype check (kpnn), and `align_inputs` on a geneless term (DrugCell probe). |
| `__version__` | none | Metadata. No application in this sample. |

## Friction

Glue that repeats in three or more sketches:

- **Depth-from-root graph shaping** (pnet, mulgonet, pathhdnn). Each
  sketch builds a virtual root (or takes `roots[0]`), runs a BFS with
  a cutoff, filters to adjacent edges, maps depths into `ranks=`, and
  in two of them pads copy chains. That is 13–15 GLUE lines per
  sketch, the largest glue in the sample. The three repositories
  share P-NET's `reactome.py` lineage, so this is one recipe with
  three users.
- **Folding observations** (kpnn, pnet, mulgonet, pathhdnn).
  `.sum("observation")` or `.mean("observation")` follows every
  `map_node_attributions` call, because `aggregate_node_attributions`
  cannot do it yet.
- **Reachability pruning before the parse** (kpnn, mulgonet, lembas,
  plus pathhdnn's ancestor subset). kpnn2 infers roles from degree,
  so an unpruned leaf or a bias-only node fails loudly, in
  `align_inputs` or `ranks=`. That is good, but the prune itself is
  user code in 4 of 6 sketches.
- **Boilerplate, not glue** (all five layered sketches). The 3-line
  `PackedLinear(h.source_index, h.target_index, h.out_features,
  h.in_features, identity=spec.fingerprint)` comprehension and the
  `saved` / `gather_hop_inputs` loop repeat everywhere, as **Typical
  `nn.Module` shape** prescribes. A model class is locked out
  (**Package philosophy**: no `LayeredNet` unless asked).

kpnn2 calls the sketches had to work around:

- `align_inputs` rejects missing names. PathHDNN and LEMBAS expect
  zeros for model inputs missing from the data, so both sketches call
  `reindex(..., fill_value=0)` first. The fill needs the matrix,
  which `align_inputs` does not take by design (**Locked contrasts**).
  In LEMBAS the zero-filled column then made the host matrix float64.
- Named readouts outside `input_nodes` (kpnn roots in several layers;
  LEMBAS TFs and drug targets) are indexed with a `node_units(...)`
  comprehension. There is no output-side or arbitrary-node
  counterpart to `align_inputs`.
- Two branches on one input (mulgonet) give input attributions per
  spec. The sketch sums them by name.
- Concatenating per-layer maps into one node table (kpnn) needs
  `xr.concat(..., coords="different", compat="equals")`. Without it,
  xarray warns about the scalar `layer` coordinate.
- Outputs read as logits (kpnn) leave a layer of only outputs with an
  unused activation, which Captum cannot hook. That is a property of
  KPNN, not a kpnn2 defect.

## Changes since the previous run

First run.

## Caveats

- **Paper code:** none was opened. The cards were the only source.
  `.paper-code/` exists for all six papers but was not needed.
- **Cards and catalog:** no card looks wrong. Three observations:
  - `README.md`'s Pilot table still lists `sctransformer-2026`, which
    `papers.csv` marks excluded (no-code).
  - N27 and N46 describe one pattern from two sides (outputs are
    named nodes). Worth a look the next time `needs.md` grows.
  - `pnet-2021` does not list N04, but kpnn2 would force a prune on
    P-NET's pathways without a data gene (the card's fragile spot).
- **Simplifications:**
  - Toy depths of 3 (the papers use 4 or 5), with 17–28 nodes per
    graph.
  - Training is one backward pass, so scores come from untrained
    weights.
  - KPNN uses Captum's gradient instead of ±ε.
  - MULGONET uses proper IG instead of the repository's variant.
  - DrugCell drops the per-annotation gene layer (a
    reparameterization).
  - LEMBAS backpropagates through the unrolled loop instead of using
    an adjoint.
- **Not sketched:** N24 (pathhdnn), N33 (kpnn, lembas), N34 (kpnn),
  N48, N50, the training-time spectral penalty, and pass-through
  removal (lembas). Their mappings are reasoned from the contract and
  from the other sketches.
- **Sketch size:** 54–68 code lines each, not counting imports,
  `def` lines, comments, prints, or blank lines. LEMBAS is the
  longest because it includes the viability drug projection. The
  sketches pack arguments onto shared lines, against `AGENTS.md`'s
  one-argument-per-line rule, to fit the ~60-line budget. Ruff skips
  `tests/paper_fit/`.
- **Line counts** are physical lines and depend on wrapping. Read
  them as orders of magnitude.
- **Environment:** Captum 0.9.0, torch 2.13.0, networkx 3.6.1 (used
  in the glue only). Captum prints harmless DeepLift hook warnings.
