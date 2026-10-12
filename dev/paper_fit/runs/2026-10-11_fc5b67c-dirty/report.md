# Paper-fit run 2026-10-11

| Field | Value |
|-------|-------|
| kpnn2 version | 0.2.0 |
| kpnn2 commit | `fc5b67c` dirty (uncommitted: `AGENTS.md`, paper-fit `README.md`, `needs.md`, `papers.csv`, four new cards, `propose_prompt.md`, `proposals/`; nothing under `src/`) |
| Scope | All included papers |
| Cards | 22: `celltics-2023`, `deepgami-2023`, `deephiscom-2022`, `deepkegg-2024`, `deepsignalingflow-2024`, `dlbcl-survival-2024`, `drugcell-2020`, `genomic-prediction-2025`, `gsnn-2024`, `kpnn-2020`, `lembas-2022`, `movida-2023`, `mulgonet-2026`, `pathexpsurv-2023`, `pathhdnn-2025`, `pideel-2023`, `pinnet-2023`, `pnet-2021`, `reliable-interpretability-2023`, `self-pruning-2025`, `sigprimednet-2023`, `tumor-cell-architecture-2024` |
| Previous run | `runs/2026-10-09_759ba0b/report.md` |
| Agent | Claude Code, Claude Opus 5.5 (`claude-opus-5-5`) |

## Summary

Verdicts are 16 R, 6 P, and 0 N over 22 papers, and all 22 sketches run.
Given a paper's final edgelist, kpnn2 builds the connectivity of every
paper in the sample, including private per-node networks
(`widths=` plus one hidden node per entity), cyclic message passing, and
frozen edges through `constraint=`. The main gaps are shared score
handling and graph statistics: N18 (8 papers fold scores over samples by
hand, because `aggregate_node_attributions` has no usable method), N33
(6 papers combine repeats), named nodes read or written outside
`input_nodes` and whole layers (5 papers, N27/N46/N55/N82 and N45), and
per-node degree or set size computed by hand (7 papers,
N24/N67/N81/N99/N103/N104). Orphans are `PackedMultiheadAttention`,
`scatter_hop_outputs`, `PackedLinear.transpose()` and
`reset_parameters()`, `Hop.column_offsets`, `AdjacencySpec.to_mask()` and
its `to_dict()`/`from_dict()`, `MaskedLinear`'s `constraint=`,
`generator=`, and methods, and the aggregation pair. Since the previous
run, `ranks=`, `parse_adjacency(widths=)`, `hop_units`, `from_dict`,
`output_index`, `constraint=`, `generator=`, and `init_bound()` gained
their first users.

## Verdicts

| Paper | Verdict | Sketch | kpnn2 lines | Glue lines | Needs left to glue or missing |
|-------|---------|--------|-------------|------------|-------------------------------|
| `celltics-2023` | R | pass | 11 | 8 | N18, N107, N108 |
| `deepgami-2023` | R | pass | 12 | 1 | N18 |
| `deephiscom-2022` | R | pass | 12 | 0 | None |
| `deepkegg-2024` | R | pass | 11 | 1 | N18, N33\* |
| `deepsignalingflow-2024` | P | pass | 12 | 12 | N25\*, N33\*, N45, N72, N81, N82 |
| `dlbcl-survival-2024` | R | pass | 14 | 2 | N67 |
| `drugcell-2020` | R | pass | 16 | 0 | None |
| `genomic-prediction-2025` | R | pass | 14 | 1 | N55 |
| `gsnn-2024` | P | pass | 11 | 8 | N64, N67\*, N68, N72\* |
| `kpnn-2020` | P | pass | 20 | 10 | N18, N27, N29, N33\*, N34\* |
| `lembas-2022` | P | pass | 21 | 9 | N33\*, N46, N47 |
| `movida-2023` | R | pass | 11 | 0 | None |
| `mulgonet-2026` | R | pass | 15 | 3 | N18 |
| `pathexpsurv-2023` | R | pass | 15 | 4 | N94 |
| `pathhdnn-2025` | R | pass | 15 | 5 | N18, N24\*, N25 |
| `pideel-2023` | R | pass | 8 | 4 | N97 |
| `pinnet-2023` | R | pass | 10 | 3 | N99 |
| `pnet-2021` | R | pass | 16 | 8 | N18, N24, N25 |
| `reliable-interpretability-2023` | R | pass | 17 | 12 | N24\*, N33, N34, N102, N103, N104, N105 |
| `self-pruning-2025` | P | pass | 12 | 11 | N33\*, N46, N47 |
| `sigprimednet-2023` | P | pass | 11 | 7 | N18, N109, N111 |
| `tumor-cell-architecture-2024` | R | pass | 19 | 4 | N76 |

\* Not sketched.

How the verdicts were drawn. The verdict says where the glue sits, not
how much there is.

- **R:** every glue line serves score analysis, training, data, or a
  per-node number the user computes for an argument kpnn2 already has
  (`widths=`, or a per-edge scale `CONTEXT.md` puts in `constraint=`).
  `reliable-interpretability-2023` is R with 12 glue lines, all of them
  score analysis.
- **P:** a need that maps the graph into the model is left to glue in
  construction or `forward`, with no kpnn2 argument to put it in:
  - named nodes read or written outside `input_nodes`, `output_nodes`,
    and whole layers;
  - per-node dropout;
  - unit-type masks for an edge-valued state;
  - matrix spectra that need a dense rebuild or a hand-written power
    iteration;
  - a prior used only to initialize a dense layer.
- **Out of scope (pre-edgelist):** a need that only decides which rows
  the edgelist holds, per the maintainer's ruling of 2026-10-09 (see
  Caveats).
- **missing vs glue:** `missing` when kpnn2 has no piece of that
  computation (degree, centrality, class gaps). `glue` when the user
  finishes something kpnn2 starts: a slot it has but cannot fill
  (`aggregate_node_attributions`), or an accessor the remaining code
  consumes (`to_edgelist`, `node_units`, hop indices, `widths=`).

Counting rules:

- **kpnn2 lines** are counted by a script. A line counts if it calls a
  `kpnn2.` name, reads a spec or hop attribute, or calls
  `effective_weight`, `init_bound`, `reset_parameters`, or `transpose`.
  The script reproduces the previous run's counts except for
  `lembas-2022`, which is 21 here and 19 there with the sketch
  unchanged.
- **Glue lines** are lines tagged `# GLUE`. A line is tagged when it
  serves a need mapped `glue` or `missing`.
- Imports, `def` and `class` lines, comments, and `print` lines are
  not counted.

## Per paper

The six papers of the previous run have the same cards, the same
sketches, and the same contract (`git diff 759ba0b fc5b67c -- src
CONTEXT.md` and their cards are empty). Their mappings were re-checked
against the cards, and none moved. They are kept short here; the full
notes are in the previous report.

### `celltics-2023`

- Sketch: `sketches/celltics-2023.py`, pass.
  - A final edgelist with 3 levels (the paper uses 5), 6 genes, and one
    copy node, `R12_copy1`.
  - Three networks, over the bottom 1, 2, and 3 graph layers.
- Needs:
  - N01 covered: `parse_layered`.
  - N02 covered: default layering. The depth cut is pre-edgelist.
  - N03, N19, N20, N21, N56, N73 out of scope (pre-edgelist).
  - N06 covered: `align_inputs`; `ENSG9` is ignored.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N10 covered.
  - N22 covered: `MaskedLinear(hop.to_mask())`.
  - N106 covered. Network k builds its own modules on
    `spec.hops[:k]`, and its output reads `spec.layer_dims[k]`. With an
    adjacent-only table, the bottom k graph layers are layers 0..k.
    The vote is one `mode` call.
  - N31 covered: `map_node_attributions(hop_output=)` on each network's
    activations.
  - N17 covered.
  - N18 glue: per-class means through an xarray groupby (2 GLUE lines).
  - N107 missing: top class mean minus the second, cutoff, top class
    (4 GLUE lines).
  - N108 missing: strip `_copy\d+` and keep the largest gap per
    pathway (2 GLUE lines).
- Author glue:
  - Read the gene-pathway table, species, IDs (~28): left,
    pre-edgelist.
  - Read the hierarchy, species, root (~45): left, pre-edgelist.
  - Map genes, prune unannotated leaves (~30): left, pre-edgelist.
  - Cut at L, attach genes, copy chains (~45): left, pre-edgelist.
  - Assign levels, build masks, align rows (~35): removed.
  - Mask variable in the graph (~6): removed. The trainable,
    randomly initialized mask variable (a fragile spot) is gone.
  - Symbols to Ensembl (~28): left (data).
  - One network per depth and the vote (~60): shrunk to a list over
    `spec.hops[:k]` and `mode`.
  - Label activations (~22): removed.
  - Class gap, copy stripping, max merge (~63): shrunk to 8 GLUE
    lines.
- Notes:
  - The class-gap cutoff is 0.01 in the sketch (paper 0.1), because
    the toy weights are untrained.
  - The regex strips any `_copyN`, so the `_copy10` fragile spot is
    gone.

### `deepgami-2023`

- Sketch: `sketches/deepgami-2023.py`, pass.
  - 5 SNPs, 3 TFs, and 3 hidden genes.
  - TCF4 is a TF input and also a hidden gene (`TCF4_g`).
- Needs:
  - N06 covered: `align_inputs` per modality; `rs9` and `FOXP2` are
    ignored.
  - N86 out of scope (pre-edgelist). Both specs then list the same
    genes in the same alphabetical order; the sketch asserts it.
  - N96, N85 covered: one `parse_layered` and one `align_inputs` per
    modality, as in `mulgonet-2026`.
  - N10 covered.
  - N22 covered: `MaskedLinear(hop.to_mask())`.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N35 covered: the noisy-prior ablation is `MaskedLinear` on
    `torch.bernoulli(0.2 + 0.6 * hop.to_mask())` (1 line).
  - N14, N16 out of scope: **Package philosophy** (Division of labor).
    Captum IG with a zero baseline, named per spec with
    `axis="inputs"`.
  - N15 out of scope: Captum `LayerConductance` on the branch-1 layer,
    named with `hop_output=`.
  - N17 covered.
  - N18 glue: mean |IG| in xarray (1 GLUE line).
  - N113 out of scope: **Package philosophy** (the user writes
    `forward()`). It relies on the shared gene order.
  - N114 out of scope (pre-edgelist): a gene that is both input and
    hidden gets two names.
  - N115 out of scope: an `nn.Linear` of width `layer_dims[1]`.
  - N116 covered for naming. Captum `NeuronConductance` runs per gene
    on the classifier input units, as DeepGAMI does. The gene × sample
    × feature stack is named with `dims=` and `coords=`, then joined to
    `to_edgelist()`.
- Author glue:
  - Read eQTL and GRN tables (~15): left, pre-edgelist.
  - Edge table to adjacency (~22): removed.
  - Gene set as an intersection (~5): left, pre-edgelist.
  - Cut GRN rows, align mask columns (~14): removed (sorted names plus
    an assert).
  - Match TFs and SNPs to data, reindex (~15): removed.
  - Write name lists, save masks (~25): removed; `spec.to_dict()`
    would hold them (not sketched).
  - Load masks, layer sizes (~8): removed.
  - Randomized masks (~23): shrunk to 1 line.
  - Masked forward per branch (~10): removed.
  - Scores labeled by feature, gene, link (~80): shrunk to
    `map_node_attributions` calls.
- Notes:
  - The fragile spot "the masks never reach the model" cannot happen:
    the mask is the layer.

### `deephiscom-2022`

- Sketch: `sketches/deephiscom-2022.py`, pass.
  - 4 pathways with hidden depths 1, 0, 2, 1 and widths 3, 0, 2, 3.
  - `m3` and `m6` each sit in two pathways. 2 covariates.
  - One real and two permuted-label runs.
- Needs:
  - N06 covered. A biomarker in two pathways is two rows, not two
    columns. `align_inputs` replaces the contiguous column blocks.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N12, N58 covered: `PackedLinear(bias=False)` and
    `gather_hop_inputs`.
  - N13 out of scope: **What kpnn2 is NOT** (Not a trainer).
  - N17 covered.
  - N54 covered: a chain of hidden nodes `p_h1 … p_hd → p` per
    pathway, each of the pathway's width (`widths=`).
  - N117 covered. Width comes from `widths=` and depth from chain
    length. `ranks=` puts every pathway in the top layer whatever its
    depth, so the output reads one layer. A width-0 pathway is a
    member → pathway skip edge.
  - N100 out of scope: **What kpnn2 is NOT** (Not a trainer). One
    `torch.manual_seed` per run; init is constant.
  - N101 out of scope: the training loop.
  - N118 out of scope: **Package philosophy** (Division of labor:
    heads). Covariates are selected by name.
  - N119 out of scope: **Typical `nn.Module` shape**, step 4.
  - N120 covered: `map_node_attributions(head.weight[0, :P],
    layer=top)` names β per pathway.
- Author glue:
  - Count members per pathway in first-appearance order (~8): removed.
  - Read width and depth per pathway (~3): unchanged (the `layerinfo`
    dict).
  - Build one subnetwork per pathway (~25): replaced by chain rows,
    `widths=`, `ranks=` (8 untagged lines that write the design) and one
    `PackedLinear` per hop.
  - Cumulative offsets and slicing (~9): removed.
  - Covariate count and concatenation (~6): shrunk to a selection by
    name.
  - Normalize the pathway layer (~2): unchanged.
  - Save β per run, unlabeled (~4): shrunk to a named Series.
- Notes:
  - Both positional fragile spots (pathway blocks, covariates by
    count) are gone.
  - Without `ranks=`, a depth-0 pathway would sit in layer 1, and the
    output would gather pathways across layers with `node_units`
    (the N27/N46 pattern).

### `deepkegg-2024`

- Sketch: `sketches/deepkegg-2024.py`, pass.
  - 4 pathways.
  - SNV, mRNA, and miRNA each have their own spec, `PackedLinear`, and
    attention.
- Needs:
  - N06 covered: `align_inputs` per data type; SNV `BRAF` is ignored.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N10, N12 covered.
  - N13 out of scope: **What kpnn2 is NOT** (Not a trainer). 1 line
    on `effective_weight()`.
  - N14, N87 out of scope: Captum `InputXGradient`, named with
    `axis="inputs"`.
  - N17 covered.
  - N18 glue: a sum over recurrence samples (1 GLUE line).
  - N33 glue: not sketched (average 5 fold tables by name).
  - N85 covered: one spec per data type.
  - N86 out of scope (pre-edgelist). Alphabetical order then makes the
    branches agree.
  - N88 out of scope: **Package philosophy** (the user writes
    `forward()`).
- Author glue:
  - Read the gene and miRNA files (~25): left, pre-edgelist.
  - Keep pathways in both files (~10): left, pre-edgelist.
  - One 0/1 matrix per data type by name (~25): removed.
  - Edge vector scattered into a kernel (~50): removed.
  - Scores by name, averaged over folds (~35): shrunk; the fold average
    is not sketched.
- Notes:
  - A pathway with no member in one data type is not a node of that
    branch's spec, because nodes come from edges. DeepKEGG keeps it as a
    constant `tanh(bias)` unit that feeds the attention, which has no
    bias to absorb it. The toy avoids the case.
  - The `b`-is-a-tuple reference and the re-initializing session are
    DeepKEGG bugs. They are not reproduced.

### `deepsignalingflow-2024`

- Sketch: `sketches/deepsignalingflow-2024.py`, pass.
  - 8 genes, 3 drugs, and 10 gene–gene edges with two cycles.
  - 3 drug–target pairs in both directions.
  - Genes `G0` and `G7` exist only so that the graph has an input and
    an output node (see Friction).
- Needs:
  - N07, N38 covered. The 4 → 12 → 108 channels are a leading tensor
    dimension on a width-1 `AdjacencySpec`. `PackedLinear` broadcasts
    over it, so every channel shares each edge.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads). Bilinear decoder.
  - N12, N69 covered: `PackedLinear`, with no index offsets per batch.
  - N25 glue: not sketched (path search on the named edge scores).
  - N32 covered: `to_edgelist()` plus the weights. Slot i is the same
    named edge in every layer and direction.
  - N33 glue: not sketched (average 5 fold tables by name, not by row).
  - N45 glue. Omics are reindexed onto `spec.nodes` (1 GLUE line), and
    the drug flags per node take 2 GLUE lines. `align_inputs` maps only
    in-degree-0 nodes, and here every node takes data.
  - N61, N62, N77 out of scope (pre-edgelist).
  - N65 out of scope: **What kpnn2 is NOT** (Not a time machine).
  - N68 out of scope: **Typical `nn.Module` shape**, step 4. On the
    channel axis it is one `F.normalize` call.
  - N72 glue: weighted degree by a groupby on `to_edgelist()` (2 GLUE
    lines).
  - N78 covered: `PackedLinear` on the node axis gives one scalar per
    edge, shared by the channels. The shared channel map is an
    `nn.Linear`.
  - N79 covered: a second `PackedLinear` with the index arrays swapped,
    with its own weights.
  - N80 out of scope: **Package philosophy** (the user writes
    `forward()`).
  - N81 glue: in- and out-degree by `bincount` on the packed indices,
    and a division in `forward` (4 GLUE lines).
  - N82 glue: the drug nodes of each sample through `node_units`, then
    a gather (3 GLUE lines).
  - N83 out of scope: **Package philosophy** (frozen values stay with
    the caller). `constraint=` with `torch.where` on `edge_location`
    slots, CONTEXT's hard freeze.
  - N84 covered: a stack and mean over layers and directions, since
    the slots line up.
- Author glue:
  - Filter KEGG, uppercase, dedupe (~35): left, pre-edgelist.
  - Drop genes without omics (~37): left, pre-edgelist.
  - Restrict DrugBank, match drug names (~45): left, pre-edgelist.
  - Number genes then drugs, renumber, mirror (~32): removed except the
    mirror (one concat).
  - Build the edge index (~45): removed.
  - Per-sample node feature rows (~80): shrunk to 3 GLUE lines.
  - Weight vectors, constant drug weights, tiling (~17): removed
    (`constraint=`, broadcasting).
  - Per-sample drug-node lookup (~15): shrunk to 3 GLUE lines, by
    name.
  - Extract, average, label edge weights; average folds (~65): shrunk
    to 2 lines; folds not sketched.
  - Weighted degree and filters (~50): shrunk to 2 GLUE lines; the
    filters are left.
  - Cell-line degree and path search (~155): left.
- Notes:
  - Three fragile spots are gone: the positional edge identity, the
    `drug_index` name trick, and fold averaging by row.
  - The per-edge `randn·√2` init is not reproduced.

### `dlbcl-survival-2024`

- Sketch: `sketches/dlbcl-survival-2024.py`, pass.
  - 6 pathways plus `root`, and 8 genes.
  - Genes feed every pathway that contains them, which gives 13 skip
    edges.
  - Widths are max(10, in-degree); `root` has 16 units (the paper
    uses 128).
- Needs:
  - N01 covered.
  - N05, N73 out of scope (pre-edgelist). The root's width comes from
    `widths=`.
  - N06 covered: `align_inputs`; `g9` is ignored.
  - N07 out of scope (pre-edgelist). Each column has its own name, so
    the duplicate-symbol fragile spot is gone.
  - N09, N41 out of scope: **Package philosophy** (Division of labor:
    heads). The clinical columns are selected by name.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
    BatchNorm per node equals BatchNorm over the layer's units.
  - N12, N58 covered: `PackedLinear` blocks and `gather_hop_inputs`.
  - N26 covered: longest-path layering with skips.
  - N38 covered: `widths=`.
  - N67 glue: in-degree from the edgelist into `widths=` (2 GLUE
    lines).
  - N70 covered. The per-row bound 1/√fan-in is the default
    `nn.Linear` bound on a node's own inputs.
- Author glue:
  - Column names to symbols (~8): left (data).
  - Read GMT (~25), filter by size (~20), hierarchy and root (~17):
    left, pre-edgelist.
  - Levels for reporting (~37): removed (`spec.layer_nodes`).
  - Index genes, pathways, root (~28): removed.
  - Child index list per node (~30): removed.
  - Random hierarchy and dense child lists (~40, unused): left.
  - Drawing (~50): left.
  - Save and reload the structure pickle (~13): replaced by
    `spec.to_dict()` (not sketched).
  - Per-node layers (~60): shrunk to one `PackedLinear` per hop.
  - Forward by gather and concat (~50): removed.
  - Split clinical columns (~8): shrunk to `X[CLIN]`.
- Notes:
  - The root-by-position fragile spots are gone: `root` is the only
    output node.
  - The paper's SHAP is sketched as Captum `GradientShap` on the
    inputs. The mean over carriers uses the input tensor named with
    `axis="inputs"` as a mask.

### `drugcell-2020`

- Sketch: `sketches/drugcell-2020.py`, pass. Unchanged from the
  previous run.
- Needs:
  - N01, N06, N12, N26, N31 covered.
  - N38 covered: `widths=`.
  - N39 covered: no absent entries to mask.
  - N09, N11, N40, N41 out of scope (Division of labor; step 4).
- Author glue: as in the previous run. Five rows are removed, one is
  shrunk, and the term checks are left (pre-edgelist).
- Notes: the per-annotation gene layer and the ×0.1 init are not
  reproduced.

### `genomic-prediction-2025`

- Sketch: `sketches/genomic-prediction-2025.py`, pass.
  - 4 genes, each with its own network: 3 SNPs, 8 hidden units (the
    paper uses 128), 1 latent.
  - SNPs `s2`, `s3`, and `s8` are shared between genes.
- Needs:
  - N06 covered. A shared SNP is one column that feeds several genes
    (the paper copies it per gene, with the same function class).
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N54 covered. Each gene is a hidden node `g_h` of width H
    (`widths=`) plus its latent node, so two `PackedLinear` hops hold
    every gene's private MLP. A SNP → gene edge carries H weights.
  - N55 glue. The loss is out of scope (**What kpnn2 is NOT**, Not a
    trainer). Measured expression is aligned to the gene layer with a
    pandas reindex on `spec.layer_nodes[2]` (1 GLUE line).
  - N56 out of scope (pre-edgelist): each split has its own table and
    spec.
  - N57 out of scope: **Package philosophy** (the user writes
    `forward()`). Its columns would be
    `X.columns.difference(spec.input_nodes)`; the branch never fires
    in the shipped models.
  - N58 covered.
- Author glue:
  - Elastic net, eQTL scan, marker tables (~85): left, pre-edgelist.
  - Input matrix as one column block per gene (~50): removed.
  - SNP → gene mask by position (~9): removed. The block-shift
    fragile spot is gone.
  - Gene → module mask (~65): left, pre-edgelist; another hop (not
    sketched).
  - Measured expression into each batch (~25): shrunk to 1 line.
  - Pass masks, switch the residual (~28): removed.
  - Rebuild inputs per split, check width at inference (~30): shrunk
    to `to_dict`/`from_dict` and `align_inputs`, which check names,
    not width.
- Notes:
  - The paper-only clamp analysis is about 8 lines. Its 1-D scores are
    named with `map_node_attributions(layer=2)`.

### `gsnn-2024`

- Sketch: `sketches/gsnn-2024.py`, pass.
  - The typed graph has 3 inputs, 5 function nodes, and 2 outputs,
    with cycles `P1→P2→P3→P1` and `R1→P3→R1`.
  - Its edge-state form has 31 state units and 68 slots, C = 4, and 3
    steps.
- Needs:
  - N06 covered: `align_inputs`; `M9` is ignored.
  - N07, N28, N35, N61, N62, N63 out of scope (pre-edgelist).
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N12, N69 covered: `PackedLinear`.
  - N31 covered: `map_node_attributions` on the list of per-step
    states (the `step` axis). Edge states are named `u->v`.
  - N46 covered. After the rewrite, each output edge is the sink node
    of its output, so `spec.output_index` reads it.
  - N54 covered: `parse_adjacency(widths=)` gives each function node a
    C-unit hidden node. Its in-edge and out-edge blocks are its private
    weights.
  - N64 glue.
    - The edge-state graph is a pre-edgelist rewrite (8 untagged lines):
      an inner edge `u→v` becomes a node between `u_h` and `v_h`.
    - The step then needs unit-type masks and two passes of one
      `PackedLinear` (6 GLUE lines).
    - Each pass computes every slot, so a step does twice the products.
  - N65, N66 out of scope: **What kpnn2 is NOT** (Not a time machine).
  - N67 glue: not sketched. `widths=` takes per-node counts; the
    sqrt-degree, min-max formula is about 2 lines.
  - N68 glue: hidden units reshaped to `(nodes, C)` for `LayerNorm`
    (1 GLUE line). This works only because the hidden blocks are
    node-major and of equal width; with N67's widths it would need a
    segment norm.
  - N70 covered: degree-aware init from present fan-in. He-normal is
    one line through `init_bound()`.
  - N71 out of scope: **Package philosophy** (Division of labor:
    attribution algorithms). Not sketched.
  - N72 glue: not sketched (as in `deepsignalingflow-2024`).
- Author glue:
  - Restrict tables, drug filters, typed edge list, pruning (~330):
    left, pre-edgelist.
  - Node index and masks (part of ~190): removed.
  - Node-indexed x and y (~90): shrunk to `align_inputs` and
    `output_index`.
  - Hidden units and sparse index lists W1/W2/W3 (~150): replaced by
    the rewrite and `widths=`. W2 is not sketched.
  - Hidden unit → node groups, node ↔ edge values (~35): shrunk to 3
    GLUE lines.
  - Sparse layer as message passing with batching (~140): removed.
  - Per-node layer norm (~30): shrunk to 1 GLUE line.
  - Randomized graph control (~25): left, pre-edgelist.
  - Explainer forward and edge table (~40): not sketched.
  - One node as a dense network (~80): not sketched;
    `edge_location`/`node_units` would locate it.
  - Edge scores → node centrality (~40): left.
- Notes:
  - Input units take `core.bias` each step (fan-in 0, starting at 0).
    This mirrors GSNN's W3 bias over all edges, a fragile spot on the
    card.

### `kpnn-2020`

- Sketch: `sketches/kpnn-2020.py`, pass. Unchanged.
- Needs:
  - N01, N06, N10, N12, N17, N26, N31, N32 covered.
  - N35 covered: the gene shuffle.
  - N04, N28 out of scope (pre-edgelist).
  - N11, N15, N30 out of scope.
  - N18 glue.
  - N27 glue: 2 lines.
  - N29 missing: 6 lines.
  - N33 glue, not sketched.
  - N34 missing, not sketched.
- Author glue: as in the previous run.

### `lembas-2022`

- Sketch: `sketches/lembas-2022.py`, pass. Unchanged.
- Needs:
  - N06, N10, N12, N17, N31, N32, N42, N45, N51, N53 covered.
  - N28, N52 out of scope (pre-edgelist).
  - N09, N43, N44, N48, N49, N50 out of scope.
  - N33 glue, not sketched.
  - N46 glue: 2 lines.
  - N47 glue: 4 lines, a dense rebuild and `eigvals`.
- Author glue: as in the previous run.

### `movida-2023`

- Sketch: `sketches/movida-2023.py`, pass.
  - 6 GO terms of 6 units each, and three data types with their own
    gene lists.
  - One enrichment input per term, which gives 12 skip edges.
- Needs:
  - N01 covered.
  - N06, N96 covered: rows from three tables with typed feature names,
    in one spec.
  - N45 covered. The enrichment score is an input node `T_es` with
    edges to `T` and to `T`'s parent; writing those rows is
    pre-edgelist. `align_inputs` maps the columns.
  - N26, N38 covered: longest-path layering, skips, `widths=`.
  - N12, N58 covered.
  - N39 covered. There are no absent entries, so MOViDA's broken
    gradient mask (a fragile spot) does not matter.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N09, N41 out of scope: **Package philosophy** (Division of labor:
    heads).
- Author glue:
  - Read the edge file, one root and one component (~24): parsing
    removed; the checks are pre-edgelist.
  - Term-to-gene-index map per type (~14): removed.
  - Enrichment table and term-to-column map (~12): removed.
  - One gene block per term and type (~4): removed. The per-annotation
    scale is dropped, as in DrugCell.
  - Peel leaves, size layers (~37): removed.
  - One-hot masks, init mask, gradient mask (~38): removed.
  - Per-term concatenation (~33): removed.
- Notes:
  - Two fragile spots are gone: the root as the last term computed,
    and the graph that the build empties.
  - RIS and DeepLIFT are in the paper only; not sketched.

### `mulgonet-2026`

- Sketch: `sketches/mulgonet-2026.py`, pass. Unchanged.
- Needs:
  - N01, N02, N06, N08, N10, N12, N17 covered.
  - N03, N04, N05, N07 out of scope (pre-edgelist).
  - N09, N11, N13, N14, N15, N16 out of scope.
  - N18 glue.
- Author glue: as in the previous run.

### `pathexpsurv-2023`

- Sketch: `sketches/pathexpsurv-2023.py`, pass.
  - 3 pathways, 10 genes, and 30 gene × pathway pairs, 8 of them in
    the prior.
  - Three seeded runs with both phases.
- Needs:
  - N06 covered.
  - N10 covered: one unit per pathway; the shift comes from BatchNorm.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N32 covered: `to_edgelist()` names both ends. Pathways are no
    longer rows by position.
  - N53 covered: no in-place mask edits are needed.
  - N89 out of scope: **Package philosophy** (per-edge policy stays
    with the caller). One spec holds every pair. The prior slots come
    from `edge_location`, and β sits inside `constraint=` (8 untagged
    lines).
  - N90, N92, N93 out of scope: **What kpnn2 is NOT** (Not a trainer).
    One line each, on `effective_weight()` or after the step.
  - N91 out of scope: the β switch. CONTEXT's other route is a reparse
    plus a name-to-name copy (**Reparse hatch**), and a transfer helper
    is locked out there.
  - N94 glue: selection counts joined to `to_edgelist()`, the cutoff,
    and the new prior as an edgelist (4 GLUE lines).
  - N95 out of scope (pre-edgelist).
- Author glue:
  - Parse gene lists, mask by name (~12): removed.
  - Mask from the expanded table (~10): removed; it is an edgelist to
    parse.
  - Masks with genes removed (~25): left, pre-edgelist.
  - Choose the mask per variant (~12): left (which table to parse).
  - Clamp and complementary masks (~10): shrunk to 1 clamp line.
  - Prior and non-prior matrices with β (~5): replaced by
    `constraint=` and `edge_location`.
  - Std and L1 penalties (3): unchanged.
  - Save weights by gene (1): `to_edgelist()`.
  - Counts, cutoff, expanded table (~45): shrunk to 4 GLUE lines.
  - Leave-one-out recovery (~30): not sketched.
- Notes:
  - The match-count mask fragile spot is gone: duplicate pairs raise
    in the parser.

### `pathhdnn-2025`

- Sketch: `sketches/pathhdnn-2025.py`, pass. Unchanged.
- Needs:
  - N01, N02, N06, N10, N17, N22 covered.
  - N03, N07, N19, N20, N21 out of scope (pre-edgelist).
  - N09, N11, N14, N15, N23 out of scope.
  - N18 glue.
  - N24 missing, not sketched.
  - N25 glue.
- Author glue: as in the previous run.

### `pideel-2023`

- Sketch: `sketches/pideel-2023.py`, pass. 6 metabolites → 6 pathways.
- Needs:
  - N06 covered by name. Rows by position (a fragile spot) are gone.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N10 covered.
  - N22 covered: `MaskedLinear(hop.to_mask())`.
  - N35 covered for the row shuffle: `align_inputs` on a permuted name
    list (1 line). The random-mask control is a new edgelist
    (pre-edgelist).
  - N97 glue: `to_edgelist()` merged with the metabolite scores, then
    each pathway's score split by member share (4 GLUE lines).
- Author glue:
  - Parse the membership CSV, copied 4 times (~12): removed.
  - Hard-coded metabolite list (~10): removed.
  - Mask in forward and on the device (~3): removed.
  - Random 0/1 matrix and Hamming distance (~13): left, pre-edgelist.
  - Row-shuffled prior (~8): shrunk to 1 line.
  - Split pathway scores into flows (~23): shrunk to 4 GLUE lines.
  - Load hierarchy tables (~35, unused): left.
- Notes:
  - SHAP is in the paper only; the sketch uses Captum `DeepLiftShap`.
  - `mask_digest` and `identity` make a reload with a different table
    raise. The two-table fragile spot fails loudly.
  - The init differs: kpnn2 uses each pathway's fan-in, the code uses
    kaiming over all 37 inputs.

### `pinnet-2023`

- Sketch: `sketches/pinnet-2023.py`, pass. 3 pathways and 14 genes,
  3 of which are in no pathway.
- Needs:
  - N06 covered by name. Genes in no pathway reach only the dense
    branch, which reads every column.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N10 covered.
  - N53 covered. `PackedLinear` weights are ordinary parameters, so
    PINNet's never-updated weights (a fragile spot) cannot happen.
  - N98 out of scope: **Package philosophy** (the user writes
    `forward()`).
  - N99 glue: `bincount` on `hop.target_index` and a division (3 GLUE
    lines).
- Author glue:
  - Load and transpose the table (~10): removed.
  - Mask to the device (1): removed.
  - Size vector (1): `bincount` (GLUE).
  - Replace the weight each forward (1): removed.
  - Divide by the size (1): left (GLUE).
- Notes:
  - CONTEXT puts per-edge scaling in `constraint=`. The sketch divides
    in `forward` instead, so that the bias is scaled as the paper
    does.

### `pnet-2021`

- Sketch: `sketches/pnet-2021.py`, pass. Unchanged.
- Needs:
  - N01, N02, N06, N10, N12, N17, N31, N32 covered.
  - N03, N07, N19 out of scope (pre-edgelist).
  - N11, N13, N14, N15, N36, N37 out of scope.
  - N18 glue.
  - N24 missing: 3 lines.
  - N25 glue: 3 lines.
- Author glue: as in the previous run.

### `reliable-interpretability-2023`

- Sketch: `sketches/reliable-interpretability-2023.py`, pass.
  - The DTox part: 8 targets, 4 pathways and `root`, units 1 to 20
    from the set size, 2 skip edges.
  - Three setups (real labels, shuffled labels, label-equal inputs),
    3 seeds each.
  - P-NET is the `pnet-2021` sketch.
- Needs:
  - N12, N26, N58 covered.
  - N73 out of scope (pre-edgelist).
  - N40 out of scope: **Package philosophy** (Division of labor:
    heads). Aux heads through `node_units`.
  - N104 glue: the size formula into `widths=` (2 GLUE lines).
  - N100 out of scope: **What kpnn2 is NOT** (Not a trainer).
    `PackedLinear(generator=)` isolates the init draws per run.
  - N101 out of scope: the training loop.
  - N14, N15 out of scope: **Package philosophy** (Division of labor:
    attribution algorithms).
  - N105 glue. Captum `LayerLRP` on each hop module, named with
    `hop_output=`. Relevance is summed over each node's units by an
    xarray groupby (2 GLUE lines).
  - N17 covered.
  - N33 glue: runs into one nodes × (setup, seed) table (1 GLUE line).
  - N34 missing: label-equal inputs (1 GLUE line).
  - N102 glue: quantile-normalize, then real minus shuffled (3 GLUE
    lines).
  - N103 missing.
    - In- plus out-degree from `to_edgelist()` and the correlation take
      3 GLUE lines.
    - Reachability and betweenness are not sketched.
  - N24 missing: P-NET's own `coef_combined`. Not re-sketched; see
    `pnet-2021`.
- Author glue:
  - Expose seeds (~18 changed): removed for DTox (`generator=`).
  - Seed loops (~60): shrunk to one loop.
  - Mount modified inputs (~10): left (data).
  - Deterministic inputs (~20) and shuffled labels (~20): 1 line each.
  - Map names to Reactome ids (~25): left.
  - Rebuild the network from weight matrices (~30): removed
    (`to_edgelist()`). The nonzero-weight fragile spot is gone.
  - Structural statistics (~15): shrunk to 2 lines (degree only).
  - Score vs centrality (~18): 1 line.
  - Quantile-normalized correction (~35): 3 lines, keyed by node name,
    not by position.
- Notes:
  - Captum has no default LRP rule for `PackedLinear` and deletes
    rules after every call, so the sketch sets `EpsilonRule` before
    each one.
  - DTox's γ/ε mix and bounded-input rule are not reproduced.

### `self-pruning-2025`

- Sketch: `sketches/self-pruning-2025.py`, pass.
  - 12 signed edges and 3 added unsigned ones, kept in an `added`
    column of the caller's table.
  - 3 TF outputs, one of them a sink.
- Needs:
  - N10, N45, N53 covered.
  - N42 covered: `parse_adjacency`; the sign is placed by
    `edge_location`.
  - N13, N48 out of scope: **What kpnn2 is NOT** (Not a trainer).
  - N22 covered: `MaskedLinear(hop.to_mask())` for the Reactome
    experiment (not sketched).
  - N32 covered: one `edge_location` slot per caller row.
  - N33 glue: not sketched. Each seed has its own added edges and so
    its own spec; the runs join by edge name.
  - N43 out of scope: **What kpnn2 is NOT** (Not a time machine).
  - N44 out of scope: **Package philosophy** (signs stay with the
    caller).
  - N46 glue: TFs by `node_units` (2 GLUE lines).
  - N47 glue. Power iteration goes through `PackedLinear` itself
    (`core(v) - core.bias`), for both the 0.9 rescale and the Jacobian
    penalty (9 GLUE lines), with no dense matrix.
  - N59 out of scope (pre-edgelist). Tracking is the caller's column
    plus `edge_location`.
  - N60 out of scope: **Package philosophy** (Division of labor:
    in-training prune). CONTEXT's `KeepMask` in `constraint=` zeroes one
    added edge at a time (3 lines).
- Author glue:
  - Sign flags to one value (~10): left (the caller's column).
  - Edge table to node index and edge list (~30): removed.
  - Sample random edges (~50): sampling left (pre-edgelist); the
    indices are replaced by the flag and `edge_location`.
  - Dense weight, absence mask, sign mask (~25): removed.
  - Re-zero after each step (2): removed. The fragile spot of nonzero
    absent entries between steps is gone.
  - Sign-violation penalty (~15): 1 line.
  - Inputs and outputs by name (~15): inputs removed, outputs 2 GLUE
    lines.
  - Real vs added weights in training (~20): a groupby on the caller's
    table.
  - Random edges per layer (~45, Reactome): left, pre-edgelist.

### `sigprimednet-2023`

- Sketch: `sketches/sigprimednet-2023.py`, pass. 4 circuits and 10
  data genes, 2 of which are in no circuit.
- Needs:
  - N06, N10, N17, N31 covered: `map_node_attributions(layer=1)` on
    the circuit activations.
  - N09 out of scope: **Package philosophy** (Division of labor:
    heads).
  - N32 covered: the trained kernel is labeled by `spec.layer_nodes[1]`
    and the data columns.
  - N105, N110 out of scope: **Package philosophy** (Division of
    labor). Captum `LRP` runs one circuit at a time. Its gene axis is
    every data column, not the spec's inputs, so pandas names it.
  - N109 glue. A dense mask over every data column is placed with the
    `align_inputs` index and multiplied into the initial kernel once
    (3 GLUE lines). kpnn2 has no init-only prior.
  - N18 glue, N111 missing: top-15 circuits per cell, counted per
    predicted type (4 GLUE lines).
  - N112 out of scope (pre-edgelist).
- Author glue:
  - Export pathways, circuits, Entrez to symbol (~210): left,
    pre-edgelist.
  - Merge into one matrix (~25): left as building the edgelist.
  - Align the matrix to the data gene order (~25): removed.
  - Zero the initial kernel (2): unchanged (3 GLUE lines).
  - Label activations and kernel (~10): 1 line.
  - Per-circuit LRP loop (~30): 2 lines. `target=j` cannot leave a
    stale selector (a fragile spot in `9.1`).
- Notes:
  - Against the paper's Eq. 4 (W ⊙ I_S in every forward), this is
    `MaskedLinear(hop.to_mask())` and an R.

### `tumor-cell-architecture-2024`

- Sketch: `sketches/tumor-cell-architecture-2024.py`, pass.
  - 5 assemblies of k = 4, with `A2` under two parents.
  - 6 genes × 3 flags, with genes at three heights.
- Needs:
  - N01, N06 covered.
  - N07 out of scope (pre-edgelist): three flag rows per gene.
  - N10 covered: a gene is one unit over its flags (hop 0).
  - N38 covered: `widths=`.
  - N26, N12, N58 covered.
  - N74 covered. Longest path from the inputs is the paper's height
    plus 2 (flags, genes). Dropout reads `hop.target_layer >= 4`.
  - N39 covered. The mask-by-name-prefix fragile spot is gone.
  - N11 out of scope: **Typical `nn.Module` shape**, step 4.
  - N40, N09 out of scope: **Package philosophy** (Division of labor:
    heads). Aux heads through `node_units`.
  - N31, N17 covered.
  - N14, N15, N75 out of scope: **Package philosophy** (Division of
    labor). `torch.autograd.grad` on every saved layer, named with
    `layer=`.
  - N76 glue. PCA, `RidgeCV`, and Spearman take 4 GLUE lines. A node's
    units come from `node_units`, and its children's units from
    `hop_units` on `gather_hop_inputs`.
- Author glue:
  - Name → index tables (~10): removed.
  - Parse the ontology; checks (~55): parsing removed; the checks are
    pre-edgelist.
  - Stack samples × genes × 3 (~15): removed.
  - Gene and direct-gene modules (~14): removed.
  - Child lists, layers, modules (~43): removed.
  - Forward wiring by name (~44): removed.
  - One-hot masks (~10) and masks at init and on gradients (~24):
    removed.
  - Dump activations and gradients by node (~40): shrunk to
    `map_node_attributions`.
  - Re-read children for the probe (~50): shrunk to `node_units` and
    `hop_units`.
- Notes:
  - The probe can no longer read too few units: `node_units` gives the
    slice.
  - The ×0.1 init and the per-annotation scalar are not reproduced.

## Gaps

Needs that are `missing` or `glue` in two or more papers.

| Need | Papers | What an abstraction would do | Lock touched |
|------|--------|------------------------------|--------------|
| N18 Aggregate scores over a sample subset | `kpnn-2020`, `pnet-2021`, `mulgonet-2026`, `pathhdnn-2025`, `deepkegg-2024`, `celltics-2023`, `sigprimednet-2023`, `deepgami-2023` (glue, all sketched) | Registered folds over `observation`: sum (`pnet`, `mulgonet`, `deepkegg`), mean (`kpnn`), mean of absolute values (`pathhdnn`, `deepgami`). Each would be limited to one label value or grouped by label, which `labels=` already carries. CellTICS needs a mean per class. | None. `AGENTS.md` **How to add an aggregation method**. |
| N33 Combine scores across training repeats | `kpnn-2020`, `lembas-2022`, `self-pruning-2025`, `deepsignalingflow-2024`, `deepkegg-2024`, `reliable-interpretability-2023` (glue; sketched only in `reliable`) | Fold a run dimension that the caller names (mean, median), optionally after a per-run filter (KPNN's test-error cutoff). Self-pruning runs have different specs per seed, so runs join by name, not by slot. | None. CONTEXT forbids guessing that `step` is `seed`. |
| N46 Read outputs from named graph nodes | `lembas-2022`, `self-pruning-2025` (glue) | A many-name index onto the state vector, or onto a layer, for names that are not `input_nodes`. The same pattern is N27 (`kpnn`, roots in several layers), N82 (`deepsignalingflow`, nodes chosen per sample), N55 (`genomic-prediction`, data aligned to a hidden layer), and N45 on non-input nodes (`deepsignalingflow`): 5 papers. Each sketch writes a `node_units` comprehension or a pandas reindex on `spec.nodes`/`layer_nodes`. | None. It would grow `__all__` or the spec methods, which `AGENTS.md` leaves to the maintainer. |
| N24 Normalize node scores by connectivity | `pnet-2021` (sketched), `pathhdnn-2025`, `reliable-interpretability-2023` (missing) | Per-node degree, or the up- plus downstream count, from a spec, as a named table. The same counts feed N67 (`gsnn`, `dlbcl`), N104 (`reliable`), N81 (`deepsignalingflow`), N99 (`pinnet`), and N103 (`reliable`): 7 papers in all. kpnn2 computes per-row fan-in for its init, but exposes it only per slot (`init_bound()`). | None. |
| N67 Units per node set by degree | `dlbcl-survival-2024` (sketched), `gsnn-2024` (not sketched) (glue) | See N24. `widths=` takes the counts; the user derives them from the edgelist in 1 or 2 lines. | None. |
| N25 Scores as a graph | `pnet-2021`, `pathhdnn-2025`, `deepsignalingflow-2024` (not sketched) (glue) | Join named node scores to `to_edgelist()`. It costs 2 or 3 lines; PiDeeL's N97 is the same join. | None. |
| N47 Keep a recurrent edge matrix contractive | `lembas-2022`, `self-pruning-2025` (glue) | A spectral-radius estimate of a packed map without a dense rebuild. Power iteration through the module works but has to subtract `bias`, because `PackedLinear` has no bias-free apply. LEMBAS uses eigenvalues and rebuilds the dense matrix. | The training penalty: **What kpnn2 is NOT** (Not a trainer). |
| N72 Node scores from scored edges | `deepsignalingflow-2024` (sketched), `gsnn-2024` (not sketched) (glue) | Per-node sums or centrality over a scored, thresholded `to_edgelist()`. 2 lines of pandas; centrality needs networkx. | None. |
| N34 Structure-only baseline from control inputs | `kpnn-2020` (not sketched), `reliable-interpretability-2023` (1 line) (missing) | Control data in which every input carries the label equally. This is data generation. | **Locked contrasts**: the data axis is the user's. |

Watch list, one paper only:

| Need | Paper | Note |
|------|-------|------|
| N27 Output units are named graph roots | `kpnn-2020` | Part of the N46 pattern. |
| N29 Dropout rate per node from the graph | `kpnn-2020` | 6 lines, still the largest single block of model glue. |
| N55 Supervise node states with measured values | `genomic-prediction-2025` | Part of the N46 pattern (data side). |
| N64 Hidden state on edges | `gsnn-2024` | A pre-edgelist rewrite plus 6 GLUE lines and twice the products per step. The only P in the sample that a node-centric layout causes. |
| N68 Normalize within each node's units | `gsnn-2024` | Ordinary at equal widths; DeepSignalingFlow's channel layout avoids it. A segment norm is needed only with N67 widths. |
| N76 Node score from a probe on its units | `tumor-cell-architecture-2024` | `node_units` and `hop_units` give the units; sklearn does the rest. |
| N81 Average over incoming edges | `deepsignalingflow-2024` | Part of the N24 family. |
| N82 Read out nodes chosen per sample | `deepsignalingflow-2024` | Part of the N46 pattern. |
| N94 Edge selection frequency over resampled fits | `pathexpsurv-2023` | A fold over runs keyed by edge, the edge-level form of N33. |
| N97 Split node scores onto edges by input scores | `pideel-2023` | A `to_edgelist()` join, like N25. |
| N99 Scale node sums by the root of in-degree | `pinnet-2023` | Part of the N24 family. |
| N102 Control-corrected node scores | `reliable-interpretability-2023` | A fold over real and control runs; a candidate for the aggregation registry. |
| N103 Node scores against network centrality | `reliable-interpretability-2023` | Part of the N24 family. |
| N104 Units per node set by gene-set size | `reliable-interpretability-2023` | Part of the N24 family (the set size is a count of input descendants). |
| N105 Layer-wise relevance propagation | `reliable-interpretability-2023` | The glue is the unit sum per node. The mapper repeats a wide node's name and does not fold, by contract. |
| N107 Class gap of mean activations | `celltics-2023` | Score analysis after N18. |
| N108 Fold copy-node scores into the original node | `celltics-2023` | Name stripping. kpnn2 requires distinct names for copies. |
| N109 Prior as initialization only | `sigprimednet-2023` | The paper's Eq. 4 is `MaskedLinear`. |
| N111 Per-class frequency of top-k active nodes | `sigprimednet-2023` | Score analysis after N18. |

These needs are not gaps, because they only decide which rows the
edgelist holds. They are out of scope (pre-edgelist):

| Need | Papers |
|------|--------|
| N03 Keep only adjacent-layer edges | `mulgonet-2026`, `pathhdnn-2025`, `pnet-2021`, `celltics-2023` |
| N04 Prune nodes with no child below | `kpnn-2020`, `mulgonet-2026` |
| N05 Filter nodes by gene-set size | `mulgonet-2026`, `dlbcl-survival-2024` |
| N07 One input per entity and data type | `mulgonet-2026`, `pathhdnn-2025`, `pnet-2021`, `gsnn-2024`, `dlbcl-survival-2024`, `tumor-cell-architecture-2024` |
| N19 Pad short branches with copy nodes | `pathhdnn-2025`, `pnet-2021`, `celltics-2023` |
| N20 Keep only nodes above the data | `pathhdnn-2025`, `celltics-2023` |
| N21 Propagate memberships up the hierarchy | `pathhdnn-2025`, `celltics-2023` |
| N28 Prune nodes that cannot reach an output | `kpnn-2020`, `lembas-2022`, `gsnn-2024` |
| N52 Prune nodes no input can reach | `lembas-2022` |
| N56 Derive the graph from training data | `genomic-prediction-2025`, `celltics-2023` |
| N59 Add random edges to the prior | `self-pruning-2025` |
| N61 Merge typed edge tables into one graph | `gsnn-2024`, `deepsignalingflow-2024` |
| N62 Restrict the graph to chosen gene sets | `gsnn-2024`, `deepsignalingflow-2024` |
| N63 Drop inputs that reach too few outputs | `gsnn-2024` |
| N73 Add a root above top-level nodes | `dlbcl-survival-2024`, `reliable-interpretability-2023`, `celltics-2023` |
| N77 Drop graph nodes that lack data | `deepsignalingflow-2024` |
| N86 Keep nodes shared by all membership tables | `deepkegg-2024`, `deepgami-2023` |
| N95 Remove chosen edges from the prior | `pathexpsurv-2023` |
| N112 Members of a sink from its upstream subgraph | `sigprimednet-2023` |
| N114 One input-to-node layer from a directed graph | `deepgami-2023` |

Partly pre-edgelist:

- N02: the depth cut.
- N27: dropping unlabeled roots.
- N35: edge swaps (`kpnn`), redrawn graphs (`gsnn`), random masks
  (`pideel`).
- N64: the edge-state rewrite of `gsnn-2024`.

## Orphans

Every name in `kpnn2.__all__`, and each documented keyword argument
and public method of those names. "All layered" means the 18 sketches
that call `parse_layered` for their main graph.

| kpnn2 name or option | Used by | Need served, or "no application in this sample" |
|----------------------|---------|-------------------------------------------------|
| `parse_layered` | 19 sketches (all but `gsnn`, `deepsignalingflow`, `self-pruning`; `lembas` for its drug hop) | N01, N02, N26 |
| `parse_layered(widths=)` | drugcell, dlbcl, tumor, movida, genomic, deephiscom, reliable | N38, N54, N67, N104, N117 |
| `parse_layered(ranks=)` | deephiscom | N117: pathways of different depth share the top layer. First user. |
| `parse_adjacency` | lembas, self-pruning, gsnn, deepsignalingflow | N42, N64, N65, N78 |
| `parse_adjacency(widths=)` | gsnn | N54 on a cyclic graph. First user. |
| `LayeredSpec` | all layered | N01 |
| `LayeredSpec.to_edgelist()` | kpnn, pnet, pathhdnn, pathexpsurv, pideel, deepgami, tumor, reliable | N25, N32, N76, N94, N97, N103, N116 |
| `LayeredSpec.edge_location()` | kpnn, pathexpsurv | N32, N89 |
| `LayeredSpec.node_units()` | drugcell, kpnn, tumor, reliable | N27, N40, N76 |
| `LayeredSpec.hop_units()` | tumor | N76: each child's units on the hop axis. First user. |
| `LayeredSpec.to_dict()` / `from_dict()` | genomic | Per-split inference rebuild, checked by name. First user. |
| `LayeredSpec.fingerprint` | 19 sketches (`identity=`) | Checkpoint identity |
| `Hop` (`in_features`, `out_features`, indices) | all layered | N12 |
| `Hop.to_mask()` | pathhdnn, pideel, celltics, deepgami, sigprimednet | N22, N35, N109 |
| `Hop.column_offsets` | none | No application in this sample. `hop_units` served the one per-node need. |
| `Skip` | none (kpnn, movida, dlbcl print `len(spec.skips)`) | Metadata. The skip edges themselves serve N26 in 6 papers. |
| `AdjacencySpec` (`state_dim`, `input_index`) | lembas, self-pruning, gsnn, deepsignalingflow | N42, N45 |
| `AdjacencySpec.nodes` | deepsignalingflow | N45 on non-input nodes (by reindex) |
| `AdjacencySpec.output_index` (field) | gsnn | N46 after the edge-state rewrite. First user. |
| `AdjacencySpec.node_units()` | lembas, self-pruning, gsnn, deepsignalingflow | N46, N64, N82 |
| `AdjacencySpec.to_edgelist()` | lembas, deepsignalingflow | N32, N72, N84 |
| `AdjacencySpec.edge_location()` | lembas, self-pruning, deepsignalingflow | N44, N51, N59, N60, N83 |
| `AdjacencySpec.to_mask()` | none | No application in this sample. N47 needs the weights, not the pattern. |
| `AdjacencySpec.to_dict()` / `from_dict()` | none | N51 (reload by name). Not sketched. |
| `AdjacencySpec.fingerprint` | lembas, self-pruning, gsnn | Checkpoint identity |
| `MaskedLinear` (`identity=`) | pathhdnn, pideel, celltics, deepgami | N22, N35 |
| `MaskedLinear(bias=)` | default only | N10 |
| `MaskedLinear(constraint=)`, `(generator=)` | none | No application in this sample. The four dense-mask papers need neither. |
| `MaskedLinear.effective_weight()`, `init_bound()`, `reset_parameters()` | none | No application in this sample |
| `PackedLinear` (`identity=`) | 17 sketches | N12, N39, N53, N54, N58, N69, N78, N79 |
| `PackedLinear(bias=False)` | lembas, pathexpsurv, deephiscom, deepsignalingflow | N06 (drug hop), N10 (PathExpSurv), N117, N78 |
| `PackedLinear(constraint=)` | self-pruning, deepsignalingflow, pathexpsurv | N60 (keep-mask ablation), N83 (frozen drug edges), N89 (prior vs other pairs). First users. |
| `PackedLinear(generator=)` | reliable | N100. First user. |
| `PackedLinear.effective_weight()` | kpnn, lembas, mulgonet, self-pruning, deepkegg, pathexpsurv | N13, N32, N44, N90, N93 |
| `PackedLinear.init_bound()` | gsnn | N70 (He-normal from present fan-in). First user. |
| `PackedLinear.reset_parameters()` | none | N33 (re-init between repeats). `reliable` rebuilds the module per seed instead. |
| `PackedLinear.transpose()` | none | No application in this sample. DeepSignalingFlow's reverse pass has its own weights, and swapping the index arrays is simpler. LEMBAS's adjoint (N43) is not sketched. |
| `PackedMultiheadAttention` and all its arguments | none | No application in this sample. DeepKEGG's attention mixes samples, not graph pairs. |
| `gather_hop_inputs` | 12 multi-hop layered sketches | N26, N76 |
| `scatter_hop_outputs` | none | No application in this sample (no decoder) |
| `align_inputs` | 21 sketches (all but deepsignalingflow) | N06, N35, N45, N85, N109 |
| `map_node_attributions` | 18 sketches | N17, N31 |
| `map_node_attributions(layer=)` | pathhdnn, genomic, tumor, sigprimednet, deephiscom | N17, N75, N120 |
| `map_node_attributions(hop_output=)` | drugcell, kpnn, mulgonet, pnet, celltics, pideel, pinnet, reliable, deepgami | N15, N31, N105 |
| `map_node_attributions(hop_input=)` | pathhdnn | N23 |
| `map_node_attributions(axis="inputs")` | mulgonet, pnet, deepkegg, pideel, dlbcl, deepgami | N14, N87 |
| `map_node_attributions(dims=, coords=)` | lembas, deepgami | N49, N116 |
| `map_node_attributions` on a tensor sequence (`step` axis) | gsnn | N31 (edge states per step). First user. |
| `aggregate_node_attributions` (`method=`, `labels=`, kwargs) | none | N18 (8 papers), N33 (6), N102. Unusable today: its only method raises. |
| `list_aggregation_methods` | none | As above |
| `Kpnn2Error` | no sketch catches it | It fired once while writing: `widths=` named a non-node because of a set-precedence bug in the GSNN sketch. |
| `__version__` | none | Metadata |

## Friction

Glue that repeats in three or more sketches:

- **Folding observations** (kpnn, pnet, mulgonet, pathhdnn, deepkegg,
  celltics, sigprimednet, deepgami). A sum or mean in xarray follows
  `map_node_attributions`, sometimes per label.
- **Named nodes outside `input_nodes` and whole layers** (kpnn, lembas,
  self-pruning, deepsignalingflow, gsnn, genomic):
  - a `node_units(...)` comprehension, or `torch.arange(n)[slice]`
    over many names;
  - or a pandas reindex on `spec.nodes` or `spec.layer_nodes[i]` for
    data that targets non-input nodes.
- **Per-node counts from the graph** (pnet, dlbcl, pinnet,
  deepsignalingflow, reliable). The counts are `value_counts` on the
  edgelist or `bincount` on packed indices. They feed `widths=`, a
  division in `forward`, or a score adjustment.
- **Boilerplate, not glue** (all 12 multi-hop layered sketches). The
  3-line `PackedLinear(...)` comprehension and the `saved` /
  `gather_hop_inputs` loop repeat, as **Typical `nn.Module` shape**
  prescribes.

kpnn2 calls the sketches had to work around:

- **Missing names in `align_inputs`** (lembas, pathhdnn). Both still
  use `reindex(..., fill_value=0)` (**Locked contrasts**).
- **`parse_adjacency` requires an input and an output node.**
  - DeepSignalingFlow writes data into every node and mirrors the
    drug–target edges. The rule protects nothing there, and a graph
    in which every node has an in-edge would be rejected.
  - The toy adds a source `G0` and a sink `G7`. Whether the real KEGG
    graph has both was not checked.
- **No bias-free apply on `PackedLinear`** (self-pruning). The power
  iteration computes `core(v) - core.bias`.
- **A spec drops edgeless nodes.** Branches that must share a node list
  (deepkegg, deepgami, mulgonet) rely on matching edge sets. DeepGAMI
  asserts it, and DeepKEGG's bias-only pathways have no place.
- **Node width by name** (reliable). It is the slice length of
  `node_units`; there is no accessor.
- **Captum LRP and custom modules** (reliable). This is not kpnn2's
  doing, but every `PackedLinear` and `MaskedLinear` user meets it:
  Captum has no default rule for these modules and deletes rules after
  each call.

## Changes since the previous run

- **Scope:** 6 papers before, 22 now (16 new). The six earlier verdicts
  are unchanged: drugcell R, mulgonet R, pathhdnn R, pnet R, kpnn P,
  lembas P.
- **Gaps that grew:**
  - N18: from 4 papers to 8.
  - N33: from 2 to 6.
  - N24: from 2 to 3, and now part of a 7-paper degree family.
  - N25: from 2 to 3.
- **Gaps that opened:**
  - N46: lembas and self-pruning. It was a watch-list item.
  - N47: lembas and self-pruning. It was a watch-list item.
  - N67, N72: both new.
  - N34: kpnn and reliable. It was a watch-list item.
- **Gaps that closed:** none.
- **Orphans that gained their first users:**
  - `parse_layered(ranks=)`, `parse_adjacency(widths=)`, and
    `LayeredSpec.hop_units()`;
  - `LayeredSpec.to_dict()`/`from_dict()` and
    `AdjacencySpec.output_index`;
  - `PackedLinear(constraint=)`, `(generator=)`, and `init_bound()`;
  - `map_node_attributions` on a tensor sequence.
  - The previous run's "no application" claim for `ranks=` no longer
    holds: DeepHisCoM needs it for pathways of different depth.
- **Orphans that lost users:** none.
- **Method:** kpnn2 lines are now counted by a script.
  `lembas-2022` reads 21 instead of 19, with the sketch unchanged.

## Caveats

- **The pre-edgelist ruling is still not in `CONTEXT.md` or in
  `assess_prompt.md`.** This run applied the maintainer's ruling of
  2026-10-09, as the previous one did. It is carried in the agent's
  memory, not in the repository.
- **Judgment calls about what counts as the final edgelist.** Each of
  the following is written as edgelist rows and marked out of scope
  (pre-edgelist):
  - `gsnn`: the edge-state rewrite, in which an inner edge becomes a
    node.
  - `genomic-prediction`, `deephiscom`: one hidden node per entity for
    its private network.
  - `movida`: an input node per enrichment score.
  - `deepgami`: a second name for a gene that is both input and
    hidden.
  - `deephiscom`: chain nodes, plus `ranks=`.

  A stricter reading would make GSNN's rewrite glue, which would not
  change its P verdict.
- **The verdict boundary is this run's.** "A per-node number for an
  existing argument" counts as R. That keeps `dlbcl`, `pinnet`, and
  `reliable` at R. Under the previous run's wording ("model structure
  derived from the graph"), `dlbcl` and `pinnet` would be P.
- **Paper code:** none was opened. The cards were the only source.
- **Cards and catalog.** No card looks wrong. Observations:
  - `README.md`'s Pilot table still lists `sctransformer-2026`, which
    `papers.csv` marks excluded.
  - N27, N46, N55, N82 (and N45 on non-input nodes) describe one
    pattern from different sides.
  - N24, N67, N81, N99, N103, N104 share one input: a per-node count
    from the graph.
- **Simplifications:**
  - Toy graphs of at most 30 nodes; depths and widths smaller than
    in the papers.
  - Training is one backward pass, or a few steps (PathExpSurv,
    DeepHisCoM), so the scores come from nearly untrained weights.
    CellTICS's cutoff was lowered to 0.01 for that reason.
  - DrugCell, MOViDA, and NeST-VNN drop the per-annotation gene scalar.
  - DTox uses an ε rule only.
  - Initializations specific to a paper (×0.1, constant, `randn·√2`)
    are kept only where the sketch needs them.
- **Not sketched:**
  - N24 (pathhdnn, reliable);
  - N25 and N33 (deepsignalingflow);
  - N33 (kpnn, lembas, self-pruning, deepkegg);
  - N34 (kpnn);
  - N67 and N72 (gsnn);
  - GSNN's explainer and two-layer node networks;
  - the Reactome experiment of self-pruning;
  - MOViDA's RIS.
- **Sketch size:** 34 to 63 counted code lines each. Several are over
  the prompt's ~60: `lembas`, `self-pruning`, `reliable`, and
  `pathexpsurv` with its training loop. The sketches pack arguments
  onto shared lines, against `AGENTS.md`'s one-argument-per-line rule;
  Ruff skips `tests/paper_fit/`.
- **Line counts** are physical lines; read them as orders of
  magnitude.
- **Environment:** torch 2.13.0, Captum 0.9.0, xarray 2026.7.0,
  pandas 2.3.3, scikit-learn 1.9.0, SciPy 1.18.1.
