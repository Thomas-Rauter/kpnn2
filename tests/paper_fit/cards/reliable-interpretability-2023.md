# reliable-interpretability-2023: Robustness and bias of P-NET and DTox node scores

| Field | Value |
|-------|-------|
| Paper | Esser-Skala W, Fortelny N, 2023. Reliable interpretability of biology-inspired deep neural networks. npj Systems Biology and Applications. 10.1038/s41540-023-00310-8 |
| Code | `https://github.com/csbg/pnet_robustness` at `fe90f4573d7bd2245c80c43d5ba021d49e3288ed` |
| Other code | Zenodo archive 10.5281/zenodo.8386694 (paper); run outputs 10.5281/zenodo.7760561 (README.md:5,14). The models come from other repositories and are changed only to expose seeds: `marakeby/pnet_prostate_paper` at `2b16264`, the same commit as the `pnet-2021` row, not re-read here (docker/pnet/setup.sh:7-9); `EpistasisLab/DTox` at `10c909b16f136358b9600768347b6d3b816a26b1` (docker/dtox/setup.sh:3-5), read for this card and cited as `DTox:path:line`. DTox paper: Hao, Romano, Moore 2022, Patterns, 10.1016/j.patter.2022.100565 |
| Framework | R 4.3.1 for data changes and analysis (README.md:51; `renv.lock`). P-NET: Python 2.7.15, TensorFlow 1.12.0, Keras 2.2.4 (docker/pnet/environment_pnet.yml:27-29). DTox: Python 3.7.3, PyTorch 1.10.1 (docker/dtox/environment_dtox.yml:11,20) |
| License | MIT (LICENSE:1-3). DTox ships `LICENSE.md` (not read) |
| Full text | PubMed Central PMC10564878 (abstract, methods, code availability, Fig. 1 caption), read through a summarizing web fetch, not verbatim |
| Extracted | 2026-10-10, Claude Code (claude-opus-5-5) |

## Summary

The paper builds no model of its own. It retrains two published biology-inspired
networks many times to test how stable and how biased their per-node importance scores
are. P-NET (prostate cancer metastasis from gene mutation and copy-number inputs,
Reactome hierarchy) runs with its original seed pair plus 50 more. DTox (compound
mitochondrial toxicity from predicted protein-target binding, Reactome hierarchy) runs
with seeds 0 to 50. Two control setups separate structure from data. In one, every input
equals the label. In the other, labels are redrawn at random. The R analysis rebuilds
P-NET's layered network from saved weight matrices and correlates node scores with
degree, reachability, and betweenness. It also proposes a corrected score:
quantile-normalize all runs, average each node, and subtract the shuffled-label average.
P-NET is also run on four MSK-IMPACT 2017 cancer types (primary vs metastatic sample).

## Prior-knowledge graph

- Source: P-NET: Reactome gene sets and pathway hierarchy (human ids only) from P-NET's
  `_database.zip` (README.md:285-306; docker/pnet/setup.sh:19-20). DTox: precomputed
  Reactome hierarchy index files for root processes `GE+IS+M+ST`, minimum pathway size
  5 (DTox:code/dtox.py:17,36-42). No code at DTox `10c909b` builds them.
- Size: P-NET: 9229 genes and 3073 pathways over 6 layers (paper, Results). DTox: not
  determined (counts are in data files not read). Both DAGs, unsigned.
- Data to nodes: P-NET keeps genes in both an expressed-gene list and a HUGO
  protein-coding list (README.md:287-289); its matching code was not read. For
  MSK-IMPACT, this repository pivots mutations and copy-number calls into sample × gene
  tables keyed by Hugo symbol (scripts/load_data_mskimpact.R:45-62) and mounts them over
  P-NET's input files (scripts/run_pnet_docker.sh:3-12). DTox inputs are predicted
  binding probabilities, one per protein target, from per-target classifiers on MACCS
  fingerprints (DTox:code/targettox.py:17-36). Column `i` feeds node `i`, so inputs are
  matched by position, not name (DTox:code/dtox_nn.py:89-91).
- Built at: P-NET upstream; the analysis rebuilds the layered edge list from saved
  weight matrices (scripts/plot_figures.R:89-119). DTox reads the index files at
  DTox:code/dtox_hierarchy.py:103-134.

## Architecture

- Layers: P-NET is unchanged at `2b16264`; the patch touches only seeds
  (docker/pnet/patch_seeds.diff:1-45); see `pnet-2021`. Its 7 saved connection files are
  an input-to-gene pair table (layer 0) and 6 matrices whose nonzero entries are edges
  (docker/pnet/entrypoint.sh:77; scripts/plot_figures.R:93-108). DTox hidden nodes carry
  a layer number from a file, used only for loss weights and statistics
  (DTox:code/dtox_loss.py:31; DTox:code/dtox_hierarchy.py:70-99). The forward pass
  computes nodes one by one in index order, each from the joined outputs of its children
  (inputs or hidden nodes of any lower index) (DTox:code/dtox_nn.py:89-102).
- Units per node: DTox: 1 unit if a node has fewer than 5 annotated genes, else
  round(1 + 19·log(size/5)/log(max size/5)) (DTox:code/dtox_hierarchy.py:28-40). Input
  width is the sum of the children's units (DTox:code/dtox_hierarchy.py:56-59).
- Connections that skip layers: DTox children can sit at any depth, inputs included
  (DTox:code/dtox_nn.py:95-97). P-NET: see `pnet-2021`.
- Outputs or losses at inner layers: every DTox hidden node has a sigmoid head
  `Linear(units, 1)`; loss = BCE(root) + 0.5·Σᵢ BCE(auxᵢ)/(nodes in i's layer); the heads
  do not feed the prediction (DTox:code/dtox_nn.py:33-49,73-74,102;
  DTox:code/dtox_loss.py:25-34; DTox:code/dtox.py:17).
- Parts the graph does not constrain: DTox's final sigmoid unit over the joined units of
  all root-pathway nodes (DTox:code/dtox_nn.py:76,104-107); the target-binding
  classifiers outside the network (DTox:code/targettox.py:28-30).
- Other structure: none. Defined at: DTox:code/dtox_nn.py:53-108; P-NET upstream.

## Connectivity mechanism

DTox gives every hidden node its own `nn.Linear` plus ReLU. Its input is the joined
output of the node's children, selected by a stored index list
(DTox:code/dtox_nn.py:20-21,70,95-100; DTox:code/dtox_hierarchy.py:116-124). No full
weight matrix exists, and absent edges have no parameter. Weights use PyTorch's default
`nn.Linear` initialization after `torch.manual_seed` (DTox:code/dtox.py:45), which the
patch changes to a caller-chosen seed (docker/dtox/patch_seeds.diff:18-19). Adam with
weight decay 1e-4 regularizes all parameters (DTox:code/dtox_learning.py:50). P-NET's
mechanism is as in `pnet-2021` and was not re-read. This repository reads P-NET's
connectivity back only from the nonzero entries of the original-seed run's trained
weight matrices (scripts/plot_figures.R:89-108).

## Interpretation

- P-NET: scores come from P-NET's own pipeline (DeepLIFT, per the paper). The analysis
  uses `coef_combined`: the raw score divided by in- plus out-degree when that degree
  exceeds the layer mean + 5 sd (README.md:201-207; scripts/plot_figures.R:56). One
  value per node, layer, and run; P-NET reduces over samples upstream. Names map to
  Reactome ids through the first human match in `ReactomePathways.txt`; genes keep their
  symbols (scripts/plot_figures.R:27-33,53-55).
- Repeats: 51 runs per setup. Seed -1 means (234, 20080808); seed s means (s, s) for s in
  0..49 (docker/pnet/entrypoint.sh:48-60). The first seeds Python, NumPy, and TensorFlow
  at start. The second is set before each model is built
  (docker/pnet/patch_seeds.diff:9-19,31-42).
- Controls: deterministic inputs set every gene's mutation to 1 and copy number to 2 in
  metastatic samples, 0 otherwise (scripts/modify_data_deterministic.R:8-30). Shuffled
  labels are redrawn i.i.d. with P(1) = 0.5, with a new seed per run, test set included
  (scripts/modify_data_shuffled.R:22-36; README.md:116-124).
- Stability: Pearson correlations of node scores between all runs of the three setups,
  clustered by 1 − r (scripts/plot_figures.R:467-488).
- Structural bias: per run and layer, the Pearson correlation of `coef_combined` with
  in-, out-, and total degree, reachability (upstream subcomponent size, the node
  included), and betweenness in the rebuilt network
  (scripts/plot_figures.R:117-136,558-575).
- Correction: quantile-normalize a nodes × runs matrix of original and shuffled runs
  (all layers together, `limma::normalizeQuantiles`). Take row means over each group and
  compare nodes by original − control (scripts/plot_figures.R:719-753). The result is
  drawn as a figure; no table is written.
- DTox: LRP (γ/ε rule, γ = 0.001, ε = 0.1, between hidden nodes; bounded-input rule on
  [0, 1] into the inputs), with unit relevance summed per node, per compound, on the
  test compounds labeled 1 (DTox:code/dtox_interpret.py:20,49,99-101;
  DTox:code/dtox_lrp.py:229-268; docker/dtox/run_dtox.py:73-81). One shuffled-label null
  model is also trained for path p-values (DTox:code/dtox_interpret.py:70-84) but not
  analyzed. The analysis correlates node scores between seeds, per compound
  (scripts/plot_figures.R:851-880).

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N12 Parameters only for present edges | `DTox:code/dtox_nn.py:20,70,95-100` | Each DTox node holds weights only for its children's units. |
| N14 Attribution to inputs | `DTox:code/dtox_lrp.py:229-237,265-268` | DTox module relevance includes the input features (layer 0). |
| N15 Attribution to hidden nodes | `scripts/plot_figures.R:35-56`; `DTox:code/dtox_lrp.py:236-237` | Per-layer hidden-node scores are what the paper tests for stability and bias. |
| N17 Map scores to node names | `scripts/plot_figures.R:27-33,53-55`; `DTox:code/dtox_interpret.py:100` | Scores are labeled by node name and joined to Reactome ids. |
| N24 Normalize node scores by connectivity | `README.md:205`; `scripts/plot_figures.R:56` | The node score analyzed is P-NET's degree-adjusted `coef_combined`. |
| N26 Unlayered DAG in topological order | `DTox:code/dtox_nn.py:93-102` | DTox computes nodes in index order, each reading children of any depth, inputs included. |
| N33 Combine scores across training repeats | `docker/pnet/entrypoint.sh:48-60`; `scripts/plot_figures.R:35-56,744-746` | 51 seeded runs per setup go into one table by experiment and seed; nodes are averaged and runs correlated pairwise. |
| N34 Structure-only baseline from control inputs | `scripts/modify_data_deterministic.R:8-30` | Every gene's mutation and amplification inputs equal the label, so scores reflect structure only. |
| N40 Auxiliary output head on every node | `DTox:code/dtox_nn.py:73-74,102`; `DTox:code/dtox_loss.py:29-33` | Each DTox hidden node has a sigmoid head; its loss is weighted by α over the size of the node's layer. |
| N58 Gather each node's inputs by index | `DTox:code/dtox_nn.py:95-100`; `DTox:code/dtox_hierarchy.py:116-124` | Children index lists from the hierarchy file select each node's inputs. |
| N73 Add a root above top-level nodes | `DTox:code/dtox_nn.py:76,104-107` | One output unit reads the joined units of every root-pathway node in the root file. |
| N100 Caller-set seed per training run | `docker/pnet/patch_seeds.diff:9-19,31-42`; `docker/dtox/patch_seeds.diff:9-19`; `docker/dtox/run_dtox.py:51-60` | Both models hard-coded their seeds; the authors patched them into arguments so that repeats differ only by seed. |
| N101 Null scores from shuffled-label runs | `scripts/modify_data_shuffled.R:22-36`; `DTox:code/dtox_interpret.py:70-82` | Control runs retrain on randomly drawn labels, one draw per seed, and are scored like the real runs. |
| N102 Control-corrected node scores | `scripts/plot_figures.R:722-753` | Quantile-normalize all runs and compare each node's real-label mean with its shuffled-label mean. |
| N103 Node scores against network centrality | `scripts/plot_figures.R:89-136,558-575` | Rebuild the network, compute degree, reachability, and betweenness per node, and correlate them with scores per layer and run. |
| N104 Units per node set by gene-set size | `DTox:code/dtox_hierarchy.py:28-40,56-59` | DTox sets each node's unit count from its annotated gene count on a log scale between 1 and 20. |
| N105 Layer-wise relevance propagation | `DTox:code/dtox_interpret.py:20,49`; `DTox:code/dtox_lrp.py:229-268` | DTox scores nodes per compound by LRP relevance summed over each node's units. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Expose P-NET's two seeds as arguments | `docker/pnet/patch_seeds.diff:1-45` | ~8 changed |
| Expose DTox's seed; save test predictions | `docker/dtox/patch_seeds.diff:1-65` | ~10 changed |
| Seed loop and result copying for P-NET | `docker/pnet/entrypoint.sh:47-79` | ~30 |
| Seed loop for DTox training, evaluation, LRP | `docker/dtox/run_dtox.py:50-81` | ~30 |
| Mount modified inputs over P-NET's files | `scripts/run_pnet_docker.sh:3-12` | ~10 |
| Deterministic inputs | `scripts/modify_data_deterministic.R:8-30` | ~20 |
| Shuffled labels | `scripts/modify_data_shuffled.R:12-36` | ~20 |
| Map node names to Reactome ids | `scripts/plot_figures.R:27-56` | ~25 |
| Rebuild the layered network from weight matrices | `scripts/plot_figures.R:89-119` | ~30 |
| Structural statistics per node | `scripts/plot_figures.R:121-136` | ~15 |
| Score vs centrality correlations | `scripts/plot_figures.R:558-575` | ~18 |
| Quantile-normalized control correction | `scripts/plot_figures.R:719-753` | ~35 |

## Fragile spots

- The P-NET network is rebuilt only from the original-seed run's trained weight
  matrices, keeping nonzero entries (scripts/plot_figures.R:89-108;
  docker/pnet/entrypoint.sh:74-78). A present edge whose trained weight is exactly 0
  would drop out. The same graph is assumed for every run and setup.
- Two degree definitions coexist. `coef_combined` is adjusted by P-NET's degree, which
  gives genes in-degree 1. The centrality analysis uses the rebuilt graph, where genes
  have in-degree 3 (scripts/plot_figures.R:121-122; README.md:204-205).
- The correction assigns columns of the normalized matrix to rows of a separately built
  `distinct(layer, reactome_id)` table by position (scripts/plot_figures.R:738-749).
  That is right only if both list the nodes in the same order. It also pivots by
  `reactome_id` after dropping `layer` (scripts/plot_figures.R:726-728), which assumes
  ids are unique across layers. Names above the gene layer with no match in
  `ReactomePathways.txt` get an NA id (scripts/plot_figures.R:54-55).
- The README writes shuffled runs to `pnet_shuffled_each` (README.md:124). The analysis
  reads `pnet_shuffled` (scripts/plot_figures.R:724).
- DTox's forward pass assumes node indices are sorted so that children come before
  parents (DTox:code/dtox_nn.py:93-100). Input columns are matched to layer-0 nodes by
  position (DTox:code/dtox_nn.py:89-91; DTox:code/targettox.py:33-34).
- DTox LRP picks one rule per node from its first child only: the input rule if that
  child is an input, the hidden rule otherwise, even when a node mixes both
  (DTox:code/dtox_lrp.py:241-248).

## Out of scope

- Plotting, styling, ROC curves, and container wrappers (`scripts/styling.R`, most of
  `scripts/plot_figures.R`, `docker/`, `scripts/run_*`).
- MSK-IMPACT loading and its seeded 80/10/10 split (scripts/load_data_mskimpact.R).
- P-NET training and loss (upstream); DTox early stopping, bootstrap intervals, path
  p-values.

## Paper vs code

- The methods summary describes shuffled labels with equal class frequencies. The code
  draws labels i.i.d. with P(1) = 0.5, so classes are equal only in expectation
  (scripts/modify_data_shuffled.R:24-32).
- DTox runs are in the code (README.md:174-182) and in figure S2
  (scripts/plot_figures.R:843-950). The methods summary named only P-NET.

## Key excerpts

`scripts/plot_figures.R:123-133`

```r
graph_stats <-
  tibble(
    reactome_id = names(V(pnet_graph)),
    indegree = degree(pnet_graph, mode = "in"),
    outdegree = degree(pnet_graph, mode = "out"),
    degree = indegree + outdegree,
    reachability = map_int(
      reactome_id,
      ~subcomponent(pnet_graph, .x, "in") %>% length()
    ),
    betweenness = betweenness(pnet_graph)
```

`scripts/plot_figures.R:722-730`

```r
  normalized_mat <-
    node_importance %>%
    filter(experiment %in% c("pnet_original", "pnet_shuffled")) %>%
    unite(experiment, seed, col = "exp_seed", sep = "/") %>%
    select(!layer) %>%
    pivot_wider(names_from = exp_seed, values_from = coef_combined) %>%
    column_to_rownames("reactome_id") %>%
    as.matrix() %>%
    normalizeQuantiles()
```

`scripts/plot_figures.R:744-747`

```r
      original_seed = normalized_mat[, "pnet_original/234_20080808"],
      original = rowMeans(normalized_mat[, idx_original]),
      control = rowMeans(normalized_mat[, !idx_original]),
      positive = original - control > 0,
```

`DTox:code/dtox_nn.py:93-102`

```python
		for scs in range(self.input_size, self.combine_size):
			# obtain input values for the current node  
			scs_children = self.node_children[scs]
			scs_children_output_list = [layer_result[sc] for sc in scs_children]
			scs_children_output = torch.cat(scs_children_output_list, 1)
			# feed input values into root loss-oriented structure to compute node output 
			scs_net_id = scs - self.input_size
			layer_result[scs] = self.net[scs_net_id](scs_children_output)
			# feed input values into auxiliary loss-oriented structure to compute auxiliary node output
			auxiliary_result[scs_net_id] = self.auxiliary[scs_net_id](layer_result[scs])
```

## Open questions

- How P-NET computes node scores (DeepLIFT layers, samples, reduction), and whether
  MSK-IMPACT genes outside its gene list are dropped: inside `pnet_prostate_paper`, not
  read; see `pnet-2021`.
- DTox hierarchy size and how its index files were built: precomputed; the analysis
  repository `yhao-compbio/DTox` was not fetched.
- How many P-NET nodes get an NA Reactome id from the name join: depends on data files
  not read.
- The methods were read through a summarizing fetch of PMC; the wording of the control
  setups is not verified verbatim.
