# gennet-2021: GenNet

| Field | Value |
|-------|-------|
| Paper | van Hilten A, Kushner SA, Kayser M, Ikram MA, Adams HHH, Klaver CCW, Niessen WJ, Roshchupkin GV, 2021. GenNet framework: interpretable deep learning for predicting phenotypes from genetic data. Communications Biology. 10.1038/s42003-021-02622-z |
| Code | `https://github.com/ArnovanHilten/GenNet` at `6507b03e6b2285b890fe5ba6d2326769d0ed0b99` (2026-07-29) |
| Other code | Zenodo archive cited by the paper; `ArnovanHilten/GenNet_Zoo` (trained networks) and `ArnovanHilten/GenNet_paper_plots`, linked from `README.md`. None fetched. Repo tags `1.0` to `v2.0` not compared with HEAD. |
| Framework | Python, TensorFlow/Keras; `tensorflow==2.11` (`requirements_GenNet.txt:24`) |
| License | Apache 2.0 (`LICENSE`) |
| Full text | Europe PMC full-text XML, PMC8448759 (supplementary notes not read) |
| Extracted | 2026-10-16, Claude Code, Claude Opus 5.5 |

## Summary

GenNet predicts a phenotype (binary, or continuous with a regression
option) from genotypes with up to about 7 million variants
(Rotterdam Study, UK Biobank exomes, Swedish schizophrenia exomes).
Prior knowledge sets the connections: variants join their annotated
gene (Annovar, RefGene), genes join KEGG pathways, pathways join
KEGG mid-level and top-level categories, and in the paper genes also
join tissues or cell types. Each layer is a sparse layer with
weights only for present connections, one unit per node, tanh, and
batch normalization without learned shift or scale; one dense unit
reads the last graph layer. Importance is the product of trained
weights along each input-to-output path, summed per node.

## Prior-knowledge graph

- Source: variant→gene from Annovar `-geneanno -dbtype refGene
  -buildver hg19` output (`Topology.py:54-59`, `:72`); gene→pathway
  from a ConsensusPathDB table filtered to KEGG
  (`Topology.py:122-124`, 333 KEGG rows) joined on KEGG id to
  `resources/pathways/pathway_overview_KEGG.csv` (330 pathways, 44
  mid-level, 6 top-level) (`:119-127`). Paper: GeneSCF and KEGG.
  Expression layers (top 10% t-statistic genes per tissue) are paper
  only; the code raises `NotImplementedError` (`:170-172`).
- Size: inputs = genotype columns (`Dataloader.py:80-85`); 21,390
  genes for 1,288,701 variants in the schizophrenia net (paper, Fig.
  2). Directed, unsigned, acyclic. Variant→gene many-to-one,
  gene→pathway many-to-many, pathway→mid→top many-to-one (a tree).
- Data to nodes: by position, not name. A variant's row in the
  Annovar input (`index_col`, `Topology.py:43-45`) is its genotype
  column and its matrix row (`:94-96`, `:103-107`). Gene strings are
  cut before the first `,` or `;` and stripped of `(...)` (`:74-78`),
  so the first listed gene is kept; `NONE` rows are dropped
  (`:79-80`). Gene ids follow chromosome and position (`:82-85`).
  Genes join a pathway by symbol in its member list (`:132-138`).
  With a topology table the first matrix has `inputsize` rows
  (`Create_network.py:259-260`): variants without a gene stay as
  unconnected inputs. Genes in no pathway keep a unit with no
  outgoing connection; empty pathways keep a bias-only unit
  (matrix shapes, `Topology.py:141-142`).
- Built at: `Topology.py:65-167`; `Create_network.py:138-180`
  (`.npz`), `:242-265` (topology table).

## Architecture

- Layers: one sparse layer per connectivity matrix; the caller
  supplies the layering, as a `topology.csv` whose rows are full
  paths with one `layer<k>_node` column per layer (`README.md`;
  `Create_network.py:242-265`), or one `.npz` matrix per adjacent
  pair, chained by `-mask_order` or by decreasing size with a shape
  assert (`:144-176`). Paper nets: variants→genes→output;
  variants→genes→pathways→mid→top→output; variants→genes→tissues→output.
- Units per node: one (`filters=1`, `Create_network.py:66`), bias
  per node (`LocallyDirected1D.py:162-168`).
- Connections that skip layers: none.
- Outputs or losses at inner layers: none.
- Parts the graph does not constrain: `Dense(1)` output on the
  flattened last graph layer, kernel L1, bias = training mean for
  regression (`Create_network.py:195-204`, `:267-272`). With `cov_*`
  columns in `subjects.csv` (`Train_network.py:59`) that output,
  after its activation, is concatenated with covariates, normalized,
  and fed to a second `Dense(1)` (`Create_network.py:100-116`).
- Other structure: after each graph layer tanh (classification) or
  ReLU (regression) (`Create_network.py:59-64`), then
  `BatchNormalization(center=False, scale=False)` (`:70-71`) or
  `PerVariantNormalization` with `-improved_norm` (`:72-73`). Each
  graph layer has an L1 activity penalty on its pre-activation
  output (`:66-67`). No recurrence, attention, or decoder.
- Defined at: `Create_network.py:58-75`, `:118-214`, `:218-284`;
  selection at `Train_network.py:266-306`.

## Connectivity mechanism

Sparse product from an index list. `LocallyDirected1D` takes a scipy
COO matrix (inputs × nodes) and creates a kernel of shape
`(len(mask.data), input_dim*filters)`, one weight per stored entry
(`LocallyDirected1D.py:152-157`), plus a list of `(col, row)` pairs
(`:160`, `:243-250`); stored values are ignored. The forward pass
builds a `SparseTensor` (nodes × inputs) from the pairs and the
kernel, computes `sparse_dense_matmul(W, X, adjoint_b=True)`,
transposes to (batch, nodes, 1) (`:223-239`), and adds the per-node
bias (`:187-188`). No full matrix exists; absent connections have no
parameter and no gradient. `glorot_uniform` on the 2D kernel
(`:106`, `:152`) takes its scale from that array's shape (the
layer's entry count), not from each node's fan-in; bias zeros
(`:107`). The mask is left out of `get_config` (`:253-255`);
reloading rebuilds the network from the topology files
(`Train_network.py:345-363`). Unused variants:
`LocallyDirected1D_scatter.py:185` (scatter to a dense matrix) and
`LocallyDirectedConnected.py:271`.

## Interpretation

Paper: weights, comparable thanks to normalization without shift or
scale; Manhattan plots use normalized absolute gene→output weights;
deeper nets multiply "the weights for each possible path from the
end node to the input SNPs" and sum "according to the groups of the
subsequent layers". Multiplying by activation standard deviation is
mentioned; not found in code.

Code: after training, `create_importance_csv`
(`Utility_functions.py:147-181`; called at `Train_network.py:238-240`)
lists each layer's `(node_layer_i, node_layer_i+1, weights_i)`, adds
node names from the topology (`:162-164`), merges layers on node ids
into one row per input-to-output path with the output weight
(`:172-179`), and sets `raw_importance` to the product (`:180`);
saved as `connection_weights.csv`. Manhattan: `|raw_importance|` over
its maximum per row (`Create_plots.py:191-193`). Sunburst: each
path's absolute product as a percentage of the total, summed per
node name level by level (`Create_plots.py:38-44`). CLI
`interpret -type get_weight_scores` does the same walk over model
layers (`Interpret.py:41-48`, `weight_importance.py:17-128`). One
score per trained network, not per sample; no combination across
runs found.

Added after the paper (no needs recorded): Deep SHAP on the network
with normalization layers removed, background = validation controls,
max over test cases (`Interpret.py:51-99`); NID (`:102-134`), DFIM
(`:143-219`), PathExplain (`:221-318`) interactions; RLIPP prints
"not implemented yet" (`:138-139`).

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N06 Connect features to nodes by membership | `Topology.py:103-107`, `:132-138` | Variants join their Annovar gene (by genotype column index); genes join KEGG pathways by symbol, possibly several. |
| N10 One unit per node | `Create_network.py:66`; `LocallyDirected1D.py:162-168` | Each gene, pathway, or category is one unit with its own bias and a shared tanh. |
| N11 Standard layers between graph layers | `Create_network.py:68-73` | Normalization without shift or scale follows every graph layer. |
| N12 Parameters only for present edges | `LocallyDirected1D.py:152-157` | One weight per connectivity entry; millions of inputs rule out a full matrix. |
| N69 Sparse product from an index list | `LocallyDirected1D.py:227-228`, `:243-250` | Fixed (node, input) pairs and a value vector feed one sparse-dense product with the transposed batch; no index offsets. |
| N119 Rescale node values before a linear readout | `Create_network.py:70-71`, `:195-204` | The last graph layer is normalized before the dense output unit so its weights compare as importances. |
| N17 Map scores to node names | `Utility_functions.py:162-164` | Path rows carry each layer's node name. |
| N32 Edge weights labeled by edge | `Utility_functions.py:157-160`; `Train_network.py:238-240` | `connection_weights.csv` gives each connection weight with ids and names of both ends; no biases. |
| N139 Prior given as one matrix per layer | `Create_network.py:144-176`, `:242-265` | Users supply the layered prior as `.npz` matrices or a path table; the code chains them. |
| N140 L1 penalty on node activity | `Create_network.py:66-67`; `GenNet.py:186-190` | Every graph layer's pre-activation output has an L1 activity penalty (`-L1_act`, default 0.01). |
| N141 Covariates after the graph's prediction | `Create_network.py:100-116` | Covariates join the graph's own output in a second output unit and enter no node. |
| N142 Path products of edge weights | `Utility_functions.py:180`; `Create_plots.py:38-44` | Variant, gene, and pathway importance is the weight product along each path, as shares, summed per node. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Annovar input from the genotype probe table | `Topology.py:8-62` | ~55 |
| Annovar output → variant-gene table and matrix | `Topology.py:65-112` | ~48 |
| Gene → KEGG pathway, mid, top matrices | `Topology.py:115-167` | ~53 |
| Topology table → `.npz` (script) | `Convert_topology_npz.py:12-43` | ~32 |
| Load, order, shape-check `.npz` matrices | `Create_network.py:138-180` | ~43 |
| Path table → one COO matrix per layer | `Create_network.py:242-265` | ~24 |
| Sparse layer: index list, sparse product | `LocallyDirected1D.py:131-250` | ~120 |
| Rebuild without normalization, re-pass matrices | `Create_network.py:520-600` | ~80 |
| Weights → named path table with products | `Utility_functions.py:147-181` | ~35 |
| Weight products through model layers | `weight_importance.py:17-128` | ~110 |
| Path shares summed per node | `Create_plots.py:23-52` | ~30 |

## Fragile spots

- Layer-wide normalization: Keras `BatchNormalization` (`axis=-1`) on
  the (batch, nodes, 1) output (`Create_network.py:70-71`,
  `LocallyDirected1D.py:236-239`) uses one mean and variance for all
  nodes; per node only where `Flatten` comes first
  (`Create_network.py:39-41`, `:330-332`).
- Repeated pairs: path-table matrices are not de-duplicated
  (`Create_network.py:256-263`); a variant→gene pair repeats once per
  pathway of the gene, one kernel entry each (`LocallyDirected1D.py:152`).
- Weights by position: `Utility_functions.py:159-160` sorts by
  `node_layer_i`, then assigns the kernel positionally, while kernel
  order follows all node columns (`Create_network.py:245`); differs
  for `i ≥ 1`. `:156` takes `chr` from the unsorted file.
- Sunburst sums per-row group totals (`Create_plots.py:42-44`), so a
  lower node counts once per row it spans.
- Several filters reuse `kernel[:, i]` (`LocallyDirected1D.py:181-183`,
  `:225-228`): one weight per connection, filters differ by bias.
- Attribution runs on a network whose normalization layers were
  dropped unfolded (`Create_network.py:539-564`).
- The mask is not saved; reload rebuilds it from current files
  (`LocallyDirected1D.py:253-255`, `Train_network.py:359-362`).
- `create_network_from_npz` ignores `batchnorm` (`Create_network.py:193`);
  a one-column last `.npz` is applied twice, then on 2D input
  (`:177-199`, `LocallyDirected1D.py:127`).
- Regex `str.replace` (`Topology.py:74-78`) is literal under pinned
  pandas 2.2.2; `np.bool` (`:140`, `:150`, `:161`) is gone in pinned
  NumPy 1.26.4 (`requirements_GenNet.txt:6-7`).

## Out of scope

- Loss (weighted BCE or MSE), Adam, early stopping (`Train_network.py:83-151`, `:308-314`).
- PLINK/VCF → HDF5 (`Convert.py`, `hase/`), batching (`Dataloader.py`), plots, SLURM, Docker.
- LASSO baseline and height net (`Create_network.py:287-308`, `:484-516`); later interaction methods.

## Paper vs code

- Penalty: paper tunes an "L1 kernel regularization penalty"; code
  has L1 on graph-layer activity (`Create_network.py:66-67`) and a
  kernel L1 only on the output unit (`:201-203`).
- Normalization: paper says per-layer zero mean and unit variance;
  code statistics are layer-wide (Fragile spots).
- Not in code: weights × activation standard deviation; tissue, cell,
  and exon layers (`Topology.py:170-172`); random controls; Adadelta
  (`Train_network.py:308`). KEGG via ConsensusPathDB, not GeneSCF
  (`Topology.py:122-124`).
- HEAD adds covariates, one-hot input via a per-variant
  `LocallyConnected1D` (`Create_network.py:92-97`), several units per
  gene (`:346-480`), `-improved_norm`, and later methods.

## Key excerpts

`GenNet_utils/LocallyDirected1D.py:152-157`

```python
        self.kernel = self.add_weight(shape=(len(self.mask.data), input_dim*self.filters ),    
        #sum of all nonzero values in mask sum(sum(mask))
                            initializer=self.kernel_initializer,
                            name='kernel',
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint)
```

`GenNet_utils/LocallyDirected1D.py:227-228`

```python
            output_flat = sparse_dense_matmul(sp_a= SparseTensor(kernel_idx, kernel, (mask.shape[1], mask.shape[0])),
                                              b=inputs_flat, adjoint_b=True)
```

`GenNet_utils/Create_network.py:256-263`

```python
    for i in range(len(columns) - 1):
        matrix_ones = np.ones(len(network_csv[[columns[i], columns[i + 1]]]), bool)
        matrix_coord = (network_csv[columns[i]].values, network_csv[columns[i + 1]].values)
        if i == 0:
            matrixshape = (inputsize, network_csv[columns[i + 1]].max() + 1)
        else:
            matrixshape = (network_csv[columns[i]].max() + 1, network_csv[columns[i + 1]].max() + 1)
        mask = scipy.sparse.coo_matrix(((matrix_ones), matrix_coord), shape = matrixshape)
```

`GenNet_utils/Create_plots.py:38-44`

```python
    csv_file["percentage_0"] = csv_file.filter(like="weights").prod(axis=1)
    csv_file["percentage_0"] = abs(csv_file["percentage_0"]) / abs(csv_file["percentage_0"]).sum() * 100
    print(csv_file["percentage_" + str(0)].sum())

    for i in range(1, first_number_layer_to_plot + 1):
        csv_file["percentage_" + str(i)] = csv_file.groupby("layer" + str(i) + '_name')[
            "percentage_" + str(i - 1)].transform('sum')
```

## Open questions

- Code version behind the paper: HEAD is 2026; tags and Zenodo not compared.
- Node counts per paper network (Supplementary Note 9): not read.
- Builders for expression, exon, and random-control networks: not found.
- Whether `sparse_dense_matmul` adds repeated pairs, and whether the
  paper used covariates: not determined (code not run; main text silent).
