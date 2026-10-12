# mulgonet-2026: MULGONET

| Field | Value |
|-------|-------|
| Paper | Lan W, Tang Z, Liao H et al., 2026 (online 2025-01-11). MULGONET: An interpretable neural network framework to integrate multi-omics data for cancer recurrence prediction and biomarker discovery. Fundamental Research 6(1):99-110. 10.1016/j.fmre.2025.01.004 |
| Code | `https://github.com/lanbiolab/MULGONET` at `1bb3ca7b54f9b6d121ee94f2cd93f500b0ef5c07` (commit of 2023-09-22) |
| Other code | None found. The paper also links a web server (`http://101.33.251.35/MULGONET/`). |
| Framework | Python 3.6.2, Keras 2.2.4, TensorFlow 1.12.0, networkx 2.5.1 (listed in `README.md:23-30`; no pinned file) |
| License | None found |
| Full text | Europe PMC JATS XML of PMC12869757 (open access) |
| Extracted | 2026-10-09, Claude Code, claude-opus-5-5 |

## Summary

Binary prediction of cancer recurrence from TCGA multi-omics data
(mRNA, DNA methylation, copy-number amplification and deletion; 1000
genes per omics after chi-square selection). Prior knowledge is the
Gene Ontology hierarchy, split into biological process (BP) and
molecular function (MF), plus gene-to-term annotations. Each namespace
becomes a five-layer sparse feedforward branch with one unit per GO
term, both on the same input, joined by a small dense head with one
sigmoid output. Input and node scores come from an
integrated-gradients-style attribution summed over recurrence samples.

## Prior-knowledge graph

- Source: GO edge table `data/GO_hierarchical_structure.csv`, columns
  term, parent term, namespace (e.g.
  `GO:0000003,GO:0008150,biological_process`); release not stated. One
  graph per namespace (`preprocessing.py:116-120`). The code labels the
  columns the other way round (`preprocessing.py:122`) and draws edges
  from column `1` to `0` (`preprocessing.py:125-126`), so edges run
  general -> specific. Annotations: pickled dicts term -> gene symbols
  (`data/gene_data_*.npy`, `preprocessing.py:87-109`); terms with more
  than 200 genes are dropped (`preprocessing.py:91-93`, `104-106`).
- Size (paper Table 2): BP layers 2819 / 2252 / 534 / 114 / 19 terms,
  MF 1428 / 522 / 172 / 37 / 10, most specific first. Inputs 4 x 1000
  features. Directed, unsigned, acyclic. Edge counts not reported.
- Data to nodes: genes match terms by exact symbol
  (`preprocessing.py:336-341`, `350-355`). Each omics gets its own
  gene x term block. The four blocks are concatenated in the order
  meth, amp, del, exp (`preprocessing.py:328`, `363-364`). A gene can
  thus be up to four inputs with the same edges. Only bottom-layer
  terms receive genes. Features whose gene has no bottom-layer term in
  either namespace are removed before feature selection
  (`preprocessing.py:238-250`). A BP-only gene has an all-zero row in
  the MF mask. A bottom-layer term without any selected gene keeps a
  unit with no incoming edge.
- Built at: `preprocessing.py:19-155`, `318-372`; called at
  `training.py:34-50`.

## Architecture

- Layers: per branch, one gene-to-term layer and four term-to-term
  layers (`MULGONET.py:183-224`). A term's layer is its shortest-path
  distance from `roots[0]`, the first parentless node
  (`preprocessing.py:19-33`, `129`), for distances 1 to 5
  (`preprocessing.py:133-135`). Data enters at distance 5 (`h0`) and
  flows toward distance 1 (`h4`). Deeper terms are dropped. From
  distance 4 up to 1, terms with no child in the next deeper layer are
  removed (`preprocessing.py:38-53`). The distance-5 layer is then cut
  to annotated terms (`preprocessing.py:142-148`). Upper layers are
  not pruned again after that cut.
- Units per node: one per GO term; one input per (gene, omics) pair.
  Every unit has a bias and tanh (`MULGONET.py:26`, `81-85`).
- Connections that skip layers: none. Only edges to a child in the
  next distance layer are kept (`preprocessing.py:63-72`).
- Outputs or losses at inner layers: none.
- Parts the graph does not constrain: `Dense(16, tanh)` and
  `Dense(1, sigmoid)` on the concatenated `h4` layers
  (`MULGONET.py:228-230`). Single-branch modes put `Dense(1, sigmoid)`
  on `h4` (`MULGONET.py:148`, `177`). Dropout 0.5 after `h0`, 0.1
  after `h1`-`h3` (`MULGONET.py:185-219`).
- Other structure: two parallel branches on one shared input
  (`MULGONET.py:119`, `183`, `203`). No recurrence, attention, or
  decoder.
- Defined at: `MULGONET.py:115-242`.

## Connectivity mechanism

A custom Keras layer stores one trainable 1-D vector with one entry
per present edge (`MULGONET.py:55-61`). The edge index list is
`np.nonzero` of the 0/1 mask, taken once at build
(`MULGONET.py:49-51`); mask values beyond the pattern are not used.
Each forward pass scatters the vector into a dense (inputs x units)
kernel with zeros elsewhere (`tf.scatter_nd`, `MULGONET.py:76-77`),
then applies matmul, bias, and tanh (`MULGONET.py:79-85`). The dense
kernel is a temporary tensor. Absent edges have no parameter and no
gradient. Init is `glorot_uniform` on the 1-D vector
(`MULGONET.py:25`, `59`); Keras 2.2.4 then takes fans from the vector
length, not layer sizes (library behavior). Masks are oriented
(inputs x units): transposed term x feature for the gene layer
(`MULGONET.py:183`), deeper x shallower terms above it
(`preprocessing.py:68-75`). L2 (1e-4) sits on the gene-to-term vectors
only (`MULGONET.py:183`, `203`). `get_config` keeps the index list for
cloning (`MULGONET.py:95`).

## Interpretation

- Method: the script asks for `deepexplain_deeplift`
  (`weight_coef.py:18-19`). The vendored DeepExplain code replaces any
  method name with `intgrad`
  (`single_inputs_IntegratedGradients/tensorflow_.py:251`). Baseline
  is an all-zero input; the layer baseline is the layer's activation
  at that input (`single_inputs_IntegratedGradients/tensorflow_.py:101-105`, `183`). Steps: 10
  (`single_inputs_IntegratedGradients/tensorflow_.py:159`; none passed at `single_inputs_IntegratedGradients/coef_weights_utils.py:98`).
  In the step loop the gradient at the interpolated input
  (`single_inputs_IntegratedGradients/tensorflow_.py:173`) is overwritten by the gradient at the real
  input (`single_inputs_IntegratedGradients/tensorflow_.py:175`). The result is that gradient times
  (activation - baseline activation) (`single_inputs_IntegratedGradients/tensorflow_.py:187-191`).
- Granularity: input features (`input_multi`) and every unit of every
  graph layer (`h0_bp` ... `h4_mf`), for the final output
  (`single_inputs_IntegratedGradients/coef_weights_utils.py:37-56`, `73-98`). Layers are selected by
  name prefix `h` or `input` (`single_inputs_IntegratedGradients/coef_weights_utils.py:40`).
- Names: inputs are labeled `<gene>_<omics>` from the mask columns
  (`weight_coef.py:27-31`); nodes take the index or columns of the
  matching relation matrix (`weight_coef.py:34-38`).
- Combination: signed sum over all recurrence-positive samples
  (`single_inputs_IntegratedGradients/coef_weights_utils.py:54`; samples from `training.py:65-70`), per
  fold, written to `weight_coef/h<fold>/<layer>.csv` sorted by score
  (`weight_coef.py:42-60`, `training.py:107-108`). Combining folds:
  not found.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N01 Read a DAG from an edge table | `preprocessing.py:116-126` | GO parent-child pairs of one namespace become a directed graph. |
| N02 Layer nodes by depth from a root | `preprocessing.py:19-33`, `133-135` | Each term sits at its shortest distance from the root, cut at depth 5; data enters at the deepest layer. |
| N03 Keep only edges between adjacent layers | `preprocessing.py:63-72` | Edges within a layer or across several layers are dropped. |
| N04 Prune nodes with no child below | `preprocessing.py:38-53` | Terms with no child in the next deeper layer are removed, bottom up. |
| N05 Filter nodes by gene-set size | `preprocessing.py:91-93` | Terms with more than 200 annotated genes are dropped. |
| N06 Connect features to nodes by membership | `preprocessing.py:142-148`, `238-250`, `336-341` | Genes join bottom-layer terms through an annotation table matched by symbol; unmatched genes are removed. |
| N07 One input per entity and data type | `preprocessing.py:328`, `363-364` | Each gene appears once per omics, each copy with the same edges. |
| N08 Parallel graph branches on one input | `MULGONET.py:183-228` | BP and MF branches read the same input and are concatenated. |
| N09 Dense head after the graph layers | `MULGONET.py:229-230` | A 16-unit dense layer and a sigmoid output sit on top. |
| N10 One unit per node | `MULGONET.py:183`, `188-200` | Layer width equals the number of terms in the layer. |
| N11 Standard layers between graph layers | `MULGONET.py:185-219` | Dropout follows each graph layer except the last. |
| N12 Parameters only for present edges | `MULGONET.py:55-61`, `76-79` | One weight per edge; absent edges have no weight and no gradient. |
| N13 Regularize edge weights per layer | `MULGONET.py:183`, `203` | L2 only on the gene-to-term edge weights. |
| N14 Attribution to inputs | `single_inputs_IntegratedGradients/coef_weights_utils.py:40`, `weight_coef.py:44-51` | Each `<gene>_<omics>` input gets a score. |
| N15 Attribution to hidden nodes | `single_inputs_IntegratedGradients/coef_weights_utils.py:37-56` | Each term unit in every layer gets a score for the output. |
| N16 Integrated gradients from a zero baseline | `single_inputs_IntegratedGradients/tensorflow_.py:157-193`, `251` | The paper's stated method; the code path differs (see Interpretation). |
| N17 Map scores to node names | `weight_coef.py:27-38` | Scores are labeled with term IDs and gene-omics names per layer. |
| N18 Aggregate scores over a sample subset | `single_inputs_IntegratedGradients/coef_weights_utils.py:54`, `training.py:65-70` | Signed sum over recurrence-positive samples. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Read GO edges of one namespace into a DAG; find roots | `preprocessing.py:114-129` | ~16 |
| Assign terms to depth layers | `preprocessing.py:19-33` | ~15 |
| Remove terms without a child in the next layer | `preprocessing.py:38-53` | ~16 |
| Build one term x term 0/1 matrix per layer pair | `preprocessing.py:58-76` | ~19 |
| Load annotations; drop terms > 200 genes | `preprocessing.py:81-109` | ~29 |
| Cut bottom layer to annotated terms | `preprocessing.py:142-148` | ~7 |
| Keep omics features whose gene has a term | `preprocessing.py:234-250` | ~17 |
| Build gene x term masks per omics; concatenate | `preprocessing.py:318-372` | ~55 |
| Concatenate omics inputs in mask order | `training.py:54`, `66-70` | ~6 |
| Edge-vector layer scattered into a dense kernel | `MULGONET.py:24-109` | ~86 |
| Wire five layers per branch from the matrices | `MULGONET.py:183-230` | ~48 |
| Per-layer attribution loop, sum over samples | `single_inputs_IntegratedGradients/coef_weights_utils.py:29-105` | ~77 |
| Label scores with names; write per layer | `weight_coef.py:16-60` | ~45 |

## Fragile spots

- Only `roots[0]` seeds the layers (`preprocessing.py:20-22`, `129`).
  With several parentless nodes, terms reachable only from the others
  would be dropped.
- Upper layers are not re-pruned after the bottom cut
  (`preprocessing.py:148`); a term that lost all children keeps a
  bias-only unit.
- Alignment is by position, never checked by name: data columns
  (`training.py:54`) vs mask rows (`preprocessing.py:363-364`), input
  score names (`weight_coef.py:27-31`), and one layer's columns vs the
  next layer's rows (`preprocessing.py:68-75`).
- Two columns with the same symbol in one omics table would connect
  only the first, via `list.index` (`preprocessing.py:338`, `352`).
- The attribution loop discards the interpolated gradient
  (`single_inputs_IntegratedGradients/tensorflow_.py:173-175`); the method name is fixed
  (`single_inputs_IntegratedGradients/tensorflow_.py:251`).
- Single-branch modes name layers `input_drop`, `bp_layer1`, ...
  (`MULGONET.py:125-145`, `154-174`), which miss the `h*` keys of the
  score code (`weight_coef.py:24-38`). `input_drop` also matches the
  `input` prefix (`single_inputs_IntegratedGradients/coef_weights_utils.py:40`).

## Out of scope

- Loss: class-weighted binary cross-entropy (`MULGONET.py:236-238`,
  `evaluates.py:35-44`).
- Training: Adam lr 1e-3, 150 epochs, batch 64, 5-fold CV, test fold
  as validation data (`training.py:59-96`).
- Data loading: PAAD only; hard-coded Windows path for the
  protein-coding list (`preprocessing.py:160-185`, `222-224`). Other
  cohorts via Baidu links (`data/other_datasets.txt`).
- Feature selection: chi-square top 1000 per omics on all samples
  before the CV split (`preprocessing.py:189-207`, `292-295`).
- CNV binarization, expression filter, scaling
  (`preprocessing.py:253-267`, `300-303`); metrics and baselines
  (`evaluates.py:11-30`, `Comparison.py`).

## Paper vs code

- Fusion: paper Eq. 11 sums the BP and MF outputs; code concatenates
  them (`MULGONET.py:228`).
- Attribution: paper uses integrated gradients with m = 20; code uses
  10 steps (`single_inputs_IntegratedGradients/tensorflow_.py:159`) and the real-input gradient in every
  step (`single_inputs_IntegratedGradients/tensorflow_.py:175`). The script requests DeepLIFT
  (`weight_coef.py:19`); it is overridden (`single_inputs_IntegratedGradients/tensorflow_.py:251`).
- Global scores: paper says per-sample scores are "weighted"; code
  takes an unweighted signed sum (`single_inputs_IntegratedGradients/coef_weights_utils.py:54`).
- Regularization: paper Eq. 14 puts L2 on all parameters; code only on
  the gene-to-term layers (`MULGONET.py:183`, `203`).
- The > 200-gene term filter (`preprocessing.py:91-93`) and the
  dropout rates (`MULGONET.py:185-219`) are not in the methods text.
- Data: paper uses PAAD, STAD, BLCA; code loads PAAD only
  (`preprocessing.py:166-175`).

## Key excerpts

`MULGONET.py:76-79`

```python
        tt = tf.scatter_nd(tf.constant(self.nonzero_ind, tf.int32), self.kernel_vector,
                           tf.constant(list(self.kernel_shape)))

        output = K.dot(inputs, tt)
```

`preprocessing.py:19-23`

```python
def get_nodes_at_level(net, distance,roots):
    nodes = set(nx.ego_graph(net, roots[0], radius=distance))
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, roots[0], radius=distance - 1))
    return list(nodes)
```

`single_inputs_IntegratedGradients/tensorflow_.py:170-176`

```python
        for alpha in list(np.linspace(1. / self.steps, 1.0, self.steps)):
            xs_mod = [b + (xs - b) * alpha for xs, b in zip(self.xs, self.baseline)] if self.has_multiple_inputs \
                else self.baseline + (self.xs - self.baseline) * alpha
            _attr = self.session_run(attributions, xs_mod)
            # print ('attributions',attributions)
            _attr = self.session_run(attributions, self.xs)
            xss = self.session_run(self.X, self.xs)
```

## Open questions

- Direct or ancestor-propagated annotations in the pickled dicts:
  only the first bytes were viewed (`GO:0000002` -> gene symbols).
- Whether the hierarchy holds only "is a" edges, as the paper says:
  the CSV has no relation column; only its first lines were read.
- Parentless nodes per namespace and edges per layer: not computed.
- Mapping of CNV Ensembl IDs and CpG probes to symbols: not in the
  repository (data ships preprocessed in `data/PAAD_data.rar`).
- How per-fold scores were combined for the figures: no code found.
- Provenance: `tensorflow_.py` carries DeepExplain's class and
  dictionary names (`single_inputs_IntegratedGradients/tensorflow_.py:196-215`). The graph and
  edge-vector code shares function names with the P-NET code release.
  Neither is credited in the repository; not checked against them.
- Keras learning phase during attribution: the code never sets it
  (`single_inputs_IntegratedGradients/tensorflow_.py:89-90`, `229`), so dropout follows Keras's default.
