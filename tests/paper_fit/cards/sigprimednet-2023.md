# sigprimednet-2023: SigPrimedNet

| Field | Value |
|-------|-------|
| Paper | Gundogdu P, Alamo I, Nepomuceno-Chamorro IA, Dopazo J, Loucera C, 2023. SigPrimedNet: A Signaling-Informed Neural Network for scRNA-seq Annotation of Known and Unknown Cell Types. Biology 12(4):579. 10.3390/biology12040579 |
| Code | `https://github.com/babelomics/sigprimednet` at `4dbb874e991648371521807d87f47a5b54aa94ea` (branch `develop`). The paper links tag `v1.0.0` (`2701f0447ec17e8eb750fba169e2abf8da3ee102`); the two differ only in `README.md`. |
| Other code | none |
| Framework | Python 3.7.9, TensorFlow 2.2.0, Keras 2.4.3, keras-tuner 1.0.1 (`environment.yml:170,299,363-364`); R `hipathia` for the prior; DeepExplain for LRP |
| License | none found |
| Full text | PubMed Central PMC10135788 (HTML), read through a summarizing fetch; the last ~7k characters (end of appendices, references) not read |
| Extracted | 2026-10-11, Claude Code, Claude Opus 5.5 |

## Summary

SigPrimedNet annotates cell types in scRNA-seq data and flags cells
of unknown type. KEGG signaling pathways are split into effector
circuits with HiPathia. Each circuit becomes one unit of the first
hidden layer, connected to that circuit's genes. The published
design has this one circuit layer followed by a softmax over cell
types. The circuit activations are the cell encoding. A Local
Outlier Factor fitted on training encodings labels test cells
below a threshold "unassigned". Circuits are interpreted through
their activations per cell type.

## Prior-knowledge graph

- Source: KEGG pathways from R `hipathia` (`load_pathways`,
  `scripts/pathway_layer_data/1.1-pg-pathway-from-hipathia.r:41`;
  the paper names v2.11.4). Pathways whose name contains
  "disease", "cancer", "virus", or similar keywords are dropped,
  along with 14 named pathways (`1.2-pg-remove-disease-cancer.py:44-55`).
  Each HiPathia effector subgraph becomes one circuit node. Its
  gene set is the union of the genes of all nodes in that subgraph
  (`1.3-pg-gene-from-hipathia.r:76-118`). HiPathia builds the
  subgraphs: an effector node and the nodes upstream of it. The
  edges and signs inside a circuit are not kept. The network uses
  only a gene × circuit 0/1 membership matrix.
- Size: genes (inputs) and circuits (hidden units). The paper
  reports 92 pathways and 1,210 circuits. A code comment says 93
  pathways and 4,502 sub-pathways (`1.3-...r:64-66`). The gene
  count and pair count were not found; the matrix file is not in
  the repo. Edges run from gene to circuit: unsigned, bipartite,
  no cycles.
- Data to nodes: Entrez IDs are mapped to symbols with
  `org.Hs.eg.db`, and IDs without a symbol are dropped
  (`1.4-pg-gene-id-entrez-converter.r:52-53`). Symbols are
  lowercased in the matrix (`1.6-pg-creating-biological-layer_circuits.py:49`)
  and in the data (`scripts/dataset_scripts.py:40`). The matrix is
  left-joined onto the data's gene order and missing values are
  filled with 0 (`notebooks/4.0-pg-model.py:165-169`). Circuit
  genes missing from the data are dropped. Data genes in no
  circuit stay as inputs with all-zero mask rows, because every run
  passes `-filter_gene_space False` (`project_bash.sh:79`); `True`
  would drop them (`4.0-pg-model.py:152-161`). A circuit with no
  data gene keeps its unit, with an all-zero column.
- Built at: `1.3-pg-gene-from-hipathia.r:76-118`,
  `1.6-pg-creating-biological-layer_circuits.py:42-54` (one matrix
  `pbk_circuit_hsa_sig.txt`), `4.0-pg-model.py:150-175` (alignment).

## Architecture

- Layers: one graph layer, from genes to circuits
  (`scripts/nn_design_scripts.py:69-76`). In the published 1-layer
  design it feeds the softmax directly (`nn_design_scripts.py:82`;
  `00_REVIEW_mdpi_rskf.sh:3`;
  `paper2_figures_for_Review/00_experiment.py:71`).
- Units per node: one ReLU unit per circuit, bias initialized to
  zero (`nn_design_scripts.py:69-74`, `project_bash.sh:77`).
- Connections that skip layers: none.
- Outputs or losses at inner layers: none. Categorical
  cross-entropy on the softmax (`nn_design_scripts.py:91-93`);
  class weights 1/count in `4.0-pg-model.py:196-202,311`.
- Parts the graph does not constrain: the output layer
  (`nn_design_scripts.py:82`). The 2-layer design adds `Dense(100)`
  before it (`:78-80`); the tuning model searches 0–200 units
  (`:137-143`). An unused option, `-first_hidden_layer_dense N`,
  appends N all-ones mask columns. That gives fully connected units
  beside the circuits (`4.0-pg-model.py:135-141,171-175`); every
  run sets N to 0 (`project_bash.sh:95`).
- Other structure: the encoding is `model.layers[-1].input`
  (`00_experiment.py:182-183`), which in the 1-layer design is the
  circuit activations. A `LocalOutlierFactor(novelty=True)` fitted
  on the training encodings relabels low-scoring test cells
  (`00_experiment.py:189-235`).
- Defined at: `scripts/nn_design_scripts.py:19-100` (`proposed_NN`),
  `:104-158` (tuning variant).

## Connectivity mechanism

The layer is a plain Keras `Dense` layer with a full gene × circuit
kernel (`nn_design_scripts.py:69-74`). Once the layer is built, the
code reads the Glorot-uniform initial kernel (seed 91), multiplies
it elementwise by the 0/1 matrix, and writes it back with zero
biases through `model.set_weights` (`:76`; the tuning model does
the same at `:135`). Nothing else touches the connectivity. A
search for `mask|constraint|callback|set_weights` finds no kernel
constraint, no forward-pass mask, no gradient mask, and no callback
other than `EarlyStopping` (`4.0-pg-model.py:215-218`). Training
is a plain `model.fit` with Adam at lr 0.001
(`nn_design_scripts.py:85`, `4.0-pg-model.py:305-311`). The
entries for absent edges start at zero and then get gradients and
updates like every other entry. The Glorot scale uses the full
layer width, not each circuit's gene count.

## Interpretation

- Circuit activations: the circuit-layer output
  (`scripts/model_scripts.py:38-39`) for all cells, labeled by
  circuit name and predicted cell type
  (`notebooks/5.0-pg-pbk-analysis.ipynb:1036-1040`). For Immune,
  100 saved models each write one CSV with predicted and true type
  (`5.1-pg-activation-each-split.ipynb:630-649`). No combination
  across models was found.
- Ranking per cell type: circuits whose per-class standard
  deviations sum to less than 1 are dropped (`5.0-...ipynb:1075,1484`).
  For each cell, the code takes up to 15 circuits with positive
  activation, highest first. It counts these per predicted cell
  type and keeps the 10 most frequent circuits (`:1594-1628`). The
  paper's Table A2 gives the "ten highest-weighted nodes" per type
  for Immune. The notebook is stored with Melanoma settings
  (`:637`).
- Weights: the trained kernel is labeled by gene and circuit. The
  100 largest-weight genes per circuit are checked against the
  circuit's effector gene (`:2248-2249,2363-2364,2651`). Each
  circuit is assigned to the cell type with its largest output
  weight (`:1515`).
- Gene-level LRP per circuit (not in the paper): DeepExplain
  epsilon-LRP (`'elrp'`, default parameters). The target is one
  circuit unit at a time, picked by multiplying the layer output
  by a one-hot vector, and scores are on genes for every cell
  (`notebooks/9.2-pg-deepExplain_detail.py:84-103`). Scores are
  averaged and ranked per true cell type (`:105-115`).
  `9.1-pg-deepExplain.py:76-106` averages over all cells instead,
  for 101 models (`:60`).

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N06 Connect features to nodes by membership | `1.3-...r:76-118`; `4.0-pg-model.py:165-169` | Genes join circuits through a membership matrix matched by lowercased symbol. Unmatched data genes stay as inputs with no edges because the filter is off. |
| N09 Dense head after the graph layers | `nn_design_scripts.py:78-82` | The 2-layer design inserts `Dense(100)` before the softmax. The published 1-layer design has none. |
| N10 One unit per node | `nn_design_scripts.py:69-76` | One ReLU unit per circuit, with its own zero-initialized bias. |
| N17 Map scores to node names | `5.0-...ipynb:1038`; `9.1-pg-deepExplain.py:105` | Activations and LRP tables are labeled by circuit and gene. |
| N18 Aggregate scores over a sample subset | `5.0-...ipynb:1610-1628`; `9.2-...py:105-107` | Counts are summed per predicted cell type, and LRP is averaged per true cell type. |
| N31 Read out node activations | `5.0-...ipynb:1036-1040`; `00_experiment.py:182-190` | Labeled circuit activations are the interpretation and the outlier detector's input. |
| N32 Edge weights labeled by edge | `5.0-...ipynb:2248-2249,2364` | The trained gene × circuit kernel is labeled by name, and the top genes per circuit are taken. |
| N105 Layer-wise relevance propagation | `9.2-...py:82-103` | Epsilon-LRP via DeepExplain, one circuit unit at a time. |
| N109 Prior as initialization only | `nn_design_scripts.py:76`; `4.0-pg-model.py:305-311` | The kernel is zeroed outside the matrix once, and then every entry trains. |
| N110 Attribute a hidden node to inputs | `9.2-...py:92-103` | Genes are scored per cell against one circuit's activation, for each circuit in turn. |
| N111 Per-class frequency of top-k active nodes | `5.0-...ipynb:1594-1628` | Top 15 circuits per cell, counted per cell type; the top 10 are reported. |
| N112 Members of a sink from its upstream subgraph | `1.3-...r:81-106` | A circuit's genes are the genes of every node in its effector subgraph. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Export pathways, drop disease pathways by name | `1.1-...r:22-59`; `1.2-...py:30-61` | ~70 |
| Genes per effector circuit, one table per pathway | `1.3-...r:25-126` | ~100 |
| Entrez to symbol | `1.4-...r:22-60` | ~40 |
| Merge into one gene × circuit matrix, lowercase | `1.6-...py:30-55` | ~25 |
| Align matrix to data gene order (repeated in `00_experiment.py:84-98` and notebooks) | `4.0-pg-model.py:150-175` | ~25 |
| Zero the initial kernel | `nn_design_scripts.py:76,135` | 2 |
| Label activations and kernel by name | `5.0-...ipynb:1036-1040,2248-2252` | ~10 |
| Per-circuit LRP loop with a one-hot selector | `9.2-...py:84-115` | ~30 |

## Fragile spots

- The mask touches only the initial kernel
  (`nn_design_scripts.py:76`), and nothing reapplies it during
  `fit` (`4.0-pg-model.py:305-311`). After training, absent entries
  generally hold nonzero weights.
- `proposed_NN` catches every exception, prints it, and returns
  `None` (`nn_design_scripts.py:45,97-100`). A matrix/data shape
  mismatch surfaces later, on `None.fit`.
- `merge(df_temp, ...)` sits outside the empty-circuit check
  (`1.3-...r:99-106`), so an empty circuit re-merges the previous
  circuit's column. Whether any circuit is empty was not checked.
- `9.1-pg-deepExplain.py:82,92` never resets the one-hot selector,
  so circuit i targets the sum of circuits 0..i. `9.2:98` resets it.
- The kernel's columns are labeled from `circuits_effector.txt`
  order (`5.0-...ipynb:2249`), not from the matrix order. No script
  builds that file.
- `no_of_nodes_in_pathways` (`5.0-...ipynb:1526`) is defined in no
  cell. `scripts/config.py:34` imports `unassigned_with_threshold`,
  which is not in the tree, and every entry point imports `config`.
- Matching relies on lowercasing in two separate scripts. Duplicate
  symbols in the matrix would add rows in the left join.

## Out of scope

- Preprocessing (`2.0-pg-preprocessing-dataset.py`,
  `dataset_scripts.py:63-140`), splits
  (`metrics_and_split_scripts.py`), and training settings: Adam,
  100 epochs, batch 10, early stopping, class weights, Hyperband,
  and one fixed seed 91 for all runs (`4.0-pg-model.py:62-72,196-303`).
- LOF and its threshold (`00_experiment.py:187-235`); imbalance
  simulations (`00_experiment_fake.py`).
- Baselines: an autoencoder that never applies the matrix
  (`autoencoder.py:49-55`), PCA, scibet, retrieval; metrics and
  plotting.
- Pathway-level variant (`1.5-pg-creating-biological-layer.py:41-50`,
  the maximum over circuits per pathway); no run uses it.

## Paper vs code

- Eq. 4 writes the kernel as `W_S = W ⊙ I_S`, and Table 2 lists
  the prior as a `kernel_initializer` ("glorot_uniform +
  sig-informed"). The code applies `I_S` once, to the initial
  kernel.
- The paper wires the circuit layer to "a new dense layer (the
  encoding layer)" before the softmax. The published 1-layer code
  has no such layer; its encoding is the circuit activations
  (`00_experiment.py:71,182-183`). The 2-layer design ("adding a
  second dense hidden layer") adds `Dense(100)`.
- Eq. 5 builds the threshold from quartiles. The code uses the mean
  over cell types of `mean − 1.5·std` of training LOF scores
  (`00_experiment.py:207-208`).
- The paper rescales genes to [−1, 1] after the log. The calls in
  `project_bash.sh:39-70` pass no gene-wise flag.
- 92 pathways in the paper, 93 in a code comment (`1.3-...r:65`).
- LRP (`notebooks/9.x`) is not mentioned in the paper.

## Key excerpts

`scripts/nn_design_scripts.py:68-82`

```python
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(units = unit_size
                                         , input_dim=input_size
                                         , kernel_initializer=init
                                         , bias_initializer='zeros'
                                         , activation=select_activation
                                         , name='layer1'))

            model.set_weights([model.get_weights()[0] * np.array(bio_layer),  np.zeros((unit_size,)) ])

            if (second_layer==True):
                print('second_layer applied!! ')
                model.add(keras.layers.Dense(100, activation=select_activation, name='layer2'))

            model.add(keras.layers.Dense(size_output_layer, activation='softmax', name='layer3'))
```

`notebooks/4.0-pg-model.py:165-169`

```python
        df_bio_with_df = pd.merge(left=pd.DataFrame(df.columns[:-1]).set_index(0)
                                         , right=df_bio
                                         , left_index=True
                                         , right_index=True
                                         , how=how_join).fillna(0.0)
```

`scripts/pathway_layer_data/1.3-pg-gene-from-hipathia.r:85-86`

```r
            genes_circuits = V(pathways$pathigraphs[[list_path_id[all_pathways_]]]$effector.subgraphs[[sub_pathways_]])$genesList
            sub_path_name = names(pathways$pathigraphs[[list_path_id[all_pathways_]]]$effector.subgraphs[sub_pathways_][1])
```

`notebooks/9.2-pg-deepExplain_detail.py:98-103`

```python
                mask_ys = np.zeros((1, T.shape[1]))
        #         filtering the target node in target layer with assigning value as 1
                mask_ys[:, i_circuit] = 1.0
                print(f'      mask sum {pd.DataFrame(mask_ys).sum(axis=1)}')
        #         compute attributions of each genes ('X') for each sample defined as 'xs' with defined mask 'T * mask_ys'
                explain = de.explain(method_name, T * mask_ys, X, xs)
```

## Open questions

- Gene count and gene–circuit pair count: the matrix
  (`data/processed/pbk_circuit_hsa_sig.txt`) is not in the repo,
  and the text I read does not give them.
- The code behind Table A2 (Immune): the closest is `5.0`, stored
  with Melanoma settings and a model name,
  `circuits_signaling_2_1_layer` (`:637`), that matches no run
  script. Whether `model.layers[2]` (`:1515`) is the output layer
  depends on that design.
- The 100 Immune models (50% test) loaded by `5.1` and `9.x` come
  from no call in `project_bash.sh`. There,
  `4.0-pg-model.py:62,67,322` saves one model with a 30% test split.
- The sources of `circuits_effector.txt` and
  `scripts/unassigned_with_threshold.py` were not found.
