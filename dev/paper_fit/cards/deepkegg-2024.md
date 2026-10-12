# deepkegg-2024: DeepKEGG

| Field | Value |
|-------|-------|
| Paper | Lan W, Liao H, Chen Q, Zhu L, Pan Y, Chen YPP, 2024. DeepKEGG: a multi-omics data integration framework with biological insights for cancer recurrence prediction and biomarker discovery. Briefings in Bioinformatics 25(3):bbae185. 10.1093/bib/bbae185 |
| Code | `https://github.com/lanbiolab/DeepKEGG` at `62201cfcfb881861ca5bf6db73d26b215a095581` |
| Other code | none |
| Framework | Python, Keras on TensorFlow in graph mode. The README pins python 3.6.2, keras 2.2.4, tensorflow 1.12.0, but the code calls `tf.compat.v1.disable_eager_execution()` (`kegg-attention_BLCA.ipynb:286-288`) |
| License | none found |
| Full text | PMC11056029 (equations are images there; prose read) |
| Extracted | 2026-10-10, Claude Code, claude-opus-5-5 |

All code is in six notebooks, `kegg-attention_{AML,BLCA,BRCA,LIHC,PRAD,WT}.ipynb`.
Line numbers are lines of the raw `.ipynb` JSON (one source line per file
line). The card cites BLCA. BRCA, LIHC, and PRAD differ only in L2 values,
epochs, and the scoring cells. AML and WT have only mRNA and miRNA inputs.

## Summary

Binary prediction of cancer recurrence from matched SNV, mRNA, and miRNA
data (TCGA BRCA, LIHC, BLCA, PRAD; TARGET AML, WT). Prior knowledge is KEGG
pathway membership of genes and of miRNAs. Each data type has its own
sparse layer from features to one tanh unit per pathway. Then comes its own
self-attention layer, which mixes the samples of a batch. The attention
outputs are concatenated into a dense head with one sigmoid output. Feature
importance is gradient × (input − reference), summed over recurrence
samples and averaged over the five CV folds.

## Prior-knowledge graph

- Source: KEGG human gene sets (`KEGG_pathways/20230205_kegg_hsa.gmt`, 297
  pathways; the paper names R EnrichmentBrowser as the source). KEGG
  miRNA sets from mirPath v3 (`KEGG_pathways/kegg_anano.txt`, 238
  pathways). Both are keyed by KEGG ID such as `hsa00010`
  (`kegg-attention_BLCA.ipynb:30-32`, `:49-56`). Only IDs present in both
  files are kept (`:66`, `:76-83`). That gives 238 pathways, counted from
  the file keys. The code has no cancer-pathway filter. The shipped GMT
  lacks pathways for specific cancer types but keeps general ones such as
  `hsa05200` (Pathways in cancer).
- Size: features (gene symbols for SNV and mRNA; miRNA names) and 238
  pathway nodes. There are no pathway–pathway edges. Edges are unsigned,
  run feature → pathway, and form no cycles. The edge count depends on
  the data columns. Not computed (data not read).
- Data to nodes: data column names are matched to set members exactly
  (`kegg-attention_BLCA.ipynb:228`, `:239`). SNV and mRNA use the gene
  sets and miRNA uses the mirPath sets (`:219`, `:236-242`). A feature in
  no pathway stays an input with an all-zero mask column, so it has no
  edge. A filter that would drop it is commented out
  (`kegg-attention_BRCA.ipynb:234`, `:247`). A pathway with no matched
  feature has an all-zero mask row, so it outputs tanh(bias). Duplicate
  miRNAs are removed with `set` (`kegg-attention_BLCA.ipynb:55`). The
  paper's KEGG matching and chi-square feature selection happen before
  the shipped CSVs and are not in the repository.
- Built at: `kegg-attention_BLCA.ipynb:22-83`, `:208-244`.

## Architecture

- Layers: one graph layer per data type, from features to pathways. There
  is no depth assignment.
- Units per node: one tanh unit with its own bias per pathway per data
  type, so 3 × 238 units (2 × 238 in AML and WT)
  (`kegg-attention_BLCA.ipynb:300`, `:342`, `:478-484`). Data types share
  no pathway units. Each has its own input, mask, and weights.
- Connections that skip layers: none.
- Outputs or losses at inner layers: none.
- Parts the graph does not constrain: per data type, one self-attention
  layer (238 → 64). Then concatenation, `Dense(32, tanh)`, and
  `Dense(1, sigmoid)` (`kegg-attention_BLCA.ipynb:487-496`). The loss is
  binary cross-entropy with balanced class weights (`:503-506`,
  `:622-623`, `:2864`).
- Other structure: attention has one weight tensor (3, 238, 64), with
  `uniform` initialization and no bias (`kegg-attention_BLCA.ipynb:407-411`).
  It computes Q, K, V (batch × 64) and a 64 × 64 matrix
  softmax(Kᵀ Q / 8), summed over the batch, and returns V times that
  matrix (`:416-432`). The scale `64**0.5` is hard-coded (`:426`). One
  sample's output depends on the other samples in its batch: 64 in
  training (`:2864`), Keras's default in `predict` (`:2866`). A
  single-omics ablation model reuses the layers (`:2913-2937`).
- Defined at: `kegg-attention_BLCA.ipynb:298-507`.

## Connectivity mechanism

`Biological_module` gets the transposed mask (features × pathways) as
`mapp` (`kegg-attention_BLCA.ipynb:478-484`). It takes the mask's nonzero
index pairs (`:326-328`) and holds one trainable `kernel_vector` entry per
present edge (`:336-340`). Each forward pass scatters the vector into a
dense kernel with `tf.scatter_nd`, then applies a dense product, bias, and
tanh (`:356-366`). Absent entries are constant zeros of the scattered
tensor. They have no parameter and get no gradient. The dense kernel is
only an intermediate. `glorot_uniform` is applied to the 1-D vector
(`:299`, `:338`). Keras takes both fans from its length, so the scale
depends on the layer's total edge count, not on each unit's fan-in. Biases
start at zero (`:300`). L2 on `kernel_vector` is 0.001 in BLCA (`:339`,
`:478-484`) and 0.01 in WT (`kegg-attention_WT.ipynb:487`, `:490`).

## Interpretation

`get_weigjts(model, p)` (`kegg-attention_BLCA.ipynb:658-678`) feeds all
recurrence-positive samples, from training and test, as one batch
(`:650-653`, `:663-665`). For each data type it computes
`tf.gradients(Y, X) * (x - b)` (`:671`). Y is the sigmoid output, summed
over the batch. X is that data type's input. `b` is meant to be the zero
reference that the paper states, but in the code it is the tuple
`(n_features,)` (see **Fragile spots**). Before evaluating, the function
opens a new session and runs the variable initializer (`:672-674`). Scores
are summed over samples (`:677`), labeled with the data column names
(`:675-676`), and written per fold to
`coef_weight/<cancer>/h<fold>/<omics>.csv` (`:678`). Another cell averages
each name over the five folds and sorts (`:2895-2902`). Because attention
mixes samples, each sample's gradient also flows through the other
samples' outputs. Granularity is input features only. No pathway scores
were found in code. The function is defined only in BLCA and LIHC. Only
LIHC calls it (`kegg-attention_LIHC.ipynb:2744`). BLCA and WT comment the
call out (`kegg-attention_BLCA.ipynb:2872-2873`,
`kegg-attention_WT.ipynb:651`).

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N06 Connect features to nodes by membership | `kegg-attention_BLCA.ipynb:224-244` | Each data type's columns are joined by name to pathways, through gene sets (SNV, mRNA) or miRNA sets; in code, unmatched columns keep no edge and are not removed. |
| N09 Dense head after the graph layers | `kegg-attention_BLCA.ipynb:491-496` | Concatenated attention outputs go through `Dense(32, tanh)` and `Dense(1, sigmoid)`. |
| N10 One unit per node | `kegg-attention_BLCA.ipynb:300`, `:342`, `:478-484` | Each pathway is one tanh unit with its own bias, once per data type. |
| N12 Parameters only for present edges | `kegg-attention_BLCA.ipynb:326-340`, `:356-359` | One trainable entry per mask nonzero, scattered into a dense kernel each forward pass. |
| N13 Regularize edge weights per layer | `kegg-attention_BLCA.ipynb:339`, `:478-484` | L2 on each graph layer's edge vector; the coefficient varies by notebook. |
| N14 Attribution to inputs | `kegg-attention_BLCA.ipynb:667-677` | Per-sample scores for every SNV, mRNA, and miRNA input. |
| N17 Map scores to node names | `kegg-attention_BLCA.ipynb:675-678` | Scores are stored with the data column names, one file per data type. |
| N18 Aggregate scores over a sample subset | `kegg-attention_BLCA.ipynb:650-653`, `:677` | Scores are summed over the recurrence-positive samples. |
| N33 Combine scores across training repeats | `kegg-attention_BLCA.ipynb:2895-2902` | Each feature's score is averaged over the models of the five CV folds. |
| N85 One graph branch per data type | `kegg-attention_BLCA.ipynb:471-491` | SNV, mRNA, and miRNA each get their own input, mask, pathway units, and attention, joined only before the head. |
| N86 Keep nodes shared by all membership tables | `kegg-attention_BLCA.ipynb:66`, `:76-83`, `:208` | Only pathways in both the gene and the miRNA annotation are kept, in one order that every branch uses. |
| N87 Gradient times input | `kegg-attention_BLCA.ipynb:671` | Feature importance is the output gradient times the input's difference from a reference. |
| N88 Attention across samples after the graph layer | `kegg-attention_BLCA.ipynb:415-436` | Each branch's pathway outputs pass through attention whose mixing matrix is summed over the batch. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Read gene and miRNA pathway files into dicts | `kegg-attention_BLCA.ipynb:22-56` | ~25 |
| Keep pathways in both files; restrict both dicts | `kegg-attention_BLCA.ipynb:66-83` | ~10 |
| One pathway × feature 0/1 matrix per data type, by name | `kegg-attention_BLCA.ipynb:208-244` | ~25 |
| Edge weights as a vector, scattered into a kernel | `kegg-attention_BLCA.ipynb:298-389` | ~50 |
| Feature scores by column name, averaged over folds | `kegg-attention_BLCA.ipynb:650-678`, `:2887-2902` | ~35 |

## Fragile spots

- Reference: `b` is the shape tuple `(n_features,)`, not zeros
  (`kegg-attention_BLCA.ipynb:671`). Broadcasting subtracts the data
  type's feature count from every input.
- Scoring weights: a new session runs the variable initializer before the
  gradients are evaluated (`kegg-attention_BLCA.ipynb:672-674`). In graph
  mode a new session holds no trained values, so this may score freshly
  initialized weights. Not run.
- Inputs by position: scoring takes the first three `model.layers` as the
  SNV, mRNA, and miRNA inputs, in that order (`:662-669`).
- Dead inputs: unmatched features stay in, and the filter is commented out
  (`kegg-attention_BRCA.ipynb:234`, `:247`).
- Pathway order comes from `list(set(...))`
  (`kegg-attention_BLCA.ipynb:66`). It can change between Python runs but
  is shared by all branches within one run.
- Batch dependence: test predictions depend on which test samples share a
  predict batch (`kegg-attention_BLCA.ipynb:423-432`, `:2866`).

## Out of scope

- Loss: binary cross-entropy with balanced class weights.
- Training: Adam, learning rate 1e-4, decay 1e-4, batch 64, 130 to 170
  epochs by notebook.
- Data loading: per-omics CSVs aligned by sample index.
- Evaluation: stratified 5-fold CV with sklearn metrics.
- Chi-square feature selection, done outside the repository.

## Paper vs code

- The paper uses DeepLIFT with reference 0. The code computes
  gradient × (input − b), with `b` as above, and has no DeepLIFT rules
  (`kegg-attention_BLCA.ipynb:671`).
- The paper scores pathways (Eq. 12–13): it divides each feature's score
  by the feature's out-degree and sums over the pathway's features. Not
  found in code. `coef_weight/` holds per-feature files only (`ls-files`).
- The paper runs five repeats of 5-fold CV. The code runs one 5-fold split
  with a fixed `random_state` (`kegg-attention_BLCA.ipynb:2843`).
- The paper removes pathways of specific cancer types. The code does not,
  but the shipped GMT already lacks them.

## Key excerpts

`kegg-attention_BLCA.ipynb:224-230`

```python
for i in range(len(mask_list)):
    pathways_genes = np.zeros((len(pathway_union), len(mask_list[i]))) 
    for p  in pathway_union:
        gs = paways_genes_dicts[p]
        g_inds = [mask_list[i].index(g) for g in gs if g in mask_list[i]]
        p_ind = pathway_union.index(p)
        pathways_genes[p_ind, g_inds] = 1
```

`kegg-attention_BLCA.ipynb:336-340`

```python
        self.kernel_vector = self.add_weight(name='kernel_vector',
                                             shape=(nonzero_count,),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             trainable=True)
```

`kegg-attention_BLCA.ipynb:356-359`

```python
        trans = tf.scatter_nd(tf.constant(self.nonzero_ind, tf.int32), self.kernel_vector,
                           tf.constant(list(self.kernel_shape)))
    
        output = K.dot(inputs, trans)
```

`kegg-attention_BLCA.ipynb:416-418`

```python
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
```

`kegg-attention_BLCA.ipynb:423-432`

```python
        QK =  K.dot(K.permute_dimensions(WK,[1,0]),WQ)
    
 
        QK = QK / (64**0.5)
 
        QK = K.softmax(QK)
 
        print("QK.shape",QK.shape)
 
        V = K.dot(WV,QK)
```

`kegg-attention_BLCA.ipynb:487-496`

```python
    atten1 = Self_Attention(64,W_regularizer=l2(0.001))(h0_snv)
    atten2 = Self_Attention(64,W_regularizer=l2(0.001))(h0_mRNA)
    atten3 = Self_Attention(64,W_regularizer=l2(0.001))(h0_miRNA)
    
    feature_tal = tf.keras.layers.concatenate([atten1,atten2,atten3])

    
    h4 = tf.keras.layers.Dense(32,activation='tanh')(feature_tal)
    
    h5 = tf.keras.layers.Dense(1,activation='sigmoid')(h4)
```

`kegg-attention_BLCA.ipynb:671-677`

```python
        ret = [g * (x - b) for g, x, b in zip(tf.gradients(Y, X), [X], [np.zeros((multi_data_inputs[k].shape))[0].shape])]
        sess = tf.compat.v1.InteractiveSession()
        sess.run(tf.compat.v1.initialize_all_variables())
        evaluated_gradients = sess.run(ret,feed_dict=feed_dict)
        gene_pd = pd.DataFrame(columns=['genes','values'])
        gene_pd['genes'] = data_pd_s[k].columns
        gene_pd['values'] = evaluated_gradients[0].sum(axis=0)
```

## Open questions

- Feature and edge counts per dataset, and whether every data column
  matches a pathway. The data CSV headers were not read. The paper's
  Table 1 gives, for example, BRCA with 1000 mRNA, 1000 SNV, and 100 miRNA.
- Which TensorFlow version produced the stored outputs. The README pin and
  the `tf.compat.v1` calls disagree.
- How the paper's pathway scores and `coef_weight/BLCA/` were produced.
  BLCA comments out its scoring call. Not in the notebooks or the README.
