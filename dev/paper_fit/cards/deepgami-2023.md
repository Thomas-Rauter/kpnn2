# deepgami-2023: DeepGAMI

| Field | Value |
|-------|-------|
| Paper | Chandrashekar PB et al., 2023. DeepGAMI: deep biologically guided auxiliary learning for multimodal integration and imputation to improve genotype–phenotype prediction. Genome Medicine 15:88. 10.1186/s13073-023-01248-6 |
| Code | `https://github.com/daifengwanglab/DeepGAMI` at `f3860f9d3016eafea3c0338a2907d0ed17825648` (only branch `main`) |
| Other code | none found |
| Framework | Python, PyTorch (README: 1.4.0 CPU or 1.10.0 GPU; no requirements file), Captum |
| License | MIT (`LICENSE`) |
| Full text | Europe PMC full-text XML of PMC10617196 (abstract, Methods, Fig. 1 caption, ablation section, availability) |
| Extracted | 2026-10-11, Claude Code, claude-opus-5-5 |

## Summary

DeepGAMI classifies samples (schizophrenia vs control, Alzheimer's
scores, cortical layer of mouse neurons) from two data modalities,
e.g. SNP genotypes and gene expression. Each modality passes
through one masked layer whose hidden units are genes. SNPs join
genes by eQTL pairs, and transcription-factor (TF) expression joins
target genes by a gene regulatory network (GRN). Both layers share
one gene list. An auxiliary loss trains a scalar affine map from one
branch's hidden states to the other's, so a missing modality can be
imputed. The branches are concatenated into a dense classifier.
Captum integrated gradients and conductance score features, hidden
genes, and input-to-gene links.

## Prior-knowledge graph

- Source: eQTL pairs (paper: 146,763 GTEx frontal-cortex BA9 eQTLs,
  or cell-type eQTLs) and GRN edges (PsychENCODE GRN, or scGRNom
  cell-type GRNs). Both are read as `source,target` tables, and
  every edge gets weight 1 (`data_preprocess.py:42-56`). The
  Patch-seq run uses no prior (`README.md:144`).
- Size: two bipartite edge sets into one shared hidden gene set.
  In the paper's schizophrenia run there are 2080 SNPs, 126 TFs,
  and 84 hidden genes. Edge counts are not in the main text. The
  edges are directed and unsigned (`data_preprocess.py:44`, `:51`,
  `:62`). The GRN adjacency covers all GRN nodes. Its rows are then
  cut to TF sources and its columns to hidden genes (`:75-77`,
  `:111-114`). A chain S1→S2→T1 thus becomes S1→S2 and S2→T1,
  with S2 both an input and a hidden node, as the paper states.
- Data to nodes: the hidden genes are eQTL targets ∩ GRN targets
  (`data_preprocess.py:99-103`). Both masks are column-sorted by
  name to share one order (`:123-124`). GRN sources are matched to
  expression genes by name and SNP ids to eQTL sources, and
  unmatched features are dropped (`:127-135`). The data are then
  reindexed to the mask row order (`:138-141`). A hidden gene that
  lost all its TFs or SNPs is kept; no pruning was found. The
  paper's t-test/ANOVA feature selection was not found in code.
- Built at: `data_preprocess.py:60-148`. Masks are saved as sparse
  `.npz` (`:173-177`) and loaded dense (`DeepGamiUtils.py:79-83`).

## Architecture

- Layers: one masked layer per modality, input → hidden genes
  (`DeepGamiModel.py:31-62`). A dense `FNN` head follows (widths
  from `--num_fc_neurons`, default `'1500'`), then a linear output
  (`:13-29`, `:88-90`).
- Units per node: one unit per gene per branch, each with its own
  bias (`DeepGamiModel.py:41-44`). The layer has no activation of
  its own. ReLU and dropout follow the merge (`:126`).
- Connections that skip layers: none. The merge adds the branch
  sum to each branch before concatenating:
  `cat[h1 + (h1 + h2), h2 + (h1 + h2)]` (`DeepGamiModel.py:107-108`,
  `:123-125`).
- Outputs or losses at inner layers: the MSE between `h2` and
  `alpha * h1 + beta` (scalars shared by all genes, initialized to
  0 and 0.1; `DeepGamiModel.py:92-99`, `:110`) is added to the
  class loss with weight `corr_reg` (`DeepGamiTrain.py:112-119`;
  default 1 at `:666`). The `'cg'` mode puts the mapped `h1` in
  place of `h2`, and `'cs'` puts `(h2 - alpha) / beta` in place of
  `h1` (`DeepGamiModel.py:111-121`). Validation reports all three
  modes (`DeepGamiTrain.py:153-174`).
- Parts the graph does not constrain: the head, the output layer,
  and `alpha` and `beta`. With `--model_type fully_connect`, two
  `nn.Linear` layers of width `latent_dim` replace the masked layer
  (`DeepGamiModel.py:84-86`, `:102-105`). The paper uses this as
  its no-prior comparison and for Patch-seq.
- Other structure: none.
- Defined at: `DeepGamiModel.py:13-144`.

## Connectivity mechanism

The mask multiplies a dense weight in every forward pass:
`input @ (weight.t() * adj) + bias` (`DeepGamiModel.py:60-61`,
`:66-67`). The full weight is a parameter of shape
`(n_hidden, n_inputs)` per branch (`:41-44`). `adj` has shape
`(n_inputs, n_hidden)`. It is a plain attribute holding a dense
NumPy matrix (`:37`, `:40`; `DeepGamiUtils.py:80-82`), not a
buffer, and no move to a tensor or device was found. Absent entries
are multiplied by 0, so they get no gradient and keep their initial
values. Init is Kaiming-uniform over the full matrix, with biases
uniform in ±1/√fan_in over the full input width
(`DeepGamiModel.py:47-57`). At this commit the training script does
not pass the masks to the model (Fragile spots). For the paper's
"Random" and "Bernoulli" ablations, `bio_dropout` redraws every
entry: Bernoulli(0.5) everywhere, or 0.8 on prior edges and 0.2 on
non-edges (`DeepGamiTrain.py:41-57`, `:624-629`).

## Interpretation

Captum runs on the saved model, which was saved in eval mode
(`DeepGamiTrain.py:161`, `:241`). The whole input file is one batch
(`DeepGamiIG.py:145`), with both modalities present
(`additional_forward_args='None'`).

- Inputs: `IntegratedGradients` (Gauss-Legendre) with no baseline
  passed, so the baseline is all zeros (`DeepGamiIG.py:60-70`). It
  runs once per class for multiclass (`:34-58`). The score is mean
  |IG| over samples, labeled with the input CSV header names
  (`:46-56`, `:68-69`, `:132`).
- Hidden genes: `LayerConductance` on `model.drop_conn`, mean |·|
  over samples. Gene names come from a hard-coded list file
  (`DeepGamiIG.py:73-81`, `:166-173`).
- Links: `NeuronConductance` on `model.ffn` input units
  `0..n_genes-1`, i.e. the `h1 + (h1 + h2)` half
  (`DeepGamiModel.py:123-125`), with
  `attribute_to_neuron_input=True`. Every feature of both
  modalities is scored per gene. The score is |mean over samples|,
  labeled `(id, target)` (`DeepGamiIG.py:84-112`). Non-edge pairs
  are included and should score 0 through the masked weights.
- Paper only: genes from the top 10% of link scores go to Metascape
  enrichment. No code for this was found.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N06 Connect features to nodes by membership | `data_preprocess.py:127-135` | SNPs join genes by eQTL pairs and TFs by GRN edges, matched by name; unmatched features are dropped. |
| N86 Keep nodes shared by all membership tables | `data_preprocess.py:99-103`, `:123-124` | Hidden genes = eQTL ∩ GRN targets, sorted so both masks share one column order. |
| N96 Own feature list and membership table per data type | `data_preprocess.py:107-111`; `DeepGamiModel.py:33-44` | SNPs with eQTLs and TFs with a GRN each join the same gene set through their own table and input. |
| N85 One graph branch per data type | `DeepGamiModel.py:64-68`, `:108`, `:123-125` | Each modality has its own copy of the gene layer, merged after the layer by sum and concatenation. |
| N10 One unit per node | `DeepGamiModel.py:41-44`, `:60-61` | Each gene is one unit with its own bias per branch. |
| N22 Mask a dense weight | `DeepGamiModel.py:60-61` | The full weight is multiplied by a fixed 0/1 mask in every forward pass. |
| N09 Dense head after the graph layers | `DeepGamiModel.py:13-29`, `:88-90` | The concatenated gene units feed an FNN and a linear output. |
| N35 Randomized prior as a control | `DeepGamiTrain.py:41-57`, `:624-629` | Random (p = 0.5) and noisy (0.8 / 0.2) masks replace the prior in an ablation. |
| N14 Attribution to inputs | `DeepGamiIG.py:60-70` | SNPs, TFs, and other features are scored by IG per sample. |
| N16 Integrated gradients from a zero baseline | `DeepGamiIG.py:62-65` | Captum IG is called with no baseline, so the baseline is zeros. |
| N15 Attribution to hidden nodes | `DeepGamiIG.py:73-81` | Layer conductance scores each hidden gene. |
| N17 Map scores to node names | `DeepGamiIG.py:68-69`, `:79`, `:96-102` | Scores are labeled by feature, gene, and (feature, gene). |
| N18 Aggregate scores over a sample subset | `DeepGamiIG.py:46-47`, `:68-69`, `:79` | Mean absolute score over the samples given (test samples in the paper). |
| N113 Predict one branch's node states from another's | `DeepGamiModel.py:92-99`, `:110-121`; `DeepGamiTrain.py:112-119`; `DeepGamiTest_SM.py:82-96` | `alpha * h1 + beta` is fit to `h2` by MSE and used in place of `h2` when modality 2 is missing. |
| N114 One input-to-node layer from a directed graph | `data_preprocess.py:75-77`, `:111-114` | GRN sources become inputs and targets become genes; a TF that is also a target sits on both sides. |
| N115 Dense twin of the graph layer | `DeepGamiModel.py:84-86`, `:102-105`; `README.md:144` | `fully_connect` swaps each masked layer for `nn.Linear`: the no-prior baseline and the Patch-seq model. |
| N116 Edge scores from attribution through a node | `DeepGamiIG.py:84-112` | Neuron conductance through each gene's unit gives one score per (feature, gene) link. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Read eQTL and GRN tables, weight 1 | `data_preprocess.py:42-56` | ~15 |
| Edge table → source × target adjacency (networkx) | `data_preprocess.py:60-81` | ~22 |
| Gene set = eQTL ∩ GRN targets; filter tables | `data_preprocess.py:99-103` | ~5 |
| Cut GRN rows to TFs; align and sort mask columns | `data_preprocess.py:111-124` | ~14 |
| Match TFs and SNPs to data; reindex data to masks | `data_preprocess.py:126-141` | ~15 |
| Write name lists; save masks as `.npz` | `data_preprocess.py:151-177` | ~25 |
| Load masks; layer sizes from mask shape | `DeepGamiUtils.py:79-83`; `DeepGamiTrain.py:619-622` | ~8 |
| Randomized masks for the ablation | `DeepGamiTrain.py:41-57`, `:624-629` | ~23 |
| Masked forward per branch | `DeepGamiModel.py:59-68` | ~10 |
| Scores labeled by feature, gene, (feature, gene) | `DeepGamiIG.py:34-112` | ~80 |

## Fragile spots

- The masks never reach the model. With `drop_connect`, `adj1` and
  `adj2` stay local (`DeepGamiTrain.py:622`). `args.adj1` and
  `args.adj2` are set only for `fully_connect` (`:618`), but the
  model reads them (`:348-349`, `:501-502`).
- `DeepGami.__init__` accepts only `'drop_connect'` and
  `'fully_connect'` (`DeepGamiModel.py:82-86`). The ablation types
  (`DeepGamiTrain.py:624-629`) build neither layer.
- Several scripts use names that are not defined:
  `data_preprocess.py:30`, `:37`, `:184`, `:202-203`, and `:209`;
  `DeepGamiIG.py:122`, `:166` (not in its parser at `:185-200`) and
  `:177` (`target_names` is set only in the `layer` branch). Both
  test scripts read `model.fcn1`, which exists only for
  `fully_connect` (`DeepGamiTest_SM.py:76`,
  `DeepGamiTest_DM.py:86-97`).
- The eQTL adjacency's row and column order relies on networkx node
  insertion order matching the stacked, de-duplicated names
  (`data_preprocess.py:62-72`).
- The two masks' gene columns are aligned only by sorting
  (`data_preprocess.py:123-124`), with no check that the sets are
  equal. `latent_dim` comes from mask 1 alone
  (`DeepGamiTrain.py:620`).
- Input order sets the imputation direction: branch 2 is always
  predicted from branch 1 (`DeepGamiModel.py:110`). The training
  code calls input 1 `snps` (`DeepGamiTrain.py:317-318`), while
  `data_preprocess.py` writes expression and the GRN as modality 1
  (`:169-177`).
- In `'cg'` mode the sum `drop_conn(inp1, inp2)` still reads the
  real second input (`DeepGamiModel.py:107`), so the
  "single-modality" validation sees modality 2
  (`DeepGamiTrain.py:168-174`). The single-modality test feeds
  standardized ones (`DeepGamiTest_SM.py:82-87`).
- `'cs'` uses `(h2 - alpha) / beta`, which is not the inverse of
  `alpha * h1 + beta` (`DeepGamiModel.py:111`).
- `LayerConductance` hooks the forward output of `drop_conn`, which
  is the sum at `DeepGamiModel.py:107`. The branch outputs at
  `:108` come from a plain method call, so gene scores may follow
  only the summed path (not run).
- Normalization differs between training and attribution. Training
  defaults to `standard` for input 1 and `log` for input 2
  (`DeepGamiTrain.py:684`). Attribution applies log to input 1 and
  scaling to input 2 (`DeepGamiIG.py:128-129`).

## Out of scope

- Loss: BCE on a sigmoid output or cross-entropy
  (`DeepGamiTrain.py:288-293`), plus the auxiliary MSE.
- Training loop: Adam; early stopping on validation balanced
  accuracy; the best model is saved whole (`DeepGamiTrain.py:139-283`).
- Data loading, normalization, and SMOTE
  (`DeepGamiUtils.py:16-93`). Splits use a random seed drawn and
  printed per run (`DeepGamiTrain.py:297-298`, `:470-473`).
- `--out_reg` (README: L2) is passed as `l1_reg`, and the penalty is
  commented out (`DeepGamiTrain.py:114-115`).
- Metrics, grid search, and Metascape enrichment.

## Paper vs code

- Eqs. 1–2 apply σ inside the masked layer. The code layer is
  linear, with ReLU after the merge (`DeepGamiModel.py:60-62`,
  `:126`).
- The paper concatenates the two latent spaces. The code
  concatenates `h1 + (h1 + h2)` and `h2 + (h1 + h2)` (`:123-125`).
- The paper selects features by t-test or ANOVA and uses
  StandardScaler. No feature selection was found, and the training
  default is `standard,log` (`DeepGamiTrain.py:684`).
- The paper says self-loops are not encoded. No self-loop filter
  was found (`data_preprocess.py:75-77`, `:113`).
- The README gives `corr_reg` a default of 0.5. The code default is
  1 (`DeepGamiTrain.py:666`).
- The paper's "random networks" are, in code, independent p = 0.5
  draws per entry, not matched to the prior's edge count
  (`DeepGamiTrain.py:48-50`).

## Key excerpts

`data_preprocess.py:99-103`

```python
    comm_genes = list(set(eqtl_genes).intersection(set(grn_tgs)))

    # Filter eqtls and grn target genes based on the common genes
    eqtl = eqtl.loc[eqtl.target.isin(comm_genes), :].reset_index(drop=True)
    grn = grn.loc[grn.target.isin(comm_genes), :].reset_index(drop=True)
```

`DeepGamiModel.py:59-62`

```python
    def forward(self, input1, input2):
        out1 = input1.matmul(self.weight1.t() * self.adj1) + self.bias1
        out2 = input2.matmul(self.weight2.t() * self.adj2) + self.bias2
        return torch.add(out1, out2)
```

`DeepGamiModel.py:107-126`

```python
            ls_comb_mod = self.drop_conn(inp1, inp2)
            ls_mod1, ls_mod2 = self.drop_conn.get_indiv_outputs(inp1, inp2)

        ls_mod2_est = (self.alpha * ls_mod1) + self.beta
        ls_mod1_est = (ls_mod2 - self.alpha)/self.beta

        ls_mod1_out, ls_mod2_out = 0, 0
        if estimate == 'None':
            ls_mod1_out, ls_mod2_out = ls_mod1, ls_mod2
        elif estimate == 'cg':
            ls_mod1_out = ls_mod1
            ls_mod2_out = ls_mod2_est
        elif estimate == 'cs':
            ls_mod1_out = ls_mod1_est
            ls_mod2_out = ls_mod2

        ls_mod1_out = torch.add(ls_mod1_out, ls_comb_mod)
        ls_mod2_out = torch.add(ls_mod2_out, ls_comb_mod)
        ls_out = torch.cat([ls_mod1_out, ls_mod2_out], dim=1)
        ls_out = self.dropout(ls_out.relu())
```

`DeepGamiIG.py:87-94`

```python
    neuron_cond = NeuronConductance(model, model.ffn)
    x1_all_neur_imp = pd.DataFrame()
    x2_all_neur_imp = pd.DataFrame()

    print('Interpreting eQTL and GRN connections importance...')
    for idx in list(range(0, len(tg_names))):
        nc_vals = neuron_cond.attribute((inp1, inp2), additional_forward_args='None',
                                        neuron_selector=idx, attribute_to_neuron_input=True)
```

## Open questions

- Whether the paper's results came from code other than this
  commit, since the `drop_connect` path does not pass the masks.
  The repository has one branch, and its history was not fetched
  (depth 1).
- Whether `tensor * numpy.matrix` works as written, on CPU or GPU
  (`DeepGamiModel.py:60`). The code was not run.
- Filtered eQTL and GRN edge counts per dataset. They may be in
  Additional file 2 (Table S2), which was not read.
- The number of IG steps is Captum's default (`DeepGamiIG.py:40`,
  `:63`). The Captum version is not pinned.
- Which samples go into `DeepGamiIG.py` is up to the caller (the
  paper says test samples).
