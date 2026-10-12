# deephiscom-2022: DeepHisCoM

| Field | Value |
|-------|-------|
| Paper | Park C, Kim B, Park T, 2022. DeepHisCoM: deep learning pathway analysis using hierarchical structural component models. Briefings in Bioinformatics 23(5):bbac171. 10.1093/bib/bbac171 |
| Code | `https://github.com/chanwoo-park-official/DeepHisCoM` at `261f59c34a10db87857daf043ba375b49e05936a` |
| Other code | `https://github.com/GooTec/DeepHisCoM_simulation` (simulation scripts, not read); `https://github.com/Chan-Hee/DeepHisCoM-MO` (later multi-omics variant, not read). The README says the paper used "some modified version of this code" (`README.md:6`). |
| Framework | Python, PyTorch; no version pinned (no requirements file) |
| License | none found |
| Full text | Publisher page at academic.oup.com (open access): abstract, Methods, data sections, Figure 1 caption |
| Extracted | 2026-10-11, Claude Code, claude-opus-5-5 |

## Summary

DeepHisCoM tests which pathways are associated with a binary
phenotype (hepatocellular carcinoma vs liver cirrhosis from
metabolomic or metagenomic data; COVID-19 hospitalization from
SNPs). Each pathway gets its own small fully connected network
that reads only that pathway's biomarkers and emits one value per
sample. The pathway values are batch-normalized, rescaled, and fed
with optional covariates into a single logistic output unit. The
output-layer weight on each pathway is the pathway's coefficient
β. The model is refit on permuted phenotypes to get a null
distribution of β for empirical p-values. The released code is a
single script (`deephiscom.py`) that trains one real run plus the
permutation runs and saves the β vector of each.

## Prior-knowledge graph

- Source: KEGG pathway membership (metabolomic and metagenomic
  data; metagenomic genes via PICRUSt); 186 KEGG pathways from
  MSigDB for the SNP data, with SNPs mapped to genes by SnpEff and
  only SNPs of a unique gene kept (paper). The code reads none of
  these; it reads a prepared `annot.csv` whose `group` column
  names the pathway of each input column (`deephiscom.py:235-236`).
- Size: two node types, biomarkers (inputs) and pathways, plus
  one output unit. Paper data: 32 metabolites in 23 pathways;
  752 genes in 23 pathways; 488 SNPs from 18 genes in 38
  pathways. Directed (biomarker → pathway → phenotype), unsigned,
  acyclic, no pathway–pathway edges ("not allowing any nodes
  between pathways", paper).
- Data to nodes: by position, not name. `nvar[i]` counts the
  `annot.csv` rows of the i-th distinct `group`, in order of first
  appearance (`deephiscom.py:236-248`); the model cuts the input
  columns into consecutive blocks of those sizes
  (`deephiscom.py:187-195`). `train_data.csv` must already hold one
  column per (biomarker, pathway) pair, grouped by pathway in that
  order. No name is checked. Dropping unmatched features and
  one-member pathways (paper) happens outside the repository.
- Built at: `deephiscom.py:235-248` (membership counts),
  `deephiscom.py:95-97` (per-pathway sizes).

## Architecture

- Layers: input columns → one private subnetwork per pathway →
  one value per pathway → output unit. Pathways sit side by side,
  with no layering among them.
- Units per node: varying. `layerinfo.csv` gives each pathway a
  hidden width `node_num` and depth `layer_num`, one row per
  pathway in pathway order (`deephiscom.py:95-97`,
  `deephiscom.py:177`). Width 0 gives `Linear(n_members, 1)`
  (`deephiscom.py:145-150`); otherwise `Linear(n_members, width)`,
  `layer_num - 1` × `Linear(width, width)`, `Linear(width, 1)`
  (`deephiscom.py:151-163`). Every linear layer is bias-free and
  followed by the activation (tanh, ReLU, LeakyReLU 0.2 by
  default, or identity) and dropout (`deephiscom.py:74`,
  `deephiscom.py:112-119`). The paper uses one hidden layer
  (DeepHisCoM) or two (Deep2HisCoM), widths tuned by
  cross-validation.
- Connections that skip layers: none.
- Outputs or losses at inner layers: none.
- Parts the graph does not constrain: the output layer
  `Linear(n_pathways + n_covariates, 1)` with sigmoid
  (`deephiscom.py:181`, `deephiscom.py:200-201`). Covariates are
  the input columns after the last pathway block, counted from
  `cov.csv` column `x` (`deephiscom.py:239-243`), and are
  concatenated unchanged to the pathway values
  (`deephiscom.py:194`, `deephiscom.py:198`).
- Other structure: on the pathway values, `BatchNorm1d` over
  pathways (`deephiscom.py:178`, `deephiscom.py:196`), then
  division of the whole batch × pathway tensor by its one L2
  (Frobenius) norm (`deephiscom.py:197`), then dropout on
  pathways and covariates together (`deephiscom.py:199`).
- Defined at: `deephiscom.py:139-202`.

## Connectivity mechanism

One small `nn.Sequential` per pathway in an `nn.ModuleList`
(`deephiscom.py:177`), each applied to a contiguous slice of the
input columns whose bounds are cumulative member counts,
recomputed every forward pass (`deephiscom.py:187-195`). No full
biomarker × pathway weight and no mask exist; a non-member has no
parameter in that pathway's subnetwork and gets no gradient from
it. All subnetwork weights start at the constant 0.01
(`deephiscom.py:121-123`, `deephiscom.py:164`), so the hidden
units of one subnetwork start identical and only dropout differs
between them. Output weights start at 0, output bias at 0.001
(`deephiscom.py:182-183`).

## Interpretation

A pathway's score is its trained output weight β, read from the
output layer's weight row, which also holds the covariate
coefficients after the pathway entries (`deephiscom.py:324`,
`deephiscom.py:352`). It is taken at the best epoch by training or
held-out loss or AUC (`stop_type` 1–4, `deephiscom.py:347-358`)
or after a fixed epoch count (`stop_type` 5,
`deephiscom.py:359-361`). Run 0 uses the real phenotype, runs 1 to
`perm - 1` a permuted one (`deephiscom.py:217-223`). Each run saves
its β vector, unlabeled, to `tmp/param<run>.txt`
(`deephiscom.py:363-364`); the order is the first-appearance
order of `group` (`deephiscom.py:236`). Empirical p-values and
Benjamini–Hochberg correction (paper) are not in the code.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N06 Connect features to nodes by membership | `deephiscom.py:235-248`, `deephiscom.py:195` | Biomarkers join pathways through a membership table (`annot.csv` `group`), but matched by column position with contiguous blocks, not by name; a biomarker in several pathways must be duplicated as columns upstream. |
| N11 Standard layers between graph layers | `deephiscom.py:149`, `deephiscom.py:155-162`, `deephiscom.py:196`, `deephiscom.py:199` | Dropout after every layer inside each pathway subnetwork, and batch normalization plus dropout on the pathway layer before the output. |
| N12 Parameters only for present edges | `deephiscom.py:145-163`, `deephiscom.py:195` | Each pathway's weights see only its member columns; non-members have no parameter (with width 0, exactly one weight per member edge). |
| N13 Regularize edge weights per layer | `deephiscom.py:296-311` | Separate L1 or L2 penalty constants for all pathway-subnetwork weights and for the output-layer weights and bias. |
| N17 Map scores to node names | `deephiscom.py:236`, `deephiscom.py:364` | β must be read as one score per named pathway; the code fixes the order from `annot.csv` but saves unlabeled numbers. |
| N54 Private subnetwork per node | `deephiscom.py:139-167`, `deephiscom.py:177` | Each pathway is computed by its own fully connected network from its members and emits one value. |
| N58 Gather each node's inputs by index | `deephiscom.py:187-195` | Each pathway's inputs are selected by start/end offsets from cumulative member counts and passed only to that pathway's subnetwork. |
| N100 Caller-set seed per training run | `deephiscom.py:20`, `deephiscom.py:220`, `deephiscom.py:232` | `--seed` fixes dropout, shuffling, and the train/test split for every run; the run index seeds the label permutation. |
| N101 Null scores from shuffled-label runs | `deephiscom.py:217-223`, `deephiscom.py:363-364` | The same network is refit on permuted phenotypes, saving β per run as the null for the real-label β. |
| N117 Subnetwork width and depth per node from a table | `deephiscom.py:95-97`, `deephiscom.py:145-163`, `deephiscom.py:177` | Each pathway's hidden width and number of hidden layers come from its row in `layerinfo.csv`, including width 0 for a single linear unit. |
| N118 Covariates straight into the output layer | `deephiscom.py:181`, `deephiscom.py:194`, `deephiscom.py:198`, `deephiscom.py:239-243` | Covariates (e.g. sex, age, BMI, PCs) are concatenated unchanged to the pathway values and get their own output coefficients. |
| N119 Rescale node values before a linear readout | `deephiscom.py:196-197` | Pathway values are batch-normalized and divided by their joint L2 norm so the output coefficients β are on one scale (the paper's Σ f² = 1 constraint). |
| N120 Output weight as node score | `deephiscom.py:324`, `deephiscom.py:352`, `deephiscom.py:364` | Each pathway is scored by its trained weight into the single logistic output unit. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Count members per pathway from the membership table, in first-appearance order | `deephiscom.py:235-248` | ~8 |
| Read per-pathway hidden width and depth | `deephiscom.py:95-97` | ~3 |
| Build one subnetwork per pathway | `deephiscom.py:139-167`, `deephiscom.py:177` | ~25 |
| Cumulative offsets and slicing of input columns per pathway | `deephiscom.py:187-195` | ~9 |
| Covariate count and concatenation | `deephiscom.py:239-243`, `deephiscom.py:198` | ~6 |
| Normalize the pathway layer | `deephiscom.py:196-197` | ~2 |
| Save β per run, unlabeled | `deephiscom.py:324`, `deephiscom.py:352`, `deephiscom.py:363-364` | ~4 |

## Fragile spots

- Alignment is positional only; nothing compares `annot.csv` to
  `train_data.csv` column names (`deephiscom.py:235-248`). If one
  group's rows are not consecutive, the counts stay right but the
  slices at `deephiscom.py:195` mix pathways.
- Covariates are found by count, not name, right after the last
  pathway block; the phenotype must be the last column
  (`deephiscom.py:128-129`, `deephiscom.py:194`,
  `deephiscom.py:240-241`).
- The norm at `deephiscom.py:197` spans the whole batch, so each
  sample's pathway values depend on the other samples in the
  batch, also at evaluation (`deephiscom.py:328-333`).
- Saved β vectors carry no pathway names (`deephiscom.py:364`).

## Out of scope

- Loss: BCE or MSE (`deephiscom.py:275-280`); the default
  `--loss BCELoss` (`deephiscom.py:34`) matches neither lowercase
  string compared there.
- Training loop: Adam, up to 10000 epochs, patience `count_lim`
  (`deephiscom.py:272-273`, `deephiscom.py:284-361`); the default
  `--stop_type 0` (`deephiscom.py:59`) never sets `param_save`.
- Data loading and per-column standardization of all inputs and
  covariates (`deephiscom.py:215-233`).
- Cross-validation (paper), p-values, FDR, plotting: not in the
  repository (`README.md:6`).

## Paper vs code

- The paper's Σ_k f_ik² = 1 (per pathway, over subjects) is coded
  as `BatchNorm1d` then division by one norm over all pathways and
  samples (`deephiscom.py:196-197`).
- Paper: ridge on network weights and on β. Code: L1 by default
  (`deephiscom.py:37`), both constants 0 (`deephiscom.py:39-42`),
  and the output penalty also covers the bias
  (`deephiscom.py:297`).
- The Methods equations have no covariates, no batch
  normalization, and no activation or dropout after each pathway's
  output unit (`deephiscom.py:160-162`); the code has all three.
- The README says the paper used a modified version of this code
  (`README.md:6`).

## Key excerpts

`deephiscom.py:152-164`

```python
            modules = []
            modules.append(nn.Linear(input_dim, hidden_dim, bias = False))
            modules.append(act_fn)
            modules.append(dropout_fn)
            for i in range(layer_num-1):
                modules.append(nn.Linear(hidden_dim, hidden_dim, bias = False))
                modules.append(act_fn)
                modules.append(dropout_fn)
            modules.append(nn.Linear(hidden_dim, 1, bias = False))
            modules.append(act_fn)
            modules.append(dropout_fn)
            self.block = nn.Sequential(*modules)
        self.block.apply(init_weights)
```

`deephiscom.py:186-202`

```python
    def forward(self, x):
        kk=0
        nvarlist = list()
        nvarlist.append(kk)
        for i in range(len(self.nvar)):
            k=kk
            kk=kk+self.nvar[i]
            nvarlist.append(kk)
        nvarlist.append(kk + self.covariate)
        pathway_layer = torch.cat([self.pathway_nn[i](x[:,nvarlist[i]:nvarlist[i+1]]) for i in range(len(self.nvar))],1)
        pathway_layer = self.bn_path(pathway_layer)
        pathway_layer = pathway_layer/(torch.norm(pathway_layer,2))
        x = torch.cat([pathway_layer, x[:, nvarlist[len(self.nvar)]:nvarlist[len(self.nvar) + 1]]], 1)
        x = self.dropout_path(x)
        x = self.fc_path_disease(x)
        x = torch.sigmoid(x)
        return(x)
```

`deephiscom.py:247-248`

```python
    for i in range(0,grouplen):  
        nvar.append(sum(annot["group"]==groupunique[i]))
```

`deephiscom.py:220-223`

```python
    torch.manual_seed(permutation); random.seed(permutation); np.random.seed(permutation)              # for permutation random seed
    train = train_base
    if permutation != 0:
        train['phenotype'] = np.random.permutation(train['phenotype'])                              # permute y-data
```

## Open questions

- How biomarkers in several pathways were laid out as columns,
  and how unmatched biomarkers and small pathways were dropped:
  the data preparation is not in the repository (looked at
  `ls-files`: two files only).
- Which test statistic turns β into p-values (β or |β|, one- or
  two-sided): neither the code nor the paper's Methods say.
- Which modifications the paper's runs had over this script
  (`README.md:6`); the two other repositories were not read.
- Hidden widths actually used per pathway: not reported in the
  paper; `layerinfo.csv` is not in the repository.
