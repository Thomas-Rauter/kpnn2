# deepomix-2021: DeepOmix

| Field | Value |
|-------|-------|
| Paper | Zhao L, Dong Q, et al., 2021. DeepOmix: A scalable and interpretable multi-omics deep learning framework and application in cancer survival analysis. Computational and Structural Biotechnology Journal 19:2719–2725. 10.1016/j.csbj.2021.04.067 |
| Code | `https://github.com/angelazhaolianhe/DeepOmix` at `e6dd44005a3c06d2737b543ba4d4a8b803d3a8f0` |
| Other code | The paper links `https://github.com/CancerProfiling/DeepOmix`, which GitHub reports as not found (2026-10-16); the account `CancerProfiling` no longer exists. The repository used here belongs to the first author (README contact `zhaolianhe@ict.ac.cn`), was created 2025-07-24, and its README still clones the old URL. `mrhan36/DeepOmix` is empty. `angelazhaolianhe/DeepOmix-ICInet` (a later model) and `jjyjoy1/ML_Data_Analysis` `DeepOmix_model/` (third party) were not read. The model, sparse-coding, and loss files match `DataX-JieHao/Cox-PASNet` `coxpasnet/Model.py` and `coxpasnet/SubNetwork_SparseCoding.py` at `cbed3ef` apart from removed comments and the class name (paper row `cox-pasnet-2018`). |
| Framework | Python 3.6, PyTorch 1.0.0 (`install_package.sh:3`), scipy, pandas, scikit-survival |
| License | GPL-3.0 (`LICENSE`) |
| Full text | Europe PMC full-text XML of PMC8131983 |
| Extracted | 2026-10-16, Claude Code, claude-opus-5-5 |

## Summary

DeepOmix predicts a Cox proportional-hazards risk score for cancer
patients from gene-level omics data (mutation, copy number,
expression, methylation in the paper; expression alone or expression
plus methylation in the repository demo). Prior knowledge is a set of
KEGG and Reactome gene sets from MSigDB, given as a pathway-by-gene
0/1 table. The first layer joins every (gene, omics type) input to
the pathways that contain the gene, through a masked dense weight;
each pathway is one tanh unit. Two unconstrained tanh layers and a
linear Cox output follow. Training drops whole pathway units per
epoch and prunes the dense layers by magnitude each epoch ("sparse
coding").

## Prior-knowledge graph

- Source: KEGG and Reactome gene sets from MSigDB (paper, Data
  section). Demo table `Data/pathway_module.csv` has 860 rows
  (186 `KEGG_*`, 674 `REACTOME_*`) and 5,534 gene columns with
  values 0/1; set sizes within those columns range from 2 to 713
  (computed from the table). The paper's 20–200 gene filter is not
  in the code. `TransferData.py:14` keeps a random 100 of the
  pathways (`random_state=2495`); the full table is a commented
  alternative (`TransferData.py:15`).
- Size: one bipartite layer, pathways × input features. Demo runs
  use 100 pathways with 2,708 inputs (expression only) or 5,075
  inputs (expression plus methylation) (`RunSingle.py:14-15`,
  `RunMultiple.py:16-17`). Unsigned, undirected in the table,
  used as gene → pathway; no pathway-to-pathway edges.
- Data to nodes: genes are matched by column name between the
  pathway table and each omics table, keeping genes that are in at
  least one selected pathway and have no missing values
  (`TransferData.py:17`, `TransferData.py:31-34`). Each omics type
  adds its own copy of every matched gene, renamed
  `<gene>_<omics>`, and the gene's pathway column is copied with it
  (`TransferData.py:35-41`). A gene missing from one omics table
  has no input for that type. Genes in no selected pathway are
  dropped. A pathway with no matched gene stays as a unit with only
  its bias (no code removes it).
- Built at: `TransferData.py:7-59`; loaded as a dense float tensor
  at `DataLoader.py:30-35`.

## Architecture

- Layers: input → pathway layer (tanh) → dense hidden layer of 80
  units (tanh) → dense layer of 30 units (tanh, no bias) → one
  linear output unit (no bias) (`DeepOmix.py:9-12`,
  `DeepOmix.py:21-31`; sizes at `RunSingle.py:14-17`). Only the
  first layer is graph-constrained.
- Units per node: one per pathway, with its own bias
  (`DeepOmix.py:9`, `DeepOmix.py:23`).
- Connections that skip layers: none.
- Outputs or losses at inner layers: none.
- Parts the graph does not constrain: the 80- and 30-unit dense
  layers and the output layer. The output layer reads the 30 units
  concatenated with a second input `x_2` (`DeepOmix.py:30-31`):
  the event indicator in the main scripts (`Train.py:29`,
  `Train.py:95`) and an `age` column in `code_with_clinical/`
  (`TransferData.py:97-98`, `code_with_clinical/TrainWithClinical.py:29`).
  After every optimizer step, the 80- and 30-unit weight matrices
  are pruned by magnitude: among weights whose gradient was nonzero
  in that step, sparsity levels 1–100 % are tried, the training loss
  is interpolated with a cubic spline, and the smallest weights are
  zeroed at the best level (`Train.py:36-87`,
  `SubNetwork.py:12-21`). The first layer is skipped
  (`Train.py:52-53`). No mask keeps pruned weights at zero in later
  steps.
- Other structure: per-epoch unit dropout on the pathway layer
  (rate 0.7) and the 80-unit layer (rate 0.5): one 0/1 vector per
  layer is drawn each epoch and multiplied into every sample's
  activations, with no rescaling; evaluation keeps all units
  (`SubNetwork.py:5-10`, `Train.py:26-27`, `DeepOmix.py:24-28`,
  `RunSingle.py:23`). Training is full-batch: one step per epoch on
  the whole training set (`Train.py:23-32`).
- Defined at: `DeepOmix.py:4-33`.

## Connectivity mechanism

The first layer is a full `nn.Linear(In_Nodes, Pathway_Nodes)`
whose weight (pathways × inputs) holds every gene–pathway pair
(`DeepOmix.py:9`). At the start of every forward call, the weight's
`.data` is replaced by its elementwise product with the 0/1 mask
(`DeepOmix.py:22`), and the same assignment runs again right after
`opt.step()` (`Train.py:34`). Because the product is written to
`.data`, the mask is not part of the autograd graph: absent entries
are zero during the forward pass but still receive the gradient
`δ_pathway · x_input`, and Adam (with `weight_decay`) updates them
(`Train.py:22`, `Train.py:31-32`); line 34 sets them back to zero.
Adam's moment estimates for absent entries are kept. Initialization
is PyTorch's default for `nn.Linear` over the full input width; the
mask is first applied in the first forward call, so a freshly built
network holds nonzero absent entries until then. The output weight
is drawn from U(−0.001, 0.001) (`DeepOmix.py:13`).

## Interpretation

The paper splits samples at the median of the output into high- and
low-risk groups, runs a Kolmogorov–Smirnov test per pathway-layer
unit between the groups with Bonferroni correction, and shows an MDS
of the second hidden layer (paper, Functional analysis; Fig. 1D).
Not found in the code: no match for `ks`, `kolmog`, `bonferr`, or
`mds`, and `forward` returns only the prediction
(`DeepOmix.py:33`), so no code returns pathway-layer activations.

`Visualization.py` (added 2025-07-24) defines
`extract_pathway_embeddings`, which returns the rows of the masked
first-layer weight as one vector per pathway
(`Visualization.py:17-31`), but `main()` never calls it. It scores
random vectors that its comments call placeholders with an ANOVA
F-test between event samples split at median survival time, and
labels the top 10 by the mask CSV's row names
(`Visualization.py:55-86`, `Visualization.py:171-188`).

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N06 Connect features to nodes by membership | `TransferData.py:17`, `TransferData.py:32-34` | Omics columns join pathways by gene name through the pathway-by-gene table; genes in no selected pathway are dropped. |
| N07 One input per entity and data type | `TransferData.py:29-42` | Each omics type adds a `<gene>_<omics>` copy of every matched gene, with that gene's pathway column copied into the mask. |
| N09 Dense head after the graph layers | `DeepOmix.py:10-12`, `DeepOmix.py:26-31` | Two dense tanh layers and a linear Cox output follow the pathway layer. |
| N10 One unit per node | `DeepOmix.py:9`, `DeepOmix.py:23` | Each pathway is one tanh unit with its own bias. |
| N53 Re-zero masked weights before each step | `DeepOmix.py:22`, `Train.py:34` | The mask is multiplied into the weight's `.data` in each forward call and after each optimizer step; absent entries get gradients and Adam updates in between. |
| N146 One unit-dropout draw per step for all samples | `SubNetwork.py:5-10`, `Train.py:26-27`, `DeepOmix.py:24-25` | Each epoch drops 70 % of the pathway units for every sample at once, without rescaling, so each step trains one sub-network of whole pathways. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Pick pathways and the genes they cover | `TransferData.py:13-17` | ~5 |
| Match each omics table to pathway genes, copy the mask columns per omics type, rename features | `TransferData.py:29-42` | ~14 |
| Write mask and data as CSVs whose column orders agree | `TransferData.py:49-58` | ~10 |
| Load mask as a dense tensor; drop label columns from data | `DataLoader.py:30-35`, `DataLoader.py:37-43` | ~13 |
| Apply mask in forward and after each optimizer step | `DeepOmix.py:22`, `Train.py:34` | 2 |
| Hard-code layer widths to the mask's shape | `RunSingle.py:14-17`, `RunMultiple.py:16-19` | ~8 |
| Read pathway names from the mask CSV for labels | `Visualization.py:171-173` | 3 |

## Fragile spots

- Inputs and mask columns are aligned by position only:
  `DataLoader.py:31` and `DataLoader.py:40` take `.values` with no
  name check. They agree because `TransferData.py:32-42` builds both
  from the same per-omics gene list, which comes from a Python `set`
  and so has no fixed order across runs.
- Absent first-layer entries hold nonzero values between
  `opt.step()` (`Train.py:32`) and the re-zeroing at `Train.py:34`,
  and in a new network before its first forward call
  (`DeepOmix.py:9`, `DeepOmix.py:22`). Code that reads or saves the
  weight at those points sees them.
- Layer widths are typed by hand (`RunSingle.py:14-15`,
  `RunMultiple.py:16-17`) and must equal the mask's shape; a
  mismatch raises at `DeepOmix.py:22`.

## Out of scope

- Loss and metric: Cox partial likelihood on time-sorted samples,
  C-index (`Survival.py:9-42`, `DataLoader.py:39`).
- Training: full-batch Adam, `weight_decay` as L2 on every
  parameter; grid over learning rate and L2 on one 80/10/10 split
  (`Train.py:22-32`, `RunSingle.py:19-51`, `TransferData.py:51-54`).
- Dense-layer magnitude pruning (`Train.py:36-87`).
- Preprocessing is not in the repository; plotting is
  `Visualization.py:120-162`.
- `code_with_clinical/` imports modules that are not in the
  repository (`code_with_clinical/TrainWithClinical.py:1`,
  `code_with_clinical/Run.py:1-2`).

## Paper vs code

- Output layer: the paper reads `h·δ` from the second hidden layer
  only. The code adds a second input next to `h`: the event
  indicator `yevent` (`Train.py:29`, `Train.py:95`), or `age` in
  `code_with_clinical/`.
- Pathways: the paper keeps sets of 20–200 genes; the code has no
  size filter (demo sets have 2–713 genes) and keeps a random 100
  (`TransferData.py:14`).
- The KS test on pathway units is not in the code.
- The paper runs 10 × 10-fold cross-validation with inner 5-fold
  tuning on four omics types. The code uses one fixed split, stops
  after the grid search (`Num_EPOCHS` at `RunSingle.py:22` is
  unused), and demos `['rna']` and `['rna', 'methy']`
  (`TransferData.py:117`, `TransferData.py:124`).

## Key excerpts

`DeepOmix.py:21-33`

```python
	def forward(self, x_1, x_2):
		self.sc1.weight.data = self.sc1.weight.data.mul(self.pathway_mask)
		x_1 = self.tanh(self.sc1(x_1))
		if self.training == True: 
			x_1 = x_1.mul(self.do_m1)
		x_1 = self.tanh(self.sc2(x_1))
		if self.training == True: 
			x_1 = x_1.mul(self.do_m2)
		x_1 = self.tanh(self.sc3(x_1))
		x_cat = torch.cat((x_1, x_2), 1)
		lin_pred = self.sc4(x_cat)
		
		return lin_pred
```

`Train.py:22-34`

```python
	opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay = L2)
	for epoch in range(Num_Epochs+1):
		net.train()
		opt.zero_grad() 
		net.do_m1 = dropout_mask(Pathway_Nodes, Dropout_Rate[0])
		net.do_m2 = dropout_mask(Hidden_Nodes, Dropout_Rate[1])

		pred = net(train_x, train_yevent) 
		loss = neg_par_log_likelihood(pred, train_ytime, train_yevent) 
		loss.backward()
		opt.step()

		net.sc1.weight.data = net.sc1.weight.data.mul(net.pathway_mask)
```

`TransferData.py:29-42`

```python
    for i in range(len(omics_data_file)):
        temp=pd.read_csv(input_path+omics_data_file[i]+'.csv.gz',index_col=0)
        temp=temp.dropna(axis=1)
        genes_intersect=list(set(pathway_gene)&set(temp.columns.tolist()))
        temp=temp.loc[patientID,genes_intersect]
        pathway_temp=pathway_select.loc[:,genes_intersect]
        x='_'+omics_data_file[i]+'.'
        genes_intersect=x.join(genes_intersect)+'_'+omics_data_file[i]
        genes_intersect=genes_intersect.split('.')
        pathway_temp.columns=genes_intersect
        temp.columns=genes_intersect
        print('there are ',len(genes_intersect),' genes in ',omics_data_file[i],' omics data')
        pathway_output=pathway_output.join(pathway_temp)
        omics_output=omics_output.join(temp)
```

## Open questions

- Whether the published `CancerProfiling/DeepOmix` code matches the
  2025 repository: the old URL is gone, and a Wayback Machine lookup
  returned HTTP 429.
- Which pathway table, filter, and inputs produced the paper's
  results: only the LGG demo ships.
- Whether the paper's runs fed `yevent`, `age`, or nothing into the
  output layer.
- How pathway-layer values for the KS test were read out: searched
  `*.py` for `ks`, `kolmog`, `bonferr`, `mds`, `median`; not found.
