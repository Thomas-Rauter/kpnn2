# tumor-cell-architecture-2024: NeST-VNN

| Field | Value |
|-------|-------|
| Paper | Park S, Silva E, Singhal A, et al. (last author Ideker T), 2024. A deep learning model of tumor cell architecture elucidates response and resistance to CDK4/6 inhibitors. Nature Cancer. 10.1038/s43018-024-00740-1 |
| Code | `https://github.com/idekerlab/nest_vnn` at `ef1b739d91cf9c91b40fb8ee4f739737cc4d0210` (2025-06-13) |
| Other code | The README says the same repository serves Zhao, Singhal et al., Cancer Discovery 2024 (`README.md:15-17`). `idekerlab/cellmaps_vnn` is a later tool that its docs say trains and predicts on data formatted for NeST-VNN; not read. The model class is still named `DrugCellNN` (`src/drugcell_nn.py:11`), from the DrugCell lineage. |
| Framework | Python 3.7, PyTorch 1.8.0 (CUDA 11.1), networkx, scikit-learn, optuna (`conda-envs/cuda11_env.yml:3-15`); only Python and PyTorch are pinned |
| License | MIT (`LICENSE`) |
| Full text | PubMed Central PMC11286358 (Methods; code and data availability) |
| Extracted | 2026-10-10, Claude Code, Claude Opus 5.5 (claude-opus-5-5) |

## Summary

The model predicts a cell line's response (area under the dose-response curve)
to one drug, mainly palbociclib. Its input is three binary flags per gene
(mutation, copy-number deletion, amplification) for 718 clinical-panel genes.
The prior is the NeST hierarchy of protein assemblies, cut to the 131
assemblies with at least five panel genes. Each gene is one unit. Each assembly
is a block of k units that reads its child assemblies and its directly
annotated genes. The root assembly feeds a small output head, and every
assembly also has an auxiliary head trained on the same label. Assemblies are
scored by how well a ridge regression on their units reproduces the prediction.

## Prior-knowledge graph

- Source: NeST, 395 nested protein assemblies from earlier work. The paper
  keeps the assemblies with at least five panel genes (131). That filter is not
  in the repository. The code reads a finished table of parent, child, and
  type. Type `default` marks an assembly edge, and any other type marks a gene
  annotation (`src/training_data_wrapper.py:51-63`; format in
  `README.md:107-125`).
- Size (`sample/ontology.txt`, matching the paper's 131): 131 assemblies, 150
  assembly edges, one root (`NEST`), 17 assemblies with more than one parent.
  1,366 annotation rows join the 718 genes to 130 assemblies, 1–21 assemblies
  per gene (median 1). Annotations are direct only: no assembly lists a gene a
  descendant lists. The graph is directed, unsigned, and acyclic.
- Data to nodes: annotation genes are matched by name to `gene2ind.txt`, and
  rows whose gene is missing are skipped
  (`src/training_data_wrapper.py:57-58`). Data columns are matched to genes by
  position only (`README.md:47`, `src/util.py:186-195`). A gene with no
  annotation would still get a unit that nothing reads (none in the sample). An
  assembly with no gene below it, more than one root, or more than one
  component stops the program (`src/training_data_wrapper.py:65-94`).
- Built at: `src/training_data_wrapper.py:44-99`.

## Architecture

- Layers: leaf-peeling by height. Assemblies with no child assembly are layer
  0; they are removed and the step repeats (`src/drugcell_nn.py:89-120`). The
  sample ontology gives 8 layers (76, 32, 13, 4, 2, 2, 1, 1), and 78 of its 150
  assembly edges join non-adjacent layers. Gene units sit below all assembly
  layers.
- Units per node: each gene is one unit, built as its own `Linear(3, 1)`, then
  tanh, then `BatchNorm1d(1)` (`src/drugcell_nn.py:62-65`, `:130-134`). Each
  assembly has the same k units (`src/drugcell_nn.py:47-56`):
  `-genotype_hiddens`, default 4 (`src/train.py:23`), 4 in
  `scripts/train_cv.sh:34`.
- Gene to assembly: per assembly, a `Linear(718, m)` over all gene units, with
  m its number of direct genes (`src/drugcell_nn.py:67-73`). The mask leaves
  one weight per row (`src/util.py:200-207`), so each output is one gene unit
  times a scalar plus a bias. No nonlinearity follows.
- Assembly update: `BatchNorm(tanh(Linear(Dropout(x))))`. Here x is the
  concatenation of the children's units and the m gene outputs, so the input
  size is k × children + m (`src/drugcell_nn.py:100-114`, `:141-159`).
- Connections that skip layers: implicit, since an assembly reads children in
  any lower layer and genes at any layer (`src/drugcell_nn.py:145-152`).
- Outputs or losses at inner layers: every assembly, the root included, has a
  head of `Linear(k, 1)`, tanh, `Linear(1, 1)` (`src/drugcell_nn.py:116-117`,
  `:160-161`). The loss is the final head's loss plus α = 0.3 times each
  assembly head's loss, all on the same label (`src/vnn_trainer.py:72-79`,
  `src/train.py:17`). These heads do not feed the prediction.
- Parts the graph does not constrain: the final head on the root's k units,
  `Linear(k, 1)`, tanh, `Linear(1, 1)` (`src/drugcell_nn.py:42-43`,
  `:163-165`).
- Other structure: dropout 0.3 on assembly inputs from layer index 2 up
  (`src/drugcell_nn.py:153-155`, `src/train.py:32-33`). No recurrence or
  attention.
- Defined at: `src/drugcell_nn.py:11-167`.

## Connectivity mechanism

1. **Between assemblies: no absent weights.** Each assembly is its own
   `nn.Linear`. The forward pass looks up the children's outputs by name from a
   stored child list, concatenates them, and applies that `Linear`
   (`src/drugcell_nn.py:83-87`, `:145-157`). No matrix spans assemblies, so
   non-edges have no parameter. Each k × k block of an assembly's weight is one
   edge.
2. **Gene to assembly: full matrix with a mask.** Each
   `<term>_direct_gene_layer` holds a full 718-wide weight. Before training it
   is multiplied by a 0/1 mask (one 1 per row) and by 0.1
   (`src/vnn_trainer.py:33-39`). After every backward pass its gradient is
   multiplied by the same mask before the AdamW step
   (`src/vnn_trainer.py:81-85`). The forward pass does not use the mask
   (`src/drugcell_nn.py:136-139`). Zero gradients keep Adam's moments at zero,
   and decoupled weight decay leaves zero entries at zero. The bias is not
   masked. The Optuna path repeats both steps
   (`src/optuna_nn_trainer.py:69-75`, `:116-120`).

Initialization: PyTorch defaults (so the gene layer's fan-in is all 718 genes),
then every parameter, batch-norm scales included, is multiplied by 0.1
(`src/vnn_trainer.py:34-39`). The mask is looked up by the part of the
parameter name before the first `_` (`src/vnn_trainer.py:35`, `:84`).

## Interpretation

- **Activations.** `src/predict.py` writes every gene's and every assembly's
  units per test sample to `<hidden>/<node>.hidden` (`src/predict.py:54-57`).
- **Gradients.** The same script writes per-sample gradients of the final
  output for every input (one file per data type) and for every node's units
  (`src/predict.py:59-76`). Nothing in the repository or the paper uses them.
- **Probe score (RLIPP variant).** For each assembly and each "drug" group of
  test samples (a placeholder column here), the code reduces the assembly's
  units to k PCA components and fits `RidgeCV(cv=5)` to the model's
  predictions. It records `P_rho`, the in-sample Spearman ρ between that fit
  and the predictions. It repeats the fit on the concatenated units of the
  assembly's children, assemblies and genes alike, to get `C_rho`, then writes
  `RLIPP = P_rho / C_rho` (`src/rlipp_calculator.py:116-137`). For each gene it
  writes the Spearman ρ between its unit and the predictions (`:141-146`). Rows
  are keyed by node name. Children are re-read from the ontology rows
  (`:76-103`).
- **Across models.** The paper averages `P_rho` over the five CV models. It
  tests that average against 500 null models with shuffled memberships that
  preserve assembly size and parent-child relations, with BH correction. Not
  found in the code.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N01 Read a DAG from an edge table | `src/training_data_wrapper.py:46-63` | Assembly edges are the `default` rows of a parent, child, type table loaded into a DiGraph. |
| N06 Connect features to nodes by membership | `src/training_data_wrapper.py:57-62`, `src/drugcell_nn.py:67-73` | Other rows annotate genes to assemblies by name (a gene can belong to up to 21), and genes outside `gene2ind.txt` are skipped. |
| N07 One input per entity and data type | `src/training_data_wrapper.py:33-36`, `src/drugcell_nn.py:64`, `:132` | A gene's three binary flags all feed that gene's own unit. |
| N10 One unit per node | `src/drugcell_nn.py:62-65`, `:130-134` | Each gene node is one unit with its own `Linear(3, 1)`, tanh, and `BatchNorm1d(1)`. |
| N38 Several units per node | `src/drugcell_nn.py:47-56`, `src/train.py:23` | Each assembly has the same k units (4 in `scripts/train_cv.sh:34`). |
| N26 Unlayered DAG in topological order | `src/drugcell_nn.py:141-159` | Assemblies run child-first, and each reads gene units and child assemblies of any depth together. |
| N74 Layer nodes by height above the data | `src/drugcell_nn.py:89-120`, `:112`, `:153` | Leaf-peeling sets the order, and the layer index turns dropout on from layer 2 up. |
| N12 Parameters only for present edges | `src/drugcell_nn.py:100-114` | Each assembly's `Linear` takes k × children + direct genes inputs, so non-edges between assemblies have no weight. |
| N58 Gather each node's inputs by index | `src/drugcell_nn.py:83-87`, `:145-152` | Each assembly concatenates outputs from its stored child-name list and feeds only those to its own weights. |
| N39 Mask weights at start and gradients each step | `src/vnn_trainer.py:33-39`, `:81-85`, `src/util.py:200-207` | Gene-to-assembly weights are 718 wide, masked once at start and in the gradient after each backward pass. |
| N11 Standard layers between graph layers | `src/drugcell_nn.py:112-115`, `:153-159` | Batch norm follows every gene and assembly, and dropout precedes assembly `Linear`s from layer 2 up. |
| N40 Auxiliary output head on every node | `src/drugcell_nn.py:116-117`, `:160-161`, `src/vnn_trainer.py:72-79` | Each assembly's k→1→1 head adds α = 0.3 times its loss. |
| N09 Dense head after the graph layers | `src/drugcell_nn.py:42-43`, `:163-165` | The root's k units go through `Linear(k, 1)`, tanh, and `Linear(1, 1)`. |
| N31 Read out node activations | `src/predict.py:54-57` | Every gene's and assembly's units are written per sample to a file named after the node. |
| N14 Attribution to inputs | `src/predict.py:63-70` | Per-sample gradients for each gene's three inputs are written out. |
| N15 Attribution to hidden nodes | `src/predict.py:59-63`, `:72-76` | Per-sample gradients for every node's units are written out. |
| N75 Plain gradients of the output | `src/predict.py:44`, `:59-76` | Raw output gradients, with no baseline, are written for inputs and node units. |
| N76 Node score from a probe on its units | `src/rlipp_calculator.py:116-146` | Assemblies get the ρ of a PCA-plus-ridge fit to the predictions, compared with the same fit on their children; genes get their unit's ρ. |
| N17 Map scores to node names | `src/rlipp_calculator.py:136`, `:145`, `src/predict.py:55`, `:74` | Activations, gradients, and probe scores are keyed by assembly or gene name. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Read name→index tables | `src/util.py:174-183` | ~10 |
| Parse ontology; split edges and annotations; check sizes, root, and components | `src/training_data_wrapper.py:44-99` | ~55 |
| Stack data into samples × genes × 3 | `src/training_data_wrapper.py:33-36`, `src/util.py:186-195` | ~15 |
| Per-gene and per-assembly direct-gene modules | `src/drugcell_nn.py:60-73` | ~14 |
| Child lists, leaf-peeling layers, per-assembly modules | `src/drugcell_nn.py:78-120` | ~43 |
| Forward wiring by name lookup | `src/drugcell_nn.py:124-167` | ~44 |
| One-hot gene masks per assembly | `src/util.py:198-207` | ~10 |
| Masks at init and on gradients, in both trainers | `src/vnn_trainer.py:33-39`, `:81-85`; `src/optuna_nn_trainer.py:69-75`, `:116-120` | ~24 |
| Dump activations and gradients by node name | `src/predict.py:34-76` | ~40 |
| Re-read children and per-node files for the probe | `src/rlipp_calculator.py:62-111` | ~50 |

## Fragile spots

- The mask key is `name.split('_')[0]` (`src/vnn_trainer.py:35`, `:84`). An
  assembly name containing `_` would fail. If its prefix were another
  assembly's name, it would silently take the wrong mask. NeST names
  (`NEST:85`) contain no `_`.
- Data columns are joined to genes by position only. Nothing checks the order
  against `gene2ind.txt` (`src/util.py:186-195`).
- The probe reads the first `-genotype_hiddens` columns of each `.hidden` file
  (`src/rlipp_calculator.py:63-69`), and the script hard-codes 4
  (`scripts/rlipp_cv.sh:25`). A model with larger k would silently be probed on
  a subset of its units.
- `predict.py` appends to the per-node files (`src/predict.py:56`, `:75`). Only
  the shell scripts clear them first (`scripts/test_cv.sh:23-27`).

## Out of scope

- Loss: `1 − CCC`, concordance correlation (`src/ccc_loss.py:9-21`).
- Training loop: AdamW, keeps the best validation loss
  (`src/vnn_trainer.py:44-133`). Patience is used only in the Optuna path
  (`src/optuna_nn_trainer.py:168-171`).
- Data loading and label standardization: `src/util.py:70-171`. The 20%
  validation split by cell line is unseeded (`src/util.py:111-117`).
- Hyperparameter search: Optuna over lr only, with k fixed at 4 and one trial
  (`src/optuna_nn_trainer.py:36-53`).
- SLURM scripts: `scripts/*.sh`.

## Paper vs code

- Loss: the paper gives MSE at the root, plus α·MSE at the other assemblies,
  plus β‖W‖. The code uses `1 − CCC` for every head
  (`src/vnn_trainer.py:72-78`). The only weight penalty is AdamW
  `weight_decay`, set to the learning rate (`src/vnn_trainer.py:44`). The
  parsed `-wd` is never used (`src/train.py:16`,
  `src/training_data_wrapper.py:18`).
- Layers and dropout: the paper reports seven layers, with dropout 0.3 on the
  last four. Leaf-peeling the shipped ontology gives eight, and the default
  dropout start (`src/train.py:32`) covers six.
- Gene input: in the paper, gene states enter the assembly directly. The code
  adds a per-edge scalar weight and a bias in between (`src/drugcell_nn.py:73`,
  `:136-139`).
- In silico activity (first principal component of an assembly's units), the
  null models, and the L1 logistic regression on NeST:85: not found (see Open
  questions).
- The CV script runs `-optimize 1` (no Optuna) with lr 0.001, k = 4, and 150
  epochs. It calls `train_drugcell.py` and `predict_drugcell.py`, which are
  absent (`scripts/train_cv.sh:28-35`, `scripts/test_cv.sh:31`).

## Key excerpts

`src/drugcell_nn.py:145-159`

```python
				child_input_list = []
				for child in self.term_neighbor_map[term]:
					child_input_list.append(hidden_embeddings_map[child])

				if term in self.term_direct_gene_map:
					child_input_list.append(term_gene_out_map[term])

				child_input = torch.cat(child_input_list, 1)
				if i >= self.min_dropout_layer:
					dropout_out = self._modules[term + '_dropout_layer'](child_input)
					term_NN_out = self._modules[term + '_linear_layer'](dropout_out)
				else:
					term_NN_out = self._modules[term + '_linear_layer'](child_input)
				Tanh_out = torch.tanh(term_NN_out)
				hidden_embeddings_map[term] = self._modules[term + '_batchnorm_layer'](Tanh_out)
```

`src/vnn_trainer.py:33-39`

```python
		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.cuda)
		for name, param in self.model.named_parameters():
			term_name = name.split('_')[0]
			if '_direct_gene_layer.weight' in name:
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1
```

`src/vnn_trainer.py:81-85`

```python
				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])
```
## Open questions

- The 395 → 131 filter and the code that writes the ontology table: not in the
  repository. Unclear whether the direct-only gene rows come from NeST or from
  that step.
- k in the pretrained palbociclib models: undetermined, because the `.pt` files
  were not loaded.
- Code for null models, fold averaging, core assemblies, and the L1 regression:
  not found (searched shuffle, null, permut, seed, jaccard, logistic).
  `cellmaps_vnn` was not read.
