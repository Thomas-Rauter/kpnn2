# drugcell-2020: DrugCell

| Field | Value |
|-------|-------|
| Paper | Kuenzi BM et al., 2020. Predicting Drug Response and Synergy Using a Deep Learning Model of Human Cancer Cells. Cancer Cell 38(5):672-684.e6. 10.1016/j.ccell.2020.09.014 |
| Code | `https://github.com/idekerlab/DrugCell` at `c507e1d821fac0201e42f831a1d772e7ef42b00e` |
| Other code | Zenodo archive of the manuscript version, 10.5281/zenodo.4011359 (not examined). `https://github.com/aksinghal5590/rlipp` (2022, a lab member's repository named after the paper's score; not linked by the paper; not examined). idekerlab web-app repositories (`drugcell-web-app`, `drugcell-analysis-webapp`, `cddrugcellfinddrug`; not examined) |
| Framework | Python 3.7, PyTorch 1.2.0, networkx 2.4 (`environment_setup/environment.yml`); README says PyTorch >= 0.4 |
| License | MIT (`LICENSE`) |
| Full text | PubMed Central, PMC7737474 (abstract, STAR Methods, Figure 1 caption, Key Resources Table) |
| Extracted | 2026-10-09, Claude Code, Claude Opus 5.5 |

## Summary

Regression of a cell line's response to a drug (area under the
dose-response curve). Inputs: a binary mutation vector over 3,008
genes and a 2,048-bit Morgan fingerprint of the drug. The genotype
branch follows a hierarchy of 2,086 GO Biological Process terms:
each term is a block of six units that reads its child terms and
its directly annotated genes. A dense branch encodes the drug; the
root term's units and the drug units feed a small dense head. Each
term also has an auxiliary output head with its own loss term.

## Prior-knowledge graph

- Source: GO Biological Process (`is_a`, `part_of`). Paper: keep
  terms with at least 10 DrugCell genes and at least 30 more than
  any child; cut to depth five above the bottom terms. Filtering
  code not in the repository; the result ships as
  `data/drugcell_ont.txt` and on NDEx.
- Size: 2,086 terms, 3,008 genes, six term layers (paper; 2,086
  x 6 = 12,516 units). Edge and annotation counts: not stated in
  the text read; data not read. Directed parent -> child
  (`util.py:28-29`), unsigned, several parents allowed. Acyclicity
  assumed, not checked. One root and one connected component
  required (`util.py:65-80`).
- Data to nodes: three-column table (parent, child term or gene,
  type `default` or `gene`; README, `util.py:24-39`). Gene names
  map to input columns through `gene2ind.txt`
  (`util.py:113-125`); annotations of unknown genes are skipped
  (`util.py:31-32`). Genes feed only the terms they are directly
  annotated to, at any height (`drugcell_NN.py:117-118`).
  Unannotated gene columns stay in the input, read by no unit;
  whether any exist is unclear. A term with no mapped gene in its
  subtree stops the program (`util.py:45-61`).
- Built at: `code/util.py:14-82`; `code/drugcell_NN.py:86-128`;
  `code/train_drugcell.py:17-34`.

## Architecture

- Layers: terms are grouped by repeatedly removing terms with no
  remaining child (`drugcell_NN.py:99-128`), so a term's layer is
  its height above the bottom. The grouping only fixes the
  computation order (`drugcell_NN.py:145-147`); there is no weight
  matrix per layer.
- Units per node: `k` per term, the same for all terms
  (`-genotype_hiddens`, default 6; `drugcell_NN.py:48-58`). The
  term's gene count is computed but only printed.
- Per term (`drugcell_NN.py:149-162`): concatenate the `k`
  outputs of each child (child-list order), then the term's gene
  values; `Linear(k*c + g, k)` (`drugcell_NN.py:111-123`), tanh,
  `BatchNorm1d(k)`.
- Gene values: per term, `Linear(n_genes, g)` over the whole gene
  vector, masked so row `i` reads only the term's `i`-th gene
  (`drugcell_NN.py:62-70`, `139-140`). Each annotation thus gets
  one weight and one bias, no nonlinearity, before the term layer.
- Connections that skip layers: kept. A term reads children of
  any layer; genes enter at every layer; a child feeds all its
  parents.
- Inner outputs: each term has a head `Linear(k, 1) -> tanh ->
  Linear(1, 1)` (`drugcell_NN.py:125-126`, `163-164`), as does
  each drug layer (`drugcell_NN.py:80-81`, `173-174`). Loss: MSE
  of the final output plus 0.2 x MSE of every auxiliary output,
  same label (`train_drugcell.py:93-99`). Prediction uses only
  the final output (`predict_drugcell.py:45-47`).
- Not constrained: drug branch 2048 -> 100 -> 50 -> 6, each
  `Linear -> tanh -> BatchNorm1d` (`drugcell_NN.py:74-83`,
  `167-171`). Head: root (6) + drug (6) -> `Linear(12, 6) -> tanh
  -> BatchNorm1d -> Linear(6, 1) -> tanh -> Linear(1, 1)`
  (`drugcell_NN.py:41-45`, `177-183`).
- Other structure: none. One input vector, genes then
  fingerprint, split by `narrow` (`drugcell_NN.py:133-134`).
- Defined at: `code/drugcell_NN.py:13-185`.

## Connectivity mechanism

Term to term: one `nn.Linear` per term over the concatenation of
its children's outputs and its gene values
(`drugcell_NN.py:111-123`, `149-159`). Each child -> parent edge
is a `k x k` block and each gene a `k`-vector column of that
weight; absent edges have no parameter.

Gene to term: per term a dense `Linear(n_genes, g)`
(`drugcell_NN.py:70`) and a 0/1 mask with one 1 per row
(`train_drugcell.py:19-34`). The weight is multiplied by the mask
once at initialization (`train_drugcell.py:61-62`) and the
gradient after every backward pass, before `optimizer.step()`
(`train_drugcell.py:103-110`). The forward pass does not use the
mask (`drugcell_NN.py:140`). With Adam and no weight decay
(`train_drugcell.py:53`) masked entries stay exactly zero. Biases
are not masked. The mask is found by the parameter name's prefix
before the first `_` (`train_drugcell.py:59`, `106`).

Initialization: PyTorch defaults, then every parameter (BatchNorm
scales included) times 0.1 (`train_drugcell.py:58-64`).

## Interpretation

Code: the prediction script writes the `k` post-BatchNorm
activations of every term, drug layer, and the head, per sample,
to a file named after the node (`predict_drugcell.py:42-52`). No
score is computed in the repository.

Paper only (code not found): RLIPP, taken from DCell. Per term,
two L2-penalized linear regressions of drug response, one on the
term's units and one on its children's units; each scored by
Spearman correlation; RLIPP is parent over children (> 1: the term
adds predictive power). Reported per drug; sample split and
penalty choice not stated in the text read.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N01 Read a DAG from an edge table | `code/util.py:24-39` | Builds the term DAG from parent-child rows whose third column is `default`; rows of type `gene` are annotations. |
| N06 Connect features to nodes by membership | `code/util.py:28-39`, `code/drugcell_NN.py:117-118` | Joins gene inputs to terms through the `gene` rows matched by name; unknown genes are skipped; a gene can join many terms, at any height. |
| N09 Dense head after the graph layers | `code/drugcell_NN.py:177-183` | Root term output (with the drug branch) goes through `Linear(12,6)`, tanh, BatchNorm, `Linear(6,1)`, tanh, `Linear(1,1)`. |
| N11 Standard layers between graph layers | `code/drugcell_NN.py:124`, `162` | Every term applies `BatchNorm1d` to its own units after tanh, before its parents read them. |
| N12 Parameters only for present edges | `code/drugcell_NN.py:111-123` | One `Linear` per term sized by its children and direct genes, so each term edge is a `k x k` block that exists only if the edge does. |
| N26 Unlayered DAG in topological order | `code/drugcell_NN.py:99-128`, `145-164` | Terms are computed one by one in leaf-peeling order; each reads children of any depth plus its direct genes. |
| N31 Read out node activations | `code/predict_drugcell.py:49-52` | Writes every term's six activations per sample to `<term>.hidden`; the paper fits RLIPP on such values. |
| N38 Several units per node | `code/drugcell_NN.py:48-58`, `123-124` | Every term has `k = 6` units; a child's six units all feed all six units of each parent. |
| N39 Mask weights at start and gradients each step | `code/train_drugcell.py:58-64`, `103-110` | Gene-to-term weights are dense, zeroed by a mask once, then held at zero by masking each gradient before the optimizer step. |
| N40 Auxiliary output head on every node | `code/drugcell_NN.py:125-126`, `163-164`; `code/train_drugcell.py:93-99` | Each term has a `Linear(k,1)`, tanh, `Linear(1,1)` head whose MSE against the label is added with weight 0.2. |
| N41 Unconstrained branch on a second input | `code/drugcell_NN.py:74-83`, `167-177` | The drug fingerprint goes through a dense 100-50-6 branch whose output is concatenated with the root term's units. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Read ontology table, split term edges from gene annotations, map gene names to columns | `code/util.py:14-43`, `113-125` | ~43 |
| Check every term has genes below it, one root, one component | `code/util.py:45-82` | ~38 |
| Build one masked gene-selection layer per term | `code/drugcell_NN.py:61-70` | ~10 |
| Order terms by peeling leaves; size and build per-term layer, BatchNorm, auxiliary head | `code/drugcell_NN.py:86-128` | ~43 |
| Forward: concatenate child outputs and gene values per term | `code/drugcell_NN.py:137-164` | ~28 |
| Build one 0/1 mask per term | `code/train_drugcell.py:17-34` | ~18 |
| Apply masks to initial weights and to every gradient | `code/train_drugcell.py:58-64`, `103-108` | ~13 |
| Write per-term activations to term-named files | `code/predict_drugcell.py:49-52` | ~4 |

## Fragile spots

- Mask lookup by `name.split('_')[0]` (`train_drugcell.py:59`,
  `106`): a term ID containing `_` gives a `KeyError` or another
  term's mask.
- The mask is applied only inside `train_model`
  (`train_drugcell.py:61-62`, `103-108`); a training path without
  these lines lets absent gene weights become nonzero silently.
- `construct_NN_graph` empties the graph it is given
  (`drugcell_NN.py:128`).
- No acyclicity check: with a cycle, leaf peeling stops early
  (`drugcell_NN.py:100-105`) and the error surfaces later as a
  missing key in `forward` (`drugcell_NN.py:152`, `177`).
- Row-to-gene order of each gene layer is a Python `set`'s
  iteration order (`train_drugcell.py:27`); only the set is kept
  on the model (`drugcell_NN.py:24`).
- Activation files are appended to (`predict_drugcell.py:50-52`);
  a rerun into the same folder mixes rows from both runs.

## Out of scope

- Loss (MSE), Adam (lr 0.001), batch 5,000, 300 epochs, model
  saved each epoch (`train_drugcell.py:53`, `69-146`).
- Data loading and per-batch feature lookup (`util.py:85-152`).
- GPU and CPU prediction scripts beyond the activation dump.
- Drug fingerprints and ontology filtering: not in the repository.

## Paper vs code

- Initialization: paper, uniform between 0.001 and 0.001 (sign
  lost); code, PyTorch defaults times 0.1
  (`train_drugcell.py:58-64`).
- Optimizer: paper, standard gradient descent; code, Adam
  (`train_drugcell.py:53`).
- Gene inputs: paper's term weight is `k x (k*c + g)` on child
  states and genes; code adds a per-annotation weight and bias
  before it (`drugcell_NN.py:62-70`, `139-140`).
- Auxiliary heads, loss weight 0.2 (`train_drugcell.py:93-99`):
  not in the methods text read, which says the VNN follows DCell
  "with minor modifications".
- Head: paper, six hidden units then one output neuron; code adds
  tanh and `Linear(1, 1)` after it (`drugcell_NN.py:182-183`).
- Training: paper stops early on validation; code trains all
  epochs and reports the best (`train_drugcell.py:140-146`).
- Ontology filtering and RLIPP: in the paper, not in the code.

## Key excerpts

`code/drugcell_NN.py:145-164`

```python
		for i, layer in enumerate(self.term_layer_list):

			for term in layer:

				child_input_list = []

				for child in self.term_neighbor_map[term]:
					child_input_list.append(term_NN_out_map[child])

				if term in self.term_direct_gene_map:
					child_input_list.append(term_gene_out_map[term])

				child_input = torch.cat(child_input_list,1)

				term_NN_out = self._modules[term+'_linear_layer'](child_input)				

				Tanh_out = torch.tanh(term_NN_out)
				term_NN_out_map[term] = self._modules[term+'_batchnorm_layer'](Tanh_out)
				aux_layer1_out = torch.tanh(self._modules[term+'_aux_linear_layer1'](term_NN_out_map[term]))
				aux_out_map[term] = self._modules[term+'_aux_linear_layer2'](aux_layer1_out)		
```

`code/train_drugcell.py:58-64`

```python
	for name, param in model.named_parameters():
		term_name = name.split('_')[0]

		if '_direct_gene_layer.weight' in name:
			param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
		else:
			param.data = param.data * 0.1
```

`code/train_drugcell.py:103-108`

```python
			for name, param in model.named_parameters():
				if '_direct_gene_layer.weight' not in name:
					continue
				term_name = name.split('_')[0]
				#print name, param.grad.data.size(), term_mask_map[term_name].size()
				param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])
```

## Open questions

- RLIPP code: not in the repository (searched for `rlipp`,
  `ridge`, `spearman`); the Zenodo archive and
  `aksinghal5590/rlipp` were not examined.
- Ontology construction (GO release, the 10-gene and 30-gene
  rules, the depth cut): not in the repository.
- Number of term edges and gene annotations, and whether every
  gene in `gene2ind.txt` is annotated to some term: data files not
  read.
- How RLIPP regressions are fit (per drug or pooled, train or test
  samples, penalty choice): not stated in the text read.
