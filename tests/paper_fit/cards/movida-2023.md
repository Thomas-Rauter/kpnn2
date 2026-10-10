# movida-2023: MOViDA

| Field | Value |
|-------|-------|
| Paper | Ferraro L, Scala G, Cerulo L, Carosati E, Ceccarelli M, 2023. MOViDA: multiomics visible drug activity prediction with a biologically informed neural network model. Bioinformatics 39(7):btad432. 10.1093/bioinformatics/btad432 |
| Code | `https://github.com/Luigi-Ferraro/MOViDA` at `2df33e1a41725a59ab3731c46d530dde1e992d06` (tip of `main`, committed 2023-01-25) |
| Other code | Zenodo 10.5281/zenodo.8180380: `data.zip`, `RIS_score.csv`, `DeepLift_drug_features.csv` (data and results, no code; not downloaded). The README lists `code/models/MOViDA_synergy.py`, absent from the tree. |
| Framework | Python, PyTorch (README: "PyTorch >= 11.3", CUDA >= 11; nothing pinned), networkx, pandas, numpy |
| License | GPL-3.0 (`LICENSE`) |
| Full text | Bioinformatics (OUP) open article page: Methods 2.1-2.4, Fig. 1 caption |
| Extracted | 2026-10-10, Claude Code, claude-opus-5-5 |

## Summary

MOViDA predicts drug response (AUC) of cancer cell lines from GDSC and
CTRP (about 384k cell line-drug pairs). A cell line enters as three
binary gene vectors (mutation, amplification, deletion) and one
precomputed expression enrichment score per GO term. A visible network
on the GO Biological Process hierarchy (2086 terms, one root) gives
each term six units, fed by its annotated genes, its children's units,
and enrichment scores. A dense branch encodes drug descriptors, and a
dense head combines the root's units with the drug embedding into one
AUC. The paper scores terms by silencing them (RIS) and drug features
with DeepLIFT; neither is in the repository.

## Prior-knowledge graph

- Source: GO Biological Process (paper: "five layers, one element as a
  root, and a total of 2086 GO terms"). GO release and term filtering:
  not stated, not in code; the code reads a prepared `ont_tree.txt`
  (`code/main.py:69`).
- Size: 2086 term nodes (paper); edge count unknown (file not read).
  Directed parent to child (`code/param_data.py:119-123`), unsigned,
  assumed acyclic. The loader exits unless there is exactly one node
  with no parent and one weakly connected component
  (`code/param_data.py:125-135`).
- Data to nodes: each data type has its own gene index file and its
  own term-gene table (`gene2ind2{mut,amp,del}.txt`,
  `ont_{mut,amp,del}.txt`, `code/main.py:61,70`); paper sizes 4870,
  2931, 2097 genes, pre-filtered to genes annotated in the hierarchy.
  Per data type, `get_go_gene_map` maps each graph term to its set of
  gene indices (`code/param_data.py:139-152`). A table gene missing
  from the index file raises `KeyError` (`:151`); rows for terms
  outside the graph are ignored; index genes with no term stay as
  input columns. A term with no genes of a type gets no gene block
  for it (`code/models/MOViDA.py:80-82,321-323`). Enrichment scores:
  `GO_enrich.tsv`, one column per term name, one row per cell line
  (`code/param_data.py:101-112`); a term without a column raises
  `KeyError` in forward (`code/models/MOViDA.py:312`). Paper: scores
  from expression with yaGST mww-GST (R); that code is not found.
- Built at: `code/param_data.py:101-152`;
  `code/models/MOViDA.py:45-87,110-122`.

## Architecture

- Layers: peel the terms with no child into a group, remove them,
  repeat until empty (`code/models/MOViDA.py:61-87`). Groups only fix
  the order; terms are computed one at a time
  (`code/models/MOViDA.py:308-335`) and read children of any depth.
- Units per node: `ccl_hiddens` per term (`code/models/MOViDA.py:84`),
  6 in `code/main.py:40`. Before them, each (term, data type) pair has
  a gene block: `nn.Linear` from all genes of the type to one unit per
  annotated gene (`code/models/MOViDA.py:45-48`), masked so unit *i*
  reads gene *i*, no nonlinearity; each annotation edge gets its own
  scale and bias before the term's layer.
- Term computation: concatenate every child's six units (in
  `dG.neighbors` order), the gene blocks, and the enrichment scores of
  the term and each child; `nn.Linear` to six units, `tanh`,
  `BatchNorm1d(6)` (`code/models/MOViDA.py:310-334`; widths `:71-82`).
- Connections that skip layers: each child's raw enrichment score also
  enters its parent's layer (`code/models/MOViDA.py:75-78,311-313`);
  gene blocks feed terms at every depth.
- Outputs or losses at inner layers: none. The last term computed
  feeds the head (`code/models/MOViDA.py:337`).
- Parts the graph does not constrain: drug branch, `Linear -> tanh ->
  BatchNorm` of widths 100, 50, 6 on 1009 drug features
  (`code/models/MOViDA.py:90-96,340-348`; `code/main.py:41`); head:
  root's six units concatenated with the drug's six, `Linear(12, 6) ->
  tanh -> BatchNorm`, `Linear(6, 1)`, leaky ReLU slope 0.1
  (`code/models/MOViDA.py:99-107,351-360`).
- Other structure: none. The paper's synergy variant (Siamese drug
  branch, 16 units per term, focal loss) is not in the tree.
- Defined at: `code/models/MOViDA.py:36-107,297-375`.

## Connectivity mechanism

Term to term: one `nn.Linear` per term over the concatenation of its
children's outputs, looked up by name in `term_children_map` and a dict
of outputs (`code/models/MOViDA.py:54-59,316-330`). An edge is a 6 x 6
block of the parent's weight; absent edges have no parameter. Gene
blocks and enrichment scores likewise enter only their own term's layer
(and, for scores, the parent's).

Gene to term: each gene block holds a full weight (annotated genes x
all genes of the type) (`code/models/MOViDA.py:48`).
`create_term_mask` builds a 0/1 matrix with one 1 per row
(`code/models/MOViDA.py:110-122`). At the start of training
(`:183`), `initialize_model` multiplies each gene-block weight by its
mask and by 0.1, and every other parameter by 0.1 (`:135-140`);
gene-block biases are not masked. `reset_grad_mask` is meant to mask
gene-block gradients (`:143-147`), but it selects names containing
`'_gene_layer.weight.weight'`, which no parameter has (they end in
`_gene_layer.weight`), and it runs after `optimizer.step()`
(`:201-207`). The forward pass does not use the mask. Absent
gene-block entries are therefore zero only at initialization and get
gradients and Adam updates from the first step. Initialization is
PyTorch's default per layer (fan-in of the full weight) times 0.1.

## Interpretation

Paper only; no code in the repository or the Zenodo file list.

- RIS (Methods 2.4), per drug-cell line pair: *p* intact prediction,
  *p_f* with one term silenced (weights and biases set to zero; for
  leaf terms the inputs), *p_cf* with all its children silenced;
  *s_f* = |*p_f* − *p*|, *s_cf* = |*p_cf* − *p*|,
  RIS = (*s_cf* − *s_f*) / (*s_cf* + *s_f*). Aggregated by drug or by
  cell line type.
- DeepLIFT (Captum) on drug features; top 20 ranked by variation
  across cell lines. Baseline not stated.
- Forward deletes per-term outputs after each call
  (`code/models/MOViDA.py:336`); nothing maps gene-block units back to
  gene names.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N01 Read a DAG from an edge table | `code/param_data.py:114-137` | Parent-child pairs from `ont_tree.txt`; one root and one component required. |
| N06 Connect features to nodes by membership | `code/param_data.py:139-152`; `code/models/MOViDA.py:45-48,110-122` | Genes join GO terms through a term-gene table matched by name; one gene can join many terms. |
| N96 Own feature list and membership table per data type | `code/param_data.py:45,48,72-73`; `code/main.py:61,70`; `code/models/MOViDA.py:80-82,321-323` | Mutation, amplification, and deletion each bring their own gene list and term-gene table into the same terms. |
| N45 Feed inputs into named graph nodes | `code/param_data.py:101-112`; `code/models/MOViDA.py:311-314,325` | One enrichment score per term, matched by column name, is an extra input of that term's layer and its parent's. |
| N26 Unlayered DAG in topological order | `code/models/MOViDA.py:61-87,308-335` | Terms computed one by one, leaves first; each reads gene inputs and children of any depth. |
| N38 Several units per node | `code/models/MOViDA.py:84,328-334`; `code/main.py:40` | Six units per term; a child's six units all feed the parent's six. |
| N12 Parameters only for present edges | `code/models/MOViDA.py:75-84,318-330` | Term-to-term weights exist only for child-parent pairs. |
| N58 Gather each node's inputs by index | `code/models/MOViDA.py:54-59,316-330` | Each term concatenates its children's outputs by a stored child list into its own weights. |
| N39 Mask weights at start and gradients each step | `code/models/MOViDA.py:110-147,201-207` | Intended for gene-to-term weights; the gradient half never runs (Fragile spots). |
| N11 Standard layers between graph layers | `code/models/MOViDA.py:85,333-334` | `tanh` then `BatchNorm1d` on each term's units. |
| N41 Unconstrained branch on a second input | `code/models/MOViDA.py:90-96,340-348,353` | Dense drug-descriptor encoder concatenated with the root's units. |
| N09 Dense head after the graph layers | `code/models/MOViDA.py:99-103,351-360` | Two dense layers from root plus drug embedding to AUC. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Read edge file; check one root and one component | `code/param_data.py:114-137` | ~24 |
| Term-to-gene-index map per data type | `code/param_data.py:139-152` | ~14 |
| Enrichment table and term-to-column map | `code/param_data.py:101-112` | ~12 |
| One gene block per (term, data type) | `code/models/MOViDA.py:45-48` | ~4 |
| Peel leaves into groups; size each term's layer | `code/models/MOViDA.py:51-87` | ~37 |
| One-hot masks; mask at init; gradient mask | `code/models/MOViDA.py:110-147` | ~38 |
| Per-term concatenation of children, genes, scores | `code/models/MOViDA.py:305-337` | ~33 |

## Fragile spots

- `reset_grad_mask` matches no parameter (`code/models/MOViDA.py:145`
  vs names at `:48`) and runs after `optimizer.step()` (`:202,207`);
  gene blocks become dense over their data type after the first step.
- Mask lookup takes the first two `_`-separated parts of every
  parameter name as term and data type (`code/models/MOViDA.py:136`);
  an underscore in either name breaks it.
- `construct_VNN_graph` empties the graph it is given
  (`code/models/MOViDA.py:87`), which is `data_mv.hierarchy_graph`
  (`:32`); a second model from the same data object has no terms.
- The graph output is the last term computed
  (`code/models/MOViDA.py:337`), the root only because the loader
  enforces one root (`code/param_data.py:130-132`); `self.root` is
  never used (`code/models/MOViDA.py:20`). A cycle ends the peeling
  early (`:64-65`) and drops its terms and their ancestors silently.
- Gene-block rows follow Python set order (`code/models/MOViDA.py:118`);
  no gene name is kept per unit.

## Out of scope

- Loss: weighted MSE, inverse AUC-bin frequency weights
  (`code/losses.py:4-8`, `code/param_data.py:154-165`); metric:
  per-bin mean MSE (`code/losses.py:13-17`).
- Training loop: Adam, weighted sampler, best model by validation loss
  (`code/models/MOViDA.py:125-240`).
- Data loading and drug descriptors (`code/param_data.py:32-99`);
  experiment folders (`code/prepare_directories.py`).
- Enrichment-score computation (yaGST, R): not in the repository.

## Paper vs code

- Paper: k + 1 units per term, k = 6. Code: 6 units per term; each
  child also passes its enrichment score, so a parent gets 7 values
  per child. Whether that is the "+1" is unclear.
- Equation 1 sums separately weighted inputs; the code uses one
  `nn.Linear` over their concatenation.
- Paper: learning rate 1e-5, 300 epochs. `code/main.py:34-36`: 2
  epochs, 1e-3, batch 30000; training file = test file
  (`code/main.py:62,64`).
- README describes command-line dictionaries; `main.py` hard-codes
  parameters and reads no arguments (`code/main.py:32-71,119-125`);
  `test()` calls `model_test` with too few arguments (`:111`).
- Synergy model, RIS, and DeepLIFT: paper only.
- The paper does not say that gene-to-term weights are masked only at
  initialization.

## Key excerpts

`code/models/MOViDA.py:45-48`

```python
    def contruct_VNN_input_layer(self):
        for idx in range(len(self.mo_names)):
            for term, gene_set in self.mo_gene2term[idx].items():
                self.add_module(term + '_' + self.mo_names[idx] + '_gene_layer', nn.Linear(self.genes_dim[idx], len(gene_set)))
```

`code/models/MOViDA.py:135-147`

```python
        for name, param in self.named_parameters():
            term_name, mo_name = name.split('_')[:2]
            if '_gene_layer.weight' in name:
                param.data = torch.mul(param.data, self.mo_term_masks[mo_name][term_name]) * 0.1
            else:
                param.data = param.data * 0.1


    def reset_grad_mask(self):
        for name, param in self.named_parameters():
            if '_gene_layer.weight.weight' in name:
                term_name, mo_name = name.split('_')[:2]
                param.grad.data = torch.mul(param.grad.data, self.mo_term_masks[mo_name][term_name])
```

`code/models/MOViDA.py:316-334`

```python
                child_input_list = []

                for child in children_term:
                    child_input_list.append(self.term_VNN_out[child])

                for mo in self.mo_names:
                    if term in self.term_gene_out_mo[mo]:
                        child_input_list.append(self.term_gene_out_mo[mo][term])

                child_input_list.append(path_act_term)
                del path_act_term

                child_input_list = torch.cat(child_input_list,1)

                term_NN_out = self._modules[term+'_linear_layer'](child_input_list)
                del child_input_list

                term_NN_out = torch.tanh(term_NN_out)
                term_NN_out = self._modules[term+'_batchnorm_layer'](term_NN_out)
```

## Open questions

- GO release, evidence codes, how the 2086 terms were chosen, edge
  count, and whether the three term-gene tables differ beyond their
  gene lists: not in the Methods as read or in the code; the files are
  in Zenodo `data.zip`, not read.
- Whether "k + 1 units" means six units plus the forwarded score.
- RIS and DeepLIFT code: not in the repository (searched `captum`,
  `deeplift`, `silenc`, `ablat`, `attribut`) or the Zenodo file list;
  how terms are silenced in code and the DeepLIFT baseline are unknown.
- Synergy model code (`MOViDA_synergy.py` in the README): not at this
  commit.
- Whether published results used code in which the gradient mask took
  effect: not determined; only the tip of `main` (2023-01-25, before
  the July 2023 publication) was fetched.
