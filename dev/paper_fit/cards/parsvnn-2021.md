# parsvnn-2021: ParsVNN

| Field | Value |
|-------|-------|
| Paper | Huang X et al., 2021. ParsVNN: parsimony visible neural networks for uncovering cancer-specific and drug-sensitive genes and pathways. NAR Genomics and Bioinformatics 3(4):lqab097. 10.1093/nargab/lqab097 |
| Code | `https://github.com/EJIUB/ParsVNN` at `99ebb22ff1d900c73c1c2a929ad154f7064549d3` (commit dated 2022-11-03) |
| Other code | None found. File names and the paper say the model is DrugCell; the DrugCell repository was not compared. |
| Framework | Python, PyTorch, networkx; no versions pinned (`__pycache__/*.cpython-38.pyc` suggests Python 3.8) |
| License | None found |
| Full text | Europe PMC full-text XML (PMC8557386); Algorithm 1 is not in the text and was not read |
| Extracted | 2026-10-16, Claude Code, claude-opus-5-5 |

## Summary

ParsVNN predicts drug response (AUC) of cell line–drug pairs within
one cancer type (CTRP v2, GDSC). It starts from the DrugCell network:
a GO biological-process hierarchy whose terms are groups of hidden
units fed by binary gene mutation vectors, plus a dense branch on drug
Morgan fingerprints. During training it removes gene→term edges with
an ℓ0 penalty and term→term edges with a group-lasso penalty, both as
proximal steps (PALM), alternating with fine-tuning in which zeroed
weights stay zero. The genes and GO terms left in the pruned network
are the paper's interpretation.

## Prior-knowledge graph

- Source: GO biological process as prepared for DrugCell; shipped as
  `data/go_bp_drugcell_min10_merge30_depth5_ontology.txt` with three
  columns: parent term, child term or gene symbol, type (`default`
  for term→term, `gene` for term→gene). No filtering in code beyond
  name matching.
- Size (counted from the shipped file): 2,086 terms, 3,167 term→term
  edges, 59,753 term→gene rows over 3,008 genes, one root
  `GO:0008150`. Directed, unsigned, acyclic; 840 terms have several
  parents; longest root-to-leaf path is 5 edges. Genes are annotated
  only to the most specific terms (no gene row repeats on a
  descendant); 18 terms have no gene row. The paper's counts differ
  (Paper vs code).
- Data to nodes: gene symbols are matched to `gene2id` names; unknown
  genes are skipped (`util.py:34-35`). Genes join terms at any depth
  (`util.py:37-40`). The input is the full mutation vector over all
  `gene2id` genes (`util.py:148-156`). A term whose subtree has no
  matched gene, more than one root, or more than one connected
  component stops the program (`util.py:48-84`).
- Built at: `util.py:17-86`, `drugcell_NN.py:91-131`.

## Architecture

- Layers: repeatedly take all terms with no child term and remove
  them (`drugcell_NN.py:102-131`); six layers in the shipped file.
  Terms are computed layer by layer in that order; a term reads all
  its child terms, whatever their layer, plus its directly annotated
  genes (`drugcell_NN.py:151-165`).
- Units per node: `num_hiddens_genotype` units for every term
  (`drugcell_NN.py:50-60`), 6 in the run script
  (`commandline_cuda.sh:20`). `term_size_map` is printed but does not
  set the width. A gene is one input unit.
- Gene input per term: each term has a private bias-free layer from
  all genes to one unit per directly annotated gene
  (`drugcell_NN.py:64-72`, `:142-143`), so each gene→term edge owns
  one scalar. The term's bias-free linear layer reads
  `[child units ..., gene copies]` densely into its k units, then
  tanh and batch norm (`drugcell_NN.py:114-127`, `:163-168`).
- Connections that skip layers: a term reads child terms of any lower
  layer and genes directly (`drugcell_NN.py:157-161`).
- Outputs or losses at inner layers: every term and every drug layer
  has a head Linear(k,1) → tanh → Linear(1,1)
  (`drugcell_NN.py:83-84`, `:128-129`, `:169-170`, `:181-182`). Loss
  = MSE(final) + 0.2 × MSE of every head against the same label
  (`train_drugcell_prune.py:382-388`). Heads do not feed the output.
- Parts the graph does not constrain: drug MLP (`100,50,6`, tanh then
  batch norm, `drugcell_NN.py:76-86`, `:175-182`); final head on the
  root's units concatenated with the drug output: Linear → tanh →
  batch norm → Linear(.,1) → tanh → Linear(1,1)
  (`drugcell_NN.py:43-47`, `:185-191`).
- Other structure: none. Defined at `drugcell_NN.py:13-196`.

## Connectivity mechanism

- Term→term: no parameter for an absent edge. Each term owns an
  `nn.Linear` over its children's outputs, looked up by name
  (`drugcell_NN.py:114-126`, `:157-165`). Child `i`'s k×k block is
  columns `i*k:(i+1)*k`, in `term_neighbor_map` order
  (`drugcell_NN.py:96-100`).
- Gene→term: each gene layer is a full (direct genes × all genes)
  weight (`drugcell_NN.py:72`); the forward pass uses no mask.
  Pretraining multiplies it by a 0/1 mask (one 1 per row) once and
  scales all parameters by 0.1 (`train_drugcell.py:142-151`, mask at
  `:17-32`). After every backward pass the gene-layer gradients are
  multiplied by the mask, in pretraining (`train_drugcell.py:191-196`)
  and pruning (`train_drugcell_prune.py:392-397`).
- Pruning, `train_drugcell_prune.py`: each outer epoch runs 10
  pruning epochs, then 10 fine-tuning epochs (`:354-357`, `:449`).
  - Pruning step per batch, no optimizer (`:390-403`; `:57-99`):
    gene layers take a gradient step of size `lip`, then entries with
    |w| < √(2·`reg_l0`·`lip`) become zero (ℓ0 proximal, `:59-66`,
    `:36-41`). Each child block of a term layer takes a gradient step
    of size `lip`, then its Frobenius norm is shrunk by
    `reg_glasso`·`lip`, the whole block becoming zero below it
    (group-lasso proximal, `:67-82`, `:44-50`). Gene-copy columns and
    all other parameters take a gradient step of size `lr` and an L2
    shrink (`:83-92`). Values `reg_l0=0.001, reg_glasso=0.5,
    reg_decay=0.001, lr=lip=0.001` (`:400`); the `_st` variant uses
    `reg_l0=0.0001, reg_glasso=0.1` (`train_drugcell_prune_st.py:443`).
  - Fine-tuning: new Adam (`:444`). Before each fine-tuning epoch,
    every gene-layer and term-layer weight gets a gradient hook
    multiplying its gradient by (weight ≠ 0) (`:457-472`, `:264-266`);
    Adam steps (`:509-516`); hooks are removed after the epoch
    (`:527-528`). Zeroed entries, off-prior gene entries included,
    stay zero until the next pruning phase (see Fragile spots).

## Interpretation

No attribution method in the repository (grep for gradient, SHAP,
DeepLIFT, integrated gradients). The interpretation is the pruned
structure:

- Remaining terms: a term→child edge is removed when its block is
  zero; the remaining graph is cut to terms connected to the root,
  undirected in `train_drugcell_prune.py:102-123`, along directed
  simple paths from the root in `train_drugcell_prune_st.py:108-137`.
  Terms are reported by GO ID. Gene→term edges do not enter this.
- Remaining genes: per remaining term, gene-layer columns with a
  nonzero row sum are kept, pooled over terms and deduplicated
  (`train_drugcell_prune_st.py:283-304`, `:584-589`). They stay
  integer indices; mapping to names was not found.
- Counts of kept terms, edges, genes, and overlap with a hard-coded
  driver-gene index list go into the checkpoint name
  (`train_drugcell_prune_st.py:335-340`, `:582-593`).
- Node activations: `predict_drugcell.py:69-72` appends each term's
  unit outputs to `<term>.hidden`. Nothing combines samples or runs.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N01 Read a DAG from an edge table | `util.py:27-44` | One table holds term→term and term→gene rows, told apart by a type column. |
| N06 Connect features to nodes by membership | `util.py:34-40` | Genes join terms by name; rows with unknown genes are skipped. |
| N09 Dense head after the graph layers | `drugcell_NN.py:43-47`, `:185-191` | Dense layers map root plus drug units to the response. |
| N11 Standard layers between graph layers | `drugcell_NN.py:127`, `:167-168` | Batch norm after each term's tanh. |
| N12 Parameters only for present edges | `drugcell_NN.py:114-126`, `:157-165` | A term's layer reads only its children's units. |
| N26 Unlayered DAG in topological order | `drugcell_NN.py:151-170` | Terms are computed one by one, each reading genes and child terms of any depth. |
| N31 Read out node activations | `predict_drugcell.py:69-72` | Each term's unit outputs go to a file named after the term. |
| N38 Several units per node | `drugcell_NN.py:50-60`, `commandline_cuda.sh:20` | Every term has 6 units; a term→term edge is a 6×6 block. |
| N39 Mask weights at start and gradients each step | `train_drugcell.py:142-151`, `:191-196`; `train_drugcell_prune.py:392-397` | Gene→term weights are full matrices masked at init and on every gradient. |
| N40 Auxiliary output head on every node | `drugcell_NN.py:128-129`, `:169-170`; `train_drugcell_prune.py:382-388` | Every term has a head with loss weight 0.2. |
| N41 Unconstrained branch on a second input | `drugcell_NN.py:76-86`, `:175-185` | Drug fingerprints pass through a dense MLP joined with the root's units. |
| N74 Layer nodes by height above the data | `drugcell_NN.py:102-131` | Layers come from repeatedly removing childless terms. |
| N136 Prune prior edges by a proximal sparsity step | `train_drugcell_prune.py:36-50`, `:57-92`, `:400` | ℓ0 hard thresholding on gene→term scalars and group-lasso shrinkage on term→term 6×6 blocks after each gradient step. |
| N137 Freeze pruned weights while fine-tuning | `train_drugcell_prune.py:444-528` | Between pruning phases, Adam retrains with hooks masking zeroed entries. |
| N138 Remaining subnetwork after pruning | `train_drugcell_prune_st.py:108-137`, `:283-304`, `:584-589` | Kept terms, edges, and genes are read from nonzero blocks and reachability from the root. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Read ontology table, map genes to ids, check root and component | `util.py:17-86` | ~70 |
| Layer terms by removing childless terms; size each term's input | `drugcell_NN.py:91-131` | ~40 |
| Per-term gene layer; child lookup and concatenation in forward | `drugcell_NN.py:64-72`, `:140-170` | ~40 |
| Build per-term gene masks; mask gene-layer gradients | `train_drugcell_prune.py:18-33`, `:392-397` | ~20 |
| Proximal operators and PALM update, sliced per child block | `train_drugcell_prune.py:35-99` | ~65 |
| Gradient hooks freezing zero weights in fine-tuning | `train_drugcell_prune.py:457-472`, `:526-528` | ~20 |
| Pruned graph from zero blocks; kept genes from nonzero columns | `train_drugcell_prune_st.py:108-137`, `:283-304` | ~50 |

## Fragile spots

- The gene-layer mask is never applied in the forward pass; absent
  gene→term entries are absent only because the loaded checkpoint
  holds zeros there and their gradients are masked
  (`drugcell_NN.py:143`, `train_drugcell_prune.py:326-328`,
  `:392-397`).
- Child blocks are found by `i*dim` column arithmetic with one `dim`
  for all terms (`train_drugcell_prune.py:69-88`); this matches the
  forward concatenation (`drugcell_NN.py:157-161`) only because every
  term has the same width (`drugcell_NN.py:55`).
- Term names come from `name.split('_')[0]`
  (`train_drugcell_prune.py:70`, `:395`), assuming no underscore in
  term IDs.
- A pruned term is still computed and its auxiliary loss still counts
  (`drugcell_NN.py:151-170`). Term-layer gradients are not masked in
  pruning phases, so a zeroed block returns if its gradient step
  exceeds the threshold (`train_drugcell_prune.py:74-79`).
- The two pruning scripts count remaining terms differently
  (`train_drugcell_prune.py:117` vs `train_drugcell_prune_st.py:124-131`).
- Checkpoint formats disagree: `train_drugcell.py:233` saves the whole
  module, `train_drugcell_learn.py:142-143` saves
  `{'model', 'optimizer'}`, `train_drugcell_prune.py:328` passes
  `torch.load(...)` to `load_state_dict`. `commandline_cuda.sh:20`
  passes no `-pretrained_model`.
- `train_drugcell.py:74` contains `i*dim.:`, which Python does not
  parse.

## Out of scope

- Loss, training loops, batching by cell and drug index
  (`util.py:89-156`), Pearson correlation named `spearman_corr`
  (`util.py:10-14`), checkpoint selection by test correlation.
- Adam-only and mixed-precision DrugCell training variants.
- Baselines and downstream analyses (not in the repository).

## Paper vs code

- Graph size: paper 3,008 GO terms, 2,086 genes, 3,176 term–term and
  19,744 term–gene connections; shipped file 2,086 terms, 3,008
  genes, 3,167 term–term and 59,753 term–gene rows.
- Data: the paper models STAD, BRCA, PAAD, kidney, LIHC; the
  repository ships LUNG files and writes `lung` into checkpoint names
  (`train_drugcell_prune.py:538`).
- Step size: Algorithm 1 estimates a Lipschitz constant (paper text);
  the code uses a fixed `lip=0.001` (`train_drugcell_prune.py:400`).
- Thresholds: paper (8b) 2η/dᵏ; `train_drugcell_prune.py:77`
  `reg_glasso`·`lip`; `train_drugcell.py:43-49`, `:73` `reg_glasso`/2.
  ℓ0: `train_drugcell_prune.py:38` vs `train_drugcell.py:37` differ.
- Fine-tuning with frozen zeros (`train_drugcell_prune.py:444-528`)
  is not in the methods text read.
- Paper: hyperparameters by 5-fold cross-validation; code: fixed
  values, no cross-validation.
- The "ParsVNN (random)" control was not found in the code.

## Key excerpts

`train_drugcell_prune.py:36-50`

```python
def proximal_l0(yvec, c):
    yvec_abs =  torch.abs(yvec)
    csqrt = torch.sqrt(2*c)
    
    xvec = (yvec_abs>=csqrt)*yvec
    return xvec

# solution for 1/2||x-y||^2_2 + c||x||_g
def proximal_glasso_nonoverlap(yvec, c):
    ynorm = torch.norm(yvec, p='fro')
    if ynorm > c:
        xvec = (yvec/ynorm)*(ynorm-c)
    else:
        xvec = torch.zeros_like(yvec)
    return xvec
```

`train_drugcell_prune.py:69-82`

```python
            dim = model.num_hiddens_genotype
            term_name = name.split('_')[0]
            child = model.term_neighbor_map[term_name]
            for i in range(len(child)):
                #dim = model.num_hiddens_genotype
                term_input = param.data[:,i*dim:(i+1)*dim]
                term_input_grad = param.grad.data[:,i*dim:(i+1)*dim]
                term_input_tmp = term_input - lip*term_input_grad
                term_input_update = proximal_glasso_nonoverlap(term_input_tmp, reg_glasso*lip)
                #print("%s child %d: before norm is %f, after #0 is %f, threshold %f" %(name, i, torch.norm(term_input, p='fro'), torch.norm(term_input_update, p='fro'), reg_glasso*lip))
                param.data[:,i*dim:(i+1)*dim] = term_input_update
                num_n0 =  len(torch.nonzero(term_input_update, as_tuple =False))
                if num_n0 == 0 :
                    dG_prune.remove_edge(term_name, child[i])
```

`train_drugcell_prune.py:460-464`

```python
                    if "direct" in name:
                        # mutation side
                        # l0 for direct edge from gene to term
                        mask = torch.where(param.data.detach()!=0, torch.ones_like(param.data.detach()), torch.zeros_like(param.data.detach()))
                        handle = param.register_hook(lambda grad, mask=mask: grad_hook_masking(grad, mask))
```

## Open questions

- Which script produced the pretrained model that pruning loads:
  `train_drugcell.py` (itself PALM with small penalties, `:198`),
  `train_drugcell_learn.py` (Adam), or DrugCell's own code. No script
  writes a plain state dict for `train_drugcell_prune.py:328`.
- Which pruning script and which per-cancer λ and η gave the paper's
  models; not found.
- Algorithm 1's update order and step-size estimate; not in the XML.
- How gene indices became the paper's gene names; not found.
- How the random-meaning control was built; not found.
