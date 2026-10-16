# cancer-dependencies-2021: BioVNN

| Field | Value |
|-------|-------|
| Paper | Lin and Lichtarge, 2021. Using interpretable deep learning to model cancer dependencies. Bioinformatics 37(17):2675–2681. 10.1093/bioinformatics/btab137 |
| Code | `https://github.com/LichtargeLab/BioVNN` at `73c1828000669b0714601ba411b9f6c9f361baa8` |
| Other code | none |
| Framework | Python 3.6.5, PyTorch 1.2.0, numpy 1.19.1, pandas 0.22.0 (`environment.yml`) |
| License | MIT (`LICENSE`) |
| Full text | PubMed Central PMC8428607 (main text; Supplementary Methods not read) |
| Extracted | 2026-10-16, Claude Code, claude-opus-5-5 |

## Summary

BioVNN predicts whether a cancer cell line depends on a gene
(binarized DepMap CRISPR score) from its RNA expression. Each row is
one (cell line, target gene) pair: expression masked to genes near the
target in the pathway hierarchy, plus a one-hot vector naming the
target; one network serves all 683 targets. The hidden part follows
the Reactome hierarchy: each pathway is a block of units fed by its
member genes (leaf pathways) or its child pathways' blocks, up to an
added root. Every pathway has a sigmoid head; the root's head predicts
and the others add a weighted auxiliary loss.

## Prior-knowledge graph

- Source: Reactome gene sets (`ReactomePathways.gmt`) and the
  parent–child table (`ReactomePathwaysRelation.txt`), human rows only
  (`src/dependency.py:398-402`); downloaded 2019-05-29 per the paper,
  release not given. GO is an option (`src/dependency.py:107-112`).
  Data files come from a lab server via `install.sh`, not git.
- Filtering: members cut to modeled genes plus targets; pathways with
  fewer than 5 members or no target gene are dropped
  (`src/dependency.py:598-622`); parent–child rows kept only when both
  ends survive (`src/dependency.py:409`).
- Size: paper reports 1,425 pathways including the root, 13 levels,
  9,501 genes. One added `root` is the parent of every pathway with no
  parent in the unfiltered table (`src/dependency.py:403-410`). Edges
  are gene→pathway memberships and child→parent links; count not
  computed. Directed, unsigned, acyclic; a pathway may have several
  parents.
- Levels: leaf pathways are level 1; parents of the current set get
  the next level, later assignments overwrite earlier ones, so level
  = longest path down to a leaf pathway
  (`src/dependency.py:698-727`). Not used to order the forward pass.
- Data to nodes: GMT members and expression columns are matched by
  gene symbol (`src/dependency.py:494-500`). Features are genes in any
  gene set, with value ≥ 1 (log2(TPM+1)) in > 1% of lines, top 10,000
  by MAD (`src/dependency.py:571-596`); paper: 9,488. Targets are
  DGIdb genes in a gene set with ≥ 6 positive and ≥ 6 negative lines
  (`src/dependency.py:267-273,541-569`). A member gene is wired to its
  expression column and, if a target, to its one-hot column
  (`src/dependency.py:1033-1050`).
- Built at: `src/dependency.py:350-413` (read), `598-622` (filter),
  `698-958` (index map, child lists, levels, controls, input masks).

## Architecture

- Order: no layer loop. Repeated sweeps build a node once all its
  child pathways are built (`src/pytorch_layer.py:249-292`); forward
  runs that `build_order` (`src/pytorch_layer.py:307`), so a parent
  reads children of any level. The paper calls this 13 hidden layers.
- Inputs: one `(batch, 1)` tensor per column
  (`src/biovnn_model.py:170`): expression columns, one-hot target
  columns, then one slot per pathway (`src/pytorch_layer.py:304`;
  order `src/dependency.py:770-774`).
- Units per node: `max(10, int(0.2 × children))`; children are child
  pathways for an inner pathway and member input columns for a leaf
  (`src/pytorch_layer.py:266-277`, `src/biovnn_model.py:71-72`).
- Inputs reach leaf pathways only: a pathway with a child pathway drops
  its own member genes (`src/pytorch_layer.py:266-269,313-314`; flag
  default `src/biovnn_model.py:70`).
- Per-node block: Linear → Mish → BatchNorm1d → Dropout(0.5)
  (`src/pytorch_layer.py:286-289,317-325`).
- Inner outputs: every node has a head Linear(units, 1) + sigmoid
  (`src/pytorch_layer.py:290,326-330`). Loss = class-weighted BCE of
  the root head + 0.3 × summed BCE of all other heads, same label, not
  averaged (`src/biovnn_model.py:65-66,194-209`).
- Not graph-constrained: the per-node heads.
- Targets as rows: dataset length is lines × targets; each item
  appends a one-hot target vector (`src/pytorch_layer.py:461,481-503`)
  and multiplies expression by that target's 0/1 mask
  (`src/pytorch_layer.py:487-489,516-517`). The mask keeps the target
  plus, for each kept pathway containing it (smallest first), up to
  100 new members by MAD (`src/dependency.py:631-685,926-943`).
- Defined at: `src/pytorch_layer.py:193-333`.

## Connectivity mechanism

One `nn.Linear(total_in, units)` per pathway, `total_in` = its own
input columns plus its child pathways' unit counts
(`src/pytorch_layer.py:285-286`). Forward concatenates exactly the
tensors listed in `self.child_map[i]` and applies that Linear
(`src/pytorch_layer.py:312-317`). No full weight matrix exists; absent
edges have no parameter and no gradient. A child-pathway edge is a
block (child units × parent units). Initialization is PyTorch's
default for `nn.Linear`, scaled by each node's own fan-in; no explicit
init call exists. The paper says Kaiming.

## Interpretation

- Node states: in prediction with `run_mode: ref`, every node's
  post-Mish, pre-BatchNorm units (`src/pytorch_layer.py:319-320,333`)
  are written per batch to HDF5 keyed by node position
  (`src/biovnn_model.py:229-233`) and reloaded by position
  (`src/biovnn_model.py:486-497`). Names come from `gene_group_idx`
  (`src/dependency.py:787-793,1769-1777`). Rows are (line, target)
  pairs; one target's rows are selected by reshape
  (`src/dependency.py:1782-1786`).
- Unused helpers: a 2-component PCA plot of one pathway's states
  (`src/dependency.py:1825-1846`), never called; `calculate_RLIPP`
  (`src/dependency.py:1746-1800`), never called and calling an
  undefined `self._RLIPP`.
- Paper: PC1/PC2 of pathway states compared between dependent and
  non-dependent lines by two-sided Mann–Whitney–Wilcoxon with
  Benjamini–Hochberg; gene importance "as measured by feature
  weights". Code for both: not found; `self.analyze_weights(...)` is
  commented out and undefined (`src/dependency.py:1394`).

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N01 Read a DAG from an edge table | `src/dependency.py:398-413` | Reactome parent–child table, human only, cut to kept pathways. |
| N73 Add a root above top-level nodes | `src/dependency.py:403-410` | `root` parents every pathway without a parent; its head predicts. |
| N06 Connect features to nodes by membership | `src/dependency.py:350-379,571-578,1033-1050` | GMT sets matched by symbol; genes in no set are not features. |
| N05 Filter nodes by gene-set size | `src/dependency.py:610-613` | Drops pathways with fewer than 5 members. |
| N145 Keep only nodes that contain a target | `src/dependency.py:614-616` | Drops pathways containing no predicted target gene. |
| N132 Features join only childless nodes | `src/pytorch_layer.py:266-269,313-314` | Pathways with child pathways ignore their own genes. |
| N26 Unlayered DAG in topological order | `src/pytorch_layer.py:254-292,307-317` | Nodes run in build order; parents read children of any level. |
| N74 Layer nodes by height above the data | `src/dependency.py:698-727,891-892,1021-1024` | Levels drive the level-matched random control and the dense baseline, not the forward order. |
| N67 Units per node set by degree | `src/pytorch_layer.py:266-277` | `max(10, int(0.2 × children))` units per pathway. |
| N58 Gather each node's inputs by index | `src/pytorch_layer.py:312-317`, `src/biovnn_model.py:170` | Each node concatenates its children's tensors by stored index. |
| N12 Parameters only for present edges | `src/pytorch_layer.py:285-286` | One Linear per node sized to its own children. |
| N70 Initialize from each unit's present fan-in | `src/pytorch_layer.py:286` | Default per-node Linear init follows that node's fan-in. |
| N11 Standard layers between graph layers | `src/pytorch_layer.py:287-289,319-325` | BatchNorm and dropout after every pathway block. |
| N40 Auxiliary output head on every node | `src/pytorch_layer.py:290,326-330`, `src/biovnn_model.py:194-209` | Non-root heads add 0.3 × summed BCE. |
| N07 One input per entity and data type | `src/dependency.py:770-774,787-788,1033-1050` | A target gene has an expression and a one-hot column, wired to the same pathways. |
| N143 Targets as rows with a one-hot target input | `src/pytorch_layer.py:461,481-503` | Rows are (line, target) pairs; one output serves all targets. |
| N144 Per-target input mask from the graph | `src/dependency.py:631-685,926-943`, `src/pytorch_layer.py:487-489,516-517` | Expression outside the target's pathways is zeroed for its rows. |
| N35 Randomized prior as a control | `src/dependency.py:813-818,923-924,960-1031` | Redraws every membership and every child link (same level), rebuilds input masks. |
| N115 Dense twin of the graph layer | `src/pytorch_layer.py:336-446` | 13 equal-width dense layers with per-layer heads. |
| N31 Read out node activations | `src/biovnn_model.py:229-233,486-497` | Saves every pathway's states per row. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Parse gene sets; read parent–child table; add root | `src/dependency.py:350-413` | ~50 |
| Cut gene sets to modeled genes; filter pathways | `src/dependency.py:598-622` | ~25 |
| Per-target input feature sets and 0/1 masks | `src/dependency.py:631-696,926-958` | ~100 |
| Levels by height | `src/dependency.py:698-727` | ~30 |
| Index map and child lists (real, random, dense) | `src/dependency.py:763-917,1033-1050` | ~170 |
| Membership and hierarchy shuffles for the control | `src/dependency.py:960-1031` | ~70 |
| One Linear per node in topological order | `src/pytorch_layer.py:249-293` | ~45 |
| Column split, gather by index, mask and one-hot | `src/biovnn_model.py:170`, `src/pytorch_layer.py:304-317,484-517` | ~45 |
| Save and reload node states by position | `src/biovnn_model.py:229-233,486-497` | ~17 |

## Fragile spots

- Root links are chosen before filtering (`src/dependency.py:404-410`).
  A kept pathway whose parents were all dropped would get no parent
  and feed only its own head; nothing checks this.
- An expression or one-hot column whose gene sits in no kept leaf
  pathway stays in the input but reaches no node
  (`src/pytorch_layer.py:266-269,313-314`); nothing reports it.
- The build loop removes from `neuron_to_build` while iterating it
  (`src/pytorch_layer.py:255,291`) and has no cycle check.
- Node `i` lives at `features[i + input_dim]`
  (`src/pytorch_layer.py:308`); `input_dim` is recomputed from data
  shapes (`src/biovnn_model.py:423-426`) and must match the index map
  (`src/dependency.py:770-774`).
- Seeds set numpy only (`src/biovnn_model.py:418`,
  `src/dependency.py:1176`); no `torch.manual_seed`, so weight init
  and dropout are not fixed by the seed.
- With several data types the random control's masks use the real
  gene set (`feat_genes`, `src/dependency.py:947`). Paper uses RNA only.
- The dense baseline's width uses hard-coded constants
  (`src/pytorch_layer.py:356`), not the built graph.

## Out of scope

- Class-ratio loss weights (`src/dependency.py:1211-1235`).
- Ranger optimizer, lr 1e-3, weight decay 1 on all parameters
  (`src/biovnn_model.py:341-351`, `src/pytorch_layer.py:31-190`).
- Training loop, early stopping (patience 2), checkpoints
  (`src/biovnn_model.py:413-460`).
- DepMap loading, per-gene z-scoring on training statistics, splits by
  tissue (`src/dependency.py:479-520,1078-1126,1279-1328`).
- Time-stamped and prospective runs; metrics; plots.

## Paper vs code

- Units: the paper scales with pathway size (about 0.3 per gene, floor
  10 below 34 genes). The code uses ratio 0.2 (`src/biovnn_model.py:71`,
  not set in `params/biovnn.yml`) on the child count, which for an
  inner pathway counts child pathways, not genes.
- Per-target inputs: the paper adds genes from the smallest to the
  largest pathway until 100 genes. The code adds up to 100 genes per
  pathway by MAD, overall cap 99,999 (`src/dependency.py:177-178,
  675-685`).
- Feature-weight importance and Mann–Whitney tests: in the paper, not
  in the code.

## Key excerpts

`src/pytorch_layer.py:266-277`

```python
                if self.only_combine_child_gene_group and len(child_com) > 0:
                    children_n = len(child_com)
                    child_feat = []
                    self.only_combine_gene_group_dict[neuron_name] = 1
                else:
                    children_n = len(children)
                    if j == len(self.child_map) - 1:
                        children_n += self.input_dim
                logging.debug("Building gene group {} with {} children".format(j, len(children)))
                if i not in neuron_n_dict:
                    neuron_n = np.max([self.neuron_min, int(children_n * self.neuron_ratio)])
                    neuron_n_dict[i] = neuron_n
```

`src/pytorch_layer.py:312-321`

```python
            children = self.child_map[i]
            if neuron_name in self.only_combine_gene_group_dict:
                children = [z for z in children if z >= self.input_dim]
            input_list = [features[z] for z in children]
            input_mat = torch.cat(input_list, axis=1)
            features[j] = com_layer(input_mat)
            ## BN after activation
            state = self.act_func(features[j])
            states[i] = state
            features[j] = bn_layer(state)
```

`src/pytorch_layer.py:497-500`

```python
        if self.use_deletion_vector:
            deletion_array = self.deletion_array.clone()
            deletion_array[j] = 1
            X_input = torch.cat([feat, deletion_array])
```

## Open questions

- Which weights give gene importance, and how they are reduced: not in
  the repository; the paper points to Supplementary Methods (not read).
- Mann–Whitney on node-state PCs: not in the repository.
- Edge count, and whether any kept pathway ends up without a parent:
  depend on data files, not downloaded.
- Reactome release: not stated in paper or code.
