# dlbcl-survival-2024: VNNSurv

| Field | Value |
|-------|-------|
| Paper | Tan J, Xie J, Huang J, Deng W, Chai H, Yang Y, 2024. An interpretable survival model for diffuse large B-cell lymphoma patients using a biologically informed visible neural network. Computational and Structural Biotechnology Journal. 10.1016/j.csbj.2024.07.019 |
| Code | `https://github.com/jie-tan/vnnsurv` at `6fc63baf0335196ff61cfc1e66b68e0ca6728380` |
| Other code | None found. `vnn/` adapts BioVNN (Lin et al. 2021, "Using interpretable deep learning to model cancer dependencies"; comment `# BioVNN` at `vnn/VNN_cell.py:186`). Web server `https://bio-web1.nscc-gz.cn/app/VNNSurv` (not code). |
| Framework | Python 3.10, PyTorch 2.0.1 (`requirements.txt:6`), pandas; torch-geometric imported but unused (`vnn/VNN_cell.py:6`) |
| License | None found |
| Full text | Europe PMC full-text XML of PMC11357880 (text and captions; figure images not viewed) |
| Extracted | 2026-10-10, Claude Code, claude-opus-5-5 |

## Summary

VNNSurv predicts a survival risk score for diffuse large B-cell
lymphoma patients from binary gene-alteration columns (117, or
the 30 highest-impact genes in a reduced model) plus three binary
clinical columns. The prior is the Reactome gene sets and pathway
hierarchy. Each kept pathway is a node with its own dense layer of
several units that reads the columns of all its member genes and
the units of its child pathways; nodes run child-before-parent up
to an added root. The root's units, concatenated with the clinical
columns, feed a two-layer dense head trained with a Cox partial
likelihood loss. The paper interprets the model with SHAP; that
code is not in the repository.

## Prior-knowledge graph

- Source: Reactome `vnn/ReactomePathways.gmt` (gene sets) and
  `vnn/ReactomePathwaysRelation.txt` (parent, child), shipped in
  the repo; release not stated. Gene sets are propagated: in all
  2611 human parent-child pairs the child's genes are a subset of
  the parent's (my count). Hierarchy rows are kept if the parent
  ID contains `HSA` (`vnn/utils_biovnn.py:498-500`). Pathways with
  fewer than 5 input genes are dropped
  (`vnn/utils_biovnn.py:100-119`); edges are kept only between
  kept pathways (`vnn/utils_biovnn.py:507`). An added `root`
  parents every kept pathway that has no parent in the human
  hierarchy (`vnn/utils_biovnn.py:502-505`, `508`).
- Size (my recount from the shipped files, code not run):
  117-gene model, 101 pathways plus root, 91 pathway edges, 12
  root edges, 1010 gene-to-pathway edges; 30-gene model, 49
  pathways plus root, 37 + 12 edges, 400 gene edges. Directed,
  unsigned, acyclic; a pathway may have several parents.
- Data to nodes: column names such as `MYD88_265` or
  `CDKN2A_OR_del` are cut at the first `_` (`train.py:17-21`) and
  matched to GMT gene symbols (`vnn/utils_biovnn.py:513-529`). A
  gene joins every kept pathway whose set contains it, at every
  depth (`vnn/utils_biovnn.py:308-310`, `378`). Genes in no kept
  pathway stay in the input but no node reads them (26 of 117 by
  my recount; 0 of 30). Duplicate symbols: see Fragile spots.
- Built at: `vnn/utils_biovnn.py:51-75`, `495-511`, `242-378`;
  saved to `vnn/BioVNN_pre.pkl` (`vnn/utils_biovnn.py:548-558`)
  and re-read by the model (`utils.py:26-27`).

## Architecture

- Layers: none in the forward pass. Levels are computed (leaves 1,
  then overwritten on each upward sweep, so longest path from a
  leaf plus 1, `vnn/utils_biovnn.py:162-192`) but only feed a unit
  tally (`vnn/VNN_cell.py:316-320`) and a drawing. Recount,
  117-gene: {1: 46, 2: 29, 3: 13, 4: 7, 5: 4, 6: 2, 7: root};
  30-gene: {1: 25, 2: 14, 3: 5, 4: 4, 5: 1, 6: root}.
- Units per node: max(10, member genes + child pathways)
  (`vnn/VNN_cell.py:299-307`; `neuron_min=10` at `198`; ratio 1
  at `utils.py:22`); root gets `output_dim` (`vnn/VNN_cell.py:300-301`),
  128 in `train.py:48`. Recount: 10 to 52 units per pathway
  (117-gene), 10 to 31 (30-gene).
- Connections that skip layers: implicit. A node reads all units
  of each child pathway, whatever its level, and the column of
  each member gene; 35 of 103 pathway edges span more than one
  level (117-gene recount).
- Outputs or losses at inner layers: none; per-node heads are
  commented out (`vnn/VNN_cell.py:326`, `374-378`).
- Parts the graph does not constrain: head
  `Linear(128 + 3, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 1)`
  (`utils.py:36-40`, `train.py:48-51`) on root units plus the last
  three input columns, which bypass the graph (`utils.py:43-49`).
- Other structure: after each node, Mish, BatchNorm1d on its
  units, Dropout 0.1 (`vnn/VNN_cell.py:323-325`, `366-373`;
  `train.py:49`).
- Defined at: `utils.py:9-52`, `vnn/VNN_cell.py:186-387`.

## Connectivity mechanism

Each pathway owns one `nn.Linear(total_in, units)`, where
`total_in` is its member-gene count plus its children's unit
counts (`vnn/VNN_cell.py:321-322`). The input matrix is split into
one `(batch, 1)` tensor per gene column (`utils.py:45-47`), node
placeholders are appended (`vnn/VNN_cell.py:345`), and each node
concatenates the stored tensors at its child indices and applies
its own layer (`vnn/VNN_cell.py:357-364`). No full weight matrix
exists; absent edges have no parameter and no gradient. A gene
edge is one column of weights; a child edge is a block of child
units by parent units. Order: repeated passes build a node once
all child pathways have layers (`vnn/VNN_cell.py:283-286`). Init:
PyTorch default `nn.Linear`, so the scale follows each node's
present fan-in; `torch.manual_seed(1234)` at `utils.py:12`.

## Interpretation

Not in the repository (no `shap` import; only the pin at
`requirements.txt:4`). Paper Methods: shap 0.41.0 values for each
input and "each pathway in the hidden layer"; explainer,
background set, and how a multi-unit pathway gets one value are
not stated. Feature impact = mean SHAP over patients carrying the
alteration. A Sankey diagram sizes nodes by SHAP and links by the
source's share of SHAP among the target's inputs (Fig. 4). The
prognostic index sums carried alterations weighted by impact;
`risk_stratification.py:27-32` loads precomputed weights
(`weight_30.npy`, absolute path on the authors' machine) with
cut-offs 0.025 and -0.015.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N01 Read a DAG from an edge table | `vnn/utils_biovnn.py:495-500` | Reactome parent-child pairs, kept when the parent ID contains `HSA`. |
| N05 Filter nodes by gene-set size | `vnn/utils_biovnn.py:100-119` | Pathways with fewer than 5 input genes are dropped. |
| N06 Connect features to nodes by membership | `train.py:17-21`, `vnn/utils_biovnn.py:51-75`, `513-529` | Column names cut to gene symbols and joined to every kept pathway whose gene set contains them. |
| N07 One input per entity and data type | `vnn/utils_biovnn.py:253-265` | Several columns name one gene (XPO1 amplification and mutation; two MYD88, EZH2, POU2F2 mutation classes); only the first is wired (Fragile spots). |
| N73 Add a root above top-level nodes | `vnn/utils_biovnn.py:502-508`, `vnn/VNN_cell.py:300-301`, `385` | An added `root` parents every pathway without a parent; its units are the graph part's output. |
| N26 Unlayered DAG in topological order | `vnn/VNN_cell.py:277-328`, `350-364` | Nodes run child-before-parent with no layers; a node reads gene columns and child pathways of any depth. |
| N38 Several units per node | `vnn/VNN_cell.py:307`, `321-322` | A node's state is a vector; a child edge joins all child units to all parent units (count varies, N67). |
| N67 Units per node set by degree | `vnn/VNN_cell.py:299-307`, `utils.py:22` | Units = max(10, member genes + child pathways); the root uses a set width. |
| N12 Parameters only for present edges | `vnn/VNN_cell.py:321-322` | Each node's layer covers only its member genes and child units. |
| N58 Gather each node's inputs by index | `vnn/VNN_cell.py:357-364` | Each node concatenates the tensors at its stored child indices. |
| N70 Initialize from each unit's present fan-in | `vnn/VNN_cell.py:322` | Implicit: default `nn.Linear` init on each node's own inputs. |
| N11 Standard layers between graph layers | `vnn/VNN_cell.py:323-325`, `366-373` | Mish, BatchNorm1d, and Dropout after every node. |
| N09 Dense head after the graph layers | `utils.py:36-40` | Two dense layers from root units (plus clinical columns) to one risk score. |
| N41 Unconstrained branch on a second input | `utils.py:43-49` | Three clinical columns bypass the graph and join the root units raw (identity branch) before the head. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Column names to gene symbols, column subset | `train.py:17-21`, `35-36` | ~8 |
| Read GMT into gene sets | `vnn/utils_biovnn.py:51-75` | ~25 |
| Filter pathways by input-gene count | `vnn/utils_biovnn.py:100-119` | ~20 |
| Read hierarchy, keep human, add root, find leaves | `vnn/utils_biovnn.py:495-511` | ~17 |
| Assign levels (reporting only) | `vnn/utils_biovnn.py:156-192` | ~37 |
| Index genes, then pathways, then root | `vnn/utils_biovnn.py:242-269` | ~28 |
| Child index list per node | `vnn/utils_biovnn.py:307-378` (part) | ~30 |
| Random hierarchy, per-level dense child lists (unused) | `vnn/utils_biovnn.py:381-417`, `459-462` | ~40 |
| Drawing of edges spanning more than 4 levels | `vnn/utils_biovnn.py:296-303`, `357-368`, `420-457` | ~50 |
| Save and reload structure pickle | `vnn/utils_biovnn.py:548-558`, `utils.py:26-27` | ~13 |
| Per-node layers in topological order | `vnn/VNN_cell.py:269-330` | ~60 |
| Forward by index gather and concat | `vnn/VNN_cell.py:338-387` | ~50 |
| Split off clinical columns, per-column list | `utils.py:42-49` | ~8 |

## Fragile spots

- Duplicate symbols: the lowest column index per symbol is kept
  (`vnn/utils_biovnn.py:265`, used at `523`). In the 117-gene
  header of `hmrn_Features.csv` XPO1, EZH2, MYD88, and POU2F2
  appear twice; the later column of each is never read, with no
  message (as for genes in no kept pathway). The 30-gene list
  (`sorted_index.npy`) has no duplicates.
- GMT parsing: gene fields start at index 3 for a path containing
  `pathway` (`vnn/utils_biovnn.py:53-55`, `71`), but GMT lines are
  name, ID, genes, so each pathway's first gene is skipped.
  Recount: 5 gene edges lost in the 117-gene model (e.g. B2M to
  R-HSA-877300), none in the 30-gene model; same kept pathways.
- The root is found by position: lowercase `root` sorts after
  `R-HSA-...` (`vnn/utils_biovnn.py:194`, `307`); unit count uses
  `i == len(self.child_map) - 1` (`vnn/VNN_cell.py:300`); output is
  `features[-1]` (`vnn/VNN_cell.py:385`).
- Node `i` sits at index `i + input_dim` (`vnn/VNN_cell.py:279`,
  `351`); `input_dim` is hard-coded to 30 (`train.py:75`), apart
  from the gene list that builds the pickle (`train.py:21`,
  `29-30`). A mismatch shifts every index.
- Column order is implicit: last three columns clinical, the rest
  in training gene order (`utils.py:43-47`); `predict.py:28`
  passes spreadsheet columns with no name check.
- Root joins only pathways with no parent anywhere in the human
  hierarchy (`vnn/utils_biovnn.py:503-505`); a kept pathway whose
  parents were all filtered would be left dangling. Propagated
  gene sets prevent it here; recount found none.
- `run_mode="random"` reads `mask_random`
  (`vnn/VNN_cell.py:234-236`), which `save_data` never writes
  (`vnn/utils_biovnn.py:550-556`); `community_list.tsv` holds the
  random gene sets (`vnn/utils_biovnn.py:459`).
- `train.py:29-30` rewrites `vnn/BioVNN_pre.pkl` on every run; the
  model is pickled whole (`train.py:112`, `predict.py:39`), so a
  saved model's graph lives only inside its `.pt` file.

## Out of scope

- Loss: Cox partial likelihood as a cumulative sum in batch order
  (`utils.py:55-69`); the loader shuffles (`train.py:71`).
- Training: 10 epochs, Adam lr 5e-3, weight decay 1e-5,
  MultiStepLR milestones 30 and 1000 (`train.py:43-107`).
- C-index via lifelines (`utils.py:72-106`); TCGA preparation
  (`preprocess.py`); prediction CLI (`predict.py`); risk
  stratification (`risk_stratification.py`).
- Unused: `Ranger`, `VNN_cell2`, `FullyNet`
  (`vnn/VNN_cell.py:24-183`, `390-770`).

## Paper vs code

- AdamW in the paper; `torch.optim.Adam` with `weight_decay` in
  code (`train.py:78`).
- Root width 283 in the paper; 128 in `train.py:48`, which is the
  30-gene configuration (`train.py:14`, `75`, `112`).
- 10-fold cross-validation, tuning, and an FCNN baseline in the
  paper; code fits once on all data, imports `KFold` unused
  (`train.py:6`), and has no FCNN baseline.
- 117 genetic inputs in the paper; in code four duplicate-symbol
  columns and 26 genes in no kept pathway (recount) are unread.
- The Sankey description has genes feeding the first pathway
  layer; in code each gene feeds every kept pathway containing
  it, at every level.
- SHAP interpretation is in the paper, not in the code.
- Agreement: "102 pathways distributed across seven layers"
  equals my recount of 101 pathways plus root over seven levels.

## Key excerpts

`vnn/utils_biovnn.py:502-509`

```python
        df_root = pd.DataFrame(columns=df.columns)
        for x in set(df[0]) - set(df[1]):
            if x in self.community_dict or 'GO:' in x:
                df_root = pd.concat([df_root, pd.DataFrame(['root', x]).T])
        # Remove those relationship of groups not in the analysis
        df = df.loc[df[1].isin(self.community_dict.keys()) & df[0].isin(self.community_dict.keys())]
        df = pd.concat([df, df_root])
        leaf_communities = sorted(list((set(df[1]) - set(df[0])) & set(self.community_dict.keys())))
```

`vnn/VNN_cell.py:321-322`

```python
                total_in = int(len(child_feat)*self.omic_dim + np.sum([self.com_layers[z].out_features for z in child_com]))
                self.com_layers[neuron_name] = nn.Linear(total_in, neuron_n) # gene个数+孩子pathway所在神经元的输出维度 ==>  child的个数
```

`vnn/VNN_cell.py:357-368`

```python
            children = self.child_map[i]
            if neuron_name in self.only_combine_gene_group_dict:
                children = [z for z in children if z >= self.input_dim]

            input_list = [features[z] for z in children]
            input_mat = torch.cat(input_list, axis=1)
            # print(neuron_name, input_mat.shape)
            features[j] = com_layer(input_mat)
            ## BN after activation
            state = self.act_func(features[j])
            states[i] = state
            features[j] = bn_layer(state)
```

`utils.py:43-49`

```python
        sup = x[:,-3:]
        x = x[:,:-3]
        x = x.unsqueeze(1)
        x = x.permute(2, 0, 1)
        x = [x[i] for i in range(len(x))]
        VNN_out = self.VNN(x)
        x = torch.concat((VNN_out, sup),1)
```

## Open questions

- SHAP: explainer, background samples, which tensor a pathway's
  value comes from (before or after BatchNorm), and how several
  units become one score. Searched the repo for `shap`,
  `explain`, `sankey`, `attribut`.
- Settings of the 117-gene model (paper root width 283); the
  committed `model/*.pt` and `vnn/BioVNN_pre.pkl` are pickles and
  were not opened.
- Reactome release: not stated in the README or paper.
- Whether sheet 1 of `Supplementary2.xlsx` (read by `train.py`)
  has the columns of `hmrn_Features.csv` (only the CSV header was
  read).
