# pideel-2023: PiDeeL

| Field | Value |
|-------|-------|
| Paper | Kaynar G et al., 2023. PiDeeL: metabolic pathway-informed deep learning model for survival analysis and pathological classification of gliomas. Bioinformatics 39(11):btad684. 10.1093/bioinformatics/btad684 |
| Code | `https://github.com/ciceklab/PiDeeL` at `de765df11926ceffb31725ff4bc6be09a871ff4b` (commit of 2026-05-06) |
| Other code | none found. The README answers a 2026 replication study by Jean-Quartier et al. (Applied Sciences); not checked for code |
| Framework | Python 3.11.4, PyTorch 2.0.1, pycox, torchtuples 0.2.2 (`PiDeeL.yml:155`, `PiDeeL.yml:281-282`) |
| License | MIT (`LICENSE`) |
| Full text | PubMed Central PMC10663986 (main text only; supplement not read) |
| Extracted | 2026-10-10, Claude Code, claude-opus-5-5 |

## Summary

PiDeeL predicts glioma survival (Cox risk) and pathology (binary class)
from 37 metabolite concentrations measured by HRMAS NMR on 384 patients.
The prior is a KEGG metabolite-to-pathway membership table: 37 metabolites
by 138 pathways. The first layer has one unit per pathway, and its weight
is multiplied elementwise by the 0/1 membership matrix in every forward
pass. A plain dense head follows (4-layer main model:
37 → 138 → 64 → 64 → 1). The paper reports SHAP scores on metabolites and
pathway units and draws metabolite-to-pathway flows as a Sankey diagram.
No SHAP code is shipped, but the code that splits scores into Sankey flows
is.

## Prior-knowledge graph

- Source: KEGG metabolite-to-pathway mapping (paper). The shipped table is
  `reproduction/scripts/result.csv`. Each row holds a KEGG compound ID and
  name (`cpd:C00025,L-Glutamate`), and the 138 columns are KEGG pathway
  names, including global maps such as "Metabolic pathways". The code that
  built it, the KEGG release, and any filtering: not found.
- Size: 37 input features and 138 pathway nodes, joined by 472 edges in
  the training copy (468 in `run/result.csv`; see Fragile spots). The
  graph is bipartite: edges are directed input → pathway, unsigned, and
  acyclic. Each pathway has 1–35 members, and each metabolite is in 2–52
  pathways (counted from the training copy).
- Hierarchy tables `result2.csv`–`result4.csv` (pathway → KEGG
  subclass → class) are loaded (`load_targeted_data.py:87-122`) but used
  by no shipped script.
- Data to nodes: by position only. CSV row *i* is taken as data column
  *i* (`load_targeted_data.py:74-85`), and data columns are named from a
  hard-coded list (`load_targeted_data.py:52-61,65-66`). The two name
  sets differ (e.g. "Allocystathionine" vs "L-Cystathionine", "NAL" vs
  "N6-Acetyl-L-lysine"), and nothing compares them. In the training copy,
  every feature has a pathway and every pathway has a member.
- Built at: `reproduction/scripts/load_targeted_data.py:74-85`. The
  loader is copied in `reproduction/scripts/pathway_integrate.py:10-21`
  and `run/pathway_integrate.py:4-15`.

## Architecture

- Layers: one graph-constrained layer (inputs → pathways). The graph is
  bipartite, so layer assignment is trivial. Variants by total depth:
  2-layer `37 → 138 → 1` (`reproduction/scripts/2layer/pathway/model.py:12-13`),
  3-layer `37 → 138 → 64 → 1` (`3layer/pathway/model.py:12-14`), and
  4-layer `37 → 138 → 64 → 64 → 1` (`4layer/pathway/model.py:12-15`),
  the main model.
- Units per node: one per pathway, with its own bias (`fc1.bias`) and a
  ReLU (`4layer/pathway/model.py:19-20`).
- Connections that skip layers: none.
- Outputs or losses at inner layers: none. A multi-task variant has two
  outputs on the last dense layer, Cox plus BCE
  (`reproduction/test_scripts/test7/2layer/pathway/main.py:56,84-88`).
- Parts the graph does not constrain: the dense head `fc2`–`fc4` with
  ReLU (`4layer/pathway/model.py:13-15,21-23`). The pathology classifier
  is the same network with a sigmoid output
  (`reproduction/test_scripts/test5/4layer/pathway/model.py:23`). The
  inputs come from a separate metabolite quantification pipeline in
  another repository (README).
- Other structure: none. PiDeeL models have no dropout.
- Defined at: `reproduction/scripts/4layer/pathway/model.py:6-30`.
  `run/predict.py:84-108` repeats it inline for the pretrained model.

## Connectivity mechanism

`fc1` is a full `nn.Linear(37, 138)`. Its 138 × 37 weight is the only
weight storage (`4layer/pathway/model.py:12`). The membership matrix is a
plain tensor attribute, not a parameter or buffer (`model.py:16`). The
forward pass computes `x @ (pathway_info * fc1.weight.t()) + fc1.bias`
(`model.py:19`). Absent entries exist and never affect the output, so
they get zero gradient from the loss. They still get gradient from the
explicit L2 term over all parameters (`4layer/pathway/main.py:89-91`) and
from Adam weight decay (`main.py:73`), which shrink them toward zero.
Initialization is `kaiming_uniform_` on the full weight, with fan-in 37
from the full width, and bias 0.01 (`model.py:27-30`, applied at
`main.py:72`). The mask is moved to the device before the model is built
(`main.py:70`), because `.to()` does not move a plain attribute.
`state_dict` saves the full weight without the mask (`main.py:170`), so
reloading (`run/predict.py:120-125`) relies on reading the same CSV.

## Interpretation

- Paper: SHAP values on the 37 inputs and the 138 pathway activations,
  ranked by "the highest mean SHAP value". A Sankey diagram sizes nodes
  by importance and sizes edges by each metabolite's contribution to a
  pathway. The explainer, background set, and pooling over folds and
  seeds are in Supplementary S1.4.3 (not read).
- Code: the repository has no SHAP call. `shap==0.42.1` is pinned
  (`PiDeeL.yml:273`) but never imported. `reproduction/scripts/names.py`
  holds the scores as constants: 14 metabolites scaled to a maximum of 100
  plus "Other metabolites" (`names.py:22-37`), and 15 pathways plus "Other
  pathways" (`names.py:40-57`). For each listed pathway, it collects the
  listed metabolites that are members in the membership CSV. It finds the
  pathway column by header name and the metabolites by row position
  (`names.py:78-90`). It then prints one flow per member edge: the
  metabolite score divided by the sum of scores of that pathway's listed
  members, times the pathway score, divided by 10 (`names.py:93-99`). The
  stored output is `reproduction/scripts/sankey.csv` (118 rows: index,
  metabolite, pathway, value).
- Granularity: input and node scores (paper) and per-edge flows (code).
  The code does not combine scores across repeats.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N06 Connect features to nodes by membership | `reproduction/scripts/load_targeted_data.py:74-85`; `4layer/pathway/model.py:19` | A 37 × 138 metabolite-to-pathway 0/1 table joins each metabolite to all its pathways; its rows are aligned to data columns by position, not by name. |
| N09 Dense head after the graph layers | `reproduction/scripts/4layer/pathway/model.py:13-15,21-23` | Two dense ReLU layers of 64 units and a linear output follow the pathway layer. |
| N10 One unit per node | `reproduction/scripts/4layer/pathway/model.py:12,19-20` | Each of the 138 pathways is one unit with its own bias and a ReLU. |
| N22 Mask a dense weight | `reproduction/scripts/4layer/pathway/model.py:16,19` | Every forward pass multiplies the full `fc1` weight elementwise by the fixed membership matrix. |
| N35 Randomized prior as a control | `reproduction/test_scripts/test2/4layer/no_pathway/main.py:42-48,69-70`; `reproduction/test_scripts/test14/4layer/no_pathway/main.py:40-43,62-63` | Control runs train the same network, over 10 draws each, on a random 0/1 matrix with 468 ones and on the membership matrix with its rows shuffled (metabolites permuted). |
| N97 Split node scores onto edges by input scores | `reproduction/scripts/names.py:78-99` | Each pathway's score is divided among its member metabolites in proportion to their own scores, giving one flow per edge for a Sankey diagram. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Parse the membership CSV into a 37 × 138 int matrix (copied 4 times) | `reproduction/scripts/load_targeted_data.py:74-85` | ~12 |
| Hard-coded metabolite list that fixes the column order (repeated per script) | `reproduction/scripts/load_targeted_data.py:52-61` | ~10 |
| Mask in the forward pass and on the device (repeated in every model) | `reproduction/scripts/4layer/pathway/model.py:16,19`; `main.py:70` | ~3 |
| Random 0/1 matrix with a fixed edge count, plus its Hamming distance to the prior | `reproduction/test_scripts/test2/4layer/no_pathway/main.py:42-54` | ~13 |
| Row-shuffled prior, plus its Hamming distance | `reproduction/test_scripts/test14/4layer/no_pathway/main.py:40-47` | ~8 |
| Split pathway scores into metabolite-to-pathway flows | `reproduction/scripts/names.py:78-100` | ~23 |
| Load hierarchy tables (unused) | `reproduction/scripts/load_targeted_data.py:87-122` | ~35 |

## Fragile spots

- The two membership tables differ. `run/result.csv`, used by
  `run/predict.py` with the pretrained weights, replaces the NAL and NAA
  rows with `cpd:C14042,Sodium iodide` and `cpd:C13014,1-Naphthylacetic
  acid`, both all zero (`run/result.csv:27-28`). That table has 468
  edges, and two of its inputs cannot reach the output. The training
  copy `reproduction/scripts/result.csv` maps those rows to
  N6-acetyl-L-lysine and N-acetyl-L-aspartate with 2 edges each.
- Rows are matched to data columns by position
  (`load_targeted_data.py:65-66,74-85`). A reordered CSV would rewire the
  network silently. `np.reshape(..., (37, ...))` hard-codes the row count
  (`load_targeted_data.py:84`).
- The CSV is parsed with `lines.split(",")` (`load_targeted_data.py:81`).
  This works only because commas in pathway names were replaced by `|`,
  as in "Glycine| serine and threonine metabolism".
- The mask is not in `state_dict` (`model.py:16`). Saved weights
  (`main.py:170`) hold the full matrix with nonzero absent entries.
  Loading them with a different CSV gives a different network without an
  error.
- The random-mask control draws columns from `np.random.choice(137)` for
  a 138-column matrix, so the last pathway never gets a random edge. It
  also hard-codes 468 edges while training uses 472, and it is unseeded
  (`test2/4layer/no_pathway/main.py:43-46`). The row-shuffle control
  shuffles in place, so shuffles accumulate across draws, and its
  generator is unseeded (`test14/4layer/no_pathway/main.py:40-43`).
- `names.py` reads the CSV from an absolute path on the author's machine
  (`names.py:78`), so it is unclear which table produced `sankey.csv`.

## Out of scope

- Loss: Cox partial likelihood via pycox `CoxPHLoss`
  (`4layer/pathway/main.py:75,91`). The classifier uses class-weighted
  `BCEWithLogitsLoss` on a sigmoid output
  (`test5/4layer/pathway/main.py:64`).
- Training: full batch, Adam lr 1e-4, weight decay 1e-5, L2 0.002 on all
  parameters, early stopping with patience 150, at most 5000 epochs
  (`4layer/pathway/config.py:1-6`, `main.py:73-122`).
- Evaluation: 5-fold CV × seeds 6, 35, 81, a 20% validation split,
  standard scaling, and the pycox time-dependent c-index
  (`main.py:56-69,160-168`).
- Baselines: dense nets (`*/no_pathway`), Cox-PH, gradient boosting,
  random survival forest, DeepHit (`test29`), PC-Hazard (`test30`), and a
  dropout sweep on the dense nets (`test13`).
- Also: NMR quantification (other repository), external validation on
  simulated labels (`figures/fig6_external_validation.py`), plotting.

## Paper vs code

- The paper states 468 connections; the training table has 472 (see
  Fragile spots).
- The paper writes the layer as "[FC(138)weights][PI_matrix]" without
  naming the operation. The code uses an elementwise product.
- The paper uses SHAP. No SHAP code is shipped, and the Sankey scores
  are constants (`names.py:22-57`).
- The paper does not mention the KEGG hierarchy. The code loads three
  hierarchy tables, uses none of them, and ships `pathway_process*`
  checkpoints without their scripts.

## Key excerpts

`reproduction/scripts/4layer/pathway/model.py:6-24`

```python
class Net(nn.Module):
    def __init__(self, input_dim, output_dim, pathway_info):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = pathway_info.shape[1]
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.pathway_info = pathway_info.float()

    def forward(self, x):
        pathway = torch.matmul(x, (self.pathway_info * self.fc1.weight.t())) + self.fc1.bias
        pathway = F.relu(pathway)
        hid = F.relu(self.fc2(pathway))
        out = F.relu((self.fc3(hid)))
        out = ((self.fc4(out)))
        return out
```

`reproduction/scripts/load_targeted_data.py:74-85`

```python
pathway_info=[]
with open(SCRIPTS_DIR / "result.csv") as f:
    for index, lines in enumerate(f):
        if index == 0:
            continue

        else:
            tokens = lines.split(",")
            pathway_info.append(tokens[2:])

pathway_info = np.reshape(pathway_info,(37,len(pathway_info[0])))
pathway_info = pathway_info.astype(np.int64)
```

`reproduction/test_scripts/test2/4layer/no_pathway/main.py:42-48`

```python
for p in range(10):
    non_pathway_info = np.zeros((37,138))
    while np.sum(non_pathway_info) < 468:
        where_row = np.random.choice(37)
        where_column = np.random.choice(137)
        if not (non_pathway_info[where_row,where_column]):
            non_pathway_info[where_row,where_column] = 1
```

## Open questions

- How the membership table was built from KEGG (no script, no release
  date), and why NAL and NAA map to different compounds in the two
  copies. Searched `reproduction/scripts/` and `run/`, and searched the
  whole repository for `kegg`, `cpd:`, and `map0`.
- The SHAP setup: explainer, background, layer, signed or absolute mean,
  and pooling over the 15 runs. It is not in the code, and the
  supplement was not read.
- Where the constant scores in `names.py:22-57` come from. The code
  prints 5 fields per row and `sankey.csv` has 4, so the exact producer
  of the stored file is unclear.
- What the `pathway_process`, `pathway_process2`, and
  `pathway_process_type` checkpoints are. Their scripts are absent, and
  the hierarchy loaders are unused.
