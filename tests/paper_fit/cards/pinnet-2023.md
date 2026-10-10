# pinnet-2023: PINNet

| Field | Value |
|-------|-------|
| Paper | Kim and Lee, 2023. PINNet: a deep neural network with pathway prior knowledge for Alzheimer's disease. Frontiers in Aging Neuroscience 15:1126156. 10.3389/fnagi.2023.1126156 |
| Code | `https://github.com/DMCB-GIST/PINNet` at `0d204153d3442b483bf2e53cb4f6fb6eb5cddb2c` |
| Other code | none found (GitHub repository search for "PINNet alzheimer" and "PINNet pathway"; the repository has no forks) |
| Framework | Python 3.7.4, PyTorch 1.8.1 (`requirements.txt:1,5`); scikit-learn and imbalanced-learn imported but not pinned |
| License | none found |
| Full text | Europe PMC full-text XML of PMC10380929 (Methods 2.1–2.4, Figure 1, abstract); the README also links arXiv 2211.15669 |
| Extracted | 2026-10-10, Claude Code, Claude Opus 5.5 |

## Summary

PINNet classifies Alzheimer's disease against controls from bulk
gene expression (8,922 genes; brain GSE33000 and blood ADNI in
the paper, brain only in the code). The prior is a binary
gene-to-pathway membership matrix from MSigDB, either KEGG (168
pathways) or GO biological process (4,026 terms). One masked
linear layer maps genes to one unit per pathway. Beside it, an
unconstrained dense layer reads all genes. The two outputs are
normalized, concatenated, and passed through one dense hidden
layer of 64 units and a 2-way softmax. The paper ranks genes and
pathways with Deep SHAP, but the repository ships no
interpretation code.

## Prior-knowledge graph

- Source: MSigDB KEGG and GO BP gene sets, no versions stated.
  The paper keeps sets with at least 10 of the 8,922 genes
  (Methods 2.1). The shipped tables are prebuilt
  (`data/pathway_kegg.csv`, `data/pathway_go_bp.csv`). No code
  builds or filters them.
- Size: two node types, genes (inputs) and pathways. KEGG table:
  8,922 gene rows × 168 pathway columns of 0/1, gene rows named by
  Entrez ID (`data/pathway_kegg.csv:1-4`, 8,923 lines). GO table
  not opened (72 MB). The paper gives 4,026 terms. Edge count was
  not counted. The paper reports a mean of 42 genes per KEGG
  pathway and 84 per GO term. Edges are directed gene → pathway
  and unsigned. Bipartite, one level, no hierarchy, no cycles.
- Data to nodes: by row order only. Both CSVs have genes as rows.
  The code drops the first (gene ID) column of each, transposes
  both, and never reads the IDs (`src/dataPre.py:17-20`,
  `src/dataPre.py:29-32`). No name matching or reordering. Genes
  in no pathway stay in the input; they reach the output only
  through the dense branch. Nothing handles pathways without
  data, because rows are assumed to line up one to one.
- Built at: `src/dataPre.py:11-20` (load and transpose the
  membership table into a pathways × genes mask).

## Architecture

- Layers: one graph-constrained layer, genes → pathways
  (`src/model.py:30`, `src/model.py:44-47`). No layer
  assignment, since the prior has one level.
- Units per node: one per pathway, with its own bias
  (`nn.Linear(input_size, num_pw)`, `src/model.py:30`). After the
  size scaling, the pathway units pass through BatchNorm1d,
  LayerNorm over all pathway units, Tanh, and Dropout(0.1)
  (`src/model.py:32-35`).
- Connections that skip layers: none in the graph. A parallel
  dense branch `fc1` reads all genes: Linear(n → num_fc),
  BatchNorm1d, LayerNorm, Tanh, Dropout(0.1)
  (`src/model.py:16-20`, `src/model.py:43`). Its output is
  concatenated with the pathway units (`src/model.py:48`).
- Size scaling: each pathway unit's pre-activation (weighted sum
  plus bias) is divided by the square root of the pathway's
  member count (`src/model.py:31`, `src/model.py:46`).
- Outputs or losses at inner layers: none.
- Parts the graph does not constrain: `fc1` (above), `fc2`
  Linear(num_fc + m → 64), BatchNorm1d, Tanh, Dropout(0.1)
  (`src/model.py:22-25`), and `fc3` Linear(64 → 2) + Softmax
  (`src/model.py:27-28`).
- Other structure: none.
- Defined at: `src/model.py:9-51`.

## Connectivity mechanism

`pw_linear` is an ordinary `nn.Linear(n, m)`. It holds the full
m × n weight, Xavier-uniform over the full layer width, with
zero bias (`src/model.py:4-7`, `src/model.py:30`,
`src/model.py:40`). The 0/1 mask is a plain tensor attribute, not
a registered buffer (`src/model.py:14`). At the start of every
forward call, in training and in evaluation, the code replaces
the weight with a new parameter that holds weight × mask
(`src/model.py:44`). The layer then runs on that stored weight
(`src/model.py:45`). Absent entries are zero in the stored weight
and do not affect the output. The product is detached into the
new parameter, so absent entries still receive gradients.

The optimizer is built at `src/train.py:87`, before the first
forward. It holds the original weight object. After the first
forward, the module's weight is a new object that is not in the
optimizer, and each later forward replaces it again. In effect,
the masked pathway weights stay at their masked Xavier start for
all of training. The pathway layer trains only its bias, plus the
BatchNorm and LayerNorm affine parameters after it. A
stand-alone toy of the same pattern on PyTorch 2.13 confirmed
this; the paper's code (PyTorch 1.8.1) was not run.
`model.to(device)` at `src/train.py:89` keeps the original
parameter objects, so the other layers train normally.

The mask is on the GPU only because `src/train.py:47` moves it
before the model is built. `.to()` does not move plain
attributes. The mask and the size vector `norm_gene` are not in
the `state_dict` saved at `src/train.py:124`. The early-stopping
restore at `src/train.py:111` unpickles the whole model
(`src/utils.py:37`), so they come back with it.

## Interpretation

None in the code. The repository has no SHAP or attribution code
(searched `shap|deeplift|captum|integrated|attribut|explainer`;
only `shape` matched). The paper (Methods 2.4) describes:

- Deep SHAP via `shap.DeepExplainer`, with a class-balanced
  training set as background.
- Granularity: input genes (on test samples) and pathway nodes.
- Combination: absolute mean SHAP per feature, z-scored within
  each of the 10 cross-validation models, then averaged.

Pathway and gene names are not kept in the code
(`src/dataPre.py:17-18`, `src/dataPre.py:29`). How scores were
mapped back to names is not shown.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N06 Connect features to nodes by membership | `src/dataPre.py:11-20` | A prebuilt genes × pathways 0/1 table becomes the mask. It is aligned with the expression matrix by row order, not by name, and genes in no pathway are kept for the dense branch. |
| N10 One unit per node | `src/model.py:30-35` | Each pathway is one unit with its own bias, then BatchNorm, LayerNorm across pathway units, and Tanh. |
| N53 Re-zero masked weights before each step | `src/model.py:44`, `src/train.py:87` | Every forward replaces the full weight with a new parameter equal to weight × mask, so absent entries are zero in the stored weight but get gradients. Because the replacement is not in the optimizer, no masked weight is updated after the first forward. |
| N09 Dense head after the graph layers | `src/model.py:22-28` | Linear(num_fc + m → 64), BatchNorm, Tanh, Dropout, then Linear(64 → 2) and Softmax after the concatenation. |
| N98 Dense branch beside a graph layer | `src/model.py:16-20`, `src/model.py:43`, `src/model.py:48` | All genes, including pathway members, also feed a dense layer of num_fc units, normalized with BatchNorm and LayerNorm, and concatenated with the pathway units before the head. |
| N99 Scale node sums by the root of in-degree | `src/model.py:31`, `src/model.py:46` | Each pathway unit's weighted sum plus bias is divided by √(member count), with the count taken from the mask's row sums. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Load the membership table, drop gene and pathway names, transpose to a pathways × genes mask | `src/dataPre.py:11-20` | ~10 |
| Move the mask to the device before building the model | `src/train.py:47` | 1 |
| Per-pathway size vector from mask row sums | `src/model.py:31` | 1 |
| Apply the mask by replacing the weight each forward | `src/model.py:44` | 1 |
| Divide pathway sums by the size vector | `src/model.py:46` | 1 |

## Fragile spots

- `src/model.py:44` creates a new parameter on every forward,
  after the optimizer was built (`src/train.py:87`), so the
  masked pathway weights never update and nothing reports it.
- Membership and expression tables are joined only by row
  position (`src/dataPre.py:17-20`, `src/dataPre.py:29-32`); a
  different gene order in either file would silently attach
  genes to the wrong pathways. `gene = data.iloc[:,1]`
  (`src/dataPre.py:30`) picks a sample column and is unused.
- Mask and size vector are plain attributes, not buffers
  (`src/model.py:14`, `src/model.py:31`): `.to()` and the saved
  `state_dict` (`src/train.py:124`) miss them.
- A pathway with no members divides by zero at
  `src/model.py:46`; the code relies on the paper's ≥10-gene
  pre-filter, which it does not run.

## Out of scope

- Loss: cross-entropy on outputs already through Softmax
  (`src/model.py:27-28`, `src/train.py:88`).
- Training: Adam, up to 200 epochs under autocast, early
  stopping on validation AUC (`src/train.py:87-108`).
- Split: stratified 10-fold, 1/9 validation, SMOTE on the
  training part (`src/train.py:53-71`).
- Labels by sample position and MinMax scaling over all samples
  (`src/dataPre.py:34-47`).
- Hyperparameter loop and per-fold selection
  (`src/train.py:82-125`, `src/main.py:12-13`).

## Paper vs code

- Size normalization: the paper divides by the member count,
  u_i = 1/Σ_j M_ij (Eq. 3). The code divides by its square root
  (`src/model.py:31`).
- Branch order: the paper has p = tanh(((W_p ∘ M) g) ∘ u) and
  f = tanh(W_f g), each LayerNormed before concatenation. The
  code runs Linear → BatchNorm → LayerNorm → Tanh → Dropout in
  both branches (`src/model.py:16-20`, `src/model.py:32-35`).
- Masked weights: the paper says the nonzero entries of W_p are
  updated by backpropagation. In the code they keep their masked
  initial values (see Connectivity mechanism).
- Dropout: 0.3 in the paper, 0.1 in the code
  (`src/model.py:20,25,35`). Optimizer: the paper names Adam,
  ReduceLROnPlateau, and SGD; the code uses Adam alone
  (`src/train.py:87`).
- Pathway counts: Methods 2.1 gives 168 KEGG and 4,026 GO BP,
  Methods 2.3 gives 186 and 7,470. The KEGG table has 168.
- Grid {32, 64, 128} × {1e-4, 5e-4, 1e-3}: commented out; the
  demo runs 32 and 1e-4 (`src/dataPre.py:63-66`).
- Not in the code: blood (ADNI) data, Deep SHAP, the dense,
  SVM, and RF baselines, and the subnetwork and enrichment
  analyses.

## Key excerpts

`src/model.py:30-51` (trailing whitespace removed)

```python
        self.pw_linear = nn.Linear(input_size, num_pw)
        self.norm_gene = torch.sqrt(pathway.sum(axis=1))
        self.pw = nn.Sequential(nn.BatchNorm1d(num_pw),
                                nn.LayerNorm(num_pw),
                                nn.Tanh(),
                                nn.Dropout(0.1))

        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)
        self.fc3.apply(weights_init)
        self.pw_linear.apply(weights_init)

    def forward(self, input):
        x1 = self.fc1(input)
        self.pw_linear.weight = nn.Parameter(self.pw_linear.weight * self.pathway)
        x2 = self.pw_linear(input)
        x2 = x2/self.norm_gene
        x2 = self.pw(x2)
        out = torch.cat((x1, x2), dim=1)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
```

`src/dataPre.py:11-20`

```python
    if (path == "GO"):
        pathway = pd.read_csv("../data/pathway_go_bp.csv", header=0)
    elif (path == "KEGG"):
        pathway = pd.read_csv("../data/pathway_kegg.csv", header=0)
    print(">> Pathway Data :",path)

    pathway_info = pathway.iloc[:,1:]
    pathway_info = pathway_info.values
    pathway_info = np.transpose (pathway_info)
    pathway_info = torch.FloatTensor(pathway_info)
```

`src/train.py:86-89`

```python
                    self.model = PINNet(input_dim,pathway_info,num_fc)
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay = 0)
                    self.criterion = nn.CrossEntropyLoss()
                    self.model = self.model.to(self.device)
```

## Open questions

- How the gene and pathway membership tables were built and
  filtered (≥10 genes), and how gene order was matched to the
  expression table. No preprocessing code ships, and the
  expression file (79 MB) was not opened.
- Which tensor the paper's pathway-node SHAP values explain: the
  raw masked sum, the size-scaled sum, or the output after
  BatchNorm, LayerNorm, and Tanh. There is no SHAP code, and the
  paper points to "p in Equation (3)".
- Whether the paper's runs used this exact code. The last commit
  (2023-03-15) predates the journal version (2023-07-14). The
  paper says the masked weights are trained; here they are not.
- GO BP table size and edge counts; the tables were not read in
  full.
