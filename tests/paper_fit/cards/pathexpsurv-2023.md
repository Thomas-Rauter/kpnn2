# pathexpsurv-2023: PathExpSurv

| Field | Value |
|-------|-------|
| Paper | Hou Z, Leng J, Yu J, Xia Z, Wu L-Y, 2023. PathExpSurv: pathway expansion for explainable survival analysis and disease gene discovery. BMC Bioinformatics 24:434. 10.1186/s12859-023-05535-2 |
| Code | `https://github.com/Wu-Lab/PathExpSurv` at `fb8af83be05e43405a59a229f4bce6f0c16abf47` |
| Other code | none |
| Framework | Python, PyTorch (unpinned; README lists torch, numpy, pandas); scikit-learn folds; lifelines, scipy in the analysis script |
| License | none found |
| Full text | PubMed Central PMC10648621 (Europe PMC XML); supplement not read |
| Extracted | 2026-10-10, Claude Code, claude-opus-5-5 |

## Summary

Cox survival model on TCGA expression (BRCA, LGG, THCA from UCSC
Xena). The prior is a few disease-related KEGG signaling pathways
per cancer, used as flat gene sets. The network is genes → one
unit per pathway → one output unit. Phase 1 (100 epochs) uses only
the prior gene–pathway edges; phase 2 (100 epochs) allows every
gene–pathway pair with an L1 penalty on the non-prior ones. These
weights are kept non-negative. Counting how often each edge is
nonzero over 100 runs on random 90% subsets gives "expanded
pathways", the paper's main interpretive output.

## Prior-knowledge graph

- Source: KEGG DISEASE signaling pathways picked per cancer
  (paper, Data acquisition). One file per cancer
  (`Dataset/<cancer>/<cancer>_kegg.csv`): one row per pathway,
  genes in one `/`-joined string, no pathway names (first lines
  of `Dataset/THCA/THCA_kegg.csv`); names appear only in the
  paper's Table 2.
- Size: bipartite gene → pathway, unsigned, acyclic, no
  pathway–pathway edges. Pathways BRCA 7, THCA 3, LGG 5; prior
  edges (Table 2 "Original") 113, 28, 69; input genes 2005, 1061,
  1126 (SD > 1 after log2(x+1)). A gene can sit in several
  pathways (CCND1, THCA rows 1 and 2).
- Data to nodes: columns are gene symbols, then event and time
  (`main.py:45`, `main.py:86`). Pathway genes match columns by
  exact string equality (`utils.py:14`); pathway genes missing
  from the data are dropped silently. Data genes in no pathway
  stay as inputs: no edge in phase 1 (all-zero mask column), free
  to gain edges in phase 2. No preprocessing code is shipped.
- Built at: `utils.py:6-17` (prior), `utils.py:19-28` (expanded
  table of a previous run), `main.py:49-60` (mask per `--model`).

## Architecture

- Layers: one graph layer, genes → pathways; no layering needed.
- Units per node: one per pathway. The gene→pathway maps have no
  bias (`PathExpSurv.py:11-12`); BatchNorm1d over pathway units
  gives a per-unit shift and scale before tanh
  (`PathExpSurv.py:13-14`, `PathExpSurv.py:26`).
- Connections that skip layers: none.
- Outputs or losses at inner layers: none.
- Parts the graph does not constrain: BatchNorm1d on the gene
  inputs (`PathExpSurv.py:9`); output `Linear(P + 1, 1,
  bias=False)`, init uniform(±0.001), then tanh
  (`PathExpSurv.py:16-17`, `PathExpSurv.py:28`). Its extra input
  `x_2` (`PathExpSurv.py:27`) is the event indicator of the same
  samples, in training (`main.py:117`) and evaluation
  (`main.py:161`).
- Other structure: the gene→pathway map is two matrices summed,
  `sc1` for prior pairs and `sc2` for all other pairs, `sc2`
  scaled by `beta` (0 in phase 1, 1 in phase 2)
  (`PathExpSurv.py:18`, `PathExpSurv.py:26`, `main.py:131-133`).
  `--model` variants: `ori` (prior mask), `adj` (expanded table
  as mask), `full` (all-ones mask, no penalty), `pathexpsurv`
  (two phases) (`main.py:49-60`, `main.py:126-137`).
- Defined at: `PathExpSurv.py:4-30`.

## Connectivity mechanism

`sc1` and `sc2` are full `nn.Linear` weights, pathways × genes
(`PathExpSurv.py:11-12`; mask P × G, `main.py:66-67`). At the
start of every forward the weight data are edited in place:
clamped to ≥ 0, `sc1` times mask, `sc2` times `1 − mask`
(`PathExpSurv.py:21-24`). The same edits run after every
optimizer step (`main.py:164-169`). The forward is an ordinary
linear call on the stored weights (`PathExpSurv.py:26`), so absent
entries get gradients and Adam updates that the next edit erases.
The mask is a plain attribute, not a buffer (`PathExpSurv.py:8`),
moved to the GPU by hand (`main.py:62-63`).

Init: `sc1` uniform [0, 1) (`main.py:97`), masked at the first
forward; `sc2` default init, zero gradient in phase 1
(`beta = 0`), set to zero at the switch (`main.py:131-133`).

Loss: phase 1 adds λ · std of `sc1` over prior entries pooled
over pathways (`main.py:127-128`); phase 2 adds 0.001 · μ ·
Σ|`sc2`| over non-prior entries (`main.py:122`, `main.py:129-130`).
At epoch 100 the learning rate drops to 1e-4, ramps up over epochs
101–120 (zero for 101–103), then decays as 0.01 · 0.95^(epoch −
120) (`main.py:141-150`); one Adam optimizer throughout
(`main.py:103`).

## Interpretation

No attribution method; the trained gene→pathway weights are the
explanation.

- Weights `sc1 + sc2` saved per fold, gene names as columns,
  pathway rows by position only (`main.py:185`).
- Expansion (`downstream_analysis.py:45-76`): per-run gene lists
  per pathway for seeds 56–155 (`downstream_analysis.py:49-51`,
  parsed at `downstream_analysis.py:31-42`) are summed into
  selection counts. Cutoff: the count at position int(K(1 + α))
  from the top, K = prior edge count, α = 0.2
  (`downstream_analysis.py:24`, `downstream_analysis.py:64-65`).
  Every pair at or above it, prior or not, forms the expanded
  table (`downstream_analysis.py:69-76`), reusable as a new mask
  (`main.py:49-52`, `utils.py:19-28`). Supplement genes =
  expanded minus prior, pooled (`downstream_analysis.py:317-341`).
- Recovery test: each gene of the first five LGG pathways is
  removed from its pathway, 100 runs count how often the pair
  comes back, and a KS test compares that with non-prior genes
  (`downstream_analysis.py:81-116`; mask `utils.py:59-66`).
  `utils.py:30-46` drops the last 10% of each pathway; no caller
  found.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N06 Connect features to nodes by membership | `utils.py:6-17` | Pathway gene lists match expression columns by symbol; one gene joins several pathways; unmatched data genes stay as inputs without prior edges. |
| N10 One unit per node | `PathExpSurv.py:11-14`, `PathExpSurv.py:26` | Each pathway is one tanh unit; its shift comes from BatchNorm, not a linear bias. |
| N11 Standard layers between graph layers | `PathExpSurv.py:9`, `PathExpSurv.py:13`, `PathExpSurv.py:26` | BatchNorm on gene inputs and on pathway pre-activations around the one graph layer. |
| N53 Re-zero masked weights before each step | `PathExpSurv.py:21-24`, `main.py:164-169` | Absent entries are zeroed in place before each forward and after each step; the forward uses the stored weight. |
| N32 Edge weights labeled by edge | `main.py:185` | Trained gene→pathway weights are saved with gene names as columns; pathways by row order. |
| N89 Prior edges plus all other pairs, kept apart | `PathExpSurv.py:11-12`, `PathExpSurv.py:23-26` | Prior pairs in `sc1` (mask), all others in `sc2` (1 − mask), summed, so each set gets its own penalty and switch. |
| N90 L1 penalty on non-prior edges only | `main.py:122`, `main.py:130` | Phase 2 adds 0.001 · μ · Σ\|w\| over non-prior entries; prior entries are unpenalized. |
| N91 Switch on more edges partway through training | `main.py:126-133`, `main.py:141-150` | After 100 prior-only epochs, non-prior edges start at zero, `beta` goes to 1, and the learning rate restarts. |
| N92 Non-negative edge weights | `PathExpSurv.py:21-22`, `main.py:165-166` | Gene→pathway weights are clamped to ≥ 0 before each forward and after each step. |
| N93 Penalize the spread of present edge weights | `main.py:128` | Phase 1 adds λ · std of all prior-edge weights pooled, pushing a pathway's genes toward equal weights. |
| N94 Edge selection frequency over resampled fits | `downstream_analysis.py:45-76` | Nonzero edges are counted over 100 runs; the top int(K(1 + α)) form the expanded table used as a new prior. |
| N95 Remove chosen edges from the prior | `utils.py:30-46`, `utils.py:59-66`, `downstream_analysis.py:96-107` | One gene is removed from one pathway to test whether phase 2 recovers that edge. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Parse gene lists, build the P × G mask by name | `utils.py:6-17` | ~12 |
| Mask from the expanded table of a previous run | `utils.py:19-28` | ~10 |
| Masks with chosen genes removed | `utils.py:30-46`, `utils.py:59-66` | ~25 |
| Choose mask per model variant | `main.py:49-60` | ~12 |
| Clamp ≥ 0 and complementary masks, in forward and after each step | `PathExpSurv.py:21-24`, `main.py:164-169` | ~10 |
| Prior and non-prior matrices with `beta` switch | `PathExpSurv.py:11-12`, `PathExpSurv.py:26`, `main.py:131-133` | ~5 |
| Std and L1 penalties on entry subsets | `main.py:122`, `main.py:128`, `main.py:130` | 3 |
| Save weights labeled by gene | `main.py:185` | 1 |
| Selection counts, cutoff, expanded table | `downstream_analysis.py:31-76` | ~45 |
| Leave-one-out recovery frequencies | `downstream_analysis.py:81-110` | ~30 |

## Fragile spots

- Mask entries are match counts (`utils.py:14`). A gene listed
  twice in a pathway, or a repeated column name, gives 2; the mask
  multiplies the weight data at every forward and step
  (`PathExpSurv.py:23`, `main.py:168`), so that weight would
  double each time, and `1 − mask` would be −1
  (`PathExpSurv.py:24`). Shipped files not checked for duplicates.
- Connectivity lives only in in-place `.data` edits. Calling `sc1`
  or `sc2` directly, or reading loaded weights without a forward
  call, sees unmasked values; the mask is not in the state dict
  (`PathExpSurv.py:8`).
- Pathway identity is row order: no names in the gene-set files,
  unnamed rows in saved weights (`main.py:185`).
- The expansion keeps every pair tied with the cutoff
  (`downstream_analysis.py:65-69`), so more than int(K(1 + α))
  can be kept; a rarely selected prior pair is left out.

## Out of scope

- Loss: Cox partial likelihood; the risk set is a lower-triangular
  matrix that ignores time values, so rows must be time-ordered
  (`Survival.py:3-19`).
- Metric: C-index (`Survival.py:21-42`); per fold the best train
  and test value over all epochs is kept (`main.py:176-178`).
- Training loop: full batch, 10-fold CV, seed 56 (`main.py:24-31`,
  `main.py:69-81`, `main.py:113-169`).
- Data: preprocessed CSVs, no preprocessing code.
- Analysis and plots: phase summaries, retraining plots,
  single-gene Kaplan–Meier at the median
  (`downstream_analysis.py:164-316`, `downstream_analysis.py:342-363`);
  GO enrichment is not in the code.

## Paper vs code

- Paper f = tanh(tanh(x · [W1 ⊙ M]) · W2), W2 ∈ R^{P×1}. Code adds
  BatchNorm on inputs and pathway pre-activations and gives the
  output layer the event indicator as an extra input
  (`PathExpSurv.py:9-16`, `PathExpSurv.py:26-28`, `main.py:117`).
- Paper phase 2: one W1 with all-ones mask E and L1 on
  W1 ⊙ (1 − M). Code: two matrices with complementary masks and a
  `beta` switch (`PathExpSurv.py:26`, `main.py:131-133`); same sum.
- L1 weight: paper μ = 1; code 0.001 · μ (`main.py:130`).
- Std term: paper Std(W1 ⊙ M); code std over prior entries only
  (`main.py:128`).
- Learning rate: paper Adam lr 0.05; code reschedules from epoch
  100 for every variant (`main.py:141-150`).
- Cutoff: paper position ⌊(1 + α)K + ½⌋; code int(K(1 + α))
  (`downstream_analysis.py:65`).
- `main.py` runs 10-fold CV and writes weight matrices
  (`main.py:81`, `main.py:185`); the expansion reads per-seed
  gene-name lists (`downstream_analysis.py:50-51`) that no shipped
  code writes.

## Key excerpts

`PathExpSurv.py:20-30`

```python
	def forward(self, x_1, x_2):
		self.sc1.weight.data.clamp_(0)
		self.sc2.weight.data.clamp_(0)
		self.sc1.weight.data = self.sc1.weight.data.mul(self.pathway_mask)
		self.sc2.weight.data = self.sc2.weight.data.mul(-self.pathway_mask+1)

		x_1 = self.tanh(self.bn(self.sc1(self.bn_input(x_1))+self.beta*self.sc2(self.bn_input(x_1))))
		x_cat = torch.cat((x_1, x_2), 1)
		lin_pred = self.tanh(self.sc4(x_cat))
		
		return lin_pred
```

`main.py:122-133`

```python
            reg_out=torch.sum(torch.abs(net.sc2.weight[pathway_mask==0]))
            reg_out_all.append(reg_out)
            alpha=0

            if args.model=='pathexpsurv':
                if epoch<100:
                    loss = main_loss + args.lambda_*torch.std(net.sc1.weight[pathway_mask>0])
                else:
                    loss = main_loss + 0.001*args.mu*reg_out
                if epoch==100:
                    net.beta = 1
                    net.sc2.weight.data=net.sc2.weight.data*0
```

`downstream_analysis.py:64-76`

```python
    p=alpha
    p_value=np.sort((W_stat2).reshape(-1))[-int(np.sum(pathway_mask)*(p+1))]
    print("Origin Pathways:", list(np.sum(pathway_mask,axis=1)))
    print("Union Pathways({}%) :".format(int(100+p * 100)),list(np.sum(W_stat2 >= p_value, axis=1)))

    Bool2=(W_stat2 >= p_value)

    pathway100=[]

    for i in range(Bool2.shape[0]):
        pathway100.append(list(set(genes[Bool2[i]])))

    pd.DataFrame(pathway100).to_csv("Results/{}/adjusted_pathway.csv".format(args.dataset))
```

## Open questions

- Not in the repository: the driver for the 100 runs on 90%
  subsets (seeds 56–155), the conversion of weights to per-pathway
  gene lists (`pathway_bio1.csv`, `pathway_bio2.csv`), and the
  leave-one-out runs. So whether "selected" means weight > 0
  exactly (paper: O = 1 if W1 > 0) or a threshold is unclear.
- Sample counts and pathway names per row are in Additional
  file 1 (Tables S1, S2), not read.
- Whether the data rows are sorted by time, as the risk set needs
  (`Survival.py:3-7`); data files not read.
