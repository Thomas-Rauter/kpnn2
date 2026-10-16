# mpvnn-2022: MPVNN

| Field | Value |
|-------|-------|
| Paper | Ghosh Roy G, Geard N, Verspoor K, He S, 2022. MPVNN: Mutated Pathway Visible Neural Network architecture for interpretable prediction of cancer-specific survival risk. Bioinformatics 38(22):5026–5032. 10.1093/bioinformatics/btac636 |
| Code | `https://github.com/gourabghoshroy/MPVNN` at `74c06951851313f2f0bf8fc2d3d9cb9f56359212` |
| Other code | `gourabghoshroy/MPVNNClassify` (same author, 2021, risk classification variant; not read); `advik11859/mpvnn` (not a GitHub fork, created 2026; not read) |
| Framework | Python 3.8.6, TensorFlow 2.4.1, Keras 2.4.3 (pinned in `README.md:9` only; no requirements file) |
| License | GPL-2.0 (`LICENSE`) |
| Full text | Publisher page (academic.oup.com, open access) and arXiv:2202.00882v1 PDF (Algorithm 1, Fig. 1 caption, Sections 2.2–2.3) |
| Extracted | 2026-10-16, Claude Code, Claude Opus 5.5 |

## Summary

Predicts cancer-specific survival risk from the expression of 1440
genes in ten TCGA cancers. The prior is the KEGG PI3K-Akt pathway as
directed gene pairs. One masked hidden layer has one unit per gene;
each reads its own gene and its pathway neighbors in both directions.
MPVNN replaces a mutation-derived fraction of edges by random pairs
before training (PVNN does not). One output unit gives the risk;
interpretation walks the largest mean absolute weights to a gene set.

## Prior-knowledge graph

- Source: KEGG PI3K-Akt signaling pathway, a two-column tab file of
  gene pairs (`Data/pi3k-akt.txt`). No KEGG release, date, or
  extraction method in the paper or code.
- Size: 3092 directed pairs over 354 gene symbols in the file; no
  self-loops, no duplicate rows, and no pair present in both
  directions (counted from the file, not stated in the paper).
  Unsigned. Direction is discarded when building the mask (see
  Architecture). Only one pathway is used in the code.
- Data to nodes: inputs are all rows of `Data/<ctype>_expr`, pathway
  genes first (symbol in either column of the pathway file), then the
  rest, in file order (`Code/mpvnn.py:106-111`,
  `Code/mpvnn.py:121-165`). An edge is kept only if both genes have
  expression (`Code/mpvnn.py:274-280`); other pathway genes are
  dropped silently. Non-pathway genes stay as inputs with a hidden
  unit. Paper: 1440 genes, 1285 of them non-pathway, so about 155
  pathway genes (inferred); kept edge count not determined.
- Built at: `Code/mpvnn.py:106-111`, `Code/mpvnn.py:274-280`,
  `Code/mpvnn.py:81-102`.

## Architecture

- Layers: input (N genes) → one graph layer (N hidden units, one per
  gene) → one output unit (`Code/mpvnn.py:306-310`). No layering of
  the graph: every edge is one connection from the input layer to
  the hidden layer.
- Units per node: one hidden unit per gene, with its own bias; the
  hidden activation is chosen by random search from tanh, sigmoid,
  ReLU (`Code/mpvnn.py:30`, `Code/mpvnn.py:288`).
- Mask: N×N, starting from the identity (each hidden unit reads its
  own gene's input), plus both `W[a,b]` and `W[b,a]` for each kept
  edge a→b, so the edge list is used as undirected
  (`Code/mpvnn.py:83`, `Code/mpvnn.py:100-101`). Non-pathway genes
  have only the self-connection unless a random edge lands on them.
- Rewiring (MPVNN only): genes at or above the 0.99 quantile of
  nonzero mutated-sample counts are "most mutated"; `thrp` is the
  share of them that are pathway genes (`Code/mpvnn.py:261-271`).
  Each edge is replaced with probability `thrp` by a random pair of
  distinct genes, from the pathway genes if a second draw is ≥
  `thrp`, else from non-pathway genes, redrawn until it matches no
  pathway edge or earlier new pair in either direction
  (`Code/mpvnn.py:85-99`). The edge count is unchanged.
- Graph as a hyperparameter: a new mask is drawn for each of 300
  random-search configurations (4-fold CV, 5 runs per fold); the
  best (`Wmax`) is used for 20 final runs (`Code/mpvnn.py:285-287`,
  `Code/mpvnn.py:326-338`, `Code/mpvnn.py:348-350`).
- Skip connections, inner outputs, other structure: none.
- Parts the graph does not constrain: the output unit, a dense layer
  with activation chosen from tanh, sigmoid, ReLU
  (`Code/mpvnn.py:309-310`). Variant MP1NN adds a dense layer of N
  units between the graph layer and the output
  (`Code/mp1nn.py:312-313`).
- Baselines: PVNN (`Code/pvnn.py:214-219`); RaNN, the PVNN mask
  shuffled within rows then columns, per configuration
  (`Code/rann.py:225-229`); ANN, dense N-unit layer
  (`Code/ann.py:210-213`); Cox-PH; Cox-PASNet1P (not read).
- Defined at: `Code/mpvnn.py:41-56`, `Code/mpvnn.py:306-310`.

## Connectivity mechanism

`CustomConnected` subclasses Keras `Dense` and computes
`K.dot(inputs, self.kernel * self.connections)` in every forward
pass (`Code/mpvnn.py:50-51`). The full N×N kernel is a trainable
`Dense` weight with the default initializer; absent entries get no
loss gradient. The L1 or L2 kernel regularizer acts on the unmasked
kernel (`Code/mpvnn.py:307-308`), so absent entries decay but never
reach the output.

## Interpretation

Weights, aggregated over the 20 final runs of the best
configuration: after each run, add `|kernel ⊙ Wmax|` and the absolute
output weights to running sums, then divide by 20
(`Code/mpvnn.py:345-346`, `Code/mpvnn.py:364-366`,
`Code/mpvnn.py:380-381`). Both matrices are written unlabeled to
`Out/<alg>_<ctype>_results.tsv` (`Code/mpvnn.py:383-397`).

`Code/interpretvnn.py` rebuilds the gene order from the expression
file (`Code/interpretvnn.py:25-40`), reads both matrices
(`Code/interpretvnn.py:52-66`), and starts from the 3 hidden units
with the largest mean absolute output weight
(`Code/interpretvnn.py:9`, `Code/interpretvnn.py:69-76`). Cutoffs:
0.85 quantile of the nonzero input-to-hidden and of the nonzero
output weights (`Code/interpretvnn.py:81-86`). Each step picks the
input with the largest weight into the current gene's unit,
excluding the current and previous gene, and stops when that weight
or the new gene's self weight is below cutoff, on a revisit (marked
`:back`), or when the new gene's output weight is below cutoff
(`Code/interpretvnn.py:92-114`). A second walk excludes the first
walk's first step (`Code/interpretvnn.py:117-142`). One gene-name
set per start is printed (`Code/interpretvnn.py:145-150`). Scores
are not per sample.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N127 Use directed edges in both directions | `Code/mpvnn.py:100-101` | Each pathway pair a→b sets both mask entries, so a reaches b's hidden unit and b reaches a's. |
| N128 Hidden unit per input with a self-connection | `Code/mpvnn.py:83`, `Code/mpvnn.py:307` | The hidden layer has one unit per input gene, joined to its own gene's input; genes outside the pathway keep only that link. |
| N129 Rewire a random fraction of prior edges | `Code/mpvnn.py:85-99`, `Code/mpvnn.py:271` | Each pathway edge is replaced with a mutation-derived probability by a new random pair, redrawn per search configuration and selected by CV. |
| N130 Greedy walk along strongest edges | `Code/interpretvnn.py:92-142` | The gene set is a walk from a top hidden unit along the largest mean absolute input weights, stopped by quantile cutoffs or a revisit. |
| N10 One unit per node | `Code/mpvnn.py:307` | One hidden unit per gene with its own bias and a shared activation. |
| N22 Mask a dense weight | `Code/mpvnn.py:50-51` | A full N×N kernel is multiplied by the fixed 0/1 mask in every forward pass. |
| N13 Regularize edge weights per layer | `Code/mpvnn.py:307-310` | L1 or L2 on the graph layer's kernel, with its own type and strength, separate from the output layer's. |
| N35 Randomized prior as a control | `Code/rann.py:225-229` | RaNN shuffles the prior mask within rows and then columns, keeping the connection count, as a control. |
| N09 Dense head after the graph layers | `Code/mp1nn.py:312-313` | The MP1NN variant inserts a dense N-unit layer between the graph layer and the output. |
| N115 Dense twin of the graph layer | `Code/ann.py:210-213` | The ANN baseline replaces the masked layer with a dense layer of the same width. |
| N120 Output weight as node score | `Code/interpretvnn.py:69-76` | Hidden units are ranked by their mean absolute weight to the output unit. |
| N32 Edge weights labeled by edge | `Code/mpvnn.py:365`, `Code/interpretvnn.py:93-102` | Mean absolute masked weights are read per (input gene, hidden gene) pair to pick the next gene. |
| N33 Combine scores across training repeats | `Code/mpvnn.py:348-381` | Absolute weights are averaged over 20 runs of the same configuration and mask. |
| N17 Map scores to node names | `Code/interpretvnn.py:25-40`, `Code/interpretvnn.py:73` | Weight rows and columns are turned back into gene symbols through the rebuilt gene list. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Read pathway gene set | `Code/mpvnn.py:106-111` | ~6 |
| Order inputs pathway-first, split by membership | `Code/mpvnn.py:121-165` | ~45 |
| Keep edges with both genes in the data | `Code/mpvnn.py:274-280` | ~7 |
| Rewiring rate from mutation counts | `Code/mpvnn.py:231-271` | ~40 |
| Build mask: identity, symmetric edges, rewiring | `Code/mpvnn.py:81-102` | ~22 |
| Masked dense layer | `Code/mpvnn.py:41-56` | ~16 |
| Average masked absolute weights over runs | `Code/mpvnn.py:345-381` | ~8 |
| Write weights, rebuild gene order, read back | `Code/mpvnn.py:383-397`, `Code/interpretvnn.py:14-66` | ~65 |
| Greedy gene-set walk | `Code/interpretvnn.py:69-150` | ~80 |

## Fragile spots

- Weight matrices are saved unlabeled; `interpretvnn.py` names them
  by rebuilding the gene order (`Code/interpretvnn.py:25-40`), which
  must match training (`Code/mpvnn.py:121-163`).
- The mask is not saved; present edges are found as nonzero averaged
  weights (`Code/interpretvnn.py:81-86`), so an edge whose weights
  are exactly zero (possible under L1) leaves the quantile.
- The regularizer sees the unmasked kernel (`Code/mpvnn.py:307-308`),
  so the penalty includes absent entries.
- Rewiring shares one numpy seed stream with the hyperparameter draws
  (`Code/mpvnn.py:228`, `Code/mpvnn.py:287-297`), so a configuration's
  mask depends on all draws before it.

## Out of scope

- Loss: concordance-index lower bound per batch (`Code/mpvnn.py:59-78`).
- Training: Keras `fit`, Adam or SGD (`Code/mpvnn.py:311-314`).
- Data loading, TCGA barcode filters (`Code/mpvnn.py:169-197`).
- Split and scaling (`Code/mpvnn.py:206-224`).
- Random search, 300 configurations (`Code/mpvnn.py:30-37`).
- Evaluation: lifelines concordance index (`Code/mpvnn.py:358-362`).
- Cox-PH and Cox-PASNet1P baselines.

## Paper vs code

- Algorithm 1 handles several pathways (per-pathway `thr_p`,
  `frac_p`, pairs drawn from other pathways); the code covers one
  pathway and computes no `frac_p` (`Code/mpvnn.py:81-102`).
- The paper checks a revisit before the new gene's self weight; the
  code checks the self weight first, so such a revisit stops without
  `:back` (`Code/interpretvnn.py:104-108`).
- The paper gives no number of walked top units; the code uses 3
  (`Code/interpretvnn.py:9`).
- The paper assumes no self-loops or reverse pairs; the code does not
  check, and the shipped file has none.

## Key excerpts

`Code/mpvnn.py:41-56`

```python
class CustomConnected(Dense):

    def __init__(self,units,connections,**kwargs):

        self.connections = connections                        

        super(CustomConnected,self).__init__(units,**kwargs)  


    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
```

`Code/mpvnn.py:81-102`

```python
def createW(fallgenes, fpgenes, edgesp, thrp):
    fogenes = [i for i in fallgenes if i not in fpgenes]
    W = np.identity(len(fallgenes))
    newedges = []
    for edge in edgesp:
        ab = edge.split("#")
        rnum1 = np.random.rand(1)[0]
        if rnum1 < thrp:
            while True:
                rnum2 = np.random.rand(1)[0]
                if rnum2 >= thrp:
                    ab = np.random.choice(fpgenes,2,replace=False)
                else:
                    ab = np.random.choice(fogenes,2,replace=False)
                newedge = ab[0] + "#" + ab[1]
                newedgerev = ab[1] + "#" + ab[0]
                if newedge not in edgesp and newedge not in newedges and newedgerev not in edgesp and newedgerev not in newedges:
                    newedges.append(newedge)
                    break        
        W[fallgenes.index(ab[0]),fallgenes.index(ab[1])] = 1
        W[fallgenes.index(ab[1]),fallgenes.index(ab[0])] = 1
    return W
```

`Code/interpretvnn.py:99-110`

```python
        while (fallgenes[maxind] == candgene or fallgenes[maxind] == topsgenes[valt]):
            counter -= 1
            maxind = maxindall[counter]
        if W1[maxind,fallgenes.index(candgene)] < thrmax1:
            break
        if W1[maxind,maxind] < thrmax1:
            break
        if fallgenes[maxind] in topsgenes:
            topsgenes.append(fallgenes[maxind]+":back")
            break
        if W2[maxind,0] < thrmax2:
            break
```

## Open questions

- KEGG release and pair-list extraction: not in paper, README, or code.
- Edges kept per cancer after the expression filter: expression files
  not read.
- Whether mutation and expression files list the same genes: the
  code does not intersect them (`Code/mpvnn.py:236-268`).
- Cox-PASNet1P's use of the pathway file (`Code/coxpasnet1p.py:59`):
  not read.
