# deepsignalingflow-2024: DeepSignalingFlow

| Field | Value |
|-------|-------|
| Paper | Zhang H, Chen Y, Payne P, Li F, 2024. Using DeepSignalingFlow to mine signaling flows interpreting mechanism of synergy of cocktails. npj Systems Biology and Applications 10:92. 10.1038/s41540-024-00421-w |
| Code | `https://github.com/FuhaiLiAiLab/DeepSignalingFlow` at `d07b483ff1ffd33a2cf16f165192e7e200e20878` (HEAD, 2025-09-29) |
| Other code | None linked from the paper. The repository also holds baseline GNNs (GAT, GCN, GIN, MixHop, graph transformer) in `enc_dec/`; this card does not cover them. |
| Framework | Python, PyTorch with PyTorch Geometric (`MessagePassing`, `degree`); networkx for analysis. No versions pinned (no requirements file). |
| License | README states MIT (`README.md:11,119`); no LICENSE file in the tree |
| Full text | nature.com article HTML (abstract, Results "Architecture of DeepSignalingFlow", Methods, Fig. 1 caption, code availability) |
| Extracted | 2026-10-10, Claude Code, Claude Opus 5.5 |

## Summary

The model predicts a synergy score for a drug pair in a cancer cell line
(NCI ALMANAC, O'Neil, DrugComb, DrugCombDB). One graph merges KEGG
signaling-pathway gene–gene edges with DrugBank drug–target edges and has
gene nodes and drug nodes. Each sample is that graph with a 4-value
vector per gene: two drug-target flags for the sample's pair, RNA, and
CNV. Drug nodes get zeros. Three message-passing layers aggregate along
and against edge direction. Each gene–gene edge has its own trainable
scalar per layer and direction, and the channel transforms are shared
across edges. A bilinear decoder scores the final states of the sample's
two drug nodes. Interpretation reads the per-edge scalars as
signaling-flow weights.

## Prior-knowledge graph

- Source: KEGG edge table `full_kegg_pathway_list.csv` (version and origin
  not found), keeping rows whose `pathway_name` contains "signaling
  pathway(s)". Symbols are uppercased and `(src, dest)` pairs deduplicated
  across pathways (`parse/init_parse.py:120-144`). DrugBank drug–target
  pairs (`parse/init_parse.py:157-165`) are restricted to graph genes
  (`parse/parse_circle.py:207-213`).
- Size: the paper gives 2016 gene nodes; the number of drug nodes varies
  by dataset. Code comments give 59241 gene–gene edges and 214 drug edges
  (107 pairs × 2) (`enc_dec/geo_webgnn_decoder.py:24,48-49`). Gene–gene
  edges are directed and unsigned (`src`, `dest` only). Drug–target edges
  appear in both directions (`load_data.py:248-249`). Cycles are not
  checked or removed.
- Data to nodes: gene symbols are matched to Cell Model Passports RNA and
  CNV tables. A KEGG gene missing from either table is deleted with all
  its edges, and the gene list is rebuilt from the remaining edges
  (`parse/parse_circle.py:181-197`). Omics rows outside the graph are
  dropped (`:199-204`), and RNA `missing` becomes 0 (`:294`). Drug names
  are matched to DrugBank after removing punctuation and uppercasing.
  Samples with an unmatched drug are dropped (`:225-255`).
- Node numbering: genes `1..n1` by sorted symbol, then drugs
  `n1+1..n1+n2` by sorted name (`parse/parse_network.py:18,34-37`).
- Built at: `parse/parse_network.py:15-46`, `load_data.py:214-258`.

## Architecture

- Layers: three message-passing layers (`conv_first`, `conv_block`,
  `conv_last`), each over the whole graph with every node present and
  with its own parameters (`enc_dec/geo_webgnn_decoder.py:111-133`).
  Nodes have no layer assignment.
- Units per node: a channel vector. It has 4 channels at input, then
  `3 × out` after each layer (up, down, and self concatenated):
  hidden_dim 4 gives 12, and output_dim 36 gives 108
  (`load_data.py:69-79`; `geo_tmain_webgnn.py:62-65`;
  `enc_dec/geo_webgnn_decoder.py:77`).
- Per layer: `up` is the mean over in-edges of `w_up[e] · (x_src M_up)`;
  `down` is the same on reversed edges with `w_down[e]` and `M_down`; and
  `self = x B`. The three are concatenated, each node's vector is
  L2-normalized, and LeakyReLU(0.1) follows
  (`enc_dec/geo_webgnn_decoder.py:41-86,127-133`). `M_up`, `M_down`, and
  `B` are bias-free Linear layers shared by all nodes and edges (`:19-21`).
- Connections that skip layers: none beyond the self path (`:45,75`).
- Outputs or losses at inner layers: none.
- Parts the graph does not constrain: BatchNorm1d on input features
  (`:106,124`). A bilinear decoder `h_a P1 P2 P1ᵀ h_bᵀ`
  (`P1` 108×150, `P2` 150×150) acts on the sample's two drug nodes
  (`:108-109,136-150`). Loss is MSE (`:156-162`).
- Other structure: none.
- Defined at: `enc_dec/geo_webgnn_decoder.py:11-162`;
  `geo_tmain_webgnn.py:93-116`.

## Connectivity mechanism

The graph is a `[2, E]` edge index. Gene–gene edges come first, in the
row order of `kegg_gene_num_interaction.csv`, followed by drug→target
and target→drug edges (`load_data.py:247-258`). PyTorch Geometric's
`propagate` gathers source rows, multiplies each by `norm[e] · w[e]`, and
scatter-adds into targets (`aggr='add'`,
`enc_dec/geo_webgnn_decoder.py:13,73-74,81-86`). Absent edges are not in
the list, so they have no parameter, product, or gradient. No mask or
n×n weight exists.

Each layer holds `up_gene_edge_weight` and `down_gene_edge_weight`,
parameter vectors of length `num_gene_edge` (`:26,29`). Constant
`torch.ones` for drug edges are appended, and these are not trained
(`:49-53`). The vector is tiled `batch_size` times to match PyG's batched
edge list; every sample shares one `edge_index` (`:55-56`;
`geo_loader/read_geograph.py:43`). The down pass reverses the edges with
`torch.flipud` and keeps the same order (`:43`). Each pass normalizes by
`1/in-degree` of the target: a mean over in-edges for up and over
original out-edges for down (`:61-69`).

Init: per-edge weights `randn · calculate_gain('relu')`, i.e. √2
(`:25-29`); PyTorch defaults for Linear; `randn` for the decoder
(`:108-109`). Adam uses weight decay 1e-6 on all parameters and gradient
clipping at 2.0 (`geo_tmain_webgnn.py:130,142`).

## Interpretation

Trained edge weights only; no gradient-based attribution.

1. Per fold, take the absolute value of the six per-edge vectors
   (3 layers × up/down). Average over layers within each direction, then
   average up and down into one `conv_bind_weight` per gene–gene edge
   (`webgnn_edge_analysis.py:20-45`).
2. Name the scores by attaching the vector as a column to
   `kegg_gene_interaction.csv`, row by row (`webgnn_edge_analysis.py:47-65`).
3. Average the 5 folds by row with `groupby(level=0).mean()` (`:68-79`).
4. Score each gene by its weighted degree in an undirected networkx graph
   of the averaged edge scores (`webgnn_dec_analysis.py:16-36`). Multiply
   by `log(RNA+1)` per cell line (`analysis_cell_bindbi_net.py:151-156`).
5. Drop genes with degree ≤ 1.0 and cell-line score ≤ 2.0
   (`analysis_cell_bindbi_net.py:169-176,548-550`). Add the drug–target
   edges, then list all simple paths between the two drugs with cutoffs
   3, 4, and 5 and sum the edge weights along them (`:256-300`). Separate
   filters write CSVs for edges < 0.2 and degree < 0.5
   (`webgnn_dec_analysis.py:38-67,81-84`).

Granularity: edge, then node. The RNA multiplier makes node scores cell
line specific.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N07 One input per entity and data type | `load_data.py:69-75` | Each gene node holds RNA, CNV, and two drug-target flags as channels that share the gene's edges. |
| N09 Dense head after the graph layers | `enc_dec/geo_webgnn_decoder.py:108-109,146-149` | An unconstrained bilinear decoder with two trainable matrices maps two node states to the score. |
| N12 Parameters only for present edges | `enc_dec/geo_webgnn_decoder.py:26,29` | One trainable scalar per gene–gene edge, layer, and direction; absent pairs have none. |
| N25 Scores as a graph | `analysis_cell_bindbi_net.py:183-190,265,287` | Averaged edge scores become a networkx graph that is searched for paths between two drug nodes. |
| N32 Edge weights labeled by edge | `webgnn_edge_analysis.py:47-65` | Per-edge scalars are joined to the KEGG edge table for source and target names. |
| N33 Combine scores across training repeats | `webgnn_edge_analysis.py:68-79` | Edge scores from the 5 cross-validation fold models are averaged into one table. |
| N38 Several units per node | `enc_dec/geo_webgnn_decoder.py:19-21,77` | Each node has a channel vector (4, 12, 12, 108 wide), and each edge carries all channels. |
| N45 Feed inputs into named graph nodes | `load_data.py:52-79`; `parse/parse_circle.py:186-197` | Omics rows are matched to gene nodes by symbol and form their initial state; drug nodes get zeros. |
| N61 Merge typed edge tables into one graph | `load_data.py:248-249`; `parse/parse_network.py:36-46` | KEGG gene–gene and DrugBank drug–target tables become one cyclic graph with gene and drug nodes. |
| N62 Restrict the graph to chosen gene sets | `parse/init_parse.py:124` | Only KEGG edges from pathways named "signaling pathway" are kept. |
| N65 Unroll the graph a fixed number of steps | `enc_dec/geo_webgnn_decoder.py:111-133` | Three message-passing steps over the whole cyclic graph, each with its own weights. |
| N68 Normalize within each node's units | `enc_dec/geo_webgnn_decoder.py:78` | Each node's concatenated channel vector is L2-normalized per sample (not layer norm). |
| N69 Sparse product from an index list | `enc_dec/geo_webgnn_decoder.py:55-56,73-74,81-86` | Gather by edge index, multiply per edge, scatter-add; batches tile the weight vector over PyG's offset edge list. |
| N72 Node scores from scored edges | `webgnn_dec_analysis.py:22-28`; `analysis_cell_bindbi_net.py:153-155` | Gene score is the weighted degree over edge scores, times log RNA per cell line. |
| N77 Drop graph nodes that lack data | `parse/parse_circle.py:189-197` | KEGG genes missing from either omics table are deleted with all their edges. |
| N78 Per-edge scalar on a shared channel transform | `enc_dec/geo_webgnn_decoder.py:19-20,42,44,85-86` | Each edge's block is its scalar times the layer's shared `up_proj` or `down_proj` matrix. |
| N79 Aggregate along and against edges | `enc_dec/geo_webgnn_decoder.py:41-44,73-77` | Each layer runs the edges forward and reversed with separate weights and concatenates both. |
| N80 Node's own state as a separate path | `enc_dec/geo_webgnn_decoder.py:21,45,75-77` | A bias-free self transform of each node is concatenated with the two aggregates. |
| N81 Average over incoming edges | `enc_dec/geo_webgnn_decoder.py:61-69` | Messages are scaled by 1/in-degree of the target, which is 1/out-degree in the reversed pass. |
| N82 Read out nodes chosen per sample | `enc_dec/geo_webgnn_decoder.py:136-145` | The head reads the two drug nodes named by each sample's drug pair. |
| N83 Fix the weight of one edge type | `enc_dec/geo_webgnn_decoder.py:49-53` | Drug–target edges keep a constant weight of 1 while gene–gene edges train. |
| N84 One score per edge from its per-step weights | `webgnn_edge_analysis.py:37-45` | Absolute per-edge weights are averaged over 3 layers, then over up and down. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Filter KEGG to signaling pathways, uppercase, deduplicate | `parse/init_parse.py:120-155` | ~35 |
| Drop genes without RNA and CNV and their edges; derive gene list | `parse/parse_circle.py:169-205` | ~37 |
| Restrict DrugBank to graph genes; match drug names to synergy data | `parse/parse_circle.py:207-255` | ~45 |
| Number genes then drugs; renumber edges; mirror drug–target pairs | `parse/parse_network.py:15-46` | ~32 |
| Build the edge index (gene edges first, then drug edges both ways) | `load_data.py:214-258` | ~45 |
| Build per-sample node feature rows (pair flags, omics) | `load_data.py:15-94` | ~80 |
| Per-edge weight vectors plus constant drug weights, tiled per batch | `enc_dec/geo_webgnn_decoder.py:23-29,47-56` | ~17 |
| Per-sample drug-node lookup in the batched graph | `enc_dec/geo_webgnn_decoder.py:136-150` | ~15 |
| Extract, take absolute values, average, and label edge weights by row; average folds | `webgnn_edge_analysis.py:15-79` | ~65 |
| Weighted degree per gene; edge and node filters | `webgnn_dec_analysis.py:16-67` | ~50 |
| Cell-line degree, node filter, drug–drug path search | `analysis_cell_bindbi_net.py:144-300` | ~155 |

## Fragile spots

- Edge identity is positional. Trainable weights line up with the first
  `num_gene_edge` edges, and drug constants are appended after them
  (`enc_dec/geo_webgnn_decoder.py:50,53`). This works only while gene
  edges come first (`load_data.py:248-249`). `num_gene_edge` is counted
  after `drop_duplicates()` (`geo_tmain_webgnn.py:105-108`), but the edge
  index is built without it (`load_data.py:217-219`).
- The drug lookup depends on PyTorch Geometric shifting every `Data`
  attribute whose name contains "index" by the node count when batching.
  `drug_index` holds 1-based global node numbers, from which the model
  subtracts 1 (`geo_loader/read_geograph.py:43`;
  `enc_dec/geo_webgnn_decoder.py:142-145`). Under another name, every
  sample would read the first sample's drug nodes.
- Edge scores are named by CSV row position
  (`webgnn_edge_analysis.py:48-49`), and folds are averaged by row
  position (`:75`).
- The degree table pairs sorted node keys with names by position, which
  holds only if every gene has an edge (`webgnn_dec_analysis.py:28-32`).
  The undirected `nx.Graph` keeps one weight for a reciprocal pair
  (`:22-27`).
- The gene count 2016 is hard-coded in the path analysis
  (`analysis_cell_bindbi_net.py:247`). Drug edge weights are created on
  `'cuda'` whatever the model's device (`enc_dec/geo_webgnn_decoder.py:49,52`).

## Out of scope

- Loss: MSE on synergy score (`enc_dec/geo_webgnn_decoder.py:156-162`).
- Training loop, step learning-rate schedule, and best epoch picked by
  test Pearson (`geo_tmain_webgnn.py:72-91,148-285`).
- Synergy dataset parsing and 5-fold splits (`parse/init_parse.py:14-100`,
  `load_data.py:264-309`).
- Baseline GNNs (`enc_dec/geo_*_decoder.py` other than webgnn),
  `ml_model.ipynb`, plotting (`vis.R`, notebooks).

## Paper vs code

- Eqs. (1)–(2) write `A' = W · A` with `W ∈ R^{n×n}`. Code holds one
  scalar per gene–gene edge (`enc_dec/geo_webgnn_decoder.py:26,29`).
- Features: the paper lists RNA, CNV, and "whether they have a connection
  spanning DA and DB". The code uses [target of one drug, target of both,
  RNA, CNV] (`load_data.py:53-75`).
- The input BatchNorm (`enc_dec/geo_webgnn_decoder.py:106,124`) is not in
  the paper's equations.
- Eqs. (9)–(14) average raw weights; the code averages absolute values
  (`webgnn_edge_analysis.py:37-45`).
- The paper defines weighted degree as the row sum (outgoing) of
  `A'_bind_final`. The code uses the undirected weighted degree
  (`webgnn_dec_analysis.py:22-28`).
- The paper multiplies degree by RNA-seq; the code multiplies by
  `log(RNA+1)` (`analysis_cell_bindbi_net.py:153-155`).
- The paper filters edges below 0.1 and uses shortest paths shorter than
  5. The code calls `filter_edge(0.2)` (`webgnn_dec_analysis.py:81-82`),
  removes no edges on the plotting path (`analysis_cell_bindbi_net.py:184`),
  and lists all simple paths with cutoffs 3–5 (`:265,287`).

## Key excerpts

`enc_dec/geo_webgnn_decoder.py:25-29`

```python
        up_std_gene_edge = torch.nn.init.calculate_gain('relu')
        self.up_gene_edge_weight = torch.nn.Parameter((torch.randn(self.num_gene_edge) * up_std_gene_edge).to(device))
        ### [down_gene_edge_weight] [num_gene_edge / 59241] ###
        down_std_gene_edge = torch.nn.init.calculate_gain('relu')
        self.down_gene_edge_weight = torch.nn.Parameter((torch.randn(self.num_gene_edge) * down_std_gene_edge).to(device))
```

`enc_dec/geo_webgnn_decoder.py:49-56`

```python
        up_drug_edge_weight = torch.ones(self.num_drug_edge).to(device='cuda') # [/107*2=214]
        up_edge_weight = torch.cat((self.up_gene_edge_weight, up_drug_edge_weight), 0)
        # [down_edge_weight] = [down_gene_edge_weight] + [down_drug_edge_weight] [59241+214=59455]
        down_drug_edge_weight = torch.ones(self.num_drug_edge).to(device='cuda')
        down_edge_weight = torch.cat((self.down_gene_edge_weight, down_drug_edge_weight), 0) # [/107*2=214]
        # [batch_up/down_edge_weight] [N*59455]
        batch_up_edge_weight = up_edge_weight.repeat(1, batch_size)
        batch_down_edge_weight = down_edge_weight.repeat(1, batch_size)
```

`enc_dec/geo_webgnn_decoder.py:85-86`

```python
        weight_norm = torch.mul(norm, edge_weight)
        return weight_norm.view(-1, 1) * x_j
```

## Open questions

- KEGG file origin and version: `full_kegg_pathway_list.csv` is read
  (`parse/init_parse.py:122`); no download or build code was found.
- Exact node and edge counts per dataset, and whether the KEGG table
  holds reciprocal pairs, cycles, or duplicates after renumbering: data
  files not read; only the paper (2016 genes) and code comments checked.
- Parsing writes under `../data/...` (`parse/parse_network.py:17`), and
  training reads `./<dataset>/filtered_data/...`
  (`geo_tmain_webgnn.py:97`). Code that moves the files between them was
  not found.
- The paper's supplementary note on preprocessing was not read.
