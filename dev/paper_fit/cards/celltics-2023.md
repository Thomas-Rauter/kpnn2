# celltics-2023: CellTICS

| Field | Value |
|-------|-------|
| Paper | Yin Q, Chen L, 2023. CellTICS: an explainable neural network for cell-type identification and interpretation based on single-cell RNA-seq data. Briefings in Bioinformatics 25(1):bbad449. 10.1093/bib/bbad449 |
| Code | `https://github.com/qyyin0516/CellTICS` at `7a9fcfe736b4cc26c22a01018ad93cce59e1e21f` (HEAD, 2026-04-16) |
| Other code | none found |
| Framework | Python 3; TensorFlow `tf.compat.v1` graph mode with v2 behavior disabled (README: TF >= 2.9, Keras >= 2.9; Keras supplies only the initializer); networkx, pandas |
| License | none found (no LICENSE file in the tree) |
| Full text | PubMed Central PMC10703497 (equations are images and were not read; Supplementary Figure S2 not read) |
| Extracted | 2026-10-11, Claude Code, Opus 5.5 |

## Summary

CellTICS classifies scRNA-seq cells into cell types, then into sub-cell
types (cell type plus a condition such as age or disease). Marker genes
from the labeled reference data are the inputs. The Reactome hierarchy is
cut to L = 5 levels below an added root, with copy nodes padding short
branches and genes only at the bottom. Each pathway is one tanh unit, and
each graph layer is a dense weight times a fixed 0/1 mask. Five separately
trained networks, over the bottom 1 to 5 pathway layers and each with a
dense softmax output, vote on the class. Pathways are scored by the gap
between their two highest per-class mean activations.

## Prior-knowledge graph

- Source: Reactome. `reactome/ReactomePathwaysRelation.txt` holds
  parent-child pathway pairs, with species chosen by an ID substring
  (`MMU`, `HSA`, `RNO`) in column 0 (`code/pathway_hierarchy.py:50-54`,
  `:66-75`). The columns are named `child`, `parent`, but each edge runs
  from the Reactome parent to its child (`:53`, `:75`).
  `reactome/Ensembl2Reactome_All_Levels.txt` maps Ensembl IDs to pathways
  at every level, with species chosen by ID string range (`:6-16`).
  Pathway names are loaded and never used (`:44-48`). Release not stated.
- Size: in the shipped relation file, mouse has 1,711 pathways, 1,726
  directed parent→child edges, and 27 parentless pathways. 31 pathways
  have several parents, so the hierarchy is a DAG, not a tree. Edges are
  unsigned, and acyclicity is not checked. Built network size depends on
  the marker genes and is not given in the paper text.
- Data to nodes: rows are Ensembl gene IDs (symbols converted through
  BioMart if `-ensembl` is false, `code/utils.py:7-34`). Gene-pathway rows
  are kept for data genes only (`code/pathway_hierarchy.py:124-147`).
  Leaf pathways with no data gene are removed until every leaf has one
  (`:149-160`). Genes join the bottom pathway level only. Genes in no
  pathway of the final network are dropped, and rows are reordered to the
  mask's gene order (`:204-206`). Inputs are marker genes chosen from the
  reference labels (`code/gene_expression.py:86-95`), so the pruned graph
  depends on the reference set.
- Built at: `code/pathway_hierarchy.py:6-210`.

## Architecture

- Layers: L graph layers (`-n_hidden_layer`, default 5,
  `code/main.py:25`). Edges go root → pathways → genes. A node's level is
  its shortest-path distance from the root (ego-graph differences,
  `code/pathway_hierarchy.py:117-121`, `:179-183`), and the graph is cut
  at L steps (`:162`). Genes are level L+1. Level 0 (the root) is replaced
  by the class labels (`:184`). Childless nodes of the cut graph
  (`:163-176`) are handled in three ways:
  - A level-L pathway that had children gets the data genes of all its
    leaf descendants in the pruned hierarchy (`:166-169`).
  - A childless level-L pathway gets its own genes (`:170-172`).
  - A leaf at level d < L gets a chain of L − d copy nodes, `<id>_copy1`
    and onward, and the last copy gets the leaf's genes (`:87-97`,
    `:173-176`).
- Units per node: one tanh unit with its own bias
  (`code/neural_network.py:42-46`, `:73-80`).
- Connections that skip layers: none. Masks cover level i → i+1 edges
  only (`code/pathway_hierarchy.py:197-202`). Edges that skip levels or
  stay within one level (parents at different depths) are dropped.
- Outputs or losses at inner layers: no shared inner heads. For each
  k = 1..L, a new network is trained from scratch, with the bottom k graph
  layers (genes → level L → ... → level L−k+1) and one dense linear output
  unit per class (`code/main.py:73-86`; `code/neural_network.py:35-56`,
  `:81-99`). Each predicts the argmax of its softmax on the query cells
  (`code/utils.py:64-71`). The final label is the most frequent of the L
  predictions (`code/main.py:103-104`).
- Parts the graph does not constrain: each network's output layer, dense
  over the top graph layer (`code/neural_network.py:49-51`, `:75-77`). An
  all-ones class × level-1 mask is built and discarded
  (`code/pathway_hierarchy.py:196`, `:207-209`).
- Other structure: in stage 2, the procedure reruns per cell type on that
  type's reference cells with sub-cell-type labels. Query cells are routed
  by their stage-1 prediction (`code/main.py:115-165`). The graph is
  rebuilt from the same genes (`:62`, `:120-132`). No recurrence,
  attention, or decoder.
- Defined at: `code/neural_network.py:35-115`, `code/main.py:59-172`.

## Connectivity mechanism

Each graph layer holds a full `W` (upper-level nodes × lower-level nodes;
data are genes × cells), Glorot normal over the full shape
(`code/neural_network.py:42-43`). The mask is a second variable `M`,
trainable and randomly initialized by default (`:47`). In the graph,
`tf.assign(M, mask)` writes the 0/1 array into `M`, and the assign's
output multiplies `W` elementwise before every matmul (`:68`, `:73`,
`:79`). TF defines no gradient through an assign, so Adam leaves `M`
alone (`:98`). Absent entries of `W` get no cross-entropy gradient, but
L2 covers every entry of every `W` (`:83-85`), so they decay toward zero
without affecting the output. Masks are pandas frames indexed by node
name, converted to numpy (`code/pathway_hierarchy.py:195-209`).

## Interpretation

Per-class mean activation. Each trained network is run on all reference
(training) cells, and the hidden activations are labeled by node and cell
name (`code/neural_network.py:113`; `code/main.py:87-98`). For each
network and hidden layer, each node's activation is averaged over each
true class (`code/utils.py:85-91`). The score is the largest class mean
minus the second. Nodes with a gap of at least 0.1 (default) are kept for
the top class (`code/utils.py:98-103`; `code/main.py:33`). Suffixes
`_copy` and `_copy1` to `_copy9` are stripped, so copies report under
their pathway's Reactome ID (`code/utils.py:106-126`). Duplicates keep the
largest gap, first within a network (`:105`, `:127`) and then across
networks (`:128-135`). Output is a CSV of (Reactome ID, class, gap) per
stage and cell type (`code/main.py:112`, `:172`), without pathway names.
Genes are not scored. A cell type with one sub-cell type copies its
stage-1 rows (`code/utils.py:76-79`). There are no repeats or seeds.

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N01 Read a DAG from an edge table | `code/pathway_hierarchy.py:50-54`, `:66-75` | Reactome parent-child pathway pairs, restricted to one species by an ID substring. |
| N73 Add a root above top-level nodes | `code/pathway_hierarchy.py:78-82` | A `root` joins every parentless pathway. Levels count from it, then its level becomes the class outputs (`:184`). |
| N06 Connect features to nodes by membership | `code/pathway_hierarchy.py:124-147`, `:204-206` | Data genes are matched by Ensembl ID to Reactome's gene-to-pathway table. Genes outside the final network are removed. |
| N20 Keep only nodes above the data | `code/pathway_hierarchy.py:149-160` | Leaf pathways without a data gene are removed until stable. |
| N56 Derive the graph from training data | `code/gene_expression.py:86-95`; `code/pathway_hierarchy.py:147-160` | Inputs are marker genes scored from reference labels. Only pathways reaching them survive, so each reference set yields its own network. |
| N02 Layer nodes by depth from a root | `code/pathway_hierarchy.py:117-121`, `:162`, `:179-184` | Shortest distance from the root sets the level. The cut is at L levels (default 5), with genes at L+1. |
| N21 Propagate memberships up the hierarchy | `code/pathway_hierarchy.py:166-169` | A cut-level pathway takes the data genes of all its leaf descendants. |
| N19 Pad short branches with copy nodes | `code/pathway_hierarchy.py:87-97`, `:173-176` | A shallow leaf gets a `_copy` chain to the bottom level, and the last copy takes its genes. |
| N03 Keep only edges between adjacent layers | `code/pathway_hierarchy.py:197-202` | Mask entries are set only for level i → i+1 edges. |
| N10 One unit per node | `code/neural_network.py:42-46`, `:73-80` | Each pathway or copy node is one tanh unit with its own bias. |
| N22 Mask a dense weight | `code/neural_network.py:47`, `:68`, `:73`, `:79` | A full `W` is multiplied by a fixed 0/1 mask every forward pass. L2 still acts on absent entries (`:83-85`). |
| N09 Dense head after the graph layers | `code/neural_network.py:49-51`, `:75-77` | One dense linear layer maps the top graph layer to the class logits, with nothing in between. |
| N106 Separate networks cut at each depth | `code/main.py:73-104` | Five independently trained networks over the bottom 1 to 5 graph layers, each with its own output layer, vote on the class. |
| N31 Read out node activations | `code/neural_network.py:113`; `code/main.py:87-98` | Hidden activations on all reference cells are kept per network, labeled by node name. |
| N18 Aggregate scores over a sample subset | `code/utils.py:85-91` | Each node's activation is averaged per class. |
| N107 Class gap of mean activations | `code/utils.py:98-103`, `:128-135` | Score = top class mean minus second, kept at ≥ 0.1, max over layers and networks. |
| N108 Fold copy-node scores into the original node | `code/utils.py:105-127` | Copy suffixes are stripped, and each pathway keeps its largest score. |
| N17 Map scores to node names | `code/main.py:91-97`; `code/utils.py:101-102` | Scores are reported under Reactome IDs, without human-readable names. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Read gene-pathway table, filter species, rewrite IDs | `code/pathway_hierarchy.py:6-33` | ~28 |
| Read hierarchy, filter species, add root | `code/pathway_hierarchy.py:36-84` | ~45 |
| Map data genes to pathways; prune unannotated leaves | `code/pathway_hierarchy.py:124-160` | ~30 |
| Cut at depth L, attach genes, add copy chains | `code/pathway_hierarchy.py:87-114`, `:162-177` | ~45 |
| Assign levels, build masks, align data rows | `code/pathway_hierarchy.py:117-121`, `:179-210` | ~35 |
| Mask variable created and assigned in the graph | `code/neural_network.py:47-48`, `:63-68` | ~6 |
| Gene symbols to Ensembl IDs via BioMart | `code/utils.py:7-34` | ~28 |
| One network per depth, then majority vote (both stages) | `code/main.py:71-105`, `:133-165` | ~60 |
| Label activations with node and cell names | `code/main.py:87-98`, `:149-158` | ~22 |
| Class-gap scoring, copy-name stripping, max merge | `code/utils.py:74-136` | ~63 |

## Fragile spots

- ID rewrite. The mouse row filter spans `ENSMUSG` to `ENSMUST`, so it
  admits protein and transcript IDs (`code/pathway_hierarchy.py:9`). Each
  ID is rewritten as `ENSMUSG` plus its own number (`:22-24`; human and
  rat alike, `:11-13`, `:26-30`). An Ensembl transcript or protein number
  is not its gene's, so such rows would join pathways to unrelated genes.
- Run grouping. `gene_mapping` builds gene → pathways from runs of
  consecutive rows (`:126-134`). The last run is never stored, and a gene
  with two runs keeps only the later one.
- Genes annotated only to internal pathways never enter. A cut-level
  pathway takes only its leaf descendants' genes (`:167-169`).
- Mask variable. `M` is trainable and randomly initialized
  (`code/neural_network.py:47`). Masking holds only because the forward
  pass uses the output of `tf.assign`, not the variable (`:68`, `:73`).
- Copy names. Stripping covers `_copy1` to `_copy9` only
  (`code/utils.py:106-126`), and chains reach L − 1 copies, so
  `-n_hidden_layer` ≥ 11 leaves `_copy10` names in the output.
- Vote ties (e.g. 2-2-1) are broken by pandas `describe()['top']`, not by
  the code (`code/main.py:104`, `:165`).

## Out of scope

- Marker genes (`code/gene_expression.py:5-38`, `:81-95`): per class,
  high- and low-expression specificity scores on class-summed counts. The
  first round(0.05·G) and round(0.1·G) genes, in table order, at or above
  the 0.95 and 0.9 quantiles are kept, then pooled.
- Normalization to counts per 10,000, then log2(x + 1), and the gene
  intersection of reference and query (`code/gene_expression.py:51-77`).
- Loss and optimizer: cross-entropy plus L2 (γ = 1e-4). Adam at 1e-3,
  batch 32, 10 or 50 epochs (`code/neural_network.py:81-112`;
  `code/main.py:27-31`). Stage-2 routing; accuracy and macro F1
  (`code/evaluate.py:16-19`). Not in the repository: the grid search and
  the R analyses (QC, split, GSEA).

## Paper vs code

- The paper says "a predictive layer is added after each hidden layer".
  The code trains L separate networks of depth 1..L, with no shared
  weights (`code/main.py:73-86`). The importance section does say "nested
  networks".
- The paper calls Reactome a tree, but the mouse file has 31 pathways
  with several parents.
- The paper connects a too-deep leaf "to upper-level pathways". The code
  joins each cut-level pathway to its leaf-pathway descendants' genes only
  (`code/pathway_hierarchy.py:166-169`).
- The README's sub-cell-type epoch default is 10. The code's is 50
  (`code/main.py:28`), as in the paper. The paper lists Keras, but the
  model is `tf.compat.v1` (`code/neural_network.py:3-4`).

## Key excerpts

`code/pathway_hierarchy.py:164-176`

```python
    for node in sub_terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node)) #len of distance: num of nodes in the shortest path
        if (distance == n_hidden + 1) & (node not in original_terminal_nodes):
            part_graph = nx.ego_graph(original_network, node, radius=100)
            corresponding_terminal_nodes = [n for n, d in part_graph.out_degree() if d == 0]
            sub_graph = add_gene_edges(sub_graph, node, corresponding_terminal_nodes, genes_df)
        elif (distance == n_hidden + 1) & (node in original_terminal_nodes):
            corresponding_terminal_nodes = node
            sub_graph = add_gene_edges(sub_graph, node, corresponding_terminal_nodes, genes_df)
        elif distance <= n_hidden:
            diff = n_hidden - distance + 1
            sub_graph, copy_node = add_duplicated_edges(sub_graph, node, diff)
            sub_graph = add_gene_edges(sub_graph, copy_node, node, genes_df)
```

`code/pathway_hierarchy.py:197-202`

```python
    for i in range(1, n_level-1):
        masking[i] = pd.DataFrame(data=0, index=layers_node[i], columns=layers_node[i + 1])
        for n in layers_node[i]:
            for node in layers_relationship[i - 1][n]:
                if node in layers_node[i + 1]:
                    masking[i].loc[n, node] = 1
```

`code/neural_network.py:66-80`

```python
        if i != (n - 1):
            M[i + 1] = parameters["M" + str(i + 1)]
            M[i + 1] = tf.assign(M[i + 1], masking[len(masking) - i - 1])
    Z = {}
    A = {}
    for i in range(1, n + 1):
        if i == 1:
            Z[i] = tf.add(tf.matmul(tf.multiply(M[i], W[i]), X),  b[i])
            A[i] = tf.nn.tanh(Z[i])
        elif i == n:
            Z[i] = tf.add(tf.matmul(W[i], A[i - 1]), b[i])
            A[i] = Z[i]
        else:
            Z[i] = tf.add(tf.matmul(tf.multiply(M[i], W[i]), A[i - 1]),  b[i])
            A[i] = tf.nn.tanh(Z[i])
```

## Open questions

- Reactome release of the shipped files: not in the paper or README.
- Whether `Ensembl2Reactome_All_Levels.txt` (Git LFS, not fetched) holds
  transcript or protein rows. This decides whether the ID rewrite and the
  run grouping (`code/pathway_hierarchy.py:22-30`, `:126-134`) change
  edges.
- Layer sizes for the paper's datasets: not in the text (Supplementary
  Figure S2 not read).
- Whether HEAD (2026-04-16) is the paper's code. The repository dates
  from 2022-10-23, and one commit was fetched.
- The class names, `_copy` naming, ego-graph levels, and comments
  (`code/pathway_hierarchy.py:60`, `:67`) resemble P-NET's Reactome
  module. Not checked against that repository.
