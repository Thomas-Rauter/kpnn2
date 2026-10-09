# pathhdnn-2025: PathHDNN

| Field | Value |
|-------|-------|
| Paper | Li X, Pan B, He Y, et al., 2025. PathHDNN: a pathway hierarchical-informed deep neural network framework for predicting immunotherapy response and mechanism interpretation. Genome Medicine. 10.1186/s13073-025-01584-9 |
| Code | `https://github.com/hanjunwei-lab/PathHDNN` at `54038946c444ef5ed8d94c74930c2f2d06568fbc` |
| Other code | None found. The local package `PathHDNN/` carries the class names and docstrings of a "Biologically Informed Neural Network (BINN)" library (`PathHDNN/binn.py:11-17`). `config.yaml:5` pins `binn` 0.0.2, and the scripts import from `binn` (see Open questions). `PathHDNN/network.py:8-12` says it was adapted from P-NET's `reactome.py`. `PathHDNN/Comparative algorithm/` holds baselines, not PathHDNN. |
| Framework | Python 3.9.13, PyTorch 2.8.0, Lightning, networkx, pandas 2.3.2, shap (unpinned) (`config.yaml:1-7`); preprocessing in R |
| License | None found |
| Full text | Europe PMC full-text XML of PMC12729056 (Methods, Data availability, captions). Equations are images there; Eq. 1 and the score-normalization formula were not readable. |
| Extracted | 2026-10-09, Claude Code, claude-opus-5-5 |

## Summary

PathHDNN predicts whether a melanoma patient responds to immune
checkpoint blockade (anti-PD1 or anti-CTLA4) from binary somatic
alterations per gene (non-silent mutation, amplification, deletion).
The prior is the Reactome pathway hierarchy plus gene-to-pathway
membership. The network is a P-NET-style feedforward stack. Alteration
features feed the deepest pathway layer, each pathway layer feeds its
parent layer through a masked dense layer, and a dense output layer
reads the top-level pathways to give two logits. Node importance is
per-layer DeepExplainer SHAP, as mean absolute value over samples.

## Prior-knowledge graph

- Source: Reactome. Hierarchy `data/pathways.tsv` (`parent`, `child`
  R-HSA ids). Memberships come from `data/ReactomePathways.gmt`
  (`PathHDNN/DataProcessing.R:27-36`, `:65-100`). Release not found.
- Size: 2,603 directed parent-to-child edges over 2,585 pathways, 29
  without a parent, 36 with several parents, no cycles, unsigned
  (counted from `data/pathways.tsv`). The code adds a node `root` above
  all parentless nodes (`PathHDNN/network.py:93-99`). The anti-PD1
  membership table (`data/SKCM144/pre-processing/reactome_data13.txt`)
  maps 4,227 features (3,996 `_mut`, 231 `_amp`, no `_del`) to 2,120
  pathways in 40,690 rows. Node counts per layer not determined.
- Subsetting: keep pathways that directly hold a data feature, plus all
  their ancestors (`PathHDNN/network.py:242-266`).
- Data to nodes: a feature is a gene name plus a data-type suffix. The
  gene's memberships are copied once per suffix
  (`PathHDNN/DataProcessing.R:93-100`). Each feature maps to its
  annotated pathways and all their ancestors
  (`PathHDNN/network.py:165-182`). Only deepest-layer nodes take
  features. Each takes every feature annotated to it or to any
  descendant, including descendants below the depth cut
  (`PathHDNN/network.py:118-133`). A feature annotated only to pathways
  above the deepest layer is not an input
  (`PathHDNN/network.py:149-153`). Data rows are reordered to the
  model's feature list by name, and missing features become zero
  (`PathHDNN/train_PathHDNN.py:13-22`, `:34`).
- Built at: `PathHDNN/network.py:39-266`;
  `PathHDNN/DataProcessing.R:27-103`.

## Architecture

- Layers: `n_layers=4` (`PathHDNN/train_PathHDNN.py:45`). A pathway's
  layer is its shortest-path distance from `root`. Only nodes within 4
  hops are kept (`PathHDNN/network.py:211`, `:222-239`). Five 0/1
  matrices are built, data side first: features × depth 4, 4 × 3,
  3 × 2, 2 × 1, 1 × `root` (`PathHDNN/network.py:136-162`). The first
  four mask the four linear layers. The fifth only labels edges in the
  importance table (`PathHDNN/explainer.py:58-60`). No `root` unit.
- Short branches: a leaf pathway above depth 4 gets a chain of copy
  nodes (`<name>_copy1`, ...) down to depth 4, one edge per copy
  (`PathHDNN/network.py:196-219`). Copies keep the pathway's name
  (`PathHDNN/network.py:124`, `:235-237`).
- Edges to a node that is not exactly one layer deeper are dropped by
  an inner join (`PathHDNN/network.py:154-157`).
- Units per node: one, ordered by sorted node name
  (`PathHDNN/network.py:152-159`).
- Per graph layer: masked `nn.Linear` with bias, `BatchNorm1d`,
  `Dropout(0.5)`, `tanh` (`PathHDNN/binn.py:50`, `:301-318`).
- Skip connections and inner outputs: none in the scripts
  (`residual=False`, `PathHDNN/train_PathHDNN.py:45`). An unused
  variant adds a dense linear+sigmoid head per graph layer and averages
  them (`PathHDNN/binn.py:249-263`, `:324-362`).
- Unconstrained: the output layer `nn.Linear(n_depth1, 2)`
  (`PathHDNN/binn.py:319`) and the BatchNorm layers.
- Other structure: none. Defined at `PathHDNN/binn.py:46-129`,
  `:292-321`.

## Connectivity mechanism

Each graph layer is a full `nn.Linear`. The code calls
`torch.nn.utils.prune.custom_from_mask` with the transposed 0/1 matrix
(`PathHDNN/binn.py:306-310`). Pruning keeps the dense trainable tensor
as `weight_orig` and the mask as a buffer, and recomputes
`weight = weight_orig * weight_mask` in a forward pre-hook. Absent
entries exist and get zero gradient from the loss. Adam's
`weight_decay=1e-3` acts on all parameters (`PathHDNN/binn.py:201-203`),
so masked entries still shrink, without effect on the output. The mask
stays fixed. Init: `xavier_uniform_(m.weight)` runs after pruning has
replaced `weight` with a derived tensor (`PathHDNN/binn.py:106`,
`:266-268`). So the masked layers appear to keep PyTorch's default
`nn.Linear` init, and only the output layer gets Xavier (read, not
run). Bias is dense, one per unit.

## Interpretation

- Method: SHAP DeepExplainer, once per `nn.Linear` (four masked layers
  and the output layer), as `(model, layer)`. That gives attributions
  to the data features and to the post-tanh activations of the depth-4
  to depth-1 pathway units (`PathHDNN/explainer.py:222-259`).
  Background and explained samples are both all training samples
  (`PathHDNN/model_explain.py:59-63`).
- Granularity: input feature and hidden node, per output class.
- Names: layer *k* uses the row names of matrix *k*
  (`PathHDNN/binn.py:77-87`). Output is a long table with one row per
  graph edge and class: source and target id, name, layer, and the
  source node's score (`PathHDNN/explainer.py:37-85`).
- Aggregation: absolute value, mean over the explained samples
  (`PathHDNN/explainer.py:61-63`). An optional normalization divides by
  log2(nodes upstream + downstream)
  (`PathHDNN/importance_network.py:316-331`). It is the class default,
  but the script passes `norm_method=False`
  (`PathHDNN/model_explain.py:74`). `explain_average`, which would
  average over re-initialized trainings, is not called
  (`PathHDNN/explainer.py:117-168`).
- The table becomes a networkx graph for up/downstream subgraphs and
  Sankey plots (`PathHDNN/importance_network.py:144-246`).

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| N01 Read a DAG from an edge table | `PathHDNN/network.py:50-52`, `:93-95` | Reactome parent/child table read into a directed graph. |
| N02 Layer nodes by depth from a root | `PathHDNN/network.py:96-99`, `:209-239` | Adds `root` above parentless pathways, layers by distance from it, keeps 4 levels, data enters at depth 4. |
| N03 Keep only edges between adjacent layers | `PathHDNN/network.py:154-157` | Inner join drops edges whose child is not in the next deeper layer. |
| N06 Connect features to nodes by membership | `PathHDNN/network.py:118-133`, `PathHDNN/DataProcessing.R:27-36` | Feature-to-pathway table from the Reactome GMT, matched by name. |
| N07 One input per entity and data type | `PathHDNN/DataProcessing.R:93-100` | `_mut`, `_amp`, `_del` features per gene, each with the gene's memberships. |
| N09 Dense head after the graph layers | `PathHDNN/binn.py:319` | One dense layer from top-level pathways to two logits. |
| N10 One unit per node | `PathHDNN/binn.py:301-302`, `PathHDNN/network.py:152-159` | Layer width is the node count. |
| N11 Standard layers between graph layers | `PathHDNN/binn.py:303-318` | BatchNorm, dropout, and tanh after every masked layer. |
| N14 Attribution to inputs | `PathHDNN/explainer.py:241-250` | SHAP on the input features. |
| N15 Attribution to hidden nodes | `PathHDNN/explainer.py:241-250` | SHAP on each pathway layer's activations. |
| N17 Map scores to node names | `PathHDNN/explainer.py:47-82` | Scores labeled with pathway ids and feature names per layer. |
| N18 Aggregate scores over a sample subset | `PathHDNN/explainer.py:61-63` | Mean absolute SHAP over the explained samples. |
| N19 Pad short branches with copy nodes | `PathHDNN/network.py:196-219` | Leaf pathways above depth 4 get copy chains down to depth 4. |
| N20 Keep only nodes above the data | `PathHDNN/network.py:252-266` | Hierarchy cut to data-annotated pathways and their ancestors. |
| N21 Propagate memberships up the hierarchy | `PathHDNN/network.py:165-182`, `:118-133` | Depth-4 nodes take the features of all descendants, including those below the cut. |
| N22 Mask a dense weight | `PathHDNN/binn.py:306-310` | Dense weight times a fixed 0/1 mask via PyTorch pruning. |
| N23 Deep SHAP against a background set | `PathHDNN/explainer.py:246-247`, `PathHDNN/model_explain.py:60-63` | DeepExplainer per layer, training samples as background. |
| N24 Normalize node scores by connectivity | `PathHDNN/importance_network.py:316-331` | Divide by log2 of the up- plus downstream node count (in the paper; off in the script). |
| N25 Scores as a graph | `PathHDNN/explainer.py:37-85`, `PathHDNN/importance_network.py:144-246` | Node scores joined to edges and layers for subgraph queries and Sankey plots. |

## Glue the authors wrote

| What | Where | Lines |
|------|-------|-------|
| Per-type feature names and membership tables | `PathHDNN/DataProcessing.R:27-100` | ~60 |
| Read hierarchy, add `root`, cut to data-annotated pathways | `PathHDNN/network.py:83-101`, `:242-266` | ~45 |
| Propagate memberships, attach features to deepest layer | `PathHDNN/network.py:103-134`, `:165-182` | ~50 |
| Layer by depth, pad short branches, build sorted 0/1 matrices | `PathHDNN/network.py:136-162`, `:185-239` | ~80 |
| Apply masks via pruning, record unit names | `PathHDNN/binn.py:74-87`, `:292-321` | ~44 |
| Align data rows to the feature list | `PathHDNN/train_PathHDNN.py:13-36` | ~24 |
| Per-layer SHAP, scores to named edge table | `PathHDNN/explainer.py:24-85`, `:222-259` | ~100 |
| Importance graph, normalization, subgraphs | `PathHDNN/importance_network.py:144-332` | ~190 |

## Fragile spots

- Init after pruning: `xavier_uniform_` writes to the derived `weight`,
  which the pre-hook overwrites on the next forward
  (`PathHDNN/binn.py:106`, `:266-268`). `reset_params` and
  `init_weights`, used by `explain_average`, hit the same tensor
  (`PathHDNN/binn.py:237-247`).
- Copies share a name (`PathHDNN/network.py:235-237`), and the
  explainer keys node ids by name (`PathHDNN/explainer.py:50-55`). So a
  pathway and its copies share one id, and the importance graph keeps
  whichever row came last (`PathHDNN/importance_network.py:152-159`).
- Only nodes with out-degree 0 at the cut take features
  (`PathHDNN/network.py:118`). A depth-4 node with an edge to another
  kept node takes none, and the next join drops it
  (`PathHDNN/network.py:154-157`).
- `explain` reads SHAP output as class × sample × feature
  (`PathHDNN/explainer.py:61-66`). shap is not pinned, and multi-output
  layouts differ across shap versions.

## Out of scope

- Loss: two-logit softmax cross-entropy (`PathHDNN/train_PathHDNN.py:105`).
- Training: manual loop, Adam lr 0.001, 10,000 epochs; predictions kept
  at a hard-coded epoch (`PathHDNN/train_PathHDNN.py:88-117`).
- Preprocessing: binarization, frequency and Cox filters, per-cohort
  StandardScaler (`PathHDNN/DataProcessing.R:8-64`,
  `PathHDNN/train_PathHDNN.py:35`).
- Plotting (`PathHDNN/plot.py`, `PathHDNN/plot_code.R`); unused
  `PathHDNN/feature_selection.py` and `PathHDNN/sklearn.py`; baselines
  under `PathHDNN/Comparative algorithm/`.

## Paper vs code

- Depth: the paper builds "a six-layer pathway hierarchical network".
  The code has 4 masked layers plus a dense output
  (`PathHDNN/train_PathHDNN.py:45`). How six is counted is unclear.
- Loss: the paper uses sigmoid binary cross-entropy. The code uses
  softmax cross-entropy on two logits (`PathHDNN/train_PathHDNN.py:105`).
- Batch size: the paper and README say 36. The script uses 27
  (`PathHDNN/train_PathHDNN.py:51`).
- The paper does not mention BatchNorm, tanh, or weight decay.
- Normalization: the paper normalizes node scores by subgraph size. The
  script turns it off (`PathHDNN/model_explain.py:74`).
- Filter: the paper keeps genes altered in at least three samples. The
  code needs more than three, and for mutations also a univariate Cox
  p < 0.05 (`PathHDNN/DataProcessing.R:19`, `:24-26`).
- Copies: the paper says a copy "replicated the terminal node's
  connectivity". The code chains copies with one edge each and attaches
  features to the last copy only (`PathHDNN/network.py:196-206`).

## Key excerpts

`PathHDNN/network.py:170-181`

```python
    for translation in mapping["input"]:
        ids = mapping[mapping["input"] == translation]["translation"]
        for id in ids:
            if graph.has_node(id):
                connections = graph.subgraph(
                    nx.single_source_shortest_path(graph, id).keys()
                ).nodes
                for connection in connections:
                    components["input"].append(translation)
                    components["connections"].append(connection)
    components = pd.DataFrame(components)
    components.drop_duplicates(inplace=True)
```

`PathHDNN/network.py:209-219`

```python
def _complete_network(G, n_levels=4):
    nr_copies = 0
    sub_graph = nx.ego_graph(G, "root", radius=n_levels)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source="root", target=node))
        if distance <= n_levels:
            nr_copies = nr_copies + n_levels - distance
            diff = n_levels - distance + 1
            sub_graph = _add_edges(sub_graph, node, diff)
    return sub_graph
```

`PathHDNN/binn.py:301-310`

```python
    for n in range(len(layer_sizes) - 1):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n + 1], bias=bias)
        layers.append((f"Layer_{n}", linear_layer))  # linear layer
        layers.append((f"BatchNorm_{n}", nn.BatchNorm1d(layer_sizes[n + 1])))
        if connectivity_matrices is not None:
            prune.custom_from_mask(
                linear_layer,
                name="weight",
                mask=torch.tensor(connectivity_matrices[n].T.values),
            )
```

## Open questions

- Which code ran: `PathHDNN/train_PathHDNN.py:1` and
  `PathHDNN/model_explain.py:57`, `:72` import from `binn`, pinned at
  0.0.2 (`config.yaml:5`), not from the local `PathHDNN/` copy. Whether
  that package matches the local files was not checked.
- Reactome release: not found in the paper, README, or code.
- Node counts per layer: not determined without running the code.
- Which class's scores the paper ranks: not found in
  `PathHDNN/model_explain.py` or `PathHDNN/plot_code.R`.
- Eq. 1 and the normalization formula are images in the full text and
  were not read.
