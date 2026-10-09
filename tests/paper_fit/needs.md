# Needs catalog

What published models need to do to get prior knowledge into,
through, or out of a neural network. Extraction sessions grow
this file. Assessment sessions read it.

## Rules

- **Paper-neutral.** Write a need as what the model has to do,
  in plain deep-learning terms. Do not use names from this
  repository's package, and do not say whether anything covers
  the need. Coverage belongs in `runs/`.
- **Evidence first.** A need enters only together with a card
  that cites code for it.
- **Granularity.** One need is one thing a library could provide
  or fail to provide. "Uses a sparse network" is too broad.
  "ReLU after the third layer" is too narrow. Test: could two
  papers from different fields share this need?
- **Stable IDs.** IDs are `N01`, `N02`, and so on, in order of
  entry. Never renumber or reuse an ID. To merge a duplicate,
  keep both rows and write `merged into Nxx` in the later row's
  description.
- **Area** is one of `graph` (reading and shaping the prior),
  `architecture` (where nodes sit and how layers connect),
  `mechanism` (how absent edges stay absent), `training`, and
  `interpretation`.

## Catalog

| ID | Area | Need | Description | Papers |
|----|------|------|-------------|--------|
| N01 | graph | Read a DAG from an edge table | Build a directed acyclic graph from a table of parent-child pairs, optionally restricted to one category column (e.g. one ontology namespace). | mulgonet-2026, pathhdnn-2025 |
| N02 | architecture | Layer nodes by depth from a root | Place each node in the layer given by its shortest-path distance from a root, keep a fixed number of levels, and let data enter at the deepest level and flow toward the root. | mulgonet-2026, pathhdnn-2025 |
| N03 | architecture | Keep only edges between adjacent layers | After layering, drop edges that join two nodes of the same layer or skip one or more layers. | mulgonet-2026, pathhdnn-2025 |
| N04 | graph | Prune nodes with no child below | Remove nodes that have no child in the next layer toward the data, working from the data side up, so every kept node can receive signal. | mulgonet-2026 |
| N05 | graph | Filter nodes by gene-set size | Drop nodes whose annotated input set is larger (or smaller) than a threshold before building the network. | mulgonet-2026 |
| N06 | graph | Connect features to nodes by membership | Join input features to first-layer nodes through a membership table (gene sets, annotations) matched by name; one feature may join many nodes; features with no node are removed. | mulgonet-2026, pathhdnn-2025 |
| N07 | architecture | One input per entity and data type | Give each entity (e.g. gene) one input feature per data type (e.g. omics layer), every copy carrying the same edges to the first node layer. | mulgonet-2026, pathhdnn-2025 |
| N08 | architecture | Parallel graph branches on one input | Build several graph-constrained subnetworks from different graphs on the same input and merge their outputs. | mulgonet-2026 |
| N09 | architecture | Dense head after the graph layers | Put unconstrained fully connected layers between the last graph layer and the output. | mulgonet-2026, pathhdnn-2025 |
| N10 | architecture | One unit per node | Represent each graph node by exactly one unit with its own bias and a shared nonlinearity. | mulgonet-2026, pathhdnn-2025 |
| N11 | architecture | Standard layers between graph layers | Insert ordinary layers such as dropout or normalization between consecutive graph-constrained layers. | mulgonet-2026, pathhdnn-2025 |
| N12 | mechanism | Parameters only for present edges | Hold one trainable weight per edge of the graph; absent edges have no parameter and receive no gradient. | mulgonet-2026 |
| N13 | training | Regularize edge weights per layer | Apply a weight penalty (e.g. L2) to the edge weights of chosen graph layers only. | mulgonet-2026 |
| N14 | interpretation | Attribution to inputs | Score each input feature for its contribution to the output, per sample. | mulgonet-2026, pathhdnn-2025 |
| N15 | interpretation | Attribution to hidden nodes | Score each hidden unit of every graph layer for its contribution to the output, per sample. | mulgonet-2026, pathhdnn-2025 |
| N16 | interpretation | Integrated gradients from a zero baseline | Compute integrated gradients of the output with respect to inputs or hidden activations, using an all-zero input as baseline and a set number of steps. | mulgonet-2026 |
| N17 | interpretation | Map scores to node names | Return attribution scores labeled with the graph node or input feature name each unit stands for, per layer. | mulgonet-2026, pathhdnn-2025 |
| N18 | interpretation | Aggregate scores over a sample subset | Combine per-sample scores over a chosen subset of samples (e.g. one class) into one score per node, e.g. by summing. | mulgonet-2026, pathhdnn-2025 |
| N19 | graph | Pad short branches with copy nodes | When a leaf node sits above the deepest layer, add a chain of copy nodes, one per missing layer, each joined to the next by one edge, so every branch reaches the data layer. | pathhdnn-2025 |
| N20 | graph | Keep only nodes above the data | Before layering, keep only nodes annotated with at least one data feature and their ancestors; drop the rest of the graph. | pathhdnn-2025 |
| N21 | graph | Propagate memberships up the hierarchy | Treat a feature annotated to a node as a member of all its ancestors, so a node at a depth cut collects the features of descendants that were cut away. | pathhdnn-2025 |
| N22 | mechanism | Mask a dense weight | Hold a full weight matrix per layer and multiply it elementwise by a fixed 0/1 mask in every forward pass; absent entries exist but do not affect the output and get no loss gradient. | pathhdnn-2025 |
| N23 | interpretation | Deep SHAP against a background set | Compute DeepLIFT-style SHAP values of the outputs with respect to inputs or hidden activations, against a set of background samples. | pathhdnn-2025 |
| N24 | interpretation | Normalize node scores by connectivity | Divide each node's score by a function of how many nodes connect to it (e.g. log of the up- plus downstream node count) so well-connected nodes do not dominate. | pathhdnn-2025 |
| N25 | interpretation | Scores as a graph | Return node scores joined to the network's edges and layers, so the scored network or the subgraph up- or downstream of one node can be queried and drawn. | pathhdnn-2025 |
