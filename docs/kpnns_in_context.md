# KPNNs in context

Neural networks can incorporate prior knowledge of a domain in
several ways. This page places
[knowledge-primed neural networks](concepts.md#kpnn) (KPNNs) among
established approaches from other domains: physics-informed,
Hamiltonian, convolutional, and equivariant networks, concept
bottleneck models, graph neural networks, and mechanistic models.
It compares them by the type of knowledge each encodes, the point
at which that knowledge enters the model, and whether it acts as a
soft or a hard constraint.

## Knowledge-primed neural networks

A KPNN is any neural network whose connectivity is derived from a
[graph](concepts.md#graph) of named entities. Each entity becomes a
[node](concepts.md#node) of the network, and two nodes are
connected only where this [prior](concepts.md#prior) relates the
corresponding entities.
[Fortelny and Bock (2020)](https://doi.org/10.1186/s13059-020-02100-5)
introduced the term for networks built from signaling and
gene-regulatory interactions and trained on single-cell
transcriptomes. The same construction underlies models published
under other names, including DCell and DrugCell, built on the Gene
Ontology ([Ma et al., 2018](https://doi.org/10.1038/nmeth.4627);
[Kuenzi et al., 2020](https://doi.org/10.1016/j.ccell.2020.09.014)),
P-NET, built on Reactome pathways
([Elmarakeby et al., 2021](https://doi.org/10.1038/s41586-021-03922-4)),
GenNet, built on variant–gene–pathway annotations
([van Hilten et al., 2021](https://doi.org/10.1038/s42003-021-02622-z)),
and LEMBAS, a recurrent network built on a signaling network
([Nilsson et al., 2022](https://doi.org/10.1038/s41467-022-30684-y)).
The construction is not specific to biology. It applies wherever
entities and the relationships between them are known.

The prior serves two purposes. As an inductive bias, it restricts
the hypothesis space to functions in which information passes only
along known relationships, which reduces the data required when the
prior is accurate. As a coordinate system, it gives every hidden
[unit](concepts.md#unit) an identity. In a fully connected network,
hidden units carry no meaning outside the model: permuting them,
together with their weights, leaves the computed function
unchanged. Irregular connectivity breaks this symmetry. Two KPNN
nodes are interchangeable only if they share all inputs and
outputs, so each node is anchored to one entity, and an
[attribution](concepts.md#attribution) assigned to that node refers
to that entity. This second purpose distinguishes KPNNs from most
other knowledge-guided architectures.

Formally, a feedforward KPNN resembles a nonlinear structural
equation model on the skeleton of the prior: each node is a learned
function of its parents, the intermediate nodes are unobserved, and
the model is fitted to predict an endpoint. The prior delimits the
explanations the model can express, and training selects among
them.

## Comparison with related approaches

All approaches compared here restrict the hypothesis space with
prior knowledge
([Battaglia et al., 2018](https://arxiv.org/abs/1806.01261)). They
differ along three axes (Figure 1, Table 1). The first is the type
of knowledge. A governing equation states how quantities relate, a
symmetry states which transformations of the input leave the output
unchanged, and a graph states which entities can influence which,
without specifying how. The second is where the knowledge enters:
the loss, the labels, the architecture, or the input. The third is
the strength of the constraint. A soft constraint, imposed through
the loss, holds approximately and only where training evaluates it,
and sufficient data can override it. A hard constraint, imposed
through the architecture, holds exactly for every input, before and
after training.

![Top row: two dense networks whose knowledge enters through a dashed line to a box, an equation residual in (a) and concept labels in (b). Bottom row: (c) two hidden units that each read three neighboring pixels with the same three line weights, and (d) four genes wired sparsely to two transcription factors, one kinase, and a receptor.](figures/knowledge_placement.png)

**Figure 1.** Where four approaches impose prior knowledge. Solid
arrows denote the forward computation; dashed lines denote training
signals that enter only the loss. (a) A physics-informed network
penalizes the residual of a governing equation. (b) A concept
bottleneck model trains a hidden layer against concept labels.
(c) A convolutional network reads neighboring pixels with weights
shared across positions; line weight marks the three shared
weights. (d) A KPNN retains only the edges of its prior, and each
node represents one entity. The top row shows soft constraints, the
bottom row hard constraints.

**Table 1.** Knowledge-guided neural networks compared by the
knowledge they encode and how they impose it.

| Approach | Knowledge encoded | Imposed through | Constraint | Internal units |
|----------|-------------------|-----------------|------------|----------------|
| Physics-informed neural network | Governing equation | Residual penalty in the loss | Soft | Anonymous; the output is the physical field |
| Hamiltonian neural network | Conservation of energy | Hamilton's equations applied to the output | Hard | Anonymous; the output is an energy |
| Convolutional and equivariant networks | Symmetry of the task | Local connectivity and shared weights | Hard | Anonymous; copies of one filter |
| Concept bottleneck model | Human-defined intermediate concepts | Concept labels on a hidden layer | Soft; the bottleneck is hard | Named, by supervision |
| Graph neural network | Relational structure of each sample | Input graph; functions shared across nodes | Hard | States of input-graph nodes; parameters shared |
| Mechanistic model | Wiring and rate laws | Model equations | Hard | Physical quantities with units |
| KPNN | Which entities can influence which | Sparse, irregular connectivity | Hard | Named, by position in the graph |

### Physics-informed neural networks

A physics-informed neural network represents a physical field, such
as temperature over space and time, with a fully connected network
whose loss penalizes the residual of a known differential equation
at sampled points
([Raissi et al., 2019](https://doi.org/10.1016/j.jcp.2018.10.045);
[Karniadakis et al., 2021](https://doi.org/10.1038/s42254-021-00314-5)).
It shares with a KPNN the premise that prior knowledge compensates
for limited data, and differs in three respects. It encodes a
quantitative, well-established law, whereas a KPNN encodes only
dependency structure, taken from curated databases that are
incomplete and combine interactions observed in different contexts.
Governing equations for thousands of interacting molecular species
are rarely available; such databases are. It imposes the law
softly, so the law can be violated where the loss is not evaluated,
whereas an interaction absent from a KPNN's prior cannot contribute
to any prediction. Its interpretable quantity is the output, while
its hidden units remain anonymous; in a KPNN it is the hidden
nodes.

### Convolutional and equivariant networks

Convolutional networks encode translation symmetry through local
connectivity and weights shared across positions
([LeCun et al., 1989](https://doi.org/10.1162/neco.1989.1.4.541)).
Equivariant networks extend this construction to rotations,
reflections, permutations, and other symmetry groups
([Cohen and Welling, 2016](https://arxiv.org/abs/1602.07576);
[Bronstein et al., 2021](https://arxiv.org/abs/2104.13478)). In
mechanism, they are the closest relatives of KPNNs: in both, the
knowledge is a hard constraint on which units connect, and both
produce sparse layers. They differ in the pattern of sparsity
(Figure 1c, d). A convolutional layer is regular: every unit of a
channel applies the same filter to a different patch, so the units
are interchangeable copies. A KPNN layer is irregular, with
distinct inputs and untied weights at every node. Regularity allows
a convolutional network to generalize across positions; irregularity
gives each KPNN node an identity. A symmetry also holds for every
instance of a task, whereas a KPNN prior describes one system as
one database release records it.

### Hamiltonian neural networks

Hamiltonian neural networks parameterize the energy of a physical
system and derive its dynamics through Hamilton's equations, so
that the learned energy is conserved by construction
([Greydanus et al., 2019](https://arxiv.org/abs/1906.01563)). They
combine a quantitative law, as in physics-informed networks, with a
hard constraint, as in KPNNs. The constraint acts on how the
network's output is used rather than on its connectivity, and the
hidden units remain anonymous.

### Concept bottleneck models

Concept bottleneck models predict the output only from a hidden
layer of human-defined concepts, each trained against annotated
labels ([Koh et al., 2020](https://arxiv.org/abs/2007.04612)). They
share with KPNNs named hidden units that can be inspected and
intervened on. They name these units by supervision, which requires
a label for every concept on every training sample. KPNNs name
units by connectivity, and the intermediate entities are never
observed. A concept unit is therefore trained to approximate its
concept, whereas a KPNN node is constrained only in its position in
the graph: its activation is a latent quantity, not an estimate of
the entity's activity.

### Graph neural networks

Graph neural networks (GNNs) also start from a graph but treat it
as input. Each sample can be a different graph, and the same
message and update functions act at every node, a relational
inductive bias that supports generalization to unseen graphs
([Battaglia et al., 2018](https://arxiv.org/abs/1806.01261)). A
KPNN treats the graph as the architecture: it is identical across
samples, and every edge carries its own parameters.
[Why not a GNN?](supported.md#why-not-a-gnn) discusses the
distinction in detail.

### Mechanistic models

Mechanistic models of signaling, typically systems of ordinary
differential equations, encode both the wiring and the functional
form of every interaction. Their state variables are concentrations
or activities with physical units, and their edges are rate laws
with few parameters. They yield quantities that can be measured and
dynamics that can be tested, but they require kinetic parameters
that are seldom known for large networks. A KPNN retains the wiring
and learns the functional form from data, at the price of hidden
values without physical units. Universal differential equations,
which learn only the unknown terms of an equation
([Rackauckas et al., 2020](https://arxiv.org/abs/2001.04385)), and
LEMBAS, which runs a KPNN along the direction of signaling, occupy
intermediate positions.

## Consequences for interpretation

Because the prior acts as a hard constraint, its quality bounds the
interpretation of the trained model. The graph determines where a
node sits, not what its activation represents, so an attribution
describes how the model uses an entity, not the entity's role in
the system. A superfluous edge can be down-weighted during
training, but a missing edge cannot be recovered: the signal is
routed through the remaining paths and attributed to their nodes.
Redundant paths admit several equally good solutions, and nodes
that aggregate many inputs can receive high scores by position
alone. Fortelny and Bock (2020) designed their training procedure
to stabilize node weights under redundancy and to control for
uneven connectivity, and
[Correcting for network bias](network-bias.ipynb) examines the
second effect. The inductive bias, finally, helps only to the
extent that the prior excludes hypotheses, which a
[rewired prior](how_we_test.md#rewired-prior) tests.

A KPNN is therefore best read as a structured hypothesis. The prior
delimits the admissible explanations, training selects among them,
and controls establish whether the selection is stable and whether
the prior mattered.
