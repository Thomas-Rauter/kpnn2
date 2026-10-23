# KPNNs in context

Neural networks can incorporate prior knowledge of a domain in
several ways. This page places
[knowledge-primed neural networks](concepts.md#kpnn) (KPNNs) among
established approaches from other domains: physics-informed,
Hamiltonian, convolutional, and equivariant networks, concept
bottleneck models, graph neural networks, and mechanistic models.
It compares them by the type of knowledge each encodes, where that
knowledge enters the model, whether it acts as a soft or a hard
constraint, and what the internal units of the model mean.

## Knowledge-primed neural networks

A KPNN is any neural network whose connectivity is derived from a
[graph](concepts.md#graph) of named entities. Each entity becomes a
[node](concepts.md#node) of the network, and two nodes are
connected only where this [prior](concepts.md#prior) relates the
corresponding entities.
[Fortelny and Bock (2020)](https://doi.org/10.1186/s13059-020-02100-5)
introduced the term for networks built from signaling and
gene-regulatory interactions. Models published under other names
are KPNNs too, including DCell and DrugCell, built on the Gene
Ontology ([Ma et al., 2018](https://doi.org/10.1038/nmeth.4627);
[Kuenzi et al., 2020](https://doi.org/10.1016/j.ccell.2020.09.014)),
P-NET, built on Reactome pathways
([Elmarakeby et al., 2021](https://doi.org/10.1038/s41586-021-03922-4)),
GenNet, built on variant–gene–pathway annotations
([van Hilten et al., 2021](https://doi.org/10.1038/s42003-021-02622-z)),
and LEMBAS, a recurrent network built on a signaling network
([Nilsson et al., 2022](https://doi.org/10.1038/s41467-022-30684-y)).
Nothing in the construction is specific to biology.

The prior does two things. First, information can flow only along
known relationships, so an accurate prior reduces the data the
model needs. Second, every hidden [unit](concepts.md#unit) gets an
identity.

In a fully connected network, hidden units mean nothing outside the
model: you can permute them, together with their weights, without
changing the function. In a KPNN, two nodes can trade places only
if they have the same inputs and outputs, so each node stays tied
to its entity, and an [attribution](concepts.md#attribution) to
that node refers to that entity. These named hidden units set KPNNs
apart from most of the approaches below (Table 1, last column).

## Comparison with related approaches

All approaches on this page use prior knowledge to restrict what a
network can learn
([Battaglia et al., 2018](https://arxiv.org/abs/1806.01261)).
Table 1 compares them in four columns.

The first column is the knowledge encoded. A governing equation
states how quantities relate, a symmetry states which
transformations of the input leave the output unchanged, and a
graph states which entities can influence which, without saying
how. The second is where that knowledge enters the model: the loss,
the labels, the architecture, or the input.

The third column is the strength of the constraint. A soft
constraint, imposed through the loss, holds only approximately and
only where training evaluates it, and enough data can override it.
A hard constraint, imposed through the architecture, holds exactly
for every input, before and after training. The fourth column says
what the internal units mean, and so whether an attribution to a
hidden unit refers to anything outside the model.

Figure 1 shows the second and third columns for four of the
approaches. The sections below follow Table 1 row by row and
contrast each approach with a KPNN, the last row.

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
([Raissi et al., 2019](https://doi.org/10.1016/j.jcp.2018.10.045);
[Karniadakis et al., 2021](https://doi.org/10.1038/s42254-021-00314-5)).
Its loss penalizes the residual of a known differential equation at
sampled points (Figure 1a). Like a KPNN, it uses prior knowledge to
make up for limited data.

It differs from a KPNN in the other three columns. Its knowledge is
a quantitative law, whereas a KPNN prior states only which entities
can influence which. For thousands of interacting molecules, such
laws are rarely known. Curated databases of their interactions
exist, though incomplete, and a KPNN needs only those.

The law is a soft constraint, so the trained network can violate it
where the loss is not evaluated. A KPNN prior, by contrast, holds
exactly: an interaction missing from it can never contribute to a
prediction. Last, its hidden units are anonymous; only its output,
the field, has a meaning outside the model. In a KPNN, every hidden
node stands for an entity.

### Hamiltonian neural networks

Hamiltonian neural networks parameterize the energy of a physical
system and derive its dynamics through Hamilton's equations, so the
learned energy is conserved by construction
([Greydanus et al., 2019](https://arxiv.org/abs/1906.01563)). They
combine a quantitative law, as in physics-informed networks, with a
hard constraint, as in KPNNs. The constraint acts on how the output
is used, not on which units connect, and the hidden units stay
anonymous.

### Convolutional and equivariant networks

Convolutional networks encode translation symmetry through local
connectivity and weights shared across positions
([LeCun et al., 1989](https://doi.org/10.1162/neco.1989.1.4.541)).
Equivariant networks extend this construction to rotations,
reflections, permutations, and other symmetry groups
([Cohen and Welling, 2016](https://arxiv.org/abs/1602.07576);
[Bronstein et al., 2021](https://arxiv.org/abs/2104.13478)). In
mechanism, they are the closest to a KPNN: the knowledge is a hard
constraint on which units connect, and the layers are sparse.

They differ in the pattern of sparsity (Figure 1c, d). In a
convolutional layer, every unit of a channel applies the same
filter to a different patch, so the units are interchangeable
copies of one filter. In a KPNN layer, every node has its own
inputs and its own weights. Shared filters let a convolutional
network generalize across positions; untied wiring gives each KPNN
node an identity.

A symmetry also holds for every instance of a task, whereas a KPNN
prior describes one system, as its source database records it.

### Concept bottleneck models

A concept bottleneck model predicts the output only from a hidden
layer of human-defined concepts, and trains each concept unit
against annotated labels (Figure 1b;
[Koh et al., 2020](https://arxiv.org/abs/2007.04612)). The labels
enter through the loss, a soft constraint. The bottleneck itself,
which routes every prediction through the concepts, is hard.

Like a KPNN, the model has named hidden units that you can inspect
and intervene on. It names them by supervision, which needs a label
for every concept on every training sample. A KPNN names them by
position in the graph and never observes the intermediate entities.
A concept unit is therefore trained to estimate its concept. A KPNN
node is only placed at its entity: its activation is a latent
quantity, not an estimate of the entity's activity.

### Graph neural networks

Graph neural networks (GNNs) also start from a graph, but take it
as input. Each sample can be a different graph, and the same
message and update functions act at every node, which lets a GNN
generalize to graphs it has not seen
([Battaglia et al., 2018](https://arxiv.org/abs/1806.01261)). A
KPNN builds its architecture from the graph: the graph is the same
for every sample, and every edge carries its own parameters.
[Why not a GNN?](supported.md#why-not-a-gnn) discusses the
distinction in detail.

### Mechanistic models

Mechanistic models of signaling, typically systems of ordinary
differential equations, encode both the wiring and the rate law of
every interaction. Their state variables are concentrations or
activities with physical units, so their predictions can be
measured and their dynamics tested. They need kinetic parameters,
which are seldom known for large networks.

A KPNN keeps the wiring and learns the functional form from data,
at the price of hidden values without physical units. Universal
differential equations, which learn only the unknown terms of an
equation ([Rackauckas et al.,
2020](https://arxiv.org/abs/2001.04385)), and LEMBAS, a KPNN that
runs along the direction of signaling, lie between the two.