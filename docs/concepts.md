# Concepts

The vocabulary these docs use, in the order the pipeline uses it:
a **graph**, written as an **edgelist**, goes through a **parser**
and comes out a **spec**, which you build a module from.

Each term below is a link target, so other pages can point here
instead of redefining. This page carries the concept; the
[API reference](reference/api.md) carries the exact contract
(fields, invariants, what raises).

| Term | In one line |
|------|-------------|
| [Graph](#graph) | Named nodes and the directed edges between them |
| [Edgelist](#edgelist) | That graph as a `source` / `target` table |
| [Node](#node) | One named entity, one unit by default |
| [Directed acyclic graph](#dag) (DAG) | A graph with no feedback loop |
| [Parser](#parser) | `parse_layered()` or `parse_adjacency()` |
| [Spec](#spec) | The parsed layout, frozen; no parameters |
| [Layer](#layer) | A depth in a `LayeredSpec` |
| [Hop](#hop) | Every edge arriving at one layer |
| [Skip edge](#skip-edge) | An edge whose endpoints are >1 layer apart |
| [Packed indices](#packed-indices) | Two parallel lists, one entry per unit pair |
| [Live edge](#live-edge) | A unit pair that exists, so it gets a weight |
| [State vector](#state-vector) | Every node as one unit, in an `AdjacencySpec` |
| [Attribution](#attribution) | A score per node or feature, after training |
| [Knowledge-primed neural network](#kpnn) (KPNN) | The domain framing that named the package |

## Graph

The named nodes and the directed edges between them. This is the
prior: the wiring you already believe in before training, which
`kpnn2` turns into an architecture.

**Named graph** is the emphatic form, used where the point is that
the nodes carry domain names rather than positions, so a score on a
node is a score on the entity it stands for.

"Graph" in these docs never means PyTorch's autograd graph, and
never a graph as *data* in the GNN sense, where each node carries a
feature vector and the batch is graphs. Here the graph is the
architecture and the batch is samples. See
[Why not a GNN?](supported.md#why-not-a-gnn).

## Edgelist

The graph as a table: a `pandas.DataFrame` with required columns
`source` and `target`, one row per directed edge, in the direction
of computation.

| source | target |
|--------|--------|
| A | H |
| B | H |
| H | C |

Extra columns are ignored, and the frame is read and never
modified. A node name is text: an integer is read through
`str(...)`, so `1` and `"1"` name the same node. A float or bool
warns, because `1.0` becomes `"1.0"`, and any other value
raises. Edgelist and graph are the same object in two forms —
the table you pass in, and the structure it encodes.

## Node

One named entity, and the name is yours to choose. By default a
node is one unit of the network. `parse_layered(..., widths=)` can
give a named node several units, so that one entity owns a small
block rather than a single scalar.

A parser splits nodes into three groups by degree, alphabetical
within each:

- **input nodes** have in-degree 0. This is the feature-axis
  order `align_inputs()` indexes into. You do not declare them,
  so the edgelist starts at your measured features: one input
  node per feature column, named like that column. Leave the
  features out and your first computed nodes become the inputs.
- **output nodes** have out-degree 0. A terminal node that sits
  below maximum depth still counts, so this is not the same tuple
  as the last layer.
- **hidden nodes** are neither.

Out-degree 0 cannot tell an output you read from a node whose
onward rows the edgelist lacks, such as a transcription factor that
no pathway row continues from. Both are output nodes, so a head
that reads every output node reads both. To confirm the outputs you
read, pass them as `outputs=` to
[`parse_layered()`][kpnn2.parse_layered] or
[`parse_adjacency()`][kpnn2.parse_adjacency]. Any other node with no
outgoing edge then raises, and the message names the rows to drop.
Without `outputs=`, a parse with more than one output node warns.

One name is one node. To feed a measurement of an entity that is
also a computed node, give the measurement its own name:
`STAT3_gene` for the gene's expression, `STAT3` for the
transcription factor node. `align_inputs()` raises on a feature
named like a hidden or output node, because the model would
never read that column.

To rename every measured gene, add the suffix in two places: on
the gene side of the gene → transcription factor rows when you
build the edgelist, and on the feature names you align. Here
`tf_targets` holds one transcription factor–target gene pair per
row, `expression` is your samples × genes table, and `spec` is
the parse of the full edgelist:

```python
gene_rows = pd.DataFrame(
    {
        "source": tf_targets["gene"] + "_gene",
        "target": tf_targets["tf"],
    }
)
col = kpnn2.align_inputs(
    expression.columns + "_gene",
    spec,
)
```

A label column named like an output node raises for the same
reason. If one `table` holds the features and a `label` column,
align `table.drop(columns="label").columns` and index that same
reduced table.

A feedback loop that returns to a measured entity ends at its
computed node, never at its input node. If the prior has a loop
between the transcription factor `STAT3` and its target gene
`SOCS3`, write the rows `SOCS3_gene → SOCS3`, `SOCS3 → STAT3`, and
`STAT3 → SOCS3`. `SOCS3_gene` stays an input, and the loop runs
through `SOCS3`.

An edge into `SOCS3_gene` would make it a hidden node, and
`align_inputs()` would raise on its column. A cyclic model also
writes the measured values into the input nodes before every step,
so an edge into one could only compute a value that the next write
overwrites. The measurement stays fixed, and the feedback changes
the computed node. [Cyclic graph example](cyclic-graph-example.ipynb)
parses this graph and counts the steps it needs.

## DAG

A directed acyclic graph: no path leads from a node back to itself.

Depth only exists without feedback, so `parse_layered()` requires a
DAG and raises on a cycle. `parse_adjacency()` has no depth to
compute, so cycles and self-loops are ordinary there. A DAG is
valid for both parsers — being a DAG tells you which parsers are
*available*, not which one to call.

## Parser

`parse_layered()` or `parse_adjacency()`. The one you call fixes
the shape of everything you build afterwards, and the package never
inspects the graph to pick for you: the layout is your choice.

- `parse_layered()` ranks nodes by depth and returns a
  `LayeredSpec`: layers, one [hop](#hop) each.
- `parse_adjacency()` puts every node in one
  [state vector](#state-vector) and returns an `AdjacencySpec`:
  no layers, cycles allowed.

[Layered vs. Adjacency](layered_vs_adjacency.md) is the full
comparison.

## Spec

What a parser returns: a `LayeredSpec` or an `AdjacencySpec`.

A spec is frozen structure — named nodes, packed indices, and the
metadata to label tensors. It is not a model. There is no
`nn.Module`, no parameters, and no stored mask. You write the
module around it, and activations, heads, losses, optimizers and
the training loop stay yours.

The word *graph* stops here. What comes out of the parser is a
chosen layout of that graph, not the graph itself, which is why
[Layered vs. Adjacency](layered_vs_adjacency.md) is a real
decision rather than something the package infers.

A checkpoint is `spec.to_dict()` plus `state_dict`, not weights
alone: the weights mean nothing without the wiring they belong to.

## Layer

A depth in a `LayeredSpec`. Layer 0 is the inputs, and
`spec.layer_nodes[i]` is the names at depth `i`, alphabetical.
There are always at least two layers. No hop produces layer 0:
the feature columns that `align_inputs()` selects fill it.

Depth is the longest path from the inputs by default.
`parse_layered(..., ranks=)` overrides that when nodes belong at
official levels — ontology tiers, say — instead of at whatever
depth the longest path puts them.

An `AdjacencySpec` has no layers at all: every node is one unit of
a single state vector instead.

## Hop

Everything that arrives at one layer: every edge whose target is a
node of that layer, whichever layer it left.

A hop holds no weights at all — those live in the layer you build
from it. It stores only its edges, as
[packed indices](#packed-indices), and its columns are its source
layers concatenated: the axis `gather_hop_inputs()` assembles.

One hop is one layer of the model, so a hop is exactly what a
single `PackedLinear` — or `MaskedLinear(hop.to_mask())`, the
dense hatch — computes. `spec.hops[i]` is the hop into layer
`i + 1`, which makes `len(spec.hops)` one less than the number of
layers.

Hops are a `LayeredSpec` idea. An `AdjacencySpec` has none.

[Feedforward example](feedforward-example.ipynb) builds one
`PackedLinear` per hop, end to end.

## Skip edge

An original edge whose endpoints are more than one layer apart.

A skip needs no separate mechanism. It is already packed inside
the hop of its *target*, sitting next to that layer's adjacent
parents, so nothing has to add it back later and nothing can
forget to. At width 1 that is one unit pair; a wider endpoint
is every unit pair of the block. `spec.skips` is metadata that
says which prior edges span layers; a forward pass never reads
it, and it is empty when no edge skips.

[Skip edges](skip-edges.ipynb) works this through on a real graph.

## Packed indices

Two parallel integer lists, `source_index` and `target_index`,
with one entry per live unit pair: pair `i` says that column
`source_index[i]` feeds row `target_index[i]`.

At width 1 that is one entry per named edge.
`widths=` on either parser expands a named edge `A -> B` into
every unit pair of the block, `k_B` by `k_A`.

This is how both specs hold connectivity. The dense alternative is
a mask — `hop.to_mask()` or `spec.to_mask()` allocates one on
demand, for `MaskedLinear` — but nothing stores a mask, because the
dense rectangle grows as the product of its two axes while the
packed lists grow with the number of live pairs.

The saving is storage, not a different kernel. `PackedLinear` is a
1-D dense weight plus `index_add` on ordinary dense tensors; there
are no sparse kernels in `kpnn2`, and none are planned.

[PackedLinear](packed_linear.md) is the layer that reads them.

## Live edge

A unit pair that exists in the graph, and therefore gets its own
trainable weight. A named edge is one live edge when both ends
have width 1, and `k_B * k_A` live edges when they are wider.

The term earns its keep by contrast. A dense `(out, in)` rectangle
has a cell for every *possible* connection, and on a real prior
almost all of them are absent edges pinned to zero. `PackedLinear`
stores one scalar per live edge and nothing for the rest, which is
the whole reason it exists. [PackedLinear](packed_linear.md) is
when that matters.

## State vector

The `AdjacencySpec` layout: every node owns a block of units of a
single alphabetical vector, `spec.nodes` — one unit unless
`widths=` gives it several — and every edge is index pairs over
it.

There are no layers, so a cycle or a self-loop is nothing special —
it is another pair. One update maps the vector to itself, and
how many times you apply it is yours to decide. Input nodes have no
incoming edges, so writing the inputs into the state at each step
is the caller's job, not the package's.

`spec.input_index` and `spec.output_index` are where to scatter
inputs into that vector and where to read outputs back out.
[Cyclic graph example](cyclic-graph-example.ipynb) and
[Time-series example](time-series-example.ipynb) both run on it.

## Attribution

A score saying how much a node or an input feature mattered,
produced after training by an attribution method.

The methods are not this package's: run Captum, or your own
gradients, or anything else. `kpnn2` does not import Captum. What
`map_node_attributions()` does is label the node axis of the
tensor that comes back, using the names the spec already holds, and
return it as an `xarray.DataArray`.
`aggregate_node_attributions()` can then fold observations (and
optional seeds) to one score per node; that step is optional and
does not run Captum.

The reason this is worth a term: in a dense network only the input
features have names, so attribution stops at the inputs. On a named
graph the hidden nodes have names too, so the same methods score
the internal entities. [Mapping attributions](map-node-attributions.ipynb)
is the naming rule in detail.

## KPNN

A knowledge-primed neural network: the use case that named the
package. Prior knowledge, encoded as a graph, constrains the
structure of the network, so the model keeps only the connections
that known relationships between named entities support.

The payoff is [attribution](#attribution) on those named entities.
Biology is the common setting — genes, transcription factors,
kinases, pathways — but nothing in `kpnn2` is biological. Any
architecture you can write as named edges works the same way.
[Feedforward example](feedforward-example.ipynb) builds one
end to end.
