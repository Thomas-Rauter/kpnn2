# Supported architectures

`kpnn2` supplies connectivity primitives, not finished models: you
write an ordinary `nn.Module` and call them from your own
`forward()`. The table below says which architecture families you
can build that way.

| Type | Supported | What to use                                                                                                                                                                                                             |
| --- |-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Feedforward | Yes       | `parse_layered` (optional `widths=` to give a node several [units](concepts.md#unit)), `PackedLinear.from_hop`, `gather_hop_inputs`; tied decode `PackedLinear.transpose` + `scatter_hop_outputs` → [Feedforward example](feedforward-example.ipynb) |
| Cyclic | Yes       | `parse_adjacency`, `PackedLinear.from_adjacency` or `MaskedLinear(spec.to_mask())`, steps you write → [Cyclic graph example](cyclic-graph-example.ipynb)                                                                               |
| Transformer | Yes       | `parse_adjacency` + `PackedMultiheadAttention.from_adjacency`; encoder / FFN / head are yours → [Transformer example](transformer-example.ipynb)                                                                                       |
| Sequence RNN / GRU / LSTM | Yes, see [Sequence models](#sequence-models) | `parse_adjacency` + `MaskedLinear` / `PackedLinear` as the maps; you write the steps over time (not `nn.RNN` / `nn.GRU` / `nn.LSTM`) → [Time-series example](time-series-example.ipynb)                                 |
| Graph NN | No, see [Why not a GNN?](#why-not-a-gnn) | Knowledge-primed GNNs are a natural edgelist model in [PyG](https://pyg.org/).                                                                                                                                          |
| Convolutional NN | No        | No sparsely connected support. Build it yourself in PyTorch (`nn.Conv1d` / `nn.Conv2d` / `nn.Conv3d`)                                                                                                                   |

## Sequence models

The sequence row needs a longer answer: PyTorch's recurrent
modules cannot carry a prior. `nn.RNN`, `nn.GRU`, and `nn.LSTM`
are fused convenience modules whose maps are dense `W_ih` /
`W_hh`. None accepts an edgelist, so none can be the recurrent map
of a [knowledge-primed neural network](concepts.md#kpnn) (KPNN);
this package does not ship `MaskedRNN`, `MaskedGRU`, or
`MaskedLSTM`.

What replaces them is the cyclic-graph recipe with a **changing**
`x_t` at each step. Memory lives in the edgelist; the steps over time
are yours:

1. Parse with `parse_adjacency()`. A self-loop or hidden→hidden
   edge lets a named [node](concepts.md#node) carry state across
   time;
   `parse_layered()` is DAG-only and rejects both.
2. One map, shared across steps:
   `MaskedLinear(spec.to_mask())`, or
   `PackedLinear.from_adjacency(spec)` when `n` is large, since
   `to_mask()` allocates `(n, n)`.
3. In your `forward()`, step through time `t`. Register
   `torch.as_tensor(spec.input_index)` as a buffer and write
   `x_t` with `index_copy` into the
   [state vector](concepts.md#state-vector), then apply the
   map. Each step Captum should hook is an `nn.Identity`.
   The shared map stays a normal child of the module.
4. You choose `n_steps`, the sequence length. `kpnn2` does not
   unroll time or pick a step count.

An Elman-style net is that recipe. A GRU or LSTM is the usual gate
equations with one `MaskedLinear` per map; whether every gate
shares the same prior is your modeling choice.

```python
state = state.index_copy(
    -1,
    self.input_index,
    x_t,
)
state = tap(self.act(self.core(state)))
```

`tap` is one `nn.Identity` from an `nn.ModuleList`. The
[Time-series example](time-series-example.ipynb) walks through
the Elman-style net, [attributions](concepts.md#attribution) on a
`step` axis, and a no-memory control.

## Why not a GNN?

A graph neural network (GNN) is the standard model class for
learning from graph-structured data, and
[PyTorch Geometric](https://pyg.org/) (PyG) is its canonical
implementation. Where the prior should become a GNN, that is the
right tool.

An edgelist does not by itself determine the model. A knowledge
graph states which
interactions exist; how that prior enters the model is a second
choice, and each choice encodes a different hypothesis about the
data-generating process. In `kpnn2` the graph is the
**architecture**, not the data: it stays the same in every sample,
and it decides which weights exist. The two subsections below take
these in turn.

### In a GNN the graph varies between samples; in a KPNN only the node values do

Picture two ways to learn about traffic. One model studies many
different city maps, each with its own streets, and learns which
street layouts tend to jam. The other has one city map and studies
many days of traffic on it: the streets stay put, and what changes
from day to day is how many cars are on each street.

The first model is how a GNN sees data. Each sample is its own
graph, with its own nodes and edges, and the shape of that graph is
part of what decides the answer. Node 5 in one graph and node 5 in
another have nothing in common, because the numbering is arbitrary.

The second is how a KPNN sees data. The graph is known before
training and is the same in every sample; what varies is the value
on each node. A node stands for the same entity in every sample, so
its name carries meaning. The edge weights change during training,
but the trained model applies the same weights to every sample.

That difference decides whether a GNN's design pays off. Across
many cities, the same kind of junction recurs in thousands of
places, so one learned rule for it serves them all. On a single
map, Main Street is not one more instance of a recurring pattern:
it is one specific street, the same every day. A standard GNN is
also built not to care which node is which, and that discards what
a KPNN keeps: the nodes have names, and the names matter.

### A GNN shares one rule across all edges; a KPNN gives each edge its own weights

Take a graph with three nodes, A → C ← B, and data in which C
follows A and ignores B. Here `a`, `b`, and `c` are the node values
in one sample, `c'` is the value of C after one GNN round, and `σ`
is the activation:

```text
KPNN:  c  = σ(w_AC · a + w_BC · b)
GNN:   c' = σ(W · (a + b) + U · c)
```

The KPNN has one weight per edge, so it learns a large `w_AC` and a
`w_BC` near zero. The GNN, in its simplest message-passing form,
has one `W` for both edges and one `U` for every node's own value.
It sees only `a + b`, so it cannot tell A from B. Graph attention
networks do give each edge its own weight, but compute it from the
values `a` and `b` with another shared function, so they can learn
that large inputs count more, not that A counts more.

In general, a GNN applies the same message and update functions at
every node and edge. In a KPNN the prior indexes the parameters
instead: the graph decides which weights exist, and training
decides their values.

The sharing is deliberate. Because `W` never mentions A or B, a GNN
can apply it to a graph it has never seen. `w_AC` means nothing on
another graph: a KPNN gives up transfer to unseen graphs, which a
fixed graph does not need, and gets one parameter per named
interaction in return.

That trade is what lets you ask which interaction the trained model
relies on. In the KPNN, the weight `w_AC` and the signal
`w_AC · a` belong to A → C alone, so the answer names A → C. In the
GNN you can still remove A → C and measure the change, but no
learned parameter belongs to that edge, so the answer describes the
shared rule at that spot.

Widening an endpoint with `widths=` (on either parser) turns an
edge's scalar weight into a block, still owned by that edge alone.
`PackedMultiheadAttention` places the prior one level up,
constraining which pairs may attend at all.

### When a GNN is the better hypothesis

A GNN is the better hypothesis where the structure is the object of
study: samples that are distinct graphs, nodes or edges unseen at
training time, or node- and link-level tasks on a single large
knowledge graph. There are knowledge-primed GNN (KPGNN) papers; this
project will not wrap or replace PyG for them.

Same edgelist, different hypothesis about the data-generating
process. `kpnn2` is the PyTorch side of that split: it builds
interpretable neural networks from a graph of named entities, with
checks that every attribution is reported under the right name.
