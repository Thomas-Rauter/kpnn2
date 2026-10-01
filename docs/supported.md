# Supported architectures

`kpnn2` supplies connectivity primitives, not finished models: you
write an ordinary `nn.Module` and call them from your own
`forward()`. The table below says which architecture families you
can build that way.

| Type | Supported | What to use                                                                                                                                                                                                             |
| --- |-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Feedforward | Yes       | `parse_layered` (optional `widths=` to give a node several units), `PackedLinear`, `gather_hop_inputs`; tied decode `PackedLinear.transpose` + `scatter_hop_outputs` → [Feedforward example](feedforward-example.ipynb) |
| Cyclic | Yes       | `parse_adjacency`, `PackedLinear` or `MaskedLinear(spec.to_mask())`, loop you own → [Cyclic graph example](cyclic-graph-example.ipynb)                                                                                  |
| Transformer | Yes       | `parse_adjacency` + `PackedMultiheadAttention`; encoder / FFN / head are yours → [Transformer example](transformer-example.ipynb)                                                                                       |
| Sequence RNN / GRU / LSTM | Yes, see [Sequence models](#sequence-models) | `parse_adjacency` + `MaskedLinear` / `PackedLinear` as the maps; you write the time loop (not `nn.RNN` / `nn.GRU` / `nn.LSTM`) → [Time-series example](time-series-example.ipynb)                                       |
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
`x_t` at each step. Memory lives in the edgelist; the time loop is
yours:

1. Parse with `parse_adjacency()`. A self-loop or hidden→hidden
   edge lets a named [node](concepts.md#node) carry state across
   time;
   `parse_layered()` is DAG-only and rejects both.
2. One map, shared across steps:
   `MaskedLinear(spec.to_mask())`, or `PackedLinear` when `n` is
   large, since `to_mask()` allocates `(n, n)`.
3. In your `forward()`, loop over time `t`. Register
   `torch.as_tensor(spec.input_index)` as a buffer and write
   `x_t` with `index_copy` into the
   [state vector](concepts.md#state-vector), then apply the
   map. Each step Captum should hook is an `nn.Identity`.
   The shared map stays a normal child of the module.
4. You choose `n_steps`, the sequence length. `kpnn2` does not
   unroll time or pick a step count.

An Elman-style net is that loop. A GRU or LSTM is the usual gate
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
the Elman loop, [attributions](concepts.md#attribution) on a
`step` axis, and a no-memory control.

## Why not a GNN?

A graph neural network (GNN) is the standard model class for
learning from graph-structured data, and
[PyTorch Geometric](https://pyg.org/) (PyG) is its canonical
implementation. Where the prior should become a GNN, that is the
right tool.

An edgelist does not by itself determine the model. A knowledge
graph (a pathway map, an ontology, a sensor network) states which
interactions exist; how that prior enters the model is a second
choice, and each choice encodes a different hypothesis about the
data-generating process. In `kpnn2` the graph is the
**architecture**, not the data.

- **Fixed structure, varying state.** A GNN assumes that the
  structure itself varies and is informative: a sample is a graph,
  its nodes carry feature vectors, and the batch is drawn from a
  distribution over graphs. A KPNN assumes the converse. The graph
  is known and identical across samples, what varies is the state
  of its named nodes, and the batch is samples. Where no structure
  varies, the regularity a GNN is built to exploit is not present.
- **Prior-indexed parameters, not one shared function.** A GNN
  applies the same message and update functions at every node and
  edge. That sharing is what permits generalization to unseen
  graphs, and it is also what leaves edge-level attribution
  ill-posed. Here the prior indexes the parameters instead: each
  named edge carries its own weights, a scalar at unit width and a
  block once `widths=` (on either parser) widens its endpoints, so
  an attribution resolves to a named interaction rather than to a
  rule shared across all of them. `PackedMultiheadAttention` places
  the prior one level up, constraining which pairs may attend at
  all.
- **Directed propagation, not k rounds of neighborhood
  aggregation.** A pathway or an ontology is a deep directed
  cascade, and the quantity of interest is what propagates along
  it. A sparsely connected feedforward network traverses that
  cascade in one pass with skip edges intact; message passing
  reaches the same depth only by stacking rounds, mixing
  neighboring node states as it proceeds.

A GNN is the better hypothesis where the structure is the object of
study: samples that are distinct graphs, nodes or edges unseen at
training time, or node- and link-level tasks on a single large
knowledge graph. There are knowledge-primed GNN papers; this
project will not wrap or replace PyG for them.

Same edgelist, different hypothesis about the data-generating
process. `kpnn2` is the PyTorch side of that split: sparsely
connected layers you assemble yourself.
