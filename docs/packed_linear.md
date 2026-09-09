# PackedLinear

`PackedLinear` stores one trainable scalar per live edge. Use it
on a `Hop` or an `AdjacencySpec` when the dense rectangle
`MaskedLinear(to_mask())` would strain RAM.

On a `LayeredSpec`, each hop is already packed:

```python
layer = kpnn2.PackedLinear(
    hop.source_index,
    hop.target_index,
    hop.out_features,
    hop.in_features,
)
```

`gather_hop_inputs` still concatenates whole source layers.
`PackedLinear` then reads only the live columns. Small graphs
may keep `MaskedLinear(hop.to_mask())` for GEMM.

On an `AdjacencySpec`, every node shares one state vector, so
`MaskedLinear(spec.to_mask())` stores an `(n, n)` parameter:

```python
core = kpnn2.PackedLinear(
    spec.source_index,
    spec.target_index,
    len(spec.nodes),
    len(spec.nodes),
)
```

Small cyclic graphs may stay on `MaskedLinear(spec.to_mask())`.

[Layered vs. Adjacency](layered_vs_adjacency.md) is the layout
split. [Feedforward example](feedforward-example.ipynb) uses
packed hops. [Cyclic graph example](cyclic-graph-example.ipynb)
is the shared-state path.

## The RAM problem

A hop that concatenates a 20k-gene input layer into a skip
makes `MaskedLinear` store an `(out, ~20k)` parameter, a dense
float32 mask, and Adam state of the same shape, even when only
a handful of those columns are live edges. An `AdjacencySpec`
with a wide input layer has the same problem as an `(n, n)`
square. Dataset size and minibatch size are not this problem.

Dead mask entries never affected learning. They were RAM (and
extra GEMM work), not extra capacity.

## What PackedLinear stores

`PackedLinear` stores one ordinary dense 1-D weight per live
edge and updates with `index_add`. It is not `torch.sparse` and
not sparse-tensor acceleration.

Use `PackedLinear` when the dense rectangle would hurt.
Otherwise `MaskedLinear` is better (GEMM, `(out, in)` weight).

## Construction

From a hop:

```python
core = kpnn2.PackedLinear(
    hop.source_index,
    hop.target_index,
    hop.out_features,
    hop.in_features,
)
x = kpnn2.gather_hop_inputs(saved, hop)
hidden = torch.relu(core(x))
```

From an `AdjacencySpec`:

```python
core = kpnn2.PackedLinear(
    spec.source_index,
    spec.target_index,
    len(spec.nodes),
    len(spec.nodes),
)
```

Scatter inputs the same way as the
[Cyclic graph example](cyclic-graph-example.ipynb). Input nodes
have in-degree 0, so writing them into the state each step is
required:

```python
x = kpnn2.align_inputs(df, spec)
n = len(spec.nodes)
state = torch.zeros(x.shape[0], n)
state[:, spec.input_index] = x
state = torch.relu(core(state))
```

The same packed indices can feed
[`PackedMultiheadAttention`](reference/PackedMultiheadAttention.md).
That is a different primitive (attention on live pairs, not one
scalar per edge). The
[Transformer example](transformer-example.ipynb) is that
walkthrough.
