# PackedLinear

`MaskedLinear` is the default. The RAM problem on this page is
`parse_adjacency()` only. That layout puts every node in one
state vector, so `MaskedLinear(spec.to_mask())` stores an
`(n, n)` parameter. Use `kpnn2.PackedLinear` on that spec's
packed indices when `n` is large enough that the square would
hurt RAM. Small recurrent graphs stay on
`MaskedLinear(spec.to_mask())`.

`parse_layered()` does not build that square. Each hop mask is
only that hop's `(out, in)`, and the layer is
`MaskedLinear(hop.mask)`. `PackedLinear` does not take a hop
mask or a `LayeredSpec`; there is no combined
`parse_layered` + `PackedLinear` path.

[Layered vs. Adjacency](layered_vs_adjacency.md) and the
[Recurrent example](recurrent-example.ipynb) stay on
`MaskedLinear`. This page is the large-n adjacency path.

## The RAM problem

`AdjacencySpec` puts every node in one state, including a wide
input layer. `MaskedLinear` then stores an `(n_nodes, n_nodes)`
parameter. That square is RAM. Dataset size and minibatch size
are not this problem.

Dead mask entries never affected learning. They were RAM (and
extra GEMM work), not extra capacity.

## What PackedLinear stores

`PackedLinear` stores one ordinary dense 1-D weight per live
edge and updates with `index_add`. It is not `torch.sparse` and
not sparse-tensor acceleration.

Use `PackedLinear` when `n_nodes` is large enough that the
square would hurt. Otherwise `MaskedLinear` is better (GEMM,
`(out, in)` weight).

## Construction

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
[Recurrent example](recurrent-example.ipynb). Input nodes have
in-degree 0, so writing them into the state each step is
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
