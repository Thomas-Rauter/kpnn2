# PackedLinear

`MaskedLinear` is the default. Usual KPNNs and small recurrent
graphs stay on `MaskedLinear(hop.mask)` or
`MaskedLinear(spec.to_mask())`. Use `kpnn2.PackedLinear` only
when `n_nodes` is large enough that the square would hurt RAM.

[Layered vs. Adjacency](layered_vs_adjacency.md) and the
[Recurrent example](recurrent-example.ipynb) stay on
`MaskedLinear`. This page is the large-n path.

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
