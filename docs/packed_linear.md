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

## Tied transpose

`PackedLinear.weight` is 1-D, so there is no `enc.weight.T`.
`layer.transpose()` is that helper: same packed slots, indices
swapped, `weight` shared by default, bias never shared.

```python
enc = kpnn2.PackedLinear(
    hop.source_index,
    hop.target_index,
    hop.out_features,
    hop.in_features,
    bias=False,
)
dec = enc.transpose()
```

Do not reparse a reversed edgelist and assign
`dec.weight = enc.weight`: packed slot order will not match.

On a hop that concatenates several source layers, split the
transposed output and add the pieces into your decoder
`saved` dict:

```python
concat = dec(restored[hop.target_layer])
for layer, piece in kpnn2.scatter_hop_outputs(
    concat,
    hop,
).items():
    if layer in restored:
        restored[layer] = restored[layer] + piece
    else:
        restored[layer] = piece
```

`MaskedLinear` can keep using `F.linear(h, enc.weight.T,
dec_bias)`. There is no autoencoder class; `forward()` is
yours.

## Frozen live edges

`constraint=` is one `nn.Module` over the packed `weight`.
A hard freeze is `torch.where` inside that module, using
slots from `edge_location`. A gradient hook that zeroes a
slot is not a freeze: AdamW's decoupled weight decay and
SGD with momentum still move the stored parameter. A loss
barrier is a soft prior, not a hold.

The stored unconstrained slot may still drift. Read
`constraint(weight)`, or write the constants back after
`optimizer.step()` if a checkpoint must match.

```python
class FreezeSlots(torch.nn.Module):
    def __init__(
        self,
        mask,
        values,
    ):
        super().__init__()
        self.register_buffer(
            "mask",
            mask,
        )
        self.register_buffer(
            "values",
            values,
        )

    def forward(
        self,
        weight,
    ):
        return torch.where(
            self.mask,
            self.values,
            weight,
        )


spec = kpnn2.parse_adjacency(edgelist)
packed = spec.edge_location("a", "b")
nnz = len(spec.source_index)
mask = torch.zeros(nnz, dtype=torch.bool)
mask[list(packed)] = True
values = torch.zeros(nnz)
values[list(packed)] = 1.5
core = kpnn2.PackedLinear(
    spec.source_index,
    spec.target_index,
    len(spec.nodes),
    len(spec.nodes),
    constraint=FreezeSlots(
        mask,
        values,
    ),
)
```

The same module works on `MaskedLinear`: freeze at
`[target_index, source_index]` of the dense rectangle,
not at packed slots. `constraint=` still runs before the
mask.

## Changing the prior (reparse)

Specs are frozen after parse. To prune or grow the prior,
edit the edgelist, parse again, build **new** layers from
the new spec, and copy surviving tensors **by name**.

Look up each surviving named edge on **both** specs with
`edge_location` and copy those packed slots. A surviving
edge can sit on a different hop with a different packed
index. Do not `Tensor.copy_` a whole `weight`, and do not
`load_state_dict` across different priors.
`PackedLinear.weight` is 1-D of length `nnz`. If you drop
one named edge and add another, `nnz` is unchanged and
slot `i` is a different named edge. `index_digest` /
`mask_digest` hash numeric indices (or the mask) and
shapes, not names. Always pass `identity=spec.fingerprint`
on the new `PackedLinear` or `MaskedLinear`.

Reparse is not "the same graph minus a row." Both parsers
recompute `input_nodes` / `output_nodes`. `parse_layered`
also recomputes longest-path depths (unless you pass the
same `ranks=`), hop membership, concat source axes, and
`skips`. Pass the same `widths=` / `ranks=` as the original
parse.

Bias is `(out_features,)`, one value per output unit, not
per edge. Copy it by named node → unit slice, not by packed
slot. On a `LayeredSpec`, `node_units` indexes the hop's
output axis; copy only when the node still exists and its
width and layer still match. On an `AdjacencySpec`, index
`spec.nodes` by name.

Construct a new optimizer on the new parameters, or accept
that Adam moments on the old `Parameter` objects are lost.

```python
old_spec = kpnn2.parse_adjacency(old_edgelist)
new_spec = kpnn2.parse_adjacency(new_edgelist)
n_old = len(old_spec.nodes)
n_new = len(new_spec.nodes)
old_core = kpnn2.PackedLinear(
    old_spec.source_index,
    old_spec.target_index,
    n_old,
    n_old,
    identity=old_spec.fingerprint,
)
new_core = kpnn2.PackedLinear(
    new_spec.source_index,
    new_spec.target_index,
    n_new,
    n_new,
    identity=new_spec.fingerprint,
)
old_edges = set(
    zip(
        old_spec.to_edgelist()["source"],
        old_spec.to_edgelist()["target"],
    )
)
new_table = new_spec.to_edgelist()
with torch.no_grad():
    for source, target in zip(
        new_table["source"],
        new_table["target"],
    ):
        pair = (source, target)
        if pair not in old_edges:
            continue
        old_pack = old_spec.edge_location(
            source,
            target,
        )
        new_pack = new_spec.edge_location(
            source,
            target,
        )
        new_core.weight[list(new_pack)] = (
            old_core.weight[list(old_pack)]
        )
    if (
        old_core.bias is not None
        and new_core.bias is not None
    ):
        old_index = {
            name: i
            for i, name in enumerate(old_spec.nodes)
        }
        for i, name in enumerate(new_spec.nodes):
            if name not in old_index:
                continue
            new_core.bias[i] = old_core.bias[
                old_index[name]
            ]
optimizer = torch.optim.Adam(new_core.parameters())
```

On a `LayeredSpec`, `edge_location` returns
`(hop_index, packed_indices)`. Copy into
`layers[hop_index]` on each spec. Copy bias with
`node_units` only when layer and width still match:

```python
old_layer, old_units = old_spec.node_units(name)
new_layer, new_units = new_spec.node_units(name)
if old_layer != new_layer:
    continue
old_width = old_units.stop - old_units.start
new_width = new_units.stop - new_units.start
if old_width != new_width or old_layer == 0:
    continue
new_layers[new_layer - 1].bias[new_units] = (
    old_layers[old_layer - 1].bias[old_units]
)
```

On `MaskedLinear`, copy the named live cells of
`parametrizations.weight.original`. Do not `copy_` the
`(out, in)` rectangle. `load_state_dict` is still
name-blind when the mask pattern and `nnz` are unchanged.
