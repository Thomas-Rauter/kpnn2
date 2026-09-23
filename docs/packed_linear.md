# PackedLinear

`PackedLinear` stores one trainable scalar per
[live edge](concepts.md#live-edge) — an edge the graph actually
has. Use it on a [`Hop`](concepts.md#hop) — everything entering
one layer — or on an `AdjacencySpec` when the dense rectangle
`MaskedLinear(to_mask())` would strain RAM.

On a `LayeredSpec`, a hop's
[packed indices](concepts.md#packed-indices) — two parallel lists,
one entry per unit pair — go straight into the constructor:

```python
layer = kpnn2.PackedLinear(
    hop.source_index,
    hop.target_index,
    hop.out_features,
    hop.in_features,
)
```

Nothing upstream changes: `gather_hop_inputs` still concatenates
whole source layers, and `PackedLinear` reads only the live
columns. Small graphs may keep `MaskedLinear(hop.to_mask())` for
GEMM.

On an `AdjacencySpec` every node shares one
[state vector](concepts.md#state-vector), a single vector over
all nodes, so `MaskedLinear(spec.to_mask())` is an `(n, n)`
parameter. The packed form is one scalar per live unit pair (one
per named edge at width 1):

```python
core = kpnn2.PackedLinear(
    spec.source_index,
    spec.target_index,
    spec.state_dim,
    spec.state_dim,
)
```

Small cyclic graphs may stay on `MaskedLinear(spec.to_mask())`.

[Layered vs. Adjacency](layered_vs_adjacency.md) is the layout
split. [Feedforward example](feedforward-example.ipynb) uses
packed hops. [Cyclic graph example](cyclic-graph-example.ipynb)
is the shared-state path.

## The RAM problem

Dense storage costs memory in proportion to the rectangle, not to
the live edges in it. A hop that concatenates a 20k-gene input
layer into a [skip](concepts.md#skip-edge), an edge spanning more
than one layer, makes `MaskedLinear` store an `(out, ~20k)`
parameter, a dense float32 mask, and Adam state of the same
shape, even when only a handful of those columns are live edges.
An `AdjacencySpec` with a wide input layer has the same problem
as an `(n, n)` square. Dataset size and minibatch size are not
this problem.

Nothing is lost by dropping the dead entries: masked-out cells
are multiplied by zero, so they never affected learning. They
were RAM (and extra GEMM work), not extra capacity.

## What PackedLinear stores

The weight is one 1-D dense tensor with an entry per live edge;
forward accumulates each edge's contribution with `index_add`
on ordinary dense tensors. It is not `torch.sparse` and not
sparse-tensor acceleration.

Use `PackedLinear` when the dense rectangle would hurt.
Otherwise `MaskedLinear` is better: its `(out, in)` weight runs
as one GEMM.

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
hidden = self.acts[index](core(x))
```

`self.acts` is `nn.ModuleList([nn.ReLU() for _ in spec.hops])`.
The hop returns the pre-activation tensor.

From an `AdjacencySpec`:

```python
core = kpnn2.PackedLinear(
    spec.source_index,
    spec.target_index,
    spec.state_dim,
    spec.state_dim,
)
```

An adjacency layer maps the state vector onto itself, so the
aligned inputs must be scattered into it. Input nodes have
in-degree 0, so no packed edge ever reaches them: writing them
in each step is required, not cosmetic. The
[Cyclic graph example](cyclic-graph-example.ipynb) does the same
scatter inside a full loop:

```python
col = kpnn2.align_inputs(df.columns, spec)
x = torch.as_tensor(
    df.to_numpy()[:, col],
    dtype=torch.float32,
)
n = spec.state_dim
state = torch.zeros(x.shape[0], n)
state[:, spec.input_index] = x
state = self.act(core(state))
```

`self.act` is an `nn.ReLU` registered on the module.

The same packed indices can feed
[`PackedMultiheadAttention`](reference/PackedMultiheadAttention.md),
a different primitive: attention on live pairs, not one scalar
per edge. That layer's training RAM is pair gathers, not an
`(n, n)` weight; pass `chunk_size` there when `nnz` is large.
The [Transformer example](transformer-example.ipynb) is that
walkthrough.

## Tied transpose

A tied decoder runs the same edges backwards: it needs the
transpose. `PackedLinear.weight` is 1-D, so there is no
`enc.weight.T` to take. `layer.transpose()` is that helper:
same packed slots, indices swapped, `weight` shared by default,
bias never shared. A `constraint=` module is shared along with
`weight`, so the decoder applies the transpose of the encoder's
live map even when that constraint has state: an edge you prune
through a [keep-mask](#pruning-during-training), or a gate you
train, changes both layers. `tie=False` copies both instead.

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
`dec.weight = enc.weight`. Packed order is lexicographic per
[spec](concepts.md#spec), so those slots will not match.

On a hop that concatenates several source layers, split the
transposed output and add the pieces into your decoder `saved`
dict; two reversed hops can write the same earlier layer, so add
rather than overwrite:

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

`MaskedLinear` has a dense rectangle, so
`F.linear(h, enc.effective_weight().T, dec_bias)` is its tied
decoder. There is no autoencoder class; `forward()` is yours.

## Constrained init

`constraint=` maps `weight` to the effective weight in
`forward`. `reset_parameters` draws the degree-aware value for
that effective weight. It reaches `weight` through the
constraint's own `right_inverse`, the same method
`torch.nn.utils.parametrize` uses. Without one, `weight` stores
the draw and the map is applied on top.

That matters for `nn.Softplus()`, which has no `right_inverse`:
every edge then starts near `softplus(0) = ln 2`, whatever its
fan-in, so a node with thousands of parents starts with a
pre-activation spread many times larger than the unconstrained
init. Give the constraint an inverse and the init keeps its
degree-aware scale:

```python
class PositiveEdges(torch.nn.Module):
    def forward(
        self,
        weight,
    ):
        return torch.nn.functional.softplus(weight)

    def right_inverse(
        self,
        weight,
    ):
        # The draw is uniform in [-bound, bound], possibly 0:
        # invert its magnitude, kept away from 0.
        magnitude = weight.abs().clamp_min(1e-6)
        return magnitude + torch.log(-torch.expm1(-magnitude))


core = kpnn2.PackedLinear(
    hop.source_index,
    hop.target_index,
    hop.out_features,
    hop.in_features,
    constraint=PositiveEdges(),
)
```

A constraint without an inverse can initialize `weight` itself
from `layer.init_bound()`, the per-entry bound
`reset_parameters` uses.

Weight decay acts on `weight`, the unconstrained tensor. Under
softplus it pulls edges toward `ln 2`, not toward 0. Give a
constrained layer `weight_decay=0` in its param group and add a
penalty on `layer.effective_weight()` to the loss instead.

## Frozen live edges

Some priors fix individual edge values. `constraint=` is where
that belongs: one `nn.Module` over the packed `weight`.
A hard freeze is a `torch.where` inside it that replaces the
slots `edge_location` reports: forward overwrites them every
step, so no optimizer can move the value the layer uses.

A gradient hook that zeroes a slot is not a freeze: AdamW's
decoupled weight decay and SGD with momentum still move the
stored parameter. A loss barrier is a soft prior, not a hold.

The stored unconstrained slot may still drift. Read
`effective_weight()`, the map `forward` uses, or write the
constants back after `optimizer.step()` if a checkpoint must
match.

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
    spec.state_dim,
    spec.state_dim,
    constraint=FreezeSlots(
        mask,
        values,
    ),
)
```

The same approach works on `MaskedLinear`: freeze at
`[target_index, source_index]` of the dense rectangle, not at
packed slots. `constraint=` still runs before the mask.

## Pruning during training

Specs stay frozen. To drop edges while training continues,
keep the spec you parsed and put a keep-mask in
`constraint=`. The mask is a persistent buffer, one entry
per packed slot, multiplied into `weight`.

```python
class KeepMask(torch.nn.Module):
    def __init__(
        self,
        nnz,
    ):
        super().__init__()
        self.register_buffer(
            "keep",
            torch.ones(nnz),
        )

    def forward(
        self,
        weight,
    ):
        return weight * self.keep


hop_index, packed = spec.edge_location(
    "A",
    "H",
)
hop = spec.hops[hop_index]
core = kpnn2.PackedLinear(
    hop.source_index,
    hop.target_index,
    hop.out_features,
    hop.in_features,
    identity=spec.fingerprint,
    constraint=KeepMask(
        len(hop.source_index),
    ),
)
core.constraint.keep[list(packed)] = 0.0
```

A zero on `keep` drops that slot from the map `forward`
uses. `effective_weight()` is 0 there, whatever scalar is
stored in `weight`. Node roles, depths, and `align_inputs`
indices stay the ones on this spec. `index_digest` stays
the digest of the parsed indices. Adam's moments stay on
the same `weight` parameter, in the same slot order, so
training continues on the optimizer you already have.

The buffer is in `state_dict` under `constraint.keep`. A
model rebuilt from the same spec, with a `KeepMask` of the
same shape, loads that checkpoint and keeps the zeros.
`PackedLinear.transpose()` shares the module, so the tied
decoder applies the same zeros.

`torch.nn.utils.prune.custom_from_mask` on `weight` also
zeroes slots, on `PackedLinear` and on `MaskedLinear`. The
checkpoint keys are then `weight_orig` and `weight_mask`.
The keep-mask checkpoint keeps the key `weight`.

On `MaskedLinear` the buffer has shape
`(out_features, in_features)` and multiplies the dense
weight. `constraint=` still runs before the connectivity
mask. Leave `layer.mask` as the parsed pattern:
`mask_digest` is that pattern, and a model rebuilt from
the original spec expects it.

## Changing the prior (reparse)

Specs are frozen after parse. Pruning during training stays
on that spec: a keep-mask, above. Reparse is how you grow
the prior, and how you remove edges and then retrain a
smaller model. Edit the edgelist, parse again, build
**new** layers from the new spec, and copy surviving
tensors **by name**.

Copy edge by edge. Look up each surviving named edge on **both**
specs with `edge_location` and copy those packed slots.
A surviving edge can sit on a different hop with a different
packed index.

Bulk shortcuts do not work. Do not `Tensor.copy_` a whole
`weight`, and do not `load_state_dict` across different priors.
`PackedLinear.weight` is 1-D of length `nnz`. If you drop
one named edge and add another, `nnz` is unchanged and
slot `i` is a different named edge. `index_digest` /
`mask_digest` hash numeric indices (or the mask) and
shapes, not names, so a rename that leaves the index pattern
unchanged passes them. Always pass `identity=spec.fingerprint`
on the new `PackedLinear` or `MaskedLinear`.

Reparse rebuilds the blueprint; it is not "the same graph minus
a row." Both parsers recompute `input_nodes` / `output_nodes`.
`parse_layered` also recomputes longest-path depths (unless you
pass the same `ranks=`), hop membership, concat source axes, and
`skips`. Pass the same `widths=` / `ranks=` as the original
parse.

Recompute `align_inputs` on the new spec. Dropping an input's
last edge removes that input, so an index kept from the old spec
is one column too wide; `PackedLinear` raises on it rather than
reading the columns by position. An index of the same width (one
input swapped for another) cannot be caught by any layer.

Bias moves by node, not by edge: it is `(out_features,)`, one
value per output unit. Copy it by named node → unit slice, not
by packed slot. On a `LayeredSpec`, `node_units` indexes the
hop's output axis; copy only when the node still exists and its
width and layer still match. On an `AdjacencySpec`, index
`spec.nodes` by name.

Optimizer state does not follow. Adam moments are keyed by
`Parameter` identity on the old module: construct a new
optimizer on the new parameters, or accept losing them.

`optimizer.load_state_dict` across that reparse is accepted
silently when each parameter shape still matches. On
`PackedLinear`, `weight` has length `nnz` and `bias` has
length `out_features`. On `MaskedLinear`, `weight` is the
`(out_features, in_features)` rectangle. Dropping one named
edge and adding another can leave those shapes unchanged.
The moments are then applied by position, so slot `i` can
be a different named edge. The load reads shapes only.
`index_digest` and `identity` are checked by the layer's
own `load_state_dict`: with `identity=spec.fingerprint`, a
different prior raises `Kpnn2Error`. A successful optimizer
load means the shapes matched.

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

That example is adjacency. On a `LayeredSpec`, `edge_location`
returns `(hop_index, packed_indices)`; copy into
`layers[hop_index]` on each spec. Copy bias with `node_units`
only when layer and width still match:

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

On `MaskedLinear`, copy the named live cells of `weight`, the
trainable tensor, as on `PackedLinear`. Do not `copy_` the whole
`(out, in)` rectangle. `load_state_dict` is still name-blind
when the mask pattern and `nnz` are unchanged.
