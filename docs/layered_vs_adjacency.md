# Layered vs. unlayered

`kpnn2` has two [parsers](concepts.md#parser), and the one you
call fixes the shape of everything you build afterwards.
`parse_layered()` ranks [nodes](concepts.md#node) by depth and
returns a `LayeredSpec`; `parse_adjacency()` puts
every node in one [state vector](concepts.md#state-vector), a
single vector over all nodes, and returns an
`AdjacencySpec`, which has no layers.
The same `source` / `target` edgelist goes through either one,
and a [directed acyclic graph](concepts.md#dag) (DAG) is valid
for both, so the layout is your choice. Only
`parse_adjacency()` allows cycles and self-loops.

This page is the difference between the two
[specs](concepts.md#spec), the frozen structures the parsers
return.
[Feedforward example](feedforward-example.ipynb) is a feedforward
`parse_layered()` workflow.
[Cyclic graph example](cyclic-graph-example.ipynb) is a
`parse_adjacency()` workflow on a graph with a cycle.
[Transformer example](transformer-example.ipynb) and
[Time-series example](time-series-example.ipynb) use that same
packed layout: they need named nodes and
[packed pairs](concepts.md#packed-indices), not
[hop](concepts.md#hop) rectangles — a hop being everything
entering one layer.

## How edges are stored

The toy is `A -> H -> C` plus the [skip](concepts.md#skip-edge)
`A -> C`, an edge whose endpoints are more than one layer apart:

| source | target |
| ------ | ------ |
| A      | H      |
| H      | C      |
| A      | C      |

<img class="figure-full" src="../figures/layered_vs_adjacency.svg" alt="Left: parse_layered puts A, H and C in layers 0 to 2, and the skip A to C becomes a second column of hops[1]. Right: parse_adjacency puts A, C and H in one state vector, and all three edges, the skip included, are ones in a 3 by 3 mask.">

**Figure 1.** The same DAG parsed two ways. Layered ranks nodes
and puts every incoming edge in one packed hop (the skip is a
pair of `hops[1]`). Adjacency puts every node in one
alphabetical state vector and every edge in packed indices.

`parse_layered()` gives every node a [depth](concepts.md#layer),
its longest path from the inputs: `A` is layer 0, `H` is layer 1,
`C` is layer 2.
Each layer after the first owns one hop, holding all of that
layer's incoming edges. `hops[0]` is the packed hop into `H`;
`hops[1]` is the hop into `C`, which reads both earlier layers
because the skip makes `A` a parent of `C` alongside `H`.
[Edges that skip layers](skip-edges.ipynb) is the algorithm for
that extra source layer.

`parse_adjacency()` does not sort nodes into layers.
`spec.nodes` gives the state-vector order, `A`, `C`, `H`,
alphabetically. Edges become
packed `source_index` / `target_index` pairs over that order,
and `spec.to_mask()` renders them as a 3×3 square where
`mask[target, source]` is `1` for an original edge. Row `A` is
all zeros because `A` is an input. The skip is one more pair:
`mask[C, A] = 1`. There are no hops and no `skips`; depth does
not exist in this layout.

`input_nodes`, `hidden_nodes`, and `output_nodes` are the same
on both specs (`A`, `H`, and `C`).

## How you compute

<img class="figure-full" src="../figures/layered_vs_adjacency_forward.svg" alt="Top: the layered forward pass applies hops[0], holds A, gathers it with H, and applies hops[1] once. Bottom: the adjacency forward pass scatters x into the state vector and applies the to_mask() layer for n steps.">

**Figure 2.** How you compute on that toy. Layered is one
sweep: hold `A`, gather it with `H`, then `hops[1]`. Adjacency
scatters `A` into the state and applies
`MaskedLinear(spec.to_mask())` for as many steps as you choose.

On a `LayeredSpec` you apply one `PackedLinear` per hop. Keep
every layer you produce in `saved`, because a later hop may read
it. `gather_hop_inputs()` concatenates the source layers a hop
reads; `hops[0]` always reads layer 0 alone, so the gather that
matters here is the one into `hops[1]`.

```python
saved = {0: x}
last = len(spec.hops) - 1
for index, hop in enumerate(spec.hops):
    sources = kpnn2.gather_hop_inputs(
        saved,
        hop,
    )
    hidden = self.hops[index](sources)
    if index < last:
        hidden = self.acts[index](hidden)
    saved[hop.target_layer] = hidden
```

`self.acts` is `nn.ModuleList([nn.ReLU() for _ in spec.hops])`.
The output hop stays linear.

`self.hops[index]` is `PackedLinear.from_hop(hop)` for that
hop. The dense alternative, `MaskedLinear(hop.to_mask())`,
builds the same model and runs faster when the hop's full
rectangle fits in memory; [PackedLinear](packed_linear.md)
explains the choice.
[Feedforward example](feedforward-example.ipynb) writes that module.
[Edges that skip layers](skip-edges.ipynb) shows why gather raises
if a source layer was never stored.

On an `AdjacencySpec` there is one square multiply. Aligned
inputs do not fit the state vector: `align_inputs()` returns
`len(spec.input_index)` positions, not `spec.state_dim`, so
you gather those columns and scatter them in at
`spec.input_index`. No edge enters an input node, so the input
rows of `to_mask()` are all zero, and the multiply drives those
[units](concepts.md#unit) to zero: the scatter is required, and
so is repeating it before every step.

```python
core = kpnn2.MaskedLinear(spec.to_mask())
state = torch.zeros(
    x.shape[0],
    spec.state_dim,
)
for tap in self.taps:
    state = state.index_copy(
        -1,
        self.input_index,
        x,
    )
    state = tap(self.act(core(state)))
y = state[:, spec.output_index]
```

`self.input_index` is a buffer,
`torch.as_tensor(spec.input_index)`. `self.taps` is an
`nn.ModuleList` of `nn.Identity`, one per step, and that is
what Captum hooks. `self.act` is an `nn.ReLU`. The shared
map stays a normal child of the module. `index_copy` returns
a new tensor, so the activation saved for backward is left
alone.

`n_steps` is yours; `kpnn2` does not unroll time. With `x` held
fixed, these steps are the shared-state update that
[Cyclic graph example](cyclic-graph-example.ipynb) trains on a
graph with a cycle. When `x` changes at each step, they make
a time-series [knowledge-primed neural network](concepts.md#kpnn)
(KPNN); see
[Time-series example](time-series-example.ipynb).

This toy is small, so `MaskedLinear` is appropriate; for large
node counts and RAM see [PackedLinear](packed_linear.md). The
same spec can feed `PackedMultiheadAttention.from_adjacency()`
(`chunk_size` when live-pair gathers strain RAM); that
walkthrough is the
[Transformer example](transformer-example.ipynb).

## How to choose a parser

- A feedforward DAG goes through `parse_layered()`. That is the
  usual KPNN.
- Use `parse_adjacency()` for cycles and self-loops
  (`parse_layered()` raises on those), a shared-state or
  time-series model, or packed attention. It is the packed
  layout: one state vector and packed edge indices.
- A DAG may go through `parse_adjacency()` if you want that
  layout. The package never inspects the
  [graph](concepts.md#graph) to pick a parser.
- Both still need at least one in-degree-0 node and one
  out-degree-0 node. A pure ring, or a lone self-loop, is
  rejected on both sides.

On a DAG, the adjacency loop in
[How you compute](#how-you-compute) reaches the layered output
only after as many steps as the longest input-to-output path
has edges. Three costs make `parse_layered()` the better choice
there:

- **Too few steps fail silently.** A node's value is final from
  step `d` on, where `d` is the number of edges on its longest
  path from the inputs. Before that it still depends on the zero
  start state. Too few steps return wrong outputs and raise
  nothing, and you have to find the step count yourself;
  `parse_layered()` computes those depths for you.
- **Attributions split across steps.** In the layered sweep a
  node has one value, in one layer tensor. In the loop it has a
  value at every step. When its paths to an output differ in
  length, its effect reaches the output from several steps, so
  a Captum tap on any one step sees only part of it.
- **Every step applies every edge.** One layered sweep applies
  each edge once. The loop applies all of them at every step,
  and backward stores the whole state vector once per step.
  The dense `to_mask()` square also spans every unit on both
  axes, where a hop's rectangle spans only its target layer and
  the layers it reads.

## Changing the prior

Specs are frozen after parse, on both layouts.

- To drop edges while training continues, keep the spec and
  put a keep-mask in `constraint=`. See
  [Pruning during training](packed_linear.md#pruning-during-training).
- To grow the prior, or to remove edges and retrain a smaller
  model, edit the edgelist, parse it again, and copy the
  surviving weights **by name**. See
  [Changing the prior (reparse)](packed_linear.md#changing-the-prior-reparse).

Do not `copy_` a whole `weight` or call `load_state_dict`
across two priors: both copy by position, so a value can
land on a different named edge.
