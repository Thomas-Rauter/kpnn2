# Supported architectures

`kpnn2` does not ship ready-made models. You write an ordinary
`nn.Module`. This table is which architecture families you can
assemble from these primitives.

| Type | Supported | What to use |
| --- | --- | --- |
| Feedforward | Yes | `parse_layered` (optional `widths=` for DCell-style nodes), `PackedLinear`, `gather_hop_inputs` → [Feedforward example](feedforward-example.ipynb) |
| Cyclic | Yes | `parse_adjacency`, `PackedLinear` or `MaskedLinear(spec.to_mask())`, loop you own → [Cyclic graph example](cyclic-graph-example.ipynb) |
| Transformer | Yes | `parse_adjacency` + `PackedMultiheadAttention`; encoder / FFN / head are yours → [Transformer example](transformer-example.ipynb) |
| Sequence RNN / GRU / LSTM | Yes | `parse_adjacency` + `MaskedLinear` / `PackedLinear` as the maps; you write the time loop (not `nn.RNN` / `nn.GRU` / `nn.LSTM`) → [Time-series example](time-series-example.ipynb) |
| Graph NN | No | Knowledge-primed GNNs are a natural edgelist model in [PyG](https://pyg.org/); we will not build or maintain `kpnn-pyg`. |
| Convolutional NN | No | Currently no sparsely connected support. Build yourself in PyTorch (`nn.Conv1d` / `nn.Conv2d` / `nn.Conv3d`) |

## Sequence models

`nn.RNN`, `nn.GRU`, and `nn.LSTM` are fused convenience modules
with dense `W_ih` / `W_hh`. They cannot take an edgelist, so they
cannot be the recurrent map of a KPNN. This package does not
ship `MaskedRNN`, `MaskedGRU`, or `MaskedLSTM`.

A time-series KPNN is the same primitives as a cyclic graph, with
a **changing** `x_t` at each step:

1. `parse_adjacency()` (a self-loop or hidden→hidden edge is how
   a named node carries state across time; `parse_layered()`
   rejects those).
2. One shared `MaskedLinear(spec.to_mask())`, or `PackedLinear`
   when `n` is large.
3. In your `forward()`, for each time `t`: write `x_t` into
   `spec.input_index`, apply the map, keep the state.
4. `n_steps` is the sequence length you chose. `kpnn2` does not
   unroll time.

An Elman-style net is that loop. A GRU or LSTM is the usual gate
equations with one `MaskedLinear` per map. Whether every gate
shares the same prior is your modeling choice.

The [Time-series example](time-series-example.ipynb) walks through
the Elman loop, attributions on a `step` axis, and a no-memory
control.
