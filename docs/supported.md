# Supported architectures

`kpnn2` supplies connectivity primitives, not finished models: you
write an ordinary `nn.Module` and call them from your own
`forward()`. The table below says which architecture families you
can build that way.

| Type | Supported | What to use |
| --- |-----------| --- |
| Feedforward | Yes       | `parse_layered` (optional `widths=` for DCell-style nodes), `PackedLinear`, `gather_hop_inputs`; tied decode `PackedLinear.transpose` + `scatter_hop_outputs` → [Feedforward example](feedforward-example.ipynb) |
| Cyclic | Yes       | `parse_adjacency`, `PackedLinear` or `MaskedLinear(spec.to_mask())`, loop you own → [Cyclic graph example](cyclic-graph-example.ipynb) |
| Transformer | Yes       | `parse_adjacency` + `PackedMultiheadAttention`; encoder / FFN / head are yours → [Transformer example](transformer-example.ipynb) |
| Sequence RNN / GRU / LSTM | Yes*      | `parse_adjacency` + `MaskedLinear` / `PackedLinear` as the maps; you write the time loop (not `nn.RNN` / `nn.GRU` / `nn.LSTM`) → [Time-series example](time-series-example.ipynb) |
| Graph NN | No        | Knowledge-primed GNNs are a natural edgelist model in [PyG](https://pyg.org/); we will not build or maintain `kpnn-pyg`. |
| Convolutional NN | No        | Currently no sparsely connected support. Build yourself in PyTorch (`nn.Conv1d` / `nn.Conv2d` / `nn.Conv3d`) |

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
