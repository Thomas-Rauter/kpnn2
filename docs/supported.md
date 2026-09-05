# Supported architectures

`kpnn2` does not ship ready-made models. You write an ordinary
`nn.Module`. This table is which architecture families you can
assemble from these primitives.

| Type | Supported | What to use |
| --- | --- | --- |
| Feedforward | Yes | `parse_layered`, `MaskedLinear`, `gather_hop_inputs` → [Getting started](getting-started.ipynb) |
| Cyclic | Yes | `parse_adjacency`, `PackedLinear` or `MaskedLinear(spec.to_mask())`, loop you own → [Cyclic graph example](cyclic-graph-example.ipynb) |
| Transformer | Yes | `parse_adjacency` + `PackedMultiheadAttention`; encoder / FFN / head are yours → [Transformer example](transformer-example.ipynb) |
| Sequence RNN / GRU / LSTM | Yes | `MaskedLinear` / `PackedLinear` as the maps; cell and time loop are yours (not `nn.RNN` / `nn.GRU` / `nn.LSTM`) |
| Graph NN | No | [`kpnn-pyg`](https://pypi.org/project/kpnn-pyg/) |
| Convolutional NN | No | Currently no sparsely connected support (`kpnn2` / `kpnn-pyg` do not cover this). Build yourself in PyTorch (`nn.Conv1d` / `nn.Conv2d` / `nn.Conv3d`) |
