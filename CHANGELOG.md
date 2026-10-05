# Changelog

Notable API and core changes only. This project follows
semantic versioning.


## [Unreleased]

### Changed

- `PackedMultiheadAttention` with a positive `chunk_size`
  checks on every call that each query's softmax sums to 1
  and its gradient to 0, and waits for the device once per
  forward and once per backward.
- `MaskedLinear` rejects a mask with a value other than 0
  and 1, an empty dimension, or no live entry. Put per-edge
  scaling or signs in `constraint=`.
- `MaskedLinear.forward` raises `Kpnn2Error` (was a raw torch
  error) when the input is not a tensor or its last dimension
  is not `in_features`.
- Flag arguments (`bias`, `tie`, `batch_first`,
  `add_self_loops`, `need_weights`, `average_attn_weights`)
  must be `bool`, and `PackedMultiheadAttention` `dropout`
  must be in [0, 1]; other values raise `Kpnn2Error`.


## [0.2.0] - 23. September 2026

### Changed

These can require edits to code written for 0.1.

- `align_inputs` returns a 1-D `int64` column index, not a
  tensor. Index your data with it: `X[:, col]`.
- `Hop` stores packed `source_index` / `target_index` instead
  of a mask, so `PackedLinear` now works on hops. Call
  `Hop.to_mask()` for the dense mask.
- `Skip.source_index` / `target_index` are renamed to
  `source_in_layer` / `target_in_layer`.
- `MaskedLinear.weight` is a plain parameter (`state_dict` key
  `weight`, was `parametrizations.weight.original`); read
  `effective_weight()` for the masked map. 0.1 checkpoints
  still load.
- `MaskedLinear`, `PackedLinear`, and
  `PackedMultiheadAttention` ignore `torch.autocast` and
  compute in their parameter dtype.
- `PackedLinear.forward` raises `Kpnn2Error` when the input's
  last dimension is not `in_features`.

### Added

- `PackedMultiheadAttention`: attention restricted to the
  prior's edges, with optional `chunk_size` to bound memory.
- `aggregate_node_attributions` and `list_aggregation_methods`:
  fold named node scores with a registered method. The default
  method, `rauter_mangano_2026`, is not yet implemented.
- `PackedLinear.transpose()` and `scatter_hop_outputs`, for
  tied decoders.
- `widths=` on both parsers, so a node can own several units,
  and `ranks=` on `parse_layered`, to set depths yourself.
- `constraint=` on `MaskedLinear` and `PackedLinear` (for
  example positive or frozen edges), with `effective_weight()`
  and `init_bound()`.
- `identity=` on all three layers: a spec fingerprint stored in
  the checkpoint and checked on load.
- `generator=` on all three layers, for isolated parameter init.
- `edge_location()` and `node_units()` on both specs, and
  `hop_units()` on `LayeredSpec`: find a named edge's weight
  slots or a node's units.
- `hop_input=`, `hop_output=`, and `axis="inputs"` on
  `map_node_attributions`: say which side of a hop, or which
  input units, a tensor holds.

### Fixed

- `map_node_attributions` no longer shares memory with the
  input tensor, and it accepts bfloat16 scores (stored as
  float32).

### Dependencies

- `xarray>=2024.11`, with no upper bound.


## [0.1.0] - 1. September 2026

First release of `kpnn2`.
