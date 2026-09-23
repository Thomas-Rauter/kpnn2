# Changelog

Notable API and core changes only. This project follows
semantic versioning.


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
