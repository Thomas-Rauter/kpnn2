# Changelog

Notable API and core changes only. Documentation,
examples, and other small edits are not listed.

This project follows semantic versioning.


## [Unreleased] — 0.2.0

### Changed

- `align_inputs` maps feature names to a 1-D `int64` column
  index. It no longer takes a DataFrame or returns a dense
  `float32` tensor of all rows. Apply the index on host X
  (`X[:, col]`).
- `MaskedLinear`, `PackedLinear`, and
  `PackedMultiheadAttention`: `forward` disables
  `torch.autocast` and computes in the parameter dtype.
  Mixed precision is still `.to(dtype=...)` / `.half()`.
- `Hop` stores packed `source_index` / `target_index`. There is
  no `mask` field; densify with `Hop.to_mask()`.
- `Skip.source_index` / `target_index` renamed to
  `source_in_layer` / `target_in_layer`.
- `PackedLinear` is the large-n path on hops as well as on an
  `AdjacencySpec`.
- `PackedLinear.forward` raises `Kpnn2Error` when the input's
  last dimension is not `in_features`. A wider input was read
  by position without error.

### Added

- `PackedMultiheadAttention`: scores only live edgelist pairs.
  `need_weights=True` returns packed per-edge weights aligned
  with `source_index` / `target_index`, not a dense `(L, S)`
  matrix. Default `False` still returns `None`.
- `PackedMultiheadAttention`: optional `chunk_size`. `None`
  gathers all live pairs at once (the default). A positive
  int softmaxes and mixes in slices of that many edges and
  rematerializes those gathers in backward.
- `PackedLinear.transpose()`: swapped packed indices, tied or
  copied `weight`, untied bias. The tied autoencoder path.
- `scatter_hop_outputs`: split a hop's concatenated source
  axis back onto source layers (inverse of
  `gather_hop_inputs`).
- `constraint=` on `MaskedLinear` and `PackedLinear`: optional
  `nn.Module` applied to live weights before the mask
  (`MaskedLinear`) or in packed space (`PackedLinear`).
  `MaskedLinear.weight` stays masked if later maps are stacked.
- `parse_layered(widths=)` / `LayeredSpec.layer_widths`: a named
  node can own several units.
- `LayeredSpec.edge_location` / `AdjacencySpec.edge_location`:
  locate a named edge's packed weight slots.
- `LayeredSpec.node_units` / `hop_units`: named node to its
  unit slice on a layer tensor or a hop source axis.
- `map_node_attributions(..., hop_input=, hop_output=)`: label
  what a hop's module reads (its concatenated source axis) or
  what it returns (its target layer). The side is always
  stated; there is no side-less `hop=`.
- `parse_layered(ranks=)`: optional user depths so official
  ontology levels need not be longest-path hops.
- `generator=` on `MaskedLinear`, `PackedLinear`,
  `PackedMultiheadAttention`, and `PackedLinear.transpose`:
  optional `torch.Generator` for isolated parameter init.
  Default `None` keeps the global stream.
- `identity=` on `MaskedLinear`, `PackedLinear`, and
  `PackedMultiheadAttention`: opaque spec fingerprint in
  `state_dict`, checked on load.
- `aggregate_node_attributions` / `list_aggregation_methods`:
  fold mapped node scores with a registered method. Default
  `rauter_mangano_2026` is registered and not yet
  implemented.

### Fixed

- `map_node_attributions` accepts bfloat16 scores and stores
  them as float32. The values are unchanged.


## [0.1.0] - 1. September 2026

First release of `kpnn2`.
