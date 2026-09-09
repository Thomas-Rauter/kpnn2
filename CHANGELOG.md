# Changelog

Notable API and core changes only. Documentation,
examples, and other small edits are not listed.

This project follows semantic versioning.


## [Unreleased]

### Added

- `constraint=` on `MaskedLinear` and `PackedLinear`: optional
  `nn.Module` applied to live weights before the mask
  (`MaskedLinear`) or in packed space (`PackedLinear`).
  `MaskedLinear.weight` stays masked if later maps are stacked.
- `parse_layered(widths=)` / `LayeredSpec.layer_widths`: a named
  node can own several units.
- `LayeredSpec.edge_location` / `AdjacencySpec.edge_location`:
  locate a named edge's packed weight slots.
- `map_node_attributions(..., hop=)`: label a hop's
  concatenated source axis.


## [0.3.0] - 9. September 2026

### Changed

- `Hop` stores packed `source_index` / `target_index`. There is
  no `mask` field; densify with `Hop.to_mask()`.
- `Skip.source_index` / `target_index` renamed to
  `source_in_layer` / `target_in_layer`.
- `PackedLinear` is the large-n path on hops as well as on an
  `AdjacencySpec`.


## [0.2.0] - 11. September 2026

### Added

- `identity=` on `MaskedLinear`, `PackedLinear`, and
  `PackedMultiheadAttention`: opaque spec fingerprint in
  `state_dict`, checked on load.


## [0.1.0] - 1. September 2026

First release of `kpnn2`.
