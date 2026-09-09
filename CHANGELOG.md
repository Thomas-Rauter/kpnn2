# Changelog

Notable API and core changes only. Documentation,
examples, and other small edits are not listed.

This project follows semantic versioning.


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
