# Changelog

Notable API and core changes only. Documentation,
examples, and other small edits are not listed.

This project follows semantic versioning.


## [Unreleased]

### Changed

- `PackedMultiheadAttention`: `need_weights=True` returns
  packed per-edge weights aligned with `source_index` /
  `target_index`, not a dense `(L, S)` matrix and not an
  error. Default `False` still returns `None`.
- `Hop` stores packed `source_index` / `target_index`. There is
  no `mask` field; densify with `Hop.to_mask()`.
- `Skip.source_index` / `target_index` renamed to
  `source_in_layer` / `target_in_layer`.
- `PackedLinear` is the large-n path on hops as well as on an
  `AdjacencySpec`.

### Added

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
- `map_node_attributions(..., hop=)`: label a hop's
  concatenated source axis.
- `parse_layered(ranks=)`: optional user depths so official
  ontology levels need not be longest-path hops.
- `generator=` on `MaskedLinear`, `PackedLinear`,
  `PackedMultiheadAttention`, and `PackedLinear.transpose`:
  optional `torch.Generator` for isolated parameter init.
  Default `None` keeps the global stream.
- `identity=` on `MaskedLinear`, `PackedLinear`, and
  `PackedMultiheadAttention`: opaque spec fingerprint in
  `state_dict`, checked on load.


## [0.1.0] - 1. September 2026

First release of `kpnn2`.
