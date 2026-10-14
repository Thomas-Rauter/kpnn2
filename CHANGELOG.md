# Changelog

Notable API and core changes only. This project follows
semantic versioning.


## [Unreleased]

### Changed

- `align_inputs` raises `Kpnn2Error` when a feature name matches
  a hidden or output node, whose column the model would never
  read. Its missing-name message counts the input nodes found
  and explains that the edgelist starts at the measured
  features.
- `PackedMultiheadAttention` with a positive `chunk_size`
  checks on every call that each query's softmax sums to 1
  (0 only for a query without a finite live score) and its
  gradient to 0, and waits for the device once per
  forward and once per backward.
- `MaskedLinear` rejects a mask with a value other than 0
  and 1, an empty dimension, or no live entry. Put per-edge
  scaling or signs in `constraint=`.
- `MaskedLinear.forward` raises `Kpnn2Error` (was a raw torch
  error) when the input is not a tensor or its last dimension
  is not `in_features`.
- `MaskedLinear`, `PackedLinear`, and `PackedMultiheadAttention`
  reject an integer, `bool`, or complex input in `forward`
  instead of casting it to the parameter dtype. Convert with
  `.float()` first.
- Arguments whose items are matched by position reject a set
  or mapping: `source_index` / `target_index` of
  `PackedLinear` and `PackedMultiheadAttention`, and `dims` and
  `coords` values of `map_node_attributions`. The packed layers
  also reject a complex index tensor.
- `gather_hop_inputs` raises `Kpnn2Error` (was a `RuntimeError`
  from `torch.cat`) when the saved source layers differ in a
  dimension other than the last, such as the batch size.
- Flag arguments (`bias`, `tie`, `batch_first`,
  `add_self_loops`, `need_weights`, `average_attn_weights`)
  must be `bool`, and `PackedMultiheadAttention` `dropout`
  must be in [0, 1]; other values raise `Kpnn2Error`.
- `parse_layered` and `parse_adjacency` reject node names that
  differ only by leading or trailing whitespace (for example
  `'A'` and `'A '`).
- Node names are text. `parse_layered`, `parse_adjacency`, and
  `from_dict` still read integers through `str()`, warn on a
  float or `bool` (`1.0` becomes `'1.0'`), and reject any other
  value, such as a list or a tensor.
- `widths=` on `parse_layered`, `parse_adjacency`, and
  `from_dict` rejects a width above 1 on an input node.
  `align_inputs` returns one column per input node and no
  longer repeats columns. Give each measurement its own input
  node.
- `map_node_attributions` gives every `LayeredSpec` result a
  `layer` coordinate along `node`, one `int64` depth per unit;
  `hop_input` results get each unit's source layer. It was a
  scalar, absent for `hop_input`, and `xr.concat` along `node`
  could give units of other layers the first array's depth.
  Read one depth with `int(da["layer"][0])`.

### Added

- `unit_index` on both specs: every unit of the named nodes,
  in order, for nodes outside `input_nodes` and
  `output_nodes`. `AdjacencySpec.unit_index(names)` indexes
  the state vector, `LayeredSpec.unit_index(names, layer=)`
  one layer.
- `edge_index` on both specs: every packed slot of the named
  edges, in order, so per-edge values kept on your own table
  (signs, prior flags) land on their slots without a loop over
  `edge_location`. `LayeredSpec.edge_index(sources, targets,
  hop=)` indexes one hop.
- `slot_table()` on both specs: the source and target name of
  every packed slot, hop by hop on `LayeredSpec`, so trained
  weights and per-slot scores get edge names by assignment.
- `node_table()` on both specs: one row per node with its
  layer (layered only), width, in- and out-degree, and counts
  of ancestors, descendants, and input ancestors, in unit
  order.
- `PackedLinear.from_hop(hop)`, `PackedLinear.from_adjacency(spec)`,
  and `PackedMultiheadAttention.from_adjacency(spec, embed_dim,
  num_heads)`: build the layer from a spec, so source and target
  indices cannot be swapped by hand. They are the recommended
  path; the index constructors stay for pairs from no spec.


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
