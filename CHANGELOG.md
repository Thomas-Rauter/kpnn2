# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.


## [Unreleased]

### Added

- `parse_adjacency()` and `AdjacencySpec`: a second layout that
  puts every node in one state vector with a single square
  `(n, n)` mask. Cycles and self-loops are allowed, so a
  recurrent network no longer needs a hand-built mask. The spec
  carries `nodes`, `input_nodes`, `output_nodes`, `hidden_nodes`,
  `mask`, `input_index`, and `output_index`.
- `parse_layered()` is unchanged and still requires a DAG. A DAG
  is valid input to both parsers; the layout is your choice.
  There is no `layout=` flag and no dispatch on cycles.

### Changed

- `map_node_attributions()` accepts a `LayeredSpec` or an
  `AdjacencySpec`, and `layer` is now optional. It stays required
  for a `LayeredSpec` and must be omitted for an `AdjacencySpec`,
  whose node axis is the whole state vector (`spec.nodes`) and
  whose result carries no scalar `layer` coordinate. Existing
  calls are unaffected: `layer` keeps its position.
- `align_inputs()` accepts a `LayeredSpec` or an `AdjacencySpec`.
  The DataFrame rules are unchanged; only `spec.input_nodes` is
  read. The result is always `len(spec.input_nodes)` wide, which
  for an `AdjacencySpec` is narrower than the square mask, so
  scatter it into the state vector with `spec.input_index`.
- Internal modules are now private (`_parse.py`, `_spec.py`,
  `_masked_linear.py`, `_skip_add.py`, `_align.py`,
  `_attributions.py`, `_errors.py`). `kpnn2/__init__.py` is the
  only public import path, so implementation files can be split
  or reorganized later without a breaking release.
- The public API is unchanged: `kpnn2.__all__` and every exported
  signature are identical. Code that imports from `kpnn2` directly
  needs no change; code that imported a module path such as
  `kpnn2.masked_linear` must import from `kpnn2` instead.
- The API reference now documents the package-level names, so
  permalinks are `#kpnn2.map_node_attributions` rather than
  `#kpnn2.map_node_attributions.map_node_attributions`.


## [0.1.0] - 2026-08-26

First release of `kpnn2`.
