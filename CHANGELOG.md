# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.


## [0.2.0] - Unreleased


### Added

- `PackedMultiheadAttention`.
- Transformer example docs notebook: a tiny
  prior-gated encoder on an edgelist, using
  `PackedMultiheadAttention`.

### Changed

- Split the API reference into a grouped index and one page
  per public name. The site nav lists only that index.
- Move the cyclic-graph and transformer notebooks under
  Additional examples in the site nav.
- Rename the Additional examples notebook from Recurrent
  example to Cyclic graph example
  (`docs/cyclic-graph-example.ipynb`). The page is a cyclic
  knowledge graph with a shared `MaskedLinear`, not an RNN.
- Rewrite the one-line summaries on that index so each name
  states its role, not an implementation detail.


## [0.1.0] - 1. September 2026

First release of `kpnn2`.
