# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.


## [0.2.0] - Unreleased


### Added

- `PackedMultiheadAttention`.
- Transformer example docs notebook: a tiny
  prior-gated encoder on an edgelist, using
  `PackedMultiheadAttention`.
- Time-series example docs notebook: a shared
  `MaskedLinear` loop over a sequence `x_t`, with
  a self-loop so named nodes carry state.
- Supported architectures docs page: a table of
  which architecture families `kpnn2` covers.

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
- Rename the Getting started notebook to Feedforward example
  (`docs/feedforward-example.ipynb`). The Getting started
  section still holds Installation, Supported architectures,
  and this page.
- Spell out the Sequence RNN / GRU / LSTM row on Supported
  architectures: maps are `MaskedLinear` / `PackedLinear`,
  the time loop is yours, and `nn.RNN` cannot take an
  edgelist.
- Generalize `parse_adjacency` docs: the packed layout is
  for shared-state loops, sequences, and attention, not only
  recurrent or cyclic graphs.


## [0.1.0] - 1. September 2026

First release of `kpnn2`.
