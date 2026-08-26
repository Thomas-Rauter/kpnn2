# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.


## [Unreleased]

### Changed

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
