# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.


## [0.1.0] - 2026-08-25

Initial release of `kpnn2`.

Parse a source/target edgelist into a frozen `GraphSpec`, then
write ordinary PyTorch with `MaskedLinear`. There is no graph
compiler and no ready-made model object.

### Added

- `parse_edgelist`
- `GraphSpec` (frozen dataclass; sequence fields are tuples)
- `Skip`
- `MaskedLinear` (edgelist mask is read-only, not in `state_dict`)
- `align_inputs`
- `map_node_attributions` (returns `xarray.DataArray`)
- `Kpnn2Error`
- `__version__`
