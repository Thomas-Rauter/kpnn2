# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.


## [Unreleased]

### Changed

- `copy.deepcopy` of `GraphSpec`, `MaskedLinear`, and user
  modules that hold them succeeds. Copied masks stay frozen
  float32 and do not share storage with the original.
- Frozen connectivity masks reject `out=` writes, mask
  replacement via `register_buffer`, and numpy aliases of
  stored storage. `MaskedLinear` clones independently of
  `GraphSpec.masks`.
- Duplicate-edge and self-loop failures from `parse_edgelist`
  name the offending pairs / nodes (unique, sorted), while
  keeping the existing extra-row / self-loop-row counts.
- `MaskedLinear` keeps a float32 frozen mask after Module
  dtype casts. Forward multiplies `raw_weight` by a mask
  cast to `raw_weight.dtype` / device so `.half()`,
  bfloat16, and `.double()` work like `nn.Linear`.
- Cycle failures from `parse_edgelist` name the unranked leftover
  nodes (Kahn set: `nodes` minus ranked depths), sorted
  alphabetically. Cycles still raise `Kpnn2Error`; DAGs only.
- `align_inputs` rejects `torch.Tensor`. Pass a pandas DataFrame
  for name alignment, or a pre-ordered tensor straight to the
  model.
- The `dev` extra now includes `captum` so tests can run the
  documented Captum mapping workflow. Captum remains optional
  for library users and is not a core dependency.
- The `dev` extra pins `ruff==0.16.4` so local and CI formatting
  cannot drift across Ruff releases.
- On Python 3.11+, `xarray` is `>=2026.4,<2026.8` so NumPy 2.5
  no longer warns about generic timedelta units. Python 3.10
  stays on `xarray>=2024.11,<2025.7`, the last line that
  supports 3.10.


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
