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
- `MaskedLinear` now applies its mask through
  `torch.nn.utils.parametrize.register_parametrization`, so the
  effective masked weight is `layer.weight` and the trainable
  tensor is `layer.parametrizations.weight.original`. The old
  `raw_weight` parameter is gone. `weight` is the attribute name
  that optimizer param-group filters, weight-decay rules,
  adapter libraries, and `state_dict` scripts look for, and
  nothing could ask this layer for its weight matrix under that
  name before. Consequences worth knowing:
  `state_dict` keys are now `parametrizations.weight.original`
  and `bias`, so 0.1.0-style checkpoints with a `raw_weight` key
  do not load; `repr` reports `ParametrizedMaskedLinear` while
  `isinstance(layer, MaskedLinear)` stays `True`; `repr` also
  gained `nn.Linear`-style `in_features` / `out_features` /
  `bias`; in-place writes to `layer.weight` are discarded, while
  assignment (`layer.weight = w`) copies into the trainable
  tensor; and pickling the module object raises, as for any
  parametrized module (`copy.deepcopy` and `state_dict` still
  work). Forward math, the degree-aware initialization, the RNG
  draw order, mask independence, dtype casts, and
  `torch.compile(fullgraph=True)` are all unchanged.
- Connectivity masks are plain `torch.Tensor` again. The
  `FrozenMask` subclass that rejected in-place writes is gone,
  together with the `Kpnn2Error` it raised from
  `spec.masks[i]`, `spec.mask`, and `MaskedLinear.mask` on
  `fill_`, item assignment, `out=`, buffer replacement, and
  `numpy()`. Those tensors are now documented as read-only
  rather than enforced, which is how PyTorch treats buffers.
  The subclass ran a `__torch_function__` hook on every
  operation: it cloned the whole mask on every forward pass and
  made `torch.compile(fullgraph=True)` fail with a graph break.
  Both are fixed, and `MaskedLinear` still stores its own copy
  of the mask, so writing to `spec.masks[i]` cannot rewire a
  layer that was already built. Mask operations also return
  standard PyTorch types again: `mask.shape` is a `torch.Size`
  rather than a plain tuple, indexing returns a view instead of
  a copy, and `numpy()` returns a writable array.
- Node positions are now internal slices rather than single
  columns. A private `_layout.py` places each node on a
  contiguous `NodeSlot` of its tensor axis, masks are written by
  block expansion, and parsing, skip indexing, input alignment,
  and attribution naming all read positions from a `Layout`.
  Every slice is one unit wide, so shapes, masks, indices, and
  the public API are unchanged. This makes per-node width an
  additive change later rather than a structural one.
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
