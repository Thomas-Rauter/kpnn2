---
toc_depth: 3
---

# API reference

Every name below is imported straight from the package
(`from kpnn2 import ...`). Module paths inside `kpnn2` are private
and may change between releases.

## Callables {: .api-group }

Names you call or construct.

::: kpnn2.parse_layered
    options:
      show_root_heading: true
      show_root_toc_entry: true

::: kpnn2.parse_adjacency
    options:
      show_root_heading: true
      show_root_toc_entry: true

::: kpnn2.MaskedLinear
    options:
      show_root_heading: true
      show_root_toc_entry: true
      merge_init_into_class: true
      members:
        - reset_parameters
        - forward

::: kpnn2.gather_hop_inputs
    options:
      show_root_heading: true
      show_root_toc_entry: true

::: kpnn2.align_inputs
    options:
      show_root_heading: true
      show_root_toc_entry: true

::: kpnn2.map_node_attributions
    options:
      show_root_heading: true
      show_root_toc_entry: true

## Specs {: .api-group }

Returned by the parsers. Frozen dataclasses; treat masks as
read-only.

::: kpnn2.LayeredSpec
    options:
      show_root_heading: true
      show_root_toc_entry: true
      members: false

::: kpnn2.Hop
    options:
      show_root_heading: true
      show_root_toc_entry: true
      members:
        - column_offsets

::: kpnn2.Skip
    options:
      show_root_heading: true
      show_root_toc_entry: true
      members: false

::: kpnn2.AdjacencySpec
    options:
      show_root_heading: true
      show_root_toc_entry: true
      members: false

## Errors and version {: .api-group }

::: kpnn2.Kpnn2Error
    options:
      show_root_heading: true
      show_root_toc_entry: true
      members: false

::: kpnn2.__version__
    options:
      show_root_heading: true
      show_root_toc_entry: true
