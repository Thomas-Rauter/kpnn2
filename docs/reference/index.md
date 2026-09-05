---
toc_depth: 2
---

# API reference

Every name below is imported straight from the package
(`from kpnn2 import ...`). Module paths inside `kpnn2` are private
and may change between releases.

## Callables {: .api-group }

Names you call or construct.

| Name | Description |
| --- | --- |
| [`parse_layered`][kpnn2.parse_layered] | Parse a source/target edgelist into a `LayeredSpec` (DAG only). |
| [`parse_adjacency`][kpnn2.parse_adjacency] | Parse a source/target edgelist into an `AdjacencySpec` (cycles allowed). |
| [`MaskedLinear`][kpnn2.MaskedLinear] | Linear layer that keeps only the connections in a fixed mask. |
| [`PackedLinear`][kpnn2.PackedLinear] | Linear layer with one weight per live edge, not a dense matrix. |
| [`PackedMultiheadAttention`][kpnn2.PackedMultiheadAttention] | Attention that scores only live edgelist pairs, not a full matrix. |
| [`gather_hop_inputs`][kpnn2.gather_hop_inputs] | Build the input for one hop from the layer activations you saved. |
| [`align_inputs`][kpnn2.align_inputs] | Align named features to the spec's input nodes so the wiring matches. |
| [`map_node_attributions`][kpnn2.map_node_attributions] | Attach spec node names to an attribution tensor you already computed. |

## Specs {: .api-group }

Returned by the parsers. Frozen dataclasses; treat masks as
read-only.

| Name | Description |
| --- | --- |
| [`LayeredSpec`][kpnn2.LayeredSpec] | A DAG as named layers and hop masks; not a ready-made model. |
| [`Hop`][kpnn2.Hop] | Incoming wiring of one layer; pass its mask to `MaskedLinear`. |
| [`Skip`][kpnn2.Skip] | Record of an edge that jumps layers; already in the hop mask. |
| [`AdjacencySpec`][kpnn2.AdjacencySpec] | Every node in one state vector, with packed edges; cycles allowed. |

## Errors and version {: .api-group }

| Name | Description |
| --- | --- |
| [`Kpnn2Error`][kpnn2.Kpnn2Error] | Error raised by the public API when a call is invalid. |
| [`__version__`][kpnn2.__version__] | Package version string. |
