# <id>: <model name or short title>

<!-- Written by tests/paper_fit/extract_prompt.md. Plain terms
only, no verdicts on library fit. Every claim about code cites
path:line at code_ref. Delete this comment and every placeholder
line in italics. -->

| Field | Value |
|-------|-------|
| Paper | First author et al., year. Title. Venue. DOI |
| Code | `code_url` at `code_ref` |
| Other code | Unofficial or later repositories, or "none" |
| Framework | Language, library, and version if pinned |
| License | Repository license, or "none found" |
| Full text | Where you read it, or "abstract only" |
| Extracted | YYYY-MM-DD, agent and model |

## Summary

*Three to five sentences: task, data, prior knowledge, model,
output.*

## Prior-knowledge graph

- Source: *database or ontology, version, filtering.*
- Size: *node types and counts; edge count; directed or not;
  signed or not; cycles or not.*
- Data to nodes: *how feature names are matched to nodes; what
  happens to unmatched features and to nodes without data.*
- Built at: *`path:start-end`.*

## Architecture

- Layers: *how many, and how a node's layer is decided.*
- Units per node: *one, or several (fixed or varying).*
- Connections that skip layers: *where, and how they are built.*
- Outputs or losses at inner layers: *which, and how combined.*
- Parts the graph does not constrain: *dense heads, embeddings,
  input encoders.*
- Other structure: *recurrence, attention, decoder, or "none".*
- Defined at: *`path:start-end`.*

## Connectivity mechanism

*How the code keeps absent edges absent. For example: a mask
multiplied into the weight, a sparse tensor, one small layer per
node, gather and scatter by index, or a constraint applied after
each optimizer step. Say where the full weight lives, whether
absent entries still get gradients, and how weights are
initialized. Cite `path:line`.*

## Interpretation

*Method (weights, DeepLIFT, integrated gradients, node activity,
...), its granularity (input, node, edge, layer), how scores are
mapped back to names, and how they are combined across samples,
repeats, or seeds. Cite `path:line`. Write "none" if the paper
does not interpret the model.*

## Needs

| Need | Evidence | How this paper needs it |
|------|----------|-------------------------|
| *N01 short name* | *`path:line`* | *one sentence* |

## Glue the authors wrote

*Code whose only job is to move between the prior-knowledge graph
and tensors: reading and filtering the graph, assigning layers,
building masks or index arrays, aligning feature names, enforcing
connectivity during training, tracking which unit belongs to
which node, mapping scores back to names.*

| What | Where | Lines |
|------|-------|-------|
| *Build one mask per layer* | *`path:start-end`* | *~40* |

## Fragile spots

*Places where the glue relies on something implicit or could
break the connectivity silently, for example a mask that one code
path does not apply. Evidence and one sentence each. "None
noticed" is fine.*

## Out of scope

*One line each: loss, training loop, data loading,
hyperparameter search, plotting, and anything else that does not
touch the graph.*

## Paper vs code

*Differences between the methods section and the code. "None
noticed" is fine.*

## Key excerpts

*At most 60 lines in total, verbatim, each headed by its path and
line range.*

`path/to/file.py:10-30`

```python
...
```

## Open questions

*What you could not determine, and where you looked.*
