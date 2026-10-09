# Needs catalog

What published models need to do to get prior knowledge into,
through, or out of a neural network. Extraction sessions grow
this file. Assessment sessions read it.

## Rules

- **Paper-neutral.** Write a need as what the model has to do,
  in plain deep-learning terms. Do not use names from this
  repository's package, and do not say whether anything covers
  the need. Coverage belongs in `runs/`.
- **Evidence first.** A need enters only together with a card
  that cites code for it.
- **Granularity.** One need is one thing a library could provide
  or fail to provide. "Uses a sparse network" is too broad.
  "ReLU after the third layer" is too narrow. Test: could two
  papers from different fields share this need?
- **Stable IDs.** IDs are `N01`, `N02`, and so on, in order of
  entry. Never renumber or reuse an ID. To merge a duplicate,
  keep both rows and write `merged into Nxx` in the later row's
  description.
- **Area** is one of `graph` (reading and shaping the prior),
  `architecture` (where nodes sit and how layers connect),
  `mechanism` (how absent edges stay absent), `training`, and
  `interpretation`.

## Catalog

| ID | Area | Need | Description | Papers |
|----|------|------|-------------|--------|
