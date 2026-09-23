In-training prune is a keep-mask inside `constraint=` on the
unchanged spec. Growing the prior, or removing edges to
retrain a smaller model, is a reparse: copy surviving
tensors **by name** with `edge_location`. Do not `copy_` or
`load_state_dict` across different priors. See
[Pruning during training](../packed_linear.md#pruning-during-training)
and
[Changing the prior (reparse)](../packed_linear.md#changing-the-prior-reparse).

::: kpnn2.AdjacencySpec
    options:
      members:
        - state_dim
        - node_units
        - to_edgelist
        - to_mask
        - edge_location
        - to_dict
        - from_dict
        - fingerprint
