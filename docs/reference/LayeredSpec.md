To prune or grow the prior, reparse and copy surviving
tensors **by name** with `edge_location`. Do not `copy_`
or `load_state_dict` across different priors. See
[Changing the prior (reparse)](../packed_linear.md#changing-the-prior-reparse).

::: kpnn2.LayeredSpec
    options:
      members:
        - to_edgelist
        - edge_location
        - node_units
        - hop_units
        - to_dict
        - from_dict
        - fingerprint
