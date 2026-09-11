To prune or grow the prior, reparse and copy surviving
weight slots **by name**. Do not `copy_` or
`load_state_dict` across different priors. Pass
`identity=spec.fingerprint` on the new layer. See
[Changing the prior (reparse)](../packed_linear.md#changing-the-prior-reparse).

::: kpnn2.PackedLinear
    options:
      members:
        - reset_parameters
        - transpose
        - forward
