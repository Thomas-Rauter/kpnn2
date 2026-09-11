To prune or grow the prior, reparse and copy surviving
live cells **by name**. Do not `copy_` the `(out, in)`
rectangle or `load_state_dict` across different priors.
Pass `identity=spec.fingerprint` on the new layer. See
[Changing the prior (reparse)](../packed_linear.md#changing-the-prior-reparse).

::: kpnn2.MaskedLinear
    options:
      members:
        - reset_parameters
        - forward
