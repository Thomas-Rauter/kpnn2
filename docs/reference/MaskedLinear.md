In-training prune is a keep-mask inside `constraint=` on the
unchanged spec. Growing the prior, or removing edges to
retrain a smaller model, is a reparse: copy surviving live
cells **by name**. Do not `copy_` the `(out, in)` rectangle
or `load_state_dict` across different priors. When the
`(out, in)` weight and the bias still have the same shapes,
`optimizer.load_state_dict` succeeds and applies moments by
position. Pass `identity=spec.fingerprint` on the new layer.
See
[Pruning during training](../packed_linear.md#pruning-during-training)
and
[Changing the prior (reparse)](../packed_linear.md#changing-the-prior-reparse).

::: kpnn2.MaskedLinear
    options:
      members:
        - effective_weight
        - init_bound
        - reset_parameters
        - forward
