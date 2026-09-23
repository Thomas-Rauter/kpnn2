In-training prune is a keep-mask inside `constraint=` on the
unchanged spec. Growing the prior, or removing edges to
retrain a smaller model, is a reparse: copy surviving weight
slots **by name**. Do not `copy_` or `load_state_dict` across
different priors. When `weight` (`nnz`) and `bias` still
have the same shapes, `optimizer.load_state_dict` succeeds
and applies moments by position. Pass
`identity=spec.fingerprint` on the new layer.
See
[Pruning during training](../packed_linear.md#pruning-during-training)
and
[Changing the prior (reparse)](../packed_linear.md#changing-the-prior-reparse).

::: kpnn2.PackedLinear
    options:
      members:
        - effective_weight
        - init_bound
        - reset_parameters
        - transpose
        - forward
