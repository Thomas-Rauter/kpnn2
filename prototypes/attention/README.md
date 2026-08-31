Packed multi-head attention prototype (not shipped in the ``kpnn2``
wheel). ``PackedMultiheadAttention`` is the packed analogue of
``nn.MultiheadAttention``: allowed pairs come from
``parse_adjacency`` packed indices. The public spec is unchanged.

Run::

    pytest prototypes/attention
