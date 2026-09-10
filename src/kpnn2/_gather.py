"""
Source axis assembly and split for one hop.
"""

from collections.abc import Mapping

import torch

from ._errors import Kpnn2Error
from ._spec import Hop


def gather_hop_inputs(
    saved: Mapping[int, torch.Tensor],
    hop: Hop,
) -> torch.Tensor:
    """
    Concatenate the saved layer tensors one hop reads, in source-column order.

    A hop's source columns are whole source layers laid side by
    side, so the tensor feeding ``PackedLinear`` or
    ``MaskedLinear`` on that hop is those layers concatenated.
    Call it between hops, keeping every layer you produce in
    ``saved``; a forgotten layer raises rather than dropping the
    edges that read it. Inputs are never modified, and a
    single-source hop returns the saved tensor itself.

    Parameters
    ----------
    saved : mapping of int to torch.Tensor
        Layer depth to that layer's activation, as far as
        ``forward()`` has produced them. ``saved[i]`` is shaped
        ``(..., layer_dims[i])``; the leading dimensions are the
        caller's, typically a batch. Only the layers in
        ``hop.source_layers`` are read, so extra keys are
        ignored, and those layers must share a dtype and a
        device. Neither the mapping nor its tensors are copied
        or modified.
    hop : Hop
        The hop about to be applied, one entry of ``spec.hops``.
        Its ``source_layers`` and ``source_dims`` decide which
        keys are read, in which order, and how wide each one
        must be; the packed indices are not used here.

    Returns
    -------
    torch.Tensor
        The hop's source axis, shape
        ``(..., hop.in_features)``, columns in concatenated
        source-unit order, dtype and device of the saved layers.
        ``source_nodes`` is one name per node, so it is shorter
        than this axis when a source node is wider than 1. When
        the hop reads a single layer, including ``hops[0]`` and
        any hop whose parents all sit at one depth, this is that
        saved tensor itself rather than a copy, so writing into
        it writes into the saved activation.

    Raises
    ------
    Kpnn2Error
        If ``saved`` is not a mapping or ``hop`` is not a
        ``Hop``; a layer in ``hop.source_layers`` is absent from
        ``saved`` or is not a tensor; a saved tensor is
        0-dimensional or is the wrong number of units wide; or
        the source layers disagree on dtype or device.
    RuntimeError
        Propagated from ``torch.cat`` when two source layers
        disagree in a dimension other than the last, such as a
        batch size.

    See Also
    --------
    scatter_hop_outputs : Split the concatenated axis this
        returns back onto source layers.
    PackedLinear : Applies the hop to the tensor returned here.
    PackedLinear.transpose : Tied decode of that hop; feed its
        output to ``scatter_hop_outputs`` when the hop reads
        several layers.
    MaskedLinear : Dense hatch via ``hop.to_mask()``.
    Hop : The record that fixes the source layers and the
        concatenated column order this follows.
    align_inputs : Builds the layer-0 tensor that seeds
        ``saved``.

    Notes
    -----
    Nothing here is graph-aware: this holds no weights, does not
    inject values into the previous layer, and does not pick
    skip sources by name. Columns that are not edges stay in the
    concatenated tensor; ``PackedLinear`` never reads them, and
    ``MaskedLinear(hop.to_mask())`` zeros them. With the skip
    ``A -> C`` reaching past layer 1, layer 0 ``[A, B]`` and
    layer 1 ``[H]`` gather to ``[A, B, H]``, and the ``A -> C``
    weight is the packed pair whose source column is ``A``.

    An ``AdjacencySpec`` has no hops and is not accepted; in
    that layout every edge is already a packed index pair.

    Examples
    --------
    The hop into ``C`` reads layers 0 and 1, so its input is
    two columns wide:

    >>> import pandas as pd
    >>> import torch
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> spec = kpnn2.parse_layered(edgelist)
    >>> saved = {
    ...     0: torch.tensor([[2.0]]),
    ...     1: torch.tensor([[5.0]]),
    ... }
    >>> x = kpnn2.gather_hop_inputs(
    ...     saved,
    ...     spec.hops[1],
    ... )
    >>> x.tolist()
    [[2.0, 5.0]]

    Forgetting to store a layer raises instead of silently
    dropping the edges that read it:

    >>> kpnn2.gather_hop_inputs(
    ...     {1: torch.tensor([[5.0]])},
    ...     spec.hops[1],
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    Kpnn2Error: saved is missing layer 0. ...
    """
    if not isinstance(hop, Hop):
        raise Kpnn2Error("'hop' must be a Hop from spec.hops.")
    if not isinstance(saved, Mapping):
        raise Kpnn2Error("'saved' must be a mapping of layer index to tensor.")

    parts: list[torch.Tensor] = []
    for layer, n_units in zip(
        hop.source_layers,
        hop.source_dims,
    ):
        if layer not in saved:
            raise Kpnn2Error(
                f"saved is missing layer {layer}. The hop into "
                f"layer {hop.target_layer} reads layers "
                f"{list(hop.source_layers)}."
            )
        tensor = saved[layer]
        if not isinstance(tensor, torch.Tensor):
            raise Kpnn2Error(f"saved[{layer}] must be a torch.Tensor.")
        if tensor.ndim < 1:
            raise Kpnn2Error(
                f"saved[{layer}] has the wrong number of units. "
                f"Expected {n_units}, got a 0-dimensional tensor."
            )
        if tensor.shape[-1] != n_units:
            raise Kpnn2Error(
                f"saved[{layer}] has the wrong number of units. "
                f"Expected {n_units}, got {tensor.shape[-1]}."
            )
        parts.append(tensor)

    first = parts[0]
    for layer, tensor in zip(
        hop.source_layers,
        parts,
    ):
        if tensor.dtype != first.dtype:
            raise Kpnn2Error(
                "Saved layers must share a dtype. "
                f"saved[{hop.source_layers[0]}] is {first.dtype} "
                f"and saved[{layer}] is {tensor.dtype}."
            )
        if tensor.device != first.device:
            raise Kpnn2Error(
                "Saved layers must share a device. "
                f"saved[{hop.source_layers[0]}] is on "
                f"{first.device} and saved[{layer}] is on "
                f"{tensor.device}."
            )

    if len(parts) == 1:
        return first
    return torch.cat(
        parts,
        dim=-1,
    )


def scatter_hop_outputs(
    tensor: object,
    hop: Hop,
) -> dict[int, torch.Tensor]:
    """
    Split a hop's concatenated source axis back onto source layers.

    Inverse of ``gather_hop_inputs`` for that axis. The encoder
    concatenates whole source layers; a tied decoder
    (``PackedLinear.transpose``) emits the same concatenated
    width. This splits it. It does not take a ``saved`` dict
    and does not add into one: return the pieces and add them
    yourself, because two reversed hops may write the same
    earlier layer.

    Parameters
    ----------
    tensor : torch.Tensor
        Concatenated source axis, last dimension
        ``hop.in_features``. Typically the output of
        ``PackedLinear.transpose()`` on this hop. Leading
        dimensions are the caller's, typically a batch.
    hop : Hop
        The hop whose source axis ``tensor`` follows, one
        entry of ``spec.hops``.

    Returns
    -------
    dict of int to torch.Tensor
        One entry per ``hop.source_layers``, in that order.
        ``result[layer]`` has last dimension
        ``hop.source_dims[i]`` for that layer. A hop with one
        source layer returns ``tensor`` itself rather than a
        copy, so writing into the piece writes into
        ``tensor``. Several source layers are ``torch.split``
        views on the last axis.

    Raises
    ------
    Kpnn2Error
        If ``hop`` is not a ``Hop``; ``tensor`` is not a
        tensor; ``tensor`` is 0-dimensional; or the last
        dimension is not ``hop.in_features``.

    See Also
    --------
    gather_hop_inputs : Concatenates the layers this splits.
    PackedLinear.transpose : Packed ``W.T``; its output on a
        skip hop is what this splits.
    Hop : ``source_layers``, ``source_dims``, and
        ``column_offsets`` are the split layout.
    PackedLinear : Encoder map whose transpose emits this axis.

    Notes
    -----
    Nothing here is graph-aware and nothing holds weights.
    Unused skip columns stay in the pieces, same as they
    stay in a gather. An ``AdjacencySpec`` has no hops and
    is not accepted.

    Examples
    --------
    The hop into ``C`` reads layers 0 and 1, so a
    concatenated decoder activation splits back onto those
    depths:

    >>> import pandas as pd
    >>> import torch
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> spec = kpnn2.parse_layered(edgelist)
    >>> concat = torch.tensor([[2.0, 5.0]])
    >>> parts = kpnn2.scatter_hop_outputs(
    ...     concat,
    ...     spec.hops[1],
    ... )
    >>> list(parts)
    [0, 1]
    >>> parts[0].tolist(), parts[1].tolist()
    ([[2.0]], [[5.0]])
    """
    if not isinstance(hop, Hop):
        raise Kpnn2Error("'hop' must be a Hop from spec.hops.")
    if not isinstance(tensor, torch.Tensor):
        raise Kpnn2Error("'tensor' must be a torch.Tensor.")
    n_units = hop.in_features
    if tensor.ndim < 1:
        raise Kpnn2Error(
            "tensor has the wrong number of units. "
            f"Expected {n_units}, got a 0-dimensional tensor."
        )
    if tensor.shape[-1] != n_units:
        raise Kpnn2Error(
            "tensor has the wrong number of units. "
            f"Expected {n_units}, got {tensor.shape[-1]}."
        )

    if len(hop.source_layers) == 1:
        return {hop.source_layers[0]: tensor}

    pieces = torch.split(
        tensor,
        list(hop.source_dims),
        dim=-1,
    )
    return dict(
        zip(
            hop.source_layers,
            pieces,
            strict=True,
        )
    )
