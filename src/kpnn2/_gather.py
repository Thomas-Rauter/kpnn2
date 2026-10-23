"""
Source axis assembly and split for one hop.
"""

from collections.abc import Mapping, Sequence

import torch

from ._errors import Kpnn2Error, caller_error, listed
from ._spec import Hop, require_hop
from ._validate import describe


def gather_hop_inputs(
    saved: Mapping[int, torch.Tensor],
    hop: Hop,
) -> torch.Tensor:
    """
    Build one hop's input from the layer activations you saved.

    Without skip edges, each layer reads only the layer directly
    below it, so one module's output is the next module's input.
    A skip edge breaks that chain: its source sits two or more
    layers below its target. kpnn2 still applies every edge into
    a layer with one module, the one built from that layer's hop
    (``PackedLinear.from_hop(hop)`` or
    ``MaskedLinear(hop.to_mask())``). That module therefore needs
    every layer its target's parents sit in, at once and as one
    tensor. This function builds that tensor. It does no
    arithmetic; it only arranges tensors you already have.

    Take ``A`` and ``B`` feeding ``H``, ``H`` feeding ``C``, and
    the skip ``A -> C``. Layer 0 is ``[A, B]``, layer 1 is
    ``[H]``, and the hop into ``C`` reads both, so its input is
    the three columns ``[A, B, H]``: whole layers, laid side by
    side in depth order. ``B`` is among them although there is
    no edge ``B -> C``. The hop says which layers to read, in
    which order, and how many units wide each must be. Without
    skip edges, every hop reads a single layer, and that layer
    is the input as it stands.

    Precisely, a hop's source columns are whole source layers
    laid side by side, so the tensor feeding ``PackedLinear`` or
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
        ignored, and those layers must share a dtype, a device,
        and their leading dimensions. Neither the mapping nor
        its tensors are copied or modified.
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
        ``Hop``; layers in ``hop.source_layers`` are absent from
        ``saved`` (the message names all of them); a saved layer
        is not a tensor; a saved tensor is
        0-dimensional or is the wrong number of units wide; or
        the source layers disagree on dtype, device, or a
        dimension other than the last, such as the batch size.

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
    align_inputs : Column index that puts named features in the
        order layer 0 expects.

    Notes
    -----
    Nothing here is graph-aware: this holds no weights, does not
    inject values into the previous layer, and does not pick
    skip sources by name. Columns that are not edges stay in the
    concatenated tensor; ``PackedLinear`` never reads them, and
    ``MaskedLinear(hop.to_mask())`` zeros them. In the
    ``[A, B, H]`` example above, the ``A -> C`` weight is the
    packed pair whose source column is ``A``.

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
    Kpnn2Error: 'saved' must hold every layer the hop into layer 2 ...
    """
    require_hop(
        hop,
        "hop",
    )
    if not isinstance(saved, Mapping):
        raise caller_error(
            "'saved' must be a mapping of layer index to tensor.",
            got=describe(saved),
        )

    # All missing layers before any other check, so one message
    # names every layer the caller forgot to store.
    missing = [layer for layer in hop.source_layers if layer not in saved]
    if missing:
        # The keys show a layer stored under another key, such as "0".
        if saved:
            held = f"'saved' has keys {listed(repr(key) for key in saved)}"
        else:
            held = "'saved' is empty"
        raise caller_error(
            "'saved' must hold every layer the hop into layer "
            f"{hop.target_layer} reads.",
            got=f"missing {_layers(missing)}; {held}",
            expected=_layers(hop.source_layers),
            fix=(
                "Store every layer the forward pass produces in 'saved', "
                "keyed by its depth as an int."
            ),
        )

    parts: list[torch.Tensor] = []
    for layer, n_units in zip(
        hop.source_layers,
        hop.source_dims,
    ):
        tensor = saved[layer]
        if not isinstance(tensor, torch.Tensor):
            raise caller_error(
                f"'saved[{layer}]' must be a torch.Tensor.",
                got=describe(tensor),
            )
        if tensor.ndim < 1 or tensor.shape[-1] != n_units:
            raise _units_error(
                tensor,
                f"'saved[{layer}]'",
                f"spec.layer_dims[{layer}]",
                n_units,
            )
        parts.append(tensor)

    first = parts[0]
    first_name = f"saved[{hop.source_layers[0]}]"
    for layer, tensor in zip(
        hop.source_layers,
        parts,
    ):
        if tensor.dtype != first.dtype:
            raise caller_error(
                "Saved layers must share a dtype.",
                got=(
                    f"{first_name} is {first.dtype}; saved[{layer}] is "
                    f"{tensor.dtype}"
                ),
            )
        if tensor.device != first.device:
            raise caller_error(
                "Saved layers must share a device.",
                got=(
                    f"{first_name} is on {first.device}; saved[{layer}] is "
                    f"on {tensor.device}"
                ),
            )
        if tensor.shape[:-1] != first.shape[:-1]:
            raise caller_error(
                "Saved layers must share every dimension but the last.",
                got=(
                    f"{first_name} has shape {tuple(first.shape)}; "
                    f"saved[{layer}] has shape {tuple(tensor.shape)}"
                ),
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
    Split a tensor laid out like one hop's input back into its layers.

    ``gather_hop_inputs`` lays the layers a hop reads side by
    side; this function cuts such a tensor apart again. You need
    it when a model runs a hop backwards, as the decoder of a
    tied autoencoder does. ``PackedLinear.transpose`` gives the
    layer built from a hop its reverse map, from the hop's
    target layer back onto the hop's input. On a hop that reads
    several layers, that output is those layers glued together,
    and the decoder can continue layer by layer only once each
    piece is filed under its own depth. This function makes the
    cut and returns the pieces keyed by depth. Like the gather,
    it does no arithmetic.

    Take ``A`` and ``B`` feeding ``H``, ``H`` feeding ``C``, and
    the skip ``A -> C``. The hop into ``C`` reads layer 0
    ``[A, B]`` and layer 1 ``[H]``, so its transposed layer maps
    ``C`` to the three columns ``[A, B, H]``, and this splits
    them into ``{0: [A, B], 1: [H]}``. Going forward, layer 0
    feeds two hops, the one into ``H`` and the one into ``C``.
    Going backward, it therefore receives a piece from each, and
    the transpose of a layer read twice is the sum of the two.
    That is why this returns pieces instead of writing them
    anywhere: the adding stays in your code.

    Precisely, this is the inverse of ``gather_hop_inputs`` for
    that axis. The encoder concatenates whole source layers; a
    tied decoder (``PackedLinear.transpose``) emits the same
    concatenated width. This splits it. It does not take a
    ``saved`` dict and does not add into one: return the pieces
    and add them yourself, because two reversed hops may write
    the same earlier layer.

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
    require_hop(
        hop,
        "hop",
    )
    if not isinstance(tensor, torch.Tensor):
        raise caller_error(
            "'tensor' must be a torch.Tensor.",
            got=describe(tensor),
        )
    n_units = hop.in_features
    if tensor.ndim < 1 or tensor.shape[-1] != n_units:
        raise _units_error(
            tensor,
            "'tensor'",
            "hop.in_features",
            n_units,
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


def _layers(layers: Sequence[int]) -> str:
    """Name one or several layers: ``"layer 0"``, ``"layers 0, 1"``."""
    noun = "layer" if len(layers) == 1 else "layers"
    return f"{noun} {listed(layers)}"


def _units_error(
    tensor: torch.Tensor,
    name: str,
    width: str,
    n_units: int,
) -> Kpnn2Error:
    """
    Build the error for a tensor whose last axis is not ``n_units``.

    ``name`` is the quoted tensor the message names, and ``width``
    the field that sets ``n_units``. A 0-dimensional tensor gets
    the same message: it has no last axis to hold the units.
    """
    return caller_error(
        f"{name} must have shape (..., {width}).",
        got=f"shape {tuple(tensor.shape)}",
        expected=f"shape (..., {n_units})",
    )
