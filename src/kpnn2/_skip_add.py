"""
Skip-edge injection onto a layer pre-activation.
"""

from collections.abc import Mapping

import torch
from torch import nn

from ._errors import Kpnn2Error
from ._layout import Layout, build_layout
from ._spec import LayeredSpec


class SkipAdd(nn.Module):
    """
    Inject skip sources into a target layer pre-activation.

    Call after that hop's ``MaskedLinear`` and before ReLU /
    BatchNorm / dropout. A skip ``A -> H2`` means ``A`` is an
    extra parent of ``H2``. This module does not send ``A``
    through the adjacent weight matrix, does not undo ReLU, and
    does not modify ``saved`` tensors. There is no skip bias;
    unit bias stays on ``MaskedLinear``.

    Construct once from a ``LayeredSpec``. Call after every hop;
    hops with no matching skip are a no-op. Empty ``spec.skips``
    is identity.

    Parameters
    ----------
    spec : LayeredSpec
        Structure whose ``skips`` this module indexes. Not
        mutated.

    Attributes
    ----------
    spec : LayeredSpec
        The constructor spec.
    skip_weights : nn.ParameterList
        One scalar parameter per ``spec.skips``, initialized to
        ``0``.

    Raises
    ------
    Kpnn2Error
        If ``spec`` is not a ``LayeredSpec``. ``forward`` also
        raises if ``target_layer`` is invalid, a needed
        ``saved`` layer is missing, or a tensor width does not
        match ``layer_nodes``.

    Notes
    -----
    ``kpnn2`` owns skip indexing. The user owns call order and
    nonlinearities. Skip edges are not written into masks.

    Forward casts each skip weight and source to
    ``hidden.dtype`` / ``hidden.device`` before the add, like
    ``MaskedLinear`` casting the mask onto the hop.

    ``copy.deepcopy`` succeeds. Parameters on the copy are
    distinct. Copied ``LayeredSpec`` masks stay float32.

    Examples
    --------
    One skip ``A -> C`` injected at layer 2. Weights start at
    ``0``, so the first call is a no-op on values:

    >>> import pandas as pd
    >>> import torch
    >>> import kpnn2 as k2
    >>> edgelist = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "A"],
    ...         "target": ["H", "C", "C"],
    ...     }
    ... )
    >>> spec = k2.parse_layered(edgelist)
    >>> skips = k2.SkipAdd(spec)
    >>> len(skips.skip_weights)
    1
    >>> hidden = torch.ones(2, 1)
    >>> saved = {0: torch.ones(2, 1)}
    >>> out = skips(
    ...     hidden,
    ...     saved,
    ...     target_layer=2,
    ... )
    >>> torch.equal(out, hidden)
    True
    """

    spec: LayeredSpec
    skip_weights: nn.ParameterList
    _layouts: list[Layout]

    def __init__(
        self,
        spec: LayeredSpec,
    ) -> None:
        super().__init__()
        if not isinstance(spec, LayeredSpec):
            raise Kpnn2Error("'spec' must be a LayeredSpec.")
        self.spec = spec
        self._layouts = [build_layout(names) for names in spec.layer_nodes]
        weights = []
        for _skip in spec.skips:
            weights.append(
                nn.Parameter(
                    torch.zeros(1),
                )
            )
        self.skip_weights = nn.ParameterList(weights)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Set every skip scalar to ``0``.
        """
        with torch.no_grad():
            for weight in self.skip_weights:
                weight.zero_()

    def forward(
        self,
        hidden: torch.Tensor,
        saved: Mapping[int, torch.Tensor],
        target_layer: int,
    ) -> torch.Tensor:
        """
        Add matching skip terms onto ``hidden``.

        Parameters
        ----------
        hidden : torch.Tensor
            Pre-activation at ``target_layer``. Last dimension
            is ``len(spec.layer_nodes[target_layer])``.
        saved : mapping of int to Tensor
            Layer index to activation. Width of ``saved[i]`` is
            ``len(spec.layer_nodes[i])``. Entries are only
            read.
        target_layer : int
            Layer index of ``hidden``. Skips whose
            ``Skip.target_layer`` matches are applied.

        Returns
        -------
        torch.Tensor
            ``hidden`` plus ``w * saved_source`` at each
            matching skip's ``target_index``.

        Raises
        ------
        Kpnn2Error
            If ``target_layer`` is not a valid layer index, a
            required ``saved`` layer is missing, or a tensor
            has the wrong number of units.
        """
        if not isinstance(hidden, torch.Tensor):
            raise Kpnn2Error("'hidden' must be a torch.Tensor.")
        if not isinstance(saved, Mapping):
            raise Kpnn2Error(
                "'saved' must be a mapping of layer index to tensor."
            )
        if not isinstance(target_layer, int) or isinstance(
            target_layer,
            bool,
        ):
            raise Kpnn2Error("'target_layer' must be an int.")

        n_layers = len(self._layouts)
        if target_layer < 0 or target_layer >= n_layers:
            raise Kpnn2Error(
                "'target_layer' must be in range "
                f"[0, {n_layers}). Got {target_layer}."
            )

        target_layout = self._layouts[target_layer]
        n_units = target_layout.n_units
        if hidden.ndim < 1:
            raise Kpnn2Error(
                "Hidden tensor has the wrong number of units. "
                f"Expected {n_units}, got a 0-dimensional "
                "tensor."
            )
        if hidden.shape[-1] != n_units:
            raise Kpnn2Error(
                "Hidden tensor has the wrong number of units. "
                f"Expected {n_units}, got {hidden.shape[-1]}."
            )

        out = hidden
        for index, skip in enumerate(self.spec.skips):
            if skip.target_layer != target_layer:
                continue
            if skip.source_layer not in saved:
                raise Kpnn2Error(f"saved is missing layer {skip.source_layer}.")
            source_tensor = saved[skip.source_layer]
            if not isinstance(source_tensor, torch.Tensor):
                raise Kpnn2Error(
                    f"saved[{skip.source_layer}] must be a torch.Tensor."
                )
            source_layout = self._layouts[skip.source_layer]
            n_source = source_layout.n_units
            if source_tensor.ndim < 1:
                raise Kpnn2Error(
                    "Saved tensor for layer "
                    f"{skip.source_layer} has the wrong "
                    "number of units. "
                    f"Expected {n_source}, got a "
                    "0-dimensional tensor."
                )
            if source_tensor.shape[-1] != n_source:
                raise Kpnn2Error(
                    "Saved tensor for layer "
                    f"{skip.source_layer} has the wrong "
                    "number of units. "
                    f"Expected {n_source}, got "
                    f"{source_tensor.shape[-1]}."
                )
            source_slot = source_layout.slot_at(skip.source_index)
            target_slot = target_layout.slot_at(skip.target_index)
            source = source_tensor[..., source_slot.units]
            weight = self.skip_weights[index].to(
                dtype=hidden.dtype,
                device=hidden.device,
            )
            source = source.to(
                dtype=hidden.dtype,
                device=hidden.device,
            )
            term = torch.zeros_like(hidden)
            term[..., target_slot.units] = weight * source
            out = out + term
        return out
