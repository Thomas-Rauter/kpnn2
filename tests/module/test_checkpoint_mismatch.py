"""Checkpoint-mismatch messages name the submodule, the difference, the fix.

A model holds many connectivity layers, so "this layer" alone does not
say which one refused the checkpoint. The headline names the submodule
path from the load prefix and what differs; an identity mismatch shows
both identities on its Got and Expected lines, and the Fix line says how
to fix the call.
"""

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

from kpnn2 import (
    Kpnn2Error,
    MaskedLinear,
    PackedLinear,
    PackedMultiheadAttention,
)

_PACKED_LINEAR_PAGE = (
    Path(__file__).resolve().parents[2] / "docs" / "packed_linear.md"
)

# Same nnz and sizes, different edges: every tensor keeps its shape,
# so only the wiring digest can tell the two apart.
_WIRING = [(0, 0), (1, 1), (2, 1)]
_REWIRED = [(0, 0), (2, 0), (1, 1)]

_SAVED_IDENTITY = hashlib.sha256(b"saved prior").hexdigest()
_LIVE_IDENTITY = hashlib.sha256(b"live prior").hexdigest()

_IDENTITY_SENTENCE = "The checkpoint identity does not match this layer"
_WIRING_REASON = ": the saved edges, sizes, or edge order differ."

Builder = Callable[[list[tuple[int, int]], str | None], nn.Module]


def _masked_linear(
    pairs: list[tuple[int, int]],
    identity: str | None,
) -> nn.Module:
    mask = torch.zeros(
        2,
        3,
    )
    for source, target in pairs:
        mask[target, source] = 1.0
    return MaskedLinear(
        mask,
        identity=identity,
    )


def _packed_linear(
    pairs: list[tuple[int, int]],
    identity: str | None,
) -> nn.Module:
    return PackedLinear(
        [source for source, _ in pairs],
        [target for _, target in pairs],
        2,
        3,
        identity=identity,
    )


def _packed_attention(
    pairs: list[tuple[int, int]],
    identity: str | None,
) -> nn.Module:
    return PackedMultiheadAttention(
        [source for source, _ in pairs],
        [target for _, target in pairs],
        2,
        3,
        4,
        2,
        identity=identity,
    )


_LAYERS = {
    "MaskedLinear": (
        _masked_linear,
        "The checkpoint mask does not match this layer",
    ),
    "PackedLinear": (
        _packed_linear,
        "The checkpoint indices do not match this layer",
    ),
    "PackedMultiheadAttention": (
        _packed_attention,
        "The checkpoint indices or num_heads do not match this layer",
    ),
}
# How each layer is rebuilt when the identity matched but the wiring
# did not.
_SAME_BUILD = {
    "MaskedLinear": "from the same hop and with the same sizes",
    "PackedLinear": "from the same hop and with the same sizes",
    "PackedMultiheadAttention": "with the same embed_dim and num_heads",
}


def _model(
    build: Builder,
    middle_pairs: list[tuple[int, int]],
    identity: str | None,
    middle_identity: str | None,
) -> nn.ModuleDict:
    """Three layers at ``hops.0`` to ``hops.2``; only ``hops.1`` varies."""
    torch.manual_seed(42)
    return nn.ModuleDict(
        {
            "hops": nn.ModuleList(
                [
                    build(
                        _WIRING,
                        identity,
                    ),
                    build(
                        middle_pairs,
                        middle_identity,
                    ),
                    build(
                        _WIRING,
                        identity,
                    ),
                ]
            )
        }
    )


def _load_error(
    destination: nn.Module,
    source: nn.Module,
) -> str:
    with pytest.raises(Kpnn2Error) as info:
        destination.load_state_dict(source.state_dict())
    return str(info.value)


def _shortened(identity: str) -> str:
    return repr(f"{identity[:12]}...")


def _snapshot(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _assert_unchanged(
    module: nn.Module,
    before: list[torch.Tensor],
) -> None:
    after = list(module.parameters())
    assert len(after) == len(before)
    for parameter, expected in zip(
        after,
        before,
    ):
        torch.testing.assert_close(
            parameter.detach(),
            expected,
        )


def _assert_names_only_the_middle_hop(message: str) -> None:
    assert " in submodule 'hops.1'" in message.splitlines()[0]
    assert "'hops.0'" not in message
    assert "'hops.2'" not in message


def _assert_names_the_fix(message: str) -> None:
    fix = message.splitlines()[-1]
    assert fix.startswith("  Fix: Load a checkpoint saved from a layer")
    assert "identity=spec.fingerprint" in fix
    assert "edge_location()" in fix
    assert "'Changing the prior (reparse)'" in fix


@pytest.mark.parametrize("layer", sorted(_LAYERS))
def test_wiring_mismatch_names_the_failing_submodule(layer: str) -> None:
    build, sentence = _LAYERS[layer]
    saved = _model(
        build,
        _WIRING,
        None,
        None,
    )
    live = _model(
        build,
        _REWIRED,
        None,
        None,
    )
    before = _snapshot(live["hops"][1])

    message = _load_error(
        live,
        saved,
    )

    lines = message.splitlines()
    assert lines[0] == f"{sentence} in submodule 'hops.1'{_WIRING_REASON}"
    assert len(lines) == 2
    _assert_names_only_the_middle_hop(message)
    _assert_names_the_fix(message)
    _assert_unchanged(
        live["hops"][1],
        before,
    )


@pytest.mark.parametrize("layer", sorted(_LAYERS))
def test_identity_mismatch_shows_both_identities_shortened(
    layer: str,
) -> None:
    build, _ = _LAYERS[layer]
    saved = _model(
        build,
        _WIRING,
        _SAVED_IDENTITY,
        _SAVED_IDENTITY,
    )
    live = _model(
        build,
        _WIRING,
        _SAVED_IDENTITY,
        _LIVE_IDENTITY,
    )
    before = _snapshot(live["hops"][1])

    message = _load_error(
        live,
        saved,
    )

    assert message.splitlines()[:3] == [
        f"{_IDENTITY_SENTENCE} in submodule 'hops.1'.",
        f"  Got: identity {_shortened(_SAVED_IDENTITY)}",
        f"  Expected: identity {_shortened(_LIVE_IDENTITY)}",
    ]
    _assert_names_only_the_middle_hop(message)
    assert _SAVED_IDENTITY not in message
    assert _LIVE_IDENTITY not in message
    _assert_names_the_fix(message)
    _assert_unchanged(
        live["hops"][1],
        before,
    )


@pytest.mark.parametrize("layer", sorted(_LAYERS))
def test_identity_mismatch_says_when_the_layer_has_no_identity(
    layer: str,
) -> None:
    build, _ = _LAYERS[layer]
    saved = _model(
        build,
        _WIRING,
        _SAVED_IDENTITY,
        _SAVED_IDENTITY,
    )
    live = _model(
        build,
        _WIRING,
        _SAVED_IDENTITY,
        None,
    )

    message = _load_error(
        live,
        saved,
    )

    assert message.splitlines()[:3] == [
        f"{_IDENTITY_SENTENCE} in submodule 'hops.1'.",
        f"  Got: identity {_shortened(_SAVED_IDENTITY)}",
        "  Expected: no identity, as this layer has identity=None",
    ]
    _assert_names_only_the_middle_hop(message)
    fix = message.splitlines()[-1]
    assert fix.startswith(
        "  Fix: Build this layer with identity=spec.fingerprint, from the "
        "spec the checkpoint was saved with."
    )
    assert "edge_location()" in fix
    assert "'Changing the prior (reparse)'" in fix


@pytest.mark.parametrize("layer", sorted(_LAYERS))
def test_wiring_mismatch_after_a_matching_identity_says_how_to_rebuild(
    layer: str,
) -> None:
    build, sentence = _LAYERS[layer]
    saved = _model(
        build,
        _WIRING,
        _SAVED_IDENTITY,
        _SAVED_IDENTITY,
    )
    live = _model(
        build,
        _REWIRED,
        _SAVED_IDENTITY,
        _SAVED_IDENTITY,
    )

    message = _load_error(
        live,
        saved,
    )

    assert message.splitlines() == [
        f"{sentence} in submodule 'hops.1'{_WIRING_REASON}",
        "  Fix: The checkpoint has this layer's identity, so it was saved "
        "from the same spec: build this submodule as it was built then, "
        f"{_SAME_BUILD[layer]}.",
    ]


@pytest.mark.parametrize("layer", sorted(_LAYERS))
@pytest.mark.parametrize("check", ["wiring", "identity"])
def test_layer_loaded_directly_names_no_submodule(
    layer: str,
    check: str,
) -> None:
    build, sentence = _LAYERS[layer]
    headline = f"{sentence}{_WIRING_REASON}"
    torch.manual_seed(42)
    if check == "wiring":
        saved = build(
            _WIRING,
            None,
        )
        live = build(
            _REWIRED,
            None,
        )
    else:
        headline = f"{_IDENTITY_SENTENCE}."
        saved = build(
            _WIRING,
            _SAVED_IDENTITY,
        )
        live = build(
            _WIRING,
            _LIVE_IDENTITY,
        )

    message = _load_error(
        live,
        saved,
    )

    assert message.splitlines()[0] == headline
    assert "submodule" not in message
    _assert_names_the_fix(message)


@pytest.mark.parametrize("layer", sorted(_LAYERS))
def test_short_identities_are_shown_whole(layer: str) -> None:
    build, _ = _LAYERS[layer]
    torch.manual_seed(42)
    saved = build(
        _WIRING,
        "left",
    )
    live = build(
        _WIRING,
        "right",
    )

    message = _load_error(
        live,
        saved,
    )

    assert (
        "\n  Got: identity 'left'\n  Expected: identity 'right'\n"
    ) in message


@pytest.mark.parametrize(
    "stored",
    [
        torch.tensor(
            [0xFF, 0xFE],
            dtype=torch.uint8,
        ),
        torch.tensor(
            list(b"abc"),
            dtype=torch.int64,
        ),
        "abc",
    ],
    ids=["invalid-utf8", "not-uint8", "not-a-tensor"],
)
@pytest.mark.parametrize("layer", sorted(_LAYERS))
def test_unreadable_saved_identity_is_named_as_such(
    layer: str,
    stored: object,
) -> None:
    build, _ = _LAYERS[layer]
    torch.manual_seed(42)
    live = build(
        _WIRING,
        "abc",
    )
    state = live.state_dict()
    state["identity"] = stored

    with pytest.raises(Kpnn2Error) as info:
        live.load_state_dict(state)

    assert (
        "\n  Got: an unreadable identity\n  Expected: identity 'abc'\n"
    ) in str(info.value)


def test_message_points_to_an_existing_docs_heading() -> None:
    torch.manual_seed(42)
    message = _load_error(
        _packed_linear(
            _REWIRED,
            None,
        ),
        _packed_linear(
            _WIRING,
            None,
        ),
    )
    found = re.search(
        r"see '([^']+)' on the PackedLinear page",
        message,
    )
    assert found, message
    headings = _PACKED_LINEAR_PAGE.read_text().splitlines()
    assert f"## {found.group(1)}" in headings
