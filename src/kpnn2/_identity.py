"""
Opaque checkpoint identity for connectivity modules.

Also builds the ``Kpnn2Error`` that every connectivity module raises
when a checkpoint's identity or wiring digest does not match it.
"""

from typing import Any

import torch

from ._errors import Kpnn2Error

IDENTITY_KEY = "identity"

# Identities in a mismatch message are cut to this many characters;
# a ``spec.fingerprint`` is 64 hex characters.
_SHOWN_IDENTITY_LENGTH = 12

_MISMATCH_FIX = (
    "Load a checkpoint saved from a layer built from the same spec "
    "(with identity=spec.fingerprint). To move weights onto a "
    "different prior, copy them by name with edge_location() on the "
    "old and the new spec; see 'Changing the prior (reparse)' on the "
    "PackedLinear page of the kpnn2 docs."
)


def as_identity(value: object) -> str | None:
    """
    Return ``None`` or a ``str``; reject every other type.

    Parameters
    ----------
    value : str or None
        Constructor identity, typically ``spec.fingerprint``.

    Returns
    -------
    str or None
        ``value`` unchanged when it is a ``str`` or ``None``.

    Raises
    ------
    Kpnn2Error
        If ``value`` is neither a ``str`` nor ``None``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise Kpnn2Error("'identity' must be a str or None.")
    return value


def identity_tensor(identity: str) -> torch.Tensor:
    """
    Encode ``identity`` as a 1-D CPU ``uint8`` tensor.

    Parameters
    ----------
    identity : str
        Opaque identity string.

    Returns
    -------
    torch.Tensor
        UTF-8 bytes of ``identity``, dtype ``uint8``.
    """
    return torch.tensor(
        tuple(identity.encode("utf-8")),
        dtype=torch.uint8,
    )


def identity_matches(
    saved: object,
    current: str | None,
) -> bool:
    """
    Return whether ``saved`` encodes ``current``.

    Parameters
    ----------
    saved : object
        Value popped from ``state_dict``.
    current : str or None
        Live layer identity.

    Returns
    -------
    bool
        ``True`` only when ``current`` is a ``str`` and
        ``saved`` is a ``uint8`` tensor of its UTF-8 bytes.
    """
    if current is None:
        return False
    if not isinstance(saved, torch.Tensor):
        return False
    expected = identity_tensor(current)
    saved_flat = saved.detach().cpu().contiguous().reshape(-1)
    if saved_flat.shape != expected.shape or saved_flat.dtype != expected.dtype:
        return False
    return bool(
        torch.equal(
            saved_flat,
            expected,
        )
    )


def save_identity(
    destination: dict[str, Any],
    prefix: str,
    identity: str | None,
) -> None:
    """
    Write ``identity`` into ``destination`` when it is set.

    Parameters
    ----------
    destination : dict
        ``state_dict`` being filled.
    prefix : str
        Module prefix, as in ``nn.Module._save_to_state_dict``.
    identity : str or None
        Live layer identity. ``None`` writes nothing.
    """
    if identity is None:
        return
    destination[prefix + IDENTITY_KEY] = identity_tensor(identity)


def check_identity(
    state_dict: dict[str, Any],
    prefix: str,
    identity: str | None,
) -> None:
    """
    Pop and check a saved identity; missing is not an error.

    Parameters
    ----------
    state_dict : dict
        Incoming ``state_dict``. The identity key is popped
        when present.
    prefix : str
        Module prefix, as in
        ``nn.Module._load_from_state_dict``.
    identity : str or None
        Live layer identity.

    Raises
    ------
    Kpnn2Error
        If a present identity does not match ``identity``. The
        message names the submodule and shows both identities,
        shortened.
    """
    key = prefix + IDENTITY_KEY
    saved = state_dict.pop(key, None)
    if saved is None:
        return
    if not identity_matches(
        saved,
        identity,
    ):
        raise _checkpoint_mismatch(
            "The checkpoint identity does not match this layer.",
            prefix,
            f"The checkpoint was saved with {_saved_identity(saved)}; "
            f"{_live_identity(identity)}.",
        )


def wiring_mismatch(
    problem: str,
    prefix: str,
) -> Kpnn2Error:
    """
    Build the error for a saved wiring digest that does not match.

    Parameters
    ----------
    problem : str
        First sentence of the message, naming the digest that
        differs.
    prefix : str
        Module prefix, as in
        ``nn.Module._load_from_state_dict``.

    Returns
    -------
    Kpnn2Error
        Error whose message starts with ``problem``, then names
        the submodule and says how to fix the load.
    """
    return _checkpoint_mismatch(
        problem,
        prefix,
        "The saved wiring (edges, sizes, or edge order) differs from "
        "this layer's.",
    )


def _checkpoint_mismatch(
    problem: str,
    prefix: str,
    difference: str,
) -> Kpnn2Error:
    if prefix:
        where = f"The mismatch is in submodule '{prefix.removesuffix('.')}'."
    else:
        where = "The mismatch is in the top-level module, loaded directly."
    return Kpnn2Error(f"{problem} {where} {difference} {_MISMATCH_FIX}")


def _saved_identity(saved: object) -> str:
    if not isinstance(saved, torch.Tensor) or saved.dtype != torch.uint8:
        return "an unreadable identity"
    raw = saved.detach().cpu().contiguous().reshape(-1).numpy().tobytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "an unreadable identity"
    return f"identity {_shown_identity(text)}"


def _live_identity(identity: str | None) -> str:
    if identity is None:
        return "this layer has no identity (identity=None)"
    return f"this layer has identity {_shown_identity(identity)}"


def _shown_identity(identity: str) -> str:
    if len(identity) > _SHOWN_IDENTITY_LENGTH:
        identity = f"{identity[:_SHOWN_IDENTITY_LENGTH]}..."
    return repr(identity)
