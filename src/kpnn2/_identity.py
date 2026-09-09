"""Opaque checkpoint identity for connectivity modules."""

from typing import Any

import torch

from ._errors import Kpnn2Error

IDENTITY_KEY = "identity"


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
        If a present identity does not match ``identity``.
    """
    key = prefix + IDENTITY_KEY
    saved = state_dict.pop(key, None)
    if saved is None:
        return
    if not identity_matches(
        saved,
        identity,
    ):
        raise Kpnn2Error("The checkpoint identity does not match this layer.")
