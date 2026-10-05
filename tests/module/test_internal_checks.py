from pathlib import Path

import pytest

import kpnn2
from kpnn2 import Kpnn2Error
from kpnn2._errors import _ISSUES_URL, internal_error

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_internal_error_is_assertion_error_not_kpnn2_error():
    error = internal_error("x")

    assert isinstance(error, AssertionError)
    assert not isinstance(error, Kpnn2Error)


def test_internal_error_message_names_bug_version_and_tracker():
    message = str(internal_error("hop edges are not conserved"))

    assert "kpnn2 internal check failed" in message
    assert "hop edges are not conserved" in message
    assert "bug in kpnn2" in message
    assert kpnn2.__version__ in message
    assert _ISSUES_URL in message


def test_issues_url_matches_pyproject():
    text = _PYPROJECT.read_text(encoding="utf-8")

    assert f'Issues = "{_ISSUES_URL}"' in text


def test_raised_internal_error_is_caught_as_assertion_error():
    with pytest.raises(
        AssertionError,
        match="internal check failed",
    ):
        raise internal_error("x")
