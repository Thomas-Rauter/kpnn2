"""Repository pytest hooks."""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--fast-only",
        action="store_true",
        default=False,
        help="run fast tests only (exclude slow)",
    )


def pytest_configure(config):
    if not config.getoption("fast_only"):
        return
    extra = "not slow"
    current = config.option.markexpr
    if current:
        config.option.markexpr = f"({current}) and ({extra})"
    else:
        config.option.markexpr = extra
