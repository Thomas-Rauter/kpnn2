"""Home URLs: absolute for GitHub and PyPI, local for the docs build.

``README.md`` links docs pages by URL on the website's ``latest``
version and raster figures by raw GitHub URL on ``main``, because
GitHub and PyPI cannot resolve relative paths to either. A figure URL
only resolves once the file is committed at that path.
``dev/docs_readme_urls.py`` maps both kinds back to local paths when
MkDocs builds Home, where ``mkdocs build --strict`` checks anchors.
"""

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_README = _REPO / "README.md"
_SCRIPT = _REPO / "dev" / "docs_readme_urls.py"

_spec = importlib.util.spec_from_file_location(
    "docs_readme_urls_script",
    _SCRIPT,
)
assert _spec is not None and _spec.loader is not None
_hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hook)


def _page(src_uri: str) -> SimpleNamespace:
    return SimpleNamespace(file=SimpleNamespace(src_uri=src_uri))


def _files(*pages: tuple[str, str]) -> SimpleNamespace:
    docs = [
        SimpleNamespace(
            src_uri=src_uri,
            url=url,
        )
        for src_uri, url in pages
    ]
    return SimpleNamespace(documentation_pages=lambda: docs)


def _readme_targets() -> list[str]:
    text = re.sub(
        r"^```.*?^```",
        "",
        _README.read_text(),
        flags=re.M | re.S,
    )
    inline = re.findall(
        r"\]\(([^)\s]+)",
        text,
    )
    reference = re.findall(
        r"^\s*\[[^\]]+\]:\s*(\S+)",
        text,
        flags=re.M,
    )
    html = re.findall(
        r"\b(?:src|href)=\"([^\"]+)\"",
        text,
    )
    return inline + reference + html


def _raw_figures() -> list[str]:
    pattern = re.escape(_hook.RAW_FIGURES) + r"([^)\s\"]+)"
    return re.findall(pattern, _README.read_text())


def _site_paths() -> list[str]:
    pattern = re.escape(_hook.SITE_LATEST) + r"([^)\s\"#]*)"
    return re.findall(pattern, _README.read_text())


def _served_from_docs(path: str) -> bool:
    if path and not path.endswith("/"):
        return False
    stem = _REPO / "docs" / (path.rstrip("/") or "index")
    return any(
        Path(f"{stem}{suffix}").is_file() for suffix in (".md", ".ipynb")
    )


def test_readme_has_no_relative_links() -> None:
    targets = _readme_targets()
    assert targets, "README.md has no links"
    relative = [t for t in targets if not t.startswith("https://")]
    assert not relative, (
        "GitHub and PyPI cannot resolve relative README links. Link "
        f"docs pages under {_hook.SITE_LATEST} and figures under "
        f"{_hook.RAW_FIGURES}: {relative}"
    )


def test_readme_site_links_name_docs_pages() -> None:
    paths = _site_paths()
    assert paths, "README.md links no docs page on the website"
    missing = [path for path in paths if not _served_from_docs(path)]
    assert not missing, f"No page under docs/ is served at: {missing}"


def test_readme_raw_figures_exist_locally() -> None:
    names = _raw_figures()
    assert names, "README.md links no figure by raw GitHub URL"
    missing = [
        name
        for name in names
        if not (_REPO / "docs" / "figures" / name).is_file()
    ]
    assert not missing, f"Not under docs/figures/: {missing}"


def test_hook_points_home_figures_at_local_files() -> None:
    url = _hook.RAW_FIGURES + "KPNNs_explained.png"
    markdown = f"![Figure]({url})"

    home = _hook.on_page_markdown(
        markdown,
        page=_page("index.md"),
        config=None,
        files=_files(),
    )
    other = _hook.on_page_markdown(
        markdown,
        page=_page("concepts.md"),
        config=None,
        files=_files(),
    )

    assert home == "![Figure](figures/KPNNs_explained.png)"
    assert other == markdown


def test_hook_points_home_links_at_source_pages() -> None:
    files = _files(
        ("index.md", "./"),
        ("concepts.md", "concepts/"),
        ("skip-edges.ipynb", "skip-edges/"),
        ("reference/api.md", "reference/api/"),
    )
    site = _hook.SITE_LATEST
    markdown = (
        f"[hop]({site}concepts/#hop), "
        f"[Skip edges]({site}skip-edges/), "
        f"[API]({site}reference/api/), "
        f"[Home]({site})"
    )

    home = _hook.on_page_markdown(
        markdown,
        page=_page("index.md"),
        config=None,
        files=files,
    )
    other = _hook.on_page_markdown(
        markdown,
        page=_page("concepts.md"),
        config=None,
        files=files,
    )

    assert home == (
        "[hop](concepts.md#hop), "
        "[Skip edges](skip-edges.ipynb), "
        "[API](reference/api.md), "
        "[Home](index.md)"
    )
    assert other == markdown


def test_hook_rejects_site_link_to_unknown_page() -> None:
    markdown = f"[Gone]({_hook.SITE_LATEST}removed-page/)"

    with pytest.raises(
        ValueError,
        match="removed-page/",
    ):
        _hook.on_page_markdown(
            markdown,
            page=_page("index.md"),
            config=None,
            files=_files(("concepts.md", "concepts/")),
        )
