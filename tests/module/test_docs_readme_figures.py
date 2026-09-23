"""Home figures: absolute URLs for PyPI, local files for the docs build.

``README.md`` links raster figures by raw GitHub URL on ``main`` so
PyPI can show them. Those URLs only resolve once the file is
committed at that path, and ``dev/docs_readme_figures.py`` maps them
to the local copy when MkDocs builds Home.
"""

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[2]
_README = _REPO / "README.md"
_SCRIPT = _REPO / "dev" / "docs_readme_figures.py"

_spec = importlib.util.spec_from_file_location(
    "docs_readme_figures_script",
    _SCRIPT,
)
assert _spec is not None and _spec.loader is not None
_hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hook)


def _page(src_uri: str) -> SimpleNamespace:
    return SimpleNamespace(file=SimpleNamespace(src_uri=src_uri))


def _raw_figures() -> list[str]:
    pattern = re.escape(_hook.RAW_FIGURES) + r"([^)\s\"]+)"
    return re.findall(pattern, _README.read_text())


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
        files=None,
    )
    other = _hook.on_page_markdown(
        markdown,
        page=_page("concepts.md"),
        config=None,
        files=None,
    )

    assert home == "![Figure](figures/KPNNs_explained.png)"
    assert other == markdown
