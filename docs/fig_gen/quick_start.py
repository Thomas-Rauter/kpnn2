"""Generate the two README quick-start figures from a real run.

Runs every ``python`` block under "## Quick start" in ``README.md``
in page order and in one namespace, as
``tests/module/test_docs_readme_quick_start.py`` does, so the
figures show the run the page prints. Writes, each with a matching
PNG:

- ``docs/figures/quick_start_graph.svg``: the parsed graph, drawn
  from ``spec``, with the signal and the noise branch boxed and
  labeled.
- ``docs/figures/quick_start_attributions.svg``: the mean absolute
  attribution per hidden node, the numbers the last block prints.

A node belongs to the branch its name says (``signal`` or
``noise``). The PNGs are 4× 96 dpi, so they stay sharp on GitHub
and PyPI, which cannot show the SVGs from a relative path. The last
block imports Captum, which is in the ``dev`` extra.
"""

import contextlib
import io
import random
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import torch
from graphviz import Digraph

_REPO = Path(__file__).resolve().parents[2]
_README = _REPO / "README.md"
_FIGURES = _REPO / "docs" / "figures"
_GRAPH_PATH = _FIGURES / "quick_start_graph.svg"
_ATTRIBUTIONS_PATH = _FIGURES / "quick_start_attributions.svg"
_PNG_DPI = 384

_HEADING = "## Quick start"
_BLOCK = re.compile(
    r"```python\n(.*?)\n```",
    re.S,
)

# Colours and type follow docs/fig_gen/kpnn2_overview.py.
_FONT = "Liberation Sans"
_FONT_SIZE = 12.5
_LABEL_SIZE = 11.5
_ACCENT = "#ee4c2c"
_ACCENT_TINT = "#fdeae5"
_GREY = "#6b6b6b"
_GREY_TINT = "#f2f2f2"
_EDGE = "#4a4a4a"
_BRANCHES = {
    "signal": ("signal branch", _ACCENT, _ACCENT_TINT),
    "noise": ("noise branch", _GREY, _GREY_TINT),
}

# Both figures are about 600 pt wide, so their text shows at one
# size when Home stretches them to the content column. Chart sizes
# are in inches, 72 pt each.
_RANKSEP = "2.12"
_CHART_WIDTH = 8.3
_CHART_HEIGHT = 1.75
_BAR_HEIGHT = 0.42


def _run_quick_start() -> dict[str, object]:
    text = _README.read_text()
    start = text.find(_HEADING)
    end = text.find(
        "\n## ",
        start + len(_HEADING),
    )
    section = text[start:] if end == -1 else text[start:end]
    random.seed(42)
    numpy.random.seed(42)
    torch.manual_seed(42)
    namespace: dict[str, object] = {"__name__": "readme_quick_start"}
    with contextlib.redirect_stdout(io.StringIO()):
        for code in _BLOCK.findall(section):
            exec(compile(code, "README.md", "exec"), namespace)  # noqa: S102
    return namespace


def _branch(name: str) -> str | None:
    for branch in _BRANCHES:
        if branch in name:
            return branch
    return None


def _graph(spec) -> Digraph:
    drawing = Digraph(format="svg")
    drawing.attr(
        rankdir="LR",
        bgcolor="white",
        pad="0.2",
        ranksep=_RANKSEP,
        nodesep="0.14",
        fontname=_FONT,
        fontsize=str(_LABEL_SIZE),
    )
    drawing.attr(
        "node",
        fontname=_FONT,
        fontsize=str(_FONT_SIZE),
        shape="box",
        style="rounded,filled",
        fillcolor="white",
        color="black",
        penwidth="1.1",
        height="0.3",
        margin="0.12,0.04",
    )
    drawing.attr(
        "edge",
        color=_EDGE,
        arrowsize="0.6",
        penwidth="1.0",
    )
    for branch, (label, color, tint) in _BRANCHES.items():
        with drawing.subgraph(name=f"cluster_{branch}") as cluster:
            cluster.attr(
                label=label,
                labeljust="l",
                fontcolor=color,
                color=color,
                style="rounded,filled",
                fillcolor=tint,
                penwidth="0.8",
                margin="10",
            )
            for name in spec.input_nodes + spec.hidden_nodes:
                if _branch(name) == branch:
                    cluster.node(
                        name,
                        color=color,
                    )
    for name in spec.output_nodes:
        drawing.node(name)
    for source, target in spec.to_edgelist().itertuples(index=False):
        drawing.edge(
            source,
            target,
        )
    return drawing


def _attribution_chart(
    scores,
    path: Path,
) -> None:
    names = list(scores.index)
    values = [float(value) for value in scores.to_numpy()]
    colors = [_BRANCHES[_branch(name)][1] for name in names]
    plt.rcParams["svg.hashsalt"] = "quick_start_attributions"
    figure, axes = plt.subplots(
        figsize=(
            _CHART_WIDTH,
            _CHART_HEIGHT,
        ),
    )
    axes.barh(
        names,
        values,
        height=_BAR_HEIGHT,
        color=colors,
    )
    for row, value in enumerate(values):
        axes.text(
            value + 0.06,
            row,
            f"{value:.2f}",
            va="center",
            fontsize=_FONT_SIZE,
            fontfamily=_FONT,
        )
    axes.set_xlim(
        0,
        max(values) * 1.12,
    )
    axes.set_xlabel(
        "Mean absolute attribution (units of the output logit)",
        fontsize=_LABEL_SIZE,
        fontfamily=_FONT,
        color=_GREY,
    )
    axes.tick_params(
        axis="y",
        length=0,
        labelsize=_FONT_SIZE,
    )
    axes.tick_params(
        axis="x",
        colors=_GREY,
        labelsize=_LABEL_SIZE,
    )
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontfamily(_FONT)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(_GREY)
    figure.patch.set_facecolor("white")
    figure.tight_layout(pad=0.6)
    figure.savefig(
        path,
        format="svg",
        metadata={"Date": None},
    )
    plt.close(figure)


def _rasterize_png(
    svg_path: Path,
    png_path: Path,
) -> None:
    subprocess.run(
        [
            "rsvg-convert",
            "--dpi-x",
            str(_PNG_DPI),
            "--dpi-y",
            str(_PNG_DPI),
            "--format",
            "png",
            "--output",
            str(png_path),
            str(svg_path),
        ],
        check=True,
    )


def write_figures() -> list[Path]:
    """Run the quick start and write both figures with their PNGs."""
    namespace = _run_quick_start()
    _FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )
    _GRAPH_PATH.write_bytes(_graph(namespace["spec"]).pipe())
    scores = abs(namespace["attributions"]).mean("observation").to_pandas()
    _attribution_chart(
        scores,
        _ATTRIBUTIONS_PATH,
    )
    paths = []
    for path in (_GRAPH_PATH, _ATTRIBUTIONS_PATH):
        _rasterize_png(
            path,
            path.with_suffix(".png"),
        )
        paths += [path, path.with_suffix(".png")]
    return paths


def main() -> None:
    for path in write_figures():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
