"""Generate the KPNNs in context Graphviz figure.

Writes ``docs/figures/knowledge_placement.svg`` and a matching PNG.
Four panels in a 2 x 2 grid show where four model families put
their prior knowledge. The top row puts it in a training signal (an
equation residual, concept labels), the bottom row in the
architecture (a symmetry, the wiring). The PNG is 4× 96 dpi so it stays
sharp on the docs site when the figure is stretched to the content
column.
"""

import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from graphviz import Digraph

_DOCS_DIR = Path(__file__).resolve().parents[1]
_OUT_PATH = _DOCS_DIR / "figures" / "knowledge_placement.svg"
_FONT = "Liberation Sans"
_FONTSIZE = "6"
_GAP_PT = 20.0
_PNG_DPI = 384
_SVG_NS = "{http://www.w3.org/2000/svg}"

# Line weights that mark the three shared weights of the
# convolution, so the panel does not rest on colour. Arrowheads
# grow with line weight, so each weight gets a smaller arrowsize.
_SHARED_STYLES = (
    ("0.5", "0.5"),
    ("1.3", "0.32"),
    ("2.4", "0.22"),
)

ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def _panel_graph(label: str) -> Digraph:
    graph = Digraph(format="svg")
    graph.attr(
        rankdir="LR",
        bgcolor="transparent",
        fontname=_FONT,
        fontsize=_FONTSIZE,
        pad="0.12",
        nodesep="0.16",
        ranksep="0.42",
        label=label,
        labelloc="t",
        fontcolor="black",
        splines="true",
    )
    graph.attr(
        "node",
        shape="ellipse",
        fontname=_FONT,
        fontsize=_FONTSIZE,
        color="black",
        fontcolor="black",
        fillcolor="none",
        margin="0.04,0.03",
        height="0.26",
        width="0.44",
    )
    graph.attr(
        "edge",
        color="black",
        arrowsize="0.5",
        penwidth="0.8",
    )
    return graph


def _anonymous_unit(
    graph: Digraph,
    name: str,
) -> None:
    graph.node(
        name,
        label="",
        width="0.18",
        height="0.18",
        shape="circle",
    )


def _loss_box(
    graph: Digraph,
    name: str,
    label: str,
) -> None:
    graph.node(
        name,
        label=label,
        shape="box",
        style="rounded",
        margin="0.06,0.04",
    )


def _training_signal(
    graph: Digraph,
    source: str,
    target: str,
) -> None:
    graph.edge(
        source,
        target,
        style="dashed",
        arrowhead="none",
    )


def _same_rank(
    graph: Digraph,
    names: list[str],
) -> None:
    with graph.subgraph() as subgraph:
        subgraph.attr(rank="same")
        for name in names:
            subgraph.node(name)
        for source, target in zip(
            names,
            names[1:],
        ):
            subgraph.edge(
                source,
                target,
                style="invis",
                weight="100",
            )


def _physics_panel() -> Digraph:
    graph = _panel_graph("(a) Physics-informed: knowledge in the loss")
    inputs = ["x", "t"]
    hidden = ["p1", "p2", "p3"]
    for name in hidden:
        _anonymous_unit(
            graph,
            name,
        )
    _same_rank(
        graph,
        inputs,
    )
    _same_rank(
        graph,
        hidden,
    )
    graph.node(
        "u",
        label="u(x, t)",
    )
    _loss_box(
        graph,
        "physics_loss",
        "equation residual",
    )
    for source in inputs:
        for target in hidden:
            graph.edge(
                source,
                target,
            )
    for source in hidden:
        graph.edge(
            source,
            "u",
        )
    _training_signal(
        graph,
        "u",
        "physics_loss",
    )
    return graph


def _symmetry_panel() -> Digraph:
    graph = _panel_graph("(c) Convolutional: knowledge as a symmetry")
    inputs = [f"pixel {index}" for index in range(1, 5)]
    hidden = ["c1", "c2"]
    for name in hidden:
        _anonymous_unit(
            graph,
            name,
        )
    _same_rank(
        graph,
        inputs,
    )
    _same_rank(
        graph,
        hidden,
    )
    for position, target in enumerate(hidden):
        for offset, (penwidth, arrowsize) in enumerate(_SHARED_STYLES):
            graph.edge(
                inputs[position + offset],
                target,
                penwidth=penwidth,
                arrowsize=arrowsize,
            )
    return graph


def _concept_panel() -> Digraph:
    graph = _panel_graph("(b) Concept bottleneck: knowledge as labels")
    inputs = ["x1", "x2", "x3"]
    concepts = ["concept 1", "concept 2"]
    _same_rank(
        graph,
        inputs,
    )
    _same_rank(
        graph,
        concepts,
    )
    graph.node("y")
    _loss_box(
        graph,
        "concept_labels",
        "concept labels",
    )
    for source in inputs:
        for target in concepts:
            graph.edge(
                source,
                target,
            )
    for source in concepts:
        graph.edge(
            source,
            "y",
        )
    for concept in concepts:
        _training_signal(
            graph,
            concept,
            "concept_labels",
        )
    return graph


def _wiring_panel() -> Digraph:
    graph = _panel_graph("(d) KPNN: knowledge as wiring")
    genes = [f"gene {index}" for index in range(1, 5)]
    factors = ["TF A", "TF B"]
    _same_rank(
        graph,
        genes,
    )
    _same_rank(
        graph,
        factors,
    )
    graph.node("kinase")
    graph.node("receptor")
    for source, target in (
        ("gene 1", "TF A"),
        ("gene 2", "TF A"),
        ("gene 2", "TF B"),
        ("gene 3", "TF B"),
        ("gene 4", "TF B"),
        ("TF A", "kinase"),
        ("TF B", "kinase"),
        ("kinase", "receptor"),
    ):
        graph.edge(
            source,
            target,
        )
    return graph


def _parse_svg(
    svg: str,
    prefix: str,
) -> tuple[ET.Element, float, float]:
    root = ET.fromstring(svg)
    viewbox = root.attrib["viewBox"].split()
    width = float(viewbox[2])
    height = float(viewbox[3])
    for element in root.iter():
        if "id" in element.attrib:
            element.attrib["id"] = prefix + element.attrib["id"]
    return root, width, height


def _nest_panel(
    parent: ET.Element,
    root: ET.Element,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    inner = ET.SubElement(
        parent,
        f"{_SVG_NS}svg",
        {
            "x": f"{x:.2f}",
            "y": f"{y:.2f}",
            "width": f"{width:.2f}",
            "height": f"{height:.2f}",
            "viewBox": (f"0.00 0.00 {width:.2f} {height:.2f}"),
        },
    )
    for child in list(root):
        inner.append(child)


def _combine_grid(svgs: list[str]) -> str:
    panels = [
        _parse_svg(
            svg,
            f"{prefix}_",
        )
        for prefix, svg in zip(
            "abcd",
            svgs,
        )
    ]
    column_width = max(width for _, width, _ in panels)
    top_height = max(panels[0][2], panels[1][2])
    bottom_height = max(panels[2][2], panels[3][2])
    total_width = 2 * column_width + _GAP_PT
    total_height = top_height + _GAP_PT + bottom_height
    combined = ET.Element(
        f"{_SVG_NS}svg",
        {
            "width": f"{total_width:.2f}pt",
            "height": f"{total_height:.2f}pt",
            "viewBox": (f"0.00 0.00 {total_width:.2f} {total_height:.2f}"),
        },
    )
    combined.set(
        "xmlns:xlink",
        "http://www.w3.org/1999/xlink",
    )
    origins = (
        (0.0, 0.0),
        (column_width + _GAP_PT, 0.0),
        (0.0, top_height + _GAP_PT),
        (column_width + _GAP_PT, top_height + _GAP_PT),
    )
    for (root, width, height), (x, y) in zip(
        panels,
        origins,
    ):
        _nest_panel(
            combined,
            root,
            x + (column_width - width) / 2,
            y,
            width,
            height,
        )
    comment = (
        " Generated by graphviz version 2.43.0 (0); "
        "panels combined in a 2 x 2 grid "
    )
    body = ET.tostring(combined, encoding="unicode")
    return (
        '<?xml version="1.0" encoding="UTF-8" '
        'standalone="no"?>\n'
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
        ' "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
        f"<!--{comment}-->\n"
        f"{body}\n"
    )


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


def write_figure(out_path: Path | None = None) -> Path:
    """Render the four panels and write the combined SVG and PNG."""
    path = _OUT_PATH if out_path is None else out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = _combine_grid(
        [
            panel().pipe().decode("utf-8")
            for panel in (
                _physics_panel,
                _concept_panel,
                _symmetry_panel,
                _wiring_panel,
            )
        ]
    )
    path.write_text(svg)
    _rasterize_png(
        path,
        path.with_suffix(".png"),
    )
    return path


def main() -> None:
    path = write_figure()
    png_path = path.with_suffix(".png")
    print(f"Wrote {path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
