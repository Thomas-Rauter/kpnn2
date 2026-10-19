"""Generate the landing-page overview schematic.

Writes ``docs/figures/kpnn2_overview.svg`` and a matching PNG.
The PNG is 4× 96 dpi, so it stays sharp on PyPI, which cannot
show the SVG from a relative path.

Inputs on the left (a named edgelist and a data table), the
network in the middle, attributions per named hidden node on the
right. The two outer panels mirror each other: names go in, and
names come out. The proteins are generic, and the graph is not the
README quick start's: the figure shows the workflow, and the quick
start runs it on its own simulated graph. One edge skips a layer
and is drawn dashed. The cells are schematic, not values from a
run: the data cells are texture, and the attribution rows show a
model that relies on one hidden protein in every sample and hardly
on the other.
"""

import random
import subprocess
from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parents[1]
_OUT_PATH = _DOCS_DIR / "figures" / "kpnn2_overview.svg"
_FONT = "Liberation Sans, sans-serif"
_PNG_DPI = 384

# User units are PostScript points, so font-size 9 is 9 pt.
_WIDTH = 656.0
_HEIGHT = 226.0
_TITLE_SIZE = 10.5
_TEXT_SIZE = 8.5
_NOTE_SIZE = 7.5
_STROKE = 0.9
_ARROW = 4.5
_FLOW_ARROW = 7.0
_SKIP_DASH = "3 2"
_ACCENT = "#ee4c2c"
_GREY = "#6b6b6b"
_RULE = "#9a9a9a"

EDGES = [
    ("protein_a", "protein_e"),
    ("protein_b", "protein_e"),
    ("protein_c", "protein_f"),
    ("protein_d", "protein_f"),
    ("protein_e", "outcome"),
    ("protein_f", "outcome"),
    ("protein_d", "outcome"),
]
# protein_d is an input and outcome sits two layers above it.
SKIP_EDGES = {("protein_d", "outcome")}

# Left column: the edgelist table and the data table.
_LEFT_X = 16.0
_TARGET_X = 80.0
_TABLE_TITLE_Y = 24.0
_TABLE_HEAD_Y = 42.0
_ROW_STEP = 12.0
_DATA_TITLE_Y = 146.0
_DATA_TOP = 156.0
_CELL = 9.0
_CELL_STEP = 11.0
_CELLS_X = 64.0
_N_SAMPLES = 6
_DATA_ROWS = [
    "protein_a",
    "protein_b",
    "protein_c",
    "protein_d",
]
_OUTPUT = "outcome"

# Middle column: the network, one pill per named node.
_MODEL_CX = 338.0
_PILL_H = 15.0
_PILL_W = 52.0
_INPUT_CX = 243.0
_HIDDEN_CX = 349.0
_OUT_CX = 433.0
_NODE_Y = {
    "protein_a": 70.0,
    "protein_b": 94.0,
    "protein_c": 140.0,
    "protein_d": 164.0,
    "protein_e": 82.0,
    "protein_f": 152.0,
    "outcome": 117.0,
}
_NODE_CX = {
    "protein_a": _INPUT_CX,
    "protein_b": _INPUT_CX,
    "protein_c": _INPUT_CX,
    "protein_d": _INPUT_CX,
    "protein_e": _HIDDEN_CX,
    "protein_f": _HIDDEN_CX,
    "outcome": _OUT_CX,
}
# The skip edge curves under the hidden layer to this depth.
_SKIP_LOW = 200.0

# Right column: one row of cells per named hidden node. Each row
# keeps one level across samples, so one protein reads as relied
# on and the other as hardly used.
_ATTR_X = 508.0
_ATTR_CELLS_X = 548.0
_ATTR_TOP = 106.0
_ATTR_LEVELS = {
    "protein_e": 1.45,
    "protein_f": 0.15,
}
_ATTR_JITTER = 0.1

# Flow arrows between the columns.
_FLOW_Y = 117.0
_FLOW_1 = (160.0, 198.0)
_FLOW_2 = (468.0, 498.0)


def _text(
    x: float,
    y: float,
    text: str,
    *,
    size: float = _TEXT_SIZE,
    anchor: str = "start",
    weight: str = "400",
    fill: str = "#000",
) -> str:
    return (
        f'  <text x="{x:g}" y="{y:g}" font-size="{size:g}" '
        f'text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="{fill}">{text}</text>'
    )


def _title(
    x: float,
    y: float,
    text: str,
    *,
    anchor: str = "start",
) -> str:
    return _text(
        x,
        y,
        text,
        size=_TITLE_SIZE,
        anchor=anchor,
        weight="700",
    )


def _hline(
    x1: float,
    x2: float,
    y: float,
) -> str:
    return (
        f'  <line x1="{x1:g}" y1="{y:g}" x2="{x2:g}" y2="{y:g}" '
        f'stroke="{_RULE}" stroke-width="0.6"/>'
    )


def _cell(
    x: float,
    y: float,
    fill: str,
) -> str:
    return (
        f'  <rect x="{x:g}" y="{y:g}" width="{_CELL:g}" '
        f'height="{_CELL:g}" fill="{fill}" stroke="#000" '
        f'stroke-width="0.4"/>'
    )


def _marker(
    marker_id: str,
    size: float,
    fill: str,
) -> str:
    # Base of the triangle sits on the line end (refX=0). The
    # shaft is shortened by the arrow length, so it does not
    # run through the tip.
    return "\n".join(
        [
            f'    <marker id="{marker_id}" viewBox="0 0 10 10" '
            'refX="0" refY="5"',
            '            markerUnits="userSpaceOnUse"',
            f'            markerWidth="{size:g}" '
            f'markerHeight="{size:g}" orient="auto">',
            f'      <path d="M 0 0 L 10 5 L 0 10 z" fill="{fill}"/>',
            "    </marker>",
        ]
    )


def _edgelist_table() -> list[str]:
    parts = [
        _title(
            _LEFT_X,
            _TABLE_TITLE_Y,
            "Your graph, as an edgelist",
        ),
        _text(
            _LEFT_X,
            _TABLE_HEAD_Y,
            "source",
            fill=_GREY,
        ),
        _text(
            _TARGET_X,
            _TABLE_HEAD_Y,
            "target",
            fill=_GREY,
        ),
        _hline(
            _LEFT_X,
            _TARGET_X + 48.0,
            _TABLE_HEAD_Y + 3.5,
        ),
    ]
    for row, (source, target) in enumerate(EDGES):
        y = _TABLE_HEAD_Y + _ROW_STEP * (row + 1)
        parts.append(
            _text(
                _LEFT_X,
                y,
                source,
            )
        )
        parts.append(
            _text(
                _TARGET_X,
                y,
                target,
            )
        )
    return parts


def _grey(value: float) -> str:
    # Map roughly -2..2 onto light..dark grey.
    level = max(0.0, min(1.0, (value + 2.0) / 4.0))
    shade = round(235 - 165 * level)
    return f"#{shade:02x}{shade:02x}{shade:02x}"


def _tint(value: float) -> str:
    # Map roughly 0..1.6 onto a pale tint..the full accent colour.
    level = 0.12 + 0.88 * max(0.0, min(1.0, value / 1.6))
    accent = [int(_ACCENT[i : i + 2], 16) for i in (1, 3, 5)]
    red, green, blue = (round(255 - (255 - c) * level) for c in accent)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _data_table() -> list[str]:
    rng = random.Random(42)
    parts = [
        _title(
            _LEFT_X,
            _DATA_TITLE_Y,
            "Your data, by name",
        ),
    ]
    rows = [*_DATA_ROWS, _OUTPUT]
    for row, name in enumerate(rows):
        top = _DATA_TOP + _CELL_STEP * row
        if name == _OUTPUT:
            top += 3.0
        parts.append(
            _text(
                _LEFT_X,
                top + _CELL - 1.5,
                name,
                size=_NOTE_SIZE,
            )
        )
        for sample in range(_N_SAMPLES):
            if name == _OUTPUT:
                fill = "#000" if rng.random() < 0.5 else "#fff"
            else:
                fill = _grey(rng.gauss(0.0, 1.0))
            parts.append(
                _cell(
                    _CELLS_X + _CELL_STEP * sample,
                    top,
                    fill,
                )
            )
    return parts


def _pill(name: str) -> str:
    cx = _NODE_CX[name]
    cy = _NODE_Y[name]
    return (
        f'  <rect x="{cx - _PILL_W / 2:g}" y="{cy - _PILL_H / 2:g}" '
        f'width="{_PILL_W:g}" height="{_PILL_H:g}" '
        f'rx="{_PILL_H / 2:g}" fill="#fff" stroke="#000" '
        f'stroke-width="{_STROKE:g}"/>\n'
        f'  <text x="{cx:g}" y="{cy + 3.0:g}" '
        f'font-size="{_NOTE_SIZE:g}" text-anchor="middle" '
        f'fill="#000">{name}</text>'
    )


def _edge(
    source: str,
    target: str,
) -> str:
    x1 = _NODE_CX[source] + _PILL_W / 2
    y1 = _NODE_Y[source]
    x2 = _NODE_CX[target] - _PILL_W / 2
    y2 = _NODE_Y[target]
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    end_x = x2 - (x2 - x1) / length * _ARROW
    end_y = y2 - (y2 - y1) / length * _ARROW
    return (
        f'  <line x1="{x1:g}" y1="{y1:g}" '
        f'x2="{end_x:.2f}" y2="{end_y:.2f}" '
        f'stroke="#000" stroke-width="{_STROKE:g}" '
        f'marker-end="url(#edge-arrow)"/>'
    )


def _skip_edge(
    source: str,
    target: str,
) -> str:
    # Curve under the hidden layer into the bottom of the target, so
    # the edge reads as passing the layer it skips. The last control
    # point sits straight below the end, so the arrow points up.
    x1 = _NODE_CX[source] + _PILL_W / 2
    y1 = _NODE_Y[source]
    x2 = _NODE_CX[target]
    y2 = _NODE_Y[target] + _PILL_H / 2 + _ARROW
    return (
        f'  <path d="M {x1:g} {y1:g} C {x1 + 60.0:g} {_SKIP_LOW:g} '
        f'{x2:g} {_SKIP_LOW:g} {x2:g} {y2:g}" fill="none" '
        f'stroke="#000" stroke-width="{_STROKE:g}" '
        f'stroke-dasharray="{_SKIP_DASH}" '
        f'marker-end="url(#edge-arrow)"/>'
    )


def _model() -> list[str]:
    parts = [
        _title(
            _MODEL_CX,
            _TABLE_TITLE_Y,
            "Your network, in PyTorch",
            anchor="middle",
        ),
        _text(
            _MODEL_CX,
            _TABLE_TITLE_Y + 13.0,
            "one neuron per node, connected only along the edges",
            size=_NOTE_SIZE,
            anchor="middle",
            fill=_GREY,
        ),
    ]
    for source, target in EDGES:
        draw = _skip_edge if (source, target) in SKIP_EDGES else _edge
        parts.append(
            draw(
                source,
                target,
            )
        )
    parts += [_pill(name) for name in _NODE_Y]
    return parts


def _flow(
    start: float,
    end: float,
    label: str,
) -> list[str]:
    shaft_end = end - _FLOW_ARROW
    return [
        f'  <line x1="{start:g}" y1="{_FLOW_Y:g}" '
        f'x2="{shaft_end:g}" y2="{_FLOW_Y:g}" '
        f'stroke="{_ACCENT}" stroke-width="2.2" '
        f'marker-end="url(#flow-arrow)"/>',
        _text(
            (start + end) / 2,
            _FLOW_Y - 7.0,
            label,
            size=_NOTE_SIZE,
            anchor="middle",
            fill=_GREY,
        ),
    ]


def _attributions() -> list[str]:
    rng = random.Random(7)
    parts = [
        _title(
            _ATTR_X,
            _TABLE_TITLE_Y,
            "Your attributions, by name",
        ),
        _text(
            _ATTR_X,
            _TABLE_TITLE_Y + 13.0,
            "one value per hidden node and sample",
            size=_NOTE_SIZE,
            fill=_GREY,
        ),
    ]
    for row, (name, level) in enumerate(_ATTR_LEVELS.items()):
        top = _ATTR_TOP + _CELL_STEP * row
        parts.append(
            _text(
                _ATTR_X,
                top + _CELL - 1.5,
                name,
                size=_NOTE_SIZE,
            )
        )
        for sample in range(_N_SAMPLES):
            parts.append(
                _cell(
                    _ATTR_CELLS_X + _CELL_STEP * sample,
                    top,
                    _tint(abs(rng.gauss(level, _ATTR_JITTER))),
                )
            )
    return parts


def svg_text() -> str:
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg"',
            f'     width="{_WIDTH:g}pt" height="{_HEIGHT:g}pt"',
            f'     viewBox="0 0 {_WIDTH:g} {_HEIGHT:g}"',
            f'     font-family="{_FONT}">',
            "  <defs>",
            _marker(
                "edge-arrow",
                _ARROW,
                "#000",
            ),
            _marker(
                "flow-arrow",
                _FLOW_ARROW,
                _ACCENT,
            ),
            "  </defs>",
            # White card, so the figure reads on dark themes too.
            f'  <rect width="{_WIDTH:g}" height="{_HEIGHT:g}" rx="8" '
            'fill="#fff"/>',
            *_edgelist_table(),
            *_data_table(),
            *_flow(
                *_FLOW_1,
                "kpnn2 + PyTorch",
            ),
            *_model(),
            *_flow(
                *_FLOW_2,
                "attribution",
            ),
            *_attributions(),
            "</svg>",
            "",
        ]
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
    """Write the overview SVG and its PNG."""
    path = _OUT_PATH if out_path is None else out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_text())
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
