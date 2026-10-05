"""Generate the landing-page overview schematic.

Writes ``docs/figures/kpnn2_overview.svg`` and a matching PNG.
The PNG is 4× 96 dpi, so it stays sharp on PyPI, which cannot
show the SVG from a relative path.

Inputs on the left (a named edgelist and a data table), the
sparsely connected model in the middle, scores per named node on
the right. The graph and node names are the README example's,
and the bar lengths follow the scores that example prints. The
data cells are texture, not the example's values; only the
output row follows the example's labelling rule.
"""

import random
import subprocess
from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parents[1]
_OUT_PATH = _DOCS_DIR / "figures" / "kpnn2_overview.svg"
_FONT = "Liberation Sans, sans-serif"
_PNG_DPI = 384

# User units are PostScript points, so font-size 9 is 9 pt.
_WIDTH = 620.0
_HEIGHT = 218.0
_TITLE_SIZE = 10.5
_TEXT_SIZE = 8.5
_NOTE_SIZE = 7.5
_STROKE = 0.9
_ARROW = 4.5
_FLOW_ARROW = 7.0
_ACCENT = "#ee4c2c"
_GREY = "#6b6b6b"
_RULE = "#9a9a9a"

EDGES = [
    ("input_signal_1", "hidden_signal"),
    ("input_signal_2", "hidden_signal"),
    ("input_noise_1", "hidden_noise"),
    ("input_noise_2", "hidden_noise"),
    ("hidden_signal", "output"),
    ("hidden_noise", "output"),
]
# Mean absolute conductance the README example prints.
SCORES = {
    "hidden_signal": 4.90,
    "hidden_noise": 0.39,
}

# Left column: the edgelist table and the data table.
_LEFT_X = 16.0
_TARGET_X = 98.0
_TABLE_TITLE_Y = 24.0
_TABLE_HEAD_Y = 42.0
_ROW_STEP = 12.0
_DATA_TITLE_Y = 136.0
_DATA_TOP = 146.0
_CELL = 9.0
_CELL_STEP = 11.0
_CELLS_X = 84.0
_N_SAMPLES = 6
_DATA_ROWS = [
    "input_signal_1",
    "input_signal_2",
    "input_noise_1",
    "input_noise_2",
]

# Middle column: the network, one pill per named node.
_MODEL_CX = 334.0
_PILL_H = 15.0
_INPUT_CX = 243.0
_INPUT_W = 74.0
_HIDDEN_CX = 349.0
_HIDDEN_W = 72.0
_OUT_CX = 433.0
_OUT_W = 48.0
_NODE_Y = {
    "input_signal_1": 70.0,
    "input_signal_2": 94.0,
    "input_noise_1": 140.0,
    "input_noise_2": 164.0,
    "hidden_signal": 82.0,
    "hidden_noise": 152.0,
    "output": 117.0,
}
_NODE_X = {
    "input_signal_1": (_INPUT_CX, _INPUT_W),
    "input_signal_2": (_INPUT_CX, _INPUT_W),
    "input_noise_1": (_INPUT_CX, _INPUT_W),
    "input_noise_2": (_INPUT_CX, _INPUT_W),
    "hidden_signal": (_HIDDEN_CX, _HIDDEN_W),
    "hidden_noise": (_HIDDEN_CX, _HIDDEN_W),
    "output": (_OUT_CX, _OUT_W),
}

# Right column: one bar per named hidden node.
_BARS_LABEL_X = 554.0
_BARS_X = 560.0
_BAR_MAX = 44.0
_BAR_H = 11.0
_BARS_Y = {
    "hidden_signal": 100.0,
    "hidden_noise": 124.0,
}

# Flow arrows between the columns.
_FLOW_Y = 117.0
_FLOW_1 = (160.0, 198.0)
_FLOW_2 = (466.0, 494.0)


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


def _hline(
    x1: float,
    x2: float,
    y: float,
) -> str:
    return (
        f'  <line x1="{x1:g}" y1="{y:g}" x2="{x2:g}" y2="{y:g}" '
        f'stroke="{_RULE}" stroke-width="0.6"/>'
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
        _text(
            _LEFT_X,
            _TABLE_TITLE_Y,
            "Your graph, as an edgelist",
            size=_TITLE_SIZE,
            weight="700",
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
            _TARGET_X + 56.0,
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


def _data_values() -> list[list[float]]:
    rng = random.Random(42)
    return [
        [rng.gauss(0.0, 1.0) for _ in range(_N_SAMPLES)] for _ in _DATA_ROWS
    ]


def _grey(value: float) -> str:
    # Map roughly -2..2 onto light..dark grey.
    level = max(0.0, min(1.0, (value + 2.0) / 4.0))
    shade = round(235 - 165 * level)
    return f"#{shade:02x}{shade:02x}{shade:02x}"


def _data_table() -> list[str]:
    values = _data_values()
    parts = [
        _text(
            _LEFT_X,
            _DATA_TITLE_Y,
            "Your data, by name",
            size=_TITLE_SIZE,
            weight="700",
        ),
    ]
    rows = [*_DATA_ROWS, "output"]
    for row, name in enumerate(rows):
        top = _DATA_TOP + _CELL_STEP * row
        if name == "output":
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
            if name == "output":
                # The README example labels a sample 1 when the
                # two signal inputs sum above zero.
                label = values[0][sample] + values[1][sample] > 0
                fill = "#000" if label else "#fff"
            else:
                fill = _grey(values[row][sample])
            x = _CELLS_X + _CELL_STEP * sample
            parts.append(
                f'  <rect x="{x:g}" y="{top:g}" width="{_CELL:g}" '
                f'height="{_CELL:g}" fill="{fill}" stroke="#000" '
                f'stroke-width="0.4"/>'
            )
    return parts


def _pill(name: str) -> str:
    cx, width = _NODE_X[name]
    cy = _NODE_Y[name]
    return (
        f'  <rect x="{cx - width / 2:g}" y="{cy - _PILL_H / 2:g}" '
        f'width="{width:g}" height="{_PILL_H:g}" '
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
    source_cx, source_w = _NODE_X[source]
    target_cx, target_w = _NODE_X[target]
    x1 = source_cx + source_w / 2
    y1 = _NODE_Y[source]
    x2 = target_cx - target_w / 2
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


def _model() -> list[str]:
    parts = [
        _text(
            _MODEL_CX,
            _TABLE_TITLE_Y,
            "Sparsely connected PyTorch model",
            size=_TITLE_SIZE,
            anchor="middle",
            weight="700",
        ),
        _text(
            _MODEL_CX,
            _TABLE_TITLE_Y + 13.0,
            "one unit per node, connected only along the edges",
            size=_NOTE_SIZE,
            anchor="middle",
            fill=_GREY,
        ),
    ]
    parts += [
        _edge(
            source,
            target,
        )
        for source, target in EDGES
    ]
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


def _bars() -> list[str]:
    top = min(_BARS_Y.values())
    parts = [
        _text(
            _BARS_X - 2.0,
            _TABLE_TITLE_Y,
            "Scores by name",
            size=_TITLE_SIZE,
            anchor="middle",
            weight="700",
        ),
        _text(
            _BARS_X - 2.0,
            _TABLE_TITLE_Y + 13.0,
            "how much the model relies",
            size=_NOTE_SIZE,
            anchor="middle",
            fill=_GREY,
        ),
        _text(
            _BARS_X - 2.0,
            _TABLE_TITLE_Y + 23.0,
            "on each hidden node",
            size=_NOTE_SIZE,
            anchor="middle",
            fill=_GREY,
        ),
        f'  <line x1="{_BARS_X:g}" y1="{top - 6.0:g}" '
        f'x2="{_BARS_X:g}" y2="{max(_BARS_Y.values()) + _BAR_H + 6.0:g}" '
        f'stroke="#000" stroke-width="0.6"/>',
    ]
    largest = max(SCORES.values())
    for name, y in _BARS_Y.items():
        width = _BAR_MAX * SCORES[name] / largest
        parts.append(
            _text(
                _BARS_LABEL_X,
                y + _BAR_H - 3.0,
                name,
                size=_NOTE_SIZE,
                anchor="end",
            )
        )
        parts.append(
            f'  <rect x="{_BARS_X:g}" y="{y:g}" width="{width:.2f}" '
            f'height="{_BAR_H:g}" fill="{_ACCENT}"/>'
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
            *_bars(),
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
