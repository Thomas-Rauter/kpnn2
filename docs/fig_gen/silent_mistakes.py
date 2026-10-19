"""Generate the three-mistakes figure at the top of How we test.

Writes ``docs/figures/silent_mistakes.svg``. Each panel shows one
mistake that leaves a model which trains and gives attributions
without an error: a skip edge that goes missing, an edge the
edgelist does not have, and attributions under swapped names. The
graph is the one of the landing-page overview, imported from
``kpnn2_overview.py`` so the two figures cannot drift apart. The
cells are schematic, not values from a run.
"""

import random
from pathlib import Path

from kpnn2_overview import EDGES, SKIP_EDGES

_DOCS_DIR = Path(__file__).resolve().parents[1]
_OUT_PATH = _DOCS_DIR / "figures" / "silent_mistakes.svg"
_FONT = "Liberation Sans, sans-serif"

# User units are PostScript points, so font-size 9 is 9 pt.
_WIDTH = 656.0
_HEIGHT = 206.0
_PANEL_W = _WIDTH / 3
_TITLE_SIZE = 10.5
_TEXT_SIZE = 8.5
_NOTE_SIZE = 7.5
_STROKE = 0.9
_EXTRA_STROKE = 1.8
_ARROW = 4.5
_SKIP_DASH = "3 2"
_ACCENT = "#ee4c2c"
_GHOST = "#c4c4c4"
_GREY = "#6b6b6b"
_RULE = "#9a9a9a"

_TITLE_Y = 24.0
_NOTE_Y = 38.0

# One network per panel, at panel-relative x.
_PILL_H = 15.0
_PILL_W = 46.0
_NODE_X = {
    "protein_a": 30.0,
    "protein_b": 30.0,
    "protein_c": 30.0,
    "protein_d": 30.0,
    "protein_e": 114.0,
    "protein_f": 114.0,
    "outcome": 190.0,
}
_NODE_Y = {
    "protein_a": 66.0,
    "protein_b": 90.0,
    "protein_c": 136.0,
    "protein_d": 160.0,
    "protein_e": 78.0,
    "protein_f": 148.0,
    "outcome": 113.0,
}
# The skip edge curves under the hidden layer to this depth.
_SKIP_LOW = 200.0

# Panel (b): an edge that is not a row of the edgelist.
EXTRA_EDGE = ("protein_a", "protein_f")

# Panel (c): the attributions of the overview, the model relying on
# protein_e in every sample and hardly on protein_f, once under the
# right names and once with the two names swapped.
_ATTR_ROWS = [
    ("protein_e", 1.45),
    ("protein_f", 0.15),
]
_ATTR_JITTER = 0.1
_N_SAMPLES = 6
_CELL = 9.0
_CELL_STEP = 11.0
_ATTR_NAME_X = 52.0
_ATTR_CELLS_X = 100.0
_ATTR_TABLES = [
    ("right names", 82.0),
    ("swapped names", 136.0),
]


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


def _panel_head(
    left: float,
    title: str,
    note: str,
) -> list[str]:
    return [
        _text(
            left + _PANEL_W / 2,
            _TITLE_Y,
            title,
            size=_TITLE_SIZE,
            anchor="middle",
            weight="700",
        ),
        _text(
            left + _PANEL_W / 2,
            _NOTE_Y,
            note,
            size=_TEXT_SIZE,
            anchor="middle",
            fill=_GREY,
        ),
    ]


def _pill(
    left: float,
    name: str,
) -> str:
    cx = left + _NODE_X[name]
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
    left: float,
    source: str,
    target: str,
    *,
    stroke: str = "#000",
    width: float = _STROKE,
    marker: str = "edge-arrow",
) -> str:
    x1 = left + _NODE_X[source] + _PILL_W / 2
    y1 = _NODE_Y[source]
    x2 = left + _NODE_X[target] - _PILL_W / 2
    y2 = _NODE_Y[target]
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    end_x = x2 - (x2 - x1) / length * _ARROW
    end_y = y2 - (y2 - y1) / length * _ARROW
    return (
        f'  <line x1="{x1:g}" y1="{y1:g}" '
        f'x2="{end_x:.2f}" y2="{end_y:.2f}" '
        f'stroke="{stroke}" stroke-width="{width:g}" '
        f'marker-end="url(#{marker})"/>'
    )


def _skip_path(
    left: float,
    source: str,
    target: str,
) -> tuple[float, float, float, float]:
    # Curve under the hidden layer into the bottom of the target, so
    # the edge reads as passing the layer it skips.
    x1 = left + _NODE_X[source] + _PILL_W / 2
    y1 = _NODE_Y[source]
    x2 = left + _NODE_X[target]
    y2 = _NODE_Y[target] + _PILL_H / 2 + _ARROW
    return x1, y1, x2, y2


def _skip_edge(
    left: float,
    source: str,
    target: str,
    *,
    stroke: str = "#000",
    marker: str = "edge-arrow",
) -> str:
    x1, y1, x2, y2 = _skip_path(
        left,
        source,
        target,
    )
    return (
        f'  <path d="M {x1:g} {y1:g} C {x1 + 50.0:g} {_SKIP_LOW:g} '
        f'{x2:g} {_SKIP_LOW:g} {x2:g} {y2:g}" fill="none" '
        f'stroke="{stroke}" stroke-width="{_STROKE:g}" '
        f'stroke-dasharray="{_SKIP_DASH}" '
        f'marker-end="url(#{marker})"/>'
    )


def _cross(
    left: float,
    source: str,
    target: str,
) -> list[str]:
    # An X on the lowest point of the skip curve, which a cubic with
    # both control points at _SKIP_LOW reaches at t = 0.5.
    x1, y1, x2, y2 = _skip_path(
        left,
        source,
        target,
    )
    cx = (x1 + 3 * (x1 + 50.0) + 3 * x2 + x2) / 8
    cy = (y1 + 6 * _SKIP_LOW + y2) / 8
    arm = 5.0
    return [
        f'  <path d="M {cx - arm:g} {cy - arm:g} '
        f"L {cx + arm:g} {cy + arm:g} "
        f"M {cx - arm:g} {cy + arm:g} "
        f'L {cx + arm:g} {cy - arm:g}" '
        f'stroke="{_ACCENT}" stroke-width="2"/>',
        _text(
            cx + arm + 4.0,
            cy + 3.0,
            "missing",
            size=_TEXT_SIZE,
            fill=_ACCENT,
        ),
    ]


def _network(
    left: float,
    *,
    missing: tuple[str, str] | None = None,
    extra: tuple[str, str] | None = None,
) -> list[str]:
    parts = []
    for source, target in EDGES:
        if (source, target) == missing:
            parts.append(
                _skip_edge(
                    left,
                    source,
                    target,
                    stroke=_GHOST,
                    marker="ghost-arrow",
                )
            )
            parts += _cross(
                left,
                source,
                target,
            )
        elif (source, target) in SKIP_EDGES:
            parts.append(
                _skip_edge(
                    left,
                    source,
                    target,
                )
            )
        else:
            parts.append(
                _edge(
                    left,
                    source,
                    target,
                )
            )
    if extra is not None:
        parts.append(
            _edge(
                left,
                *extra,
                stroke=_ACCENT,
                width=_EXTRA_STROKE,
                marker="extra-arrow",
            )
        )
    parts += [
        _pill(
            left,
            name,
        )
        for name in _NODE_Y
    ]
    return parts


def _tint(value: float) -> str:
    # Map roughly 0..1.6 onto a pale tint..the full accent colour.
    level = 0.12 + 0.88 * max(0.0, min(1.0, value / 1.6))
    accent = [int(_ACCENT[i : i + 2], 16) for i in (1, 3, 5)]
    red, green, blue = (round(255 - (255 - c) * level) for c in accent)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _attribution_tables(left: float) -> list[str]:
    # Both tables hold the same values, drawn once. The second
    # attaches the two names in the wrong order.
    rng = random.Random(7)
    values = [
        [
            abs(
                rng.gauss(
                    level,
                    _ATTR_JITTER,
                )
            )
            for _ in range(_N_SAMPLES)
        ]
        for _, level in _ATTR_ROWS
    ]
    names = [name for name, _ in _ATTR_ROWS]
    parts = []
    for table, (label, top) in enumerate(_ATTR_TABLES):
        row_names = names if table == 0 else names[::-1]
        parts.append(
            _text(
                left + _ATTR_NAME_X,
                top - 5.0,
                label,
                size=_NOTE_SIZE,
                fill=_GREY,
            )
        )
        for row, name in enumerate(row_names):
            y = top + _CELL_STEP * row
            parts.append(
                _text(
                    left + _ATTR_NAME_X,
                    y + _CELL - 1.5,
                    name,
                    size=_NOTE_SIZE,
                    weight="400" if table == 0 else "700",
                )
            )
            for sample, value in enumerate(values[row]):
                x = left + _ATTR_CELLS_X + _CELL_STEP * sample
                parts.append(
                    f'  <rect x="{x:g}" y="{y:g}" '
                    f'width="{_CELL:g}" height="{_CELL:g}" '
                    f'fill="{_tint(value)}" stroke="#000" '
                    f'stroke-width="0.4"/>'
                )
    return parts


def _divider(x: float) -> str:
    return (
        f'  <line x1="{x:g}" y1="12" x2="{x:g}" y2="{_HEIGHT - 12:g}" '
        f'stroke="{_RULE}" stroke-width="0.6"/>'
    )


def svg_text() -> str:
    missing = next(iter(SKIP_EDGES))
    panel_b = _PANEL_W
    panel_c = 2 * _PANEL_W
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
                "ghost-arrow",
                _ARROW,
                _GHOST,
            ),
            _marker(
                "extra-arrow",
                _ARROW * 1.4,
                _ACCENT,
            ),
            "  </defs>",
            # White card, so the figure reads on dark themes too.
            f'  <rect width="{_WIDTH:g}" height="{_HEIGHT:g}" rx="8" '
            'fill="#fff"/>',
            *_panel_head(
                0.0,
                "(a) An edge goes missing",
                f"{missing[0]} → {missing[1]} is lost",
            ),
            *_network(
                0.0,
                missing=missing,
            ),
            _divider(panel_b),
            *_panel_head(
                panel_b,
                "(b) An edge appears",
                f"{EXTRA_EDGE[0]} → {EXTRA_EDGE[1]} is not in the edgelist",
            ),
            *_network(
                panel_b,
                extra=EXTRA_EDGE,
            ),
            _divider(panel_c),
            *_panel_head(
                panel_c,
                "(c) The wrong name",
                "protein_e's values reported as protein_f's",
            ),
            *_attribution_tables(panel_c),
            "</svg>",
            "",
        ]
    )


def write_figure(out_path: Path | None = None) -> Path:
    """Write the three-mistakes SVG."""
    path = _OUT_PATH if out_path is None else out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_text())
    return path


def main() -> None:
    print(f"Wrote {write_figure()}")


if __name__ == "__main__":
    main()
