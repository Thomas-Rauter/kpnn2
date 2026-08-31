"""Generate the layered-versus-adjacency forward-pass schematic.

Writes ``docs/figures/layered_vs_adjacency_forward.svg``,
108 mm by 64 mm.

The same toy DAG as ``layered_vs_adjacency.svg``. Panel (a) is
one layered sweep: hops, ReLU, gather. Panel (b) scatters the
input into a state vector and applies ``spec.mask`` in a loop
the user owns.
"""

from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parents[1]
_OUT_PATH = _DOCS_DIR / "figures" / "layered_vs_adjacency_forward.svg"
_FONT = "Liberation Sans, sans-serif"
_FONTSIZE = "6"

_MM_TO_PT = 72.0 / 25.4
_WIDTH_MM = 108.0
_HEIGHT_MM = 64.0
_WIDTH_PT = _WIDTH_MM * _MM_TO_PT
_HEIGHT_PT = _HEIGHT_MM * _MM_TO_PT

_STROKE = 0.6
_DASH = "2 1.2"
_ARROW = 3.5

# Strip (a) operator boxes: (x, y, width, height).
_AX = (10.0, 26.0, 22.0, 14.0)
_H0 = (48.0, 26.0, 38.0, 14.0)
_RE = (102.0, 26.0, 28.0, 14.0)
_GA = (146.0, 26.0, 38.0, 14.0)
_H1 = (200.0, 26.0, 38.0, 14.0)
_AY = (254.0, 26.0, 22.0, 14.0)
_HOLD_DEPTH = 58.0

# Strip (b).
_BX = (10.0, 108.0, 22.0, 14.0)
_SC = (48.0, 108.0, 38.0, 14.0)
_ST = (102.0, 108.0, 72.0, 14.0)
_SM = (190.0, 108.0, 48.0, 14.0)
_BY = (254.0, 108.0, 22.0, 14.0)
_LOOP_DEPTH = 140.0

_DIVIDER_Y = 80.0


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, width, height = box
    return (
        x + width / 2.0,
        y + height / 2.0,
    )


def _right_center(
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    x, y, width, height = box
    return (
        x + width,
        y + height / 2.0,
    )


def _left_center(
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    x, y, width, height = box
    return (
        x,
        y + height / 2.0,
    )


def _bottom_center(
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    x, y, width, height = box
    return (
        x + width / 2.0,
        y + height,
    )


def _shorten(
    start: tuple[float, float],
    end: tuple[float, float],
    amount: float,
) -> tuple[float, float]:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    ux = dx / length
    uy = dy / length
    return (
        x2 - ux * amount,
        y2 - uy * amount,
    )


def _box(
    box: tuple[float, float, float, float],
    label: str,
) -> str:
    x, y, width, height = box
    cx, cy = _center(box)
    return (
        f'  <rect x="{x:g}" y="{y:g}" width="{width:g}" '
        f'height="{height:g}" fill="#fff" stroke="#000" '
        f'stroke-width="{_STROKE:g}"/>\n'
        f'  <text x="{cx:g}" y="{cy + 2.2:g}" '
        f'text-anchor="middle" fill="#000">{label}</text>'
    )


def _line(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
) -> str:
    x1, y1 = start
    x2, y2 = _shorten(
        start,
        end,
        _ARROW,
    )
    dash = ""
    if dashed:
        dash = f' stroke-dasharray="{_DASH}"'
    return (
        f'  <line x1="{x1:g}" y1="{y1:g}" '
        f'x2="{x2:g}" y2="{y2:g}" '
        f'stroke="#000" stroke-width="{_STROKE:g}"{dash}\n'
        f'        marker-end="url(#arrow)"/>'
    )


def _hold(
    start: tuple[float, float],
    end: tuple[float, float],
    depth: float,
) -> str:
    x1, y1 = start
    x2, y2 = end
    shaft_y = y2 + _ARROW
    return (
        f'  <path d="M{x1:g} {y1:g} '
        f"C{x1:g} {depth:g}, {x2:g} {depth:g}, "
        f'{x2:g} {shaft_y:g}"\n'
        f'        stroke="#000" stroke-width="{_STROKE:g}" '
        f'stroke-dasharray="{_DASH}"\n'
        f'        fill="none" marker-end="url(#arrow)"/>'
    )


def _loop(
    box: tuple[float, float, float, float],
    depth: float,
) -> str:
    x, y, width, height = box
    x1 = x + width
    y1 = y + height
    x2 = x
    y2 = y + height
    shaft_y = y2 + _ARROW
    return (
        f'  <path d="M{x1:g} {y1:g} '
        f"C{x1:g} {depth:g}, {x2:g} {depth:g}, "
        f'{x2:g} {shaft_y:g}"\n'
        f'        stroke="#000" stroke-width="{_STROKE:g}"\n'
        f'        fill="none" marker-end="url(#arrow)"/>'
    )


def _label(
    x: float,
    y: float,
    text: str,
    *,
    anchor: str = "middle",
    weight: str = "400",
) -> str:
    return (
        f'  <text x="{x:g}" y="{y:g}" '
        f'text-anchor="{anchor}" fill="#000" '
        f'font-weight="{weight}">{text}</text>'
    )


def _marker() -> str:
    return "\n".join(
        [
            '    <marker id="arrow" viewBox="0 0 10 10" refX="0" refY="5"',
            '            markerUnits="userSpaceOnUse"',
            f'            markerWidth="{_ARROW:g}" '
            f'markerHeight="{_ARROW:g}" orient="auto">',
            '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#000"/>',
            "    </marker>",
        ]
    )


def _state(box: tuple[float, float, float, float]) -> str:
    x, y, width, height = box
    cell = width / 3.0
    cy = y + height / 2.0 + 2.2
    parts = [
        f'  <rect x="{x:g}" y="{y:g}" width="{width:g}" '
        f'height="{height:g}" fill="#fff" stroke="#000" '
        f'stroke-width="{_STROKE:g}"/>',
        f'  <line x1="{x + cell:g}" y1="{y:g}" '
        f'x2="{x + cell:g}" y2="{y + height:g}" '
        f'stroke="#000" stroke-width="{_STROKE:g}"/>',
        f'  <line x1="{x + 2.0 * cell:g}" y1="{y:g}" '
        f'x2="{x + 2.0 * cell:g}" y2="{y + height:g}" '
        f'stroke="#000" stroke-width="{_STROKE:g}"/>',
        _label(
            x + 0.5 * cell,
            cy,
            "A",
            weight="700",
        ),
        _label(
            x + 1.5 * cell,
            cy,
            "C",
        ),
        _label(
            x + 2.5 * cell,
            cy,
            "H",
        ),
    ]
    return "\n".join(parts)


def svg_text() -> str:
    ax_mid = _center(_AX)
    h0_mid = _center(_H0)
    ga_mid = _center(_GA)
    h1_mid = _center(_H1)
    ay_mid = _center(_AY)
    bx_mid = _center(_BX)
    sm_mid = _center(_SM)
    by_mid = _center(_BY)
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg"',
            f'     width="{_WIDTH_MM:g}mm" height="{_HEIGHT_MM:g}mm"',
            f'     viewBox="0 0 {_WIDTH_PT:.3f} {_HEIGHT_PT:.3f}"',
            f'     font-family="{_FONT}"',
            f'     font-size="{_FONTSIZE}"',
            '     fill="none">',
            "  <defs>",
            _marker(),
            "  </defs>",
            f'  <line x1="8" y1="{_DIVIDER_Y:g}" '
            f'x2="{_WIDTH_PT - 8.0:.3f}" y2="{_DIVIDER_Y:g}"',
            '        stroke="#000" stroke-width="0.4"/>',
            "",
            _label(
                8,
                12,
                "(a) parse_layered — one sweep",
                anchor="start",
                weight="700",
            ),
            _box(_AX, "x"),
            _box(_H0, "hops[0]"),
            _box(_RE, "ReLU"),
            _box(_GA, "gather"),
            _box(_H1, "hops[1]"),
            _box(_AY, "y"),
            _line(
                _right_center(_AX),
                _left_center(_H0),
            ),
            _line(
                _right_center(_H0),
                _left_center(_RE),
            ),
            _line(
                _right_center(_RE),
                _left_center(_GA),
            ),
            _line(
                _right_center(_GA),
                _left_center(_H1),
            ),
            _line(
                _right_center(_H1),
                _left_center(_AY),
            ),
            _hold(
                _bottom_center(_AX),
                _bottom_center(_GA),
                _HOLD_DEPTH,
            ),
            _label(
                ax_mid[0],
                22,
                "A",
            ),
            _label(
                h0_mid[0],
                22,
                "H",
            ),
            _label(
                ga_mid[0],
                22,
                "A, H",
            ),
            _label(
                h1_mid[0],
                22,
                "C",
            ),
            _label(
                ay_mid[0],
                22,
                "C",
            ),
            _label(
                40,
                _HOLD_DEPTH + 8.0,
                "hold A",
                anchor="start",
            ),
            "",
            _label(
                8,
                94,
                "(b) parse_adjacency — user loop",
                anchor="start",
                weight="700",
            ),
            _box(_BX, "x"),
            _box(_SC, "scatter"),
            _state(_ST),
            _box(_SM, "spec.mask"),
            _box(_BY, "y"),
            _line(
                _right_center(_BX),
                _left_center(_SC),
            ),
            _line(
                _right_center(_SC),
                _left_center(_ST),
            ),
            _line(
                _right_center(_ST),
                _left_center(_SM),
            ),
            _line(
                _right_center(_SM),
                _left_center(_BY),
            ),
            _loop(
                _SM,
                _LOOP_DEPTH,
            ),
            _label(
                bx_mid[0],
                102,
                "A",
            ),
            _label(
                sm_mid[0],
                _LOOP_DEPTH + 8.0,
                "n steps",
            ),
            _label(
                by_mid[0],
                102,
                "C",
            ),
            "</svg>",
            "",
        ]
    )


def write_figure(out_path: Path | None = None) -> Path:
    """Write the forward-pass schematic SVG."""
    path = _OUT_PATH if out_path is None else out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_text())
    return path


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
