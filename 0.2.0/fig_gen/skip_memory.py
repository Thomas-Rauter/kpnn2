"""Generate the skip-edge memory-lane schematic.

Writes ``docs/figures/skip_memory.svg``, 60 mm by 53 mm.
A toy DAG sits above the computational steps.
Graph edges are solid. Dashed arrows are computational.
Stored activations are written z with a node subscript.
"""

from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parents[1]
_OUT_PATH = _DOCS_DIR / "figures" / "skip_memory.svg"
_FONT = "Liberation Sans, sans-serif"
_FONTSIZE = "6"

# Paper size. User units are PostScript points so font-size 6 is 6 pt.
_MM_TO_PT = 72.0 / 25.4
_WIDTH_MM = 60.0
_HEIGHT_MM = 53.0
_WIDTH_PT = _WIDTH_MM * _MM_TO_PT
_HEIGHT_PT = _HEIGHT_MM * _MM_TO_PT

_GRAPH_STROKE = 1.0
_OP_STROKE = 0.6
_OP_DASH = "2 1.2"
_OP_GAP = 3.0
_SUB_SIZE = "5"
_GRAPH_ARROW = 4.0
_OP_ARROW = 3.5

_A_CX = 28.0
_B_CX = 90.0
_C_CX = 152.0

_TOP_CY = 36.0
_TOP_R = 6.0
_SKIP_DEPTH = 16.0
_TOP_LAYER_Y = 51.0

_Z_Y = 76.0
_ALG_CY = 110.0
_ALG_R = 6.0
_ALG_LAYER_Y = 125.0
# Half-width of a z_a label.
_Z_HALF = 6.0
_LEGEND_SIZE = "5"
_LEGEND_Y = 141.0
_LEGEND_R = 3.5
_LEGEND_CIRCLE_CX = 36.0


def _circle(
    cx: float,
    cy: float,
    radius: float,
    label: str,
    *,
    stroke: float,
) -> str:
    return (
        f'  <circle cx="{cx:g}" cy="{cy:g}" r="{radius:g}" '
        f'fill="#fff" stroke="#000" '
        f'stroke-width="{stroke:g}"/>\n'
        f'  <text x="{cx:g}" y="{cy + 2.2:g}" '
        f'text-anchor="middle" fill="#000">{label}</text>'
    )


def _z(
    x: float,
    y: float,
    sub: str,
    *,
    anchor: str = "middle",
    size: str | None = None,
) -> str:
    size_attr = ""
    sub_size = _SUB_SIZE
    if size is not None:
        size_attr = f' font-size="{size}"'
        sub_size = size
    return (
        f'  <text x="{x:g}" y="{y:g}" '
        f'text-anchor="{anchor}" fill="#000"'
        f"{size_attr}>"
        f'<tspan font-style="italic">z</tspan>'
        f'<tspan dy="2" font-size="{sub_size}">{sub}</tspan>'
        f"</text>"
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
    return (x2 - ux * amount, y2 - uy * amount)


def _line(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    kind: str,
    arrow: bool = True,
) -> str:
    x1, y1 = start
    if kind == "graph":
        width = _GRAPH_STROKE
        marker = "graph-arrow"
        dash = ""
        amount = _GRAPH_ARROW
    else:
        width = _OP_STROKE
        marker = "op-arrow"
        dash = f' stroke-dasharray="{_OP_DASH}"'
        amount = _OP_ARROW
    if arrow:
        x2, y2 = _shorten(
            start,
            end,
            amount,
        )
        marker_attr = f'\n        marker-end="url(#{marker})"'
    else:
        x2, y2 = end
        marker_attr = ""
    return (
        f'  <line x1="{x1:g}" y1="{y1:g}" '
        f'x2="{x2:g}" y2="{y2:g}" '
        f'stroke="#000" stroke-width="{width:g}"{dash}\n'
        f'        stroke-linecap="butt"{marker_attr}/>'
    )


def _skip(
    start: tuple[float, float],
    end: tuple[float, float],
    depth: float,
) -> str:
    x1, y1 = start
    x2, y2 = end
    # The skip arrives from above; shorten along that tangent
    # so the shaft does not run through the arrow tip.
    shaft_y = y2 - _GRAPH_ARROW
    return (
        f'  <path d="M{x1:g} {y1:g} '
        f"C{x1:g} {depth:g}, {x2:g} {depth:g}, "
        f'{x2:g} {shaft_y:g}"\n'
        f'        stroke="#000" '
        f'stroke-width="{_GRAPH_STROKE}"\n'
        f'        fill="none" stroke-linecap="butt"\n'
        f'        marker-end="url(#graph-arrow)"/>'
    )


def _label(
    x: float,
    y: float,
    text: str,
    *,
    anchor: str = "middle",
    weight: str = "400",
    size: str | None = None,
) -> str:
    size_attr = ""
    if size is not None:
        size_attr = f' font-size="{size}"'
    return (
        f'  <text x="{x:g}" y="{y:g}" '
        f'text-anchor="{anchor}" fill="#000" '
        f'font-weight="{weight}"{size_attr}>{text}</text>'
    )


def _marker(
    marker_id: str,
    size: str,
) -> str:
    # Base of the triangle sits on the line end (refX=0). The
    # shaft is shortened by the arrow length, so it does not
    # run through the tip.
    return "\n".join(
        [
            f'    <marker id="{marker_id}" viewBox="0 0 10 10" '
            'refX="0" refY="5"',
            '            markerUnits="userSpaceOnUse"',
            f'            markerWidth="{size}" markerHeight="{size}" '
            'orient="auto">',
            '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#000"/>',
            "    </marker>",
        ]
    )


def _attach(
    cx: float,
    cy: float,
    radius: float,
    stroke: float,
    side: str,
) -> tuple[float, float]:
    distance = radius + stroke / 2.0
    if side == "right":
        return (cx + distance, cy)
    if side == "left":
        return (cx - distance, cy)
    if side == "top":
        return (cx, cy - distance)
    return (cx, cy + distance)


def svg_text() -> str:
    z_line_y = _Z_Y - 1.0
    hold_start = _A_CX + _Z_HALF + _OP_GAP
    hold_end = _B_CX - _Z_HALF - _OP_GAP
    pass_start = _B_CX + _Z_HALF + _OP_GAP
    alg_top = _attach(
        _A_CX,
        _ALG_CY,
        _ALG_R,
        _OP_STROKE,
        "top",
    )[1]
    c_top = _attach(
        _C_CX,
        _ALG_CY,
        _ALG_R,
        _OP_STROKE,
        "top",
    )
    store_end_a = (
        _A_CX,
        _Z_Y + 3.0,
    )
    store_start_a = (
        _A_CX,
        alg_top - _OP_GAP,
    )
    pass_down_start = (
        _C_CX,
        z_line_y,
    )
    pass_down_end = (
        _C_CX,
        c_top[1] - _OP_GAP,
    )
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg"',
            f'     width="{_WIDTH_MM:g}mm" height="{_HEIGHT_MM:g}mm"',
            f'     viewBox="0 0 {_WIDTH_PT:.3f} {_HEIGHT_PT:.3f}"',
            f'     font-family="{_FONT}"',
            f'     font-size="{_FONTSIZE}"',
            '     fill="none">',
            "  <defs>",
            _marker(
                "graph-arrow",
                f"{_GRAPH_ARROW:g}",
            ),
            _marker(
                "op-arrow",
                f"{_OP_ARROW:g}",
            ),
            "  </defs>",
            _label(
                8,
                11,
                "Toy graph",
                anchor="start",
                weight="700",
            ),
            _line(
                _attach(
                    _A_CX,
                    _TOP_CY,
                    _TOP_R,
                    _GRAPH_STROKE,
                    "right",
                ),
                _attach(
                    _B_CX,
                    _TOP_CY,
                    _TOP_R,
                    _GRAPH_STROKE,
                    "left",
                ),
                kind="graph",
            ),
            _line(
                _attach(
                    _B_CX,
                    _TOP_CY,
                    _TOP_R,
                    _GRAPH_STROKE,
                    "right",
                ),
                _attach(
                    _C_CX,
                    _TOP_CY,
                    _TOP_R,
                    _GRAPH_STROKE,
                    "left",
                ),
                kind="graph",
            ),
            _skip(
                _attach(
                    _A_CX,
                    _TOP_CY,
                    _TOP_R,
                    _GRAPH_STROKE,
                    "top",
                ),
                _attach(
                    _C_CX,
                    _TOP_CY,
                    _TOP_R,
                    _GRAPH_STROKE,
                    "top",
                ),
                _SKIP_DEPTH,
            ),
            _circle(
                _A_CX,
                _TOP_CY,
                _TOP_R,
                "A",
                stroke=_GRAPH_STROKE,
            ),
            _circle(
                _B_CX,
                _TOP_CY,
                _TOP_R,
                "B",
                stroke=_GRAPH_STROKE,
            ),
            _circle(
                _C_CX,
                _TOP_CY,
                _TOP_R,
                "C",
                stroke=_GRAPH_STROKE,
            ),
            _label(
                _A_CX,
                _TOP_LAYER_Y,
                "Layer 0",
            ),
            _label(
                _B_CX,
                _TOP_LAYER_Y,
                "Layer 1",
            ),
            _label(
                _C_CX,
                _TOP_LAYER_Y,
                "Layer 2",
            ),
            "",
            _label(
                8,
                64,
                "Algorithm",
                anchor="start",
                weight="700",
            ),
            _z(
                _A_CX,
                _Z_Y,
                "a",
            ),
            _z(
                _B_CX,
                _Z_Y,
                "a",
            ),
            _line(
                (hold_start, z_line_y),
                (hold_end, z_line_y),
                kind="op",
            ),
            _label(
                (hold_start + hold_end) / 2.0,
                z_line_y - 5.0,
                "hold",
            ),
            _line(
                (pass_start, z_line_y),
                (_C_CX, z_line_y),
                kind="op",
                arrow=False,
            ),
            _line(
                pass_down_start,
                pass_down_end,
                kind="op",
            ),
            _label(
                (pass_start + _C_CX) / 2.0,
                z_line_y - 5.0,
                "Pass forward",
            ),
            "",
            _line(
                store_start_a,
                store_end_a,
                kind="op",
            ),
            _label(
                _A_CX - 4,
                (store_start_a[1] + store_end_a[1]) / 2.0 + 2.0,
                "store",
                anchor="end",
            ),
            _line(
                (
                    _A_CX + _ALG_R + _OP_GAP,
                    _ALG_CY,
                ),
                (
                    _B_CX - _ALG_R - _OP_GAP,
                    _ALG_CY,
                ),
                kind="op",
            ),
            _label(
                (_A_CX + _B_CX) / 2.0,
                _ALG_CY - 6.0,
                "Pass forward",
            ),
            _line(
                (
                    _B_CX + _ALG_R + _OP_GAP,
                    _ALG_CY,
                ),
                (
                    _C_CX - _ALG_R - _OP_GAP,
                    _ALG_CY,
                ),
                kind="op",
            ),
            _label(
                (_B_CX + _C_CX) / 2.0,
                _ALG_CY - 6.0,
                "Pass forward",
            ),
            _circle(
                _A_CX,
                _ALG_CY,
                _ALG_R,
                "A",
                stroke=_OP_STROKE,
            ),
            _circle(
                _B_CX,
                _ALG_CY,
                _ALG_R,
                "B",
                stroke=_OP_STROKE,
            ),
            _circle(
                _C_CX,
                _ALG_CY,
                _ALG_R,
                "C",
                stroke=_OP_STROKE,
            ),
            _label(
                _A_CX,
                _ALG_LAYER_Y,
                "Layer 0",
            ),
            _label(
                _B_CX,
                _ALG_LAYER_Y,
                "Layer 1",
            ),
            _label(
                _C_CX,
                _ALG_LAYER_Y,
                "Layer 2",
            ),
            "",
            _label(
                8,
                _LEGEND_Y,
                "Legend",
                anchor="start",
                weight="700",
                size=_LEGEND_SIZE,
            ),
            (
                f'  <circle cx="{_LEGEND_CIRCLE_CX:g}" '
                f'cy="{_LEGEND_Y - 1.6:g}" r="{_LEGEND_R:g}" '
                f'fill="#fff" stroke="#000" '
                f'stroke-width="{_OP_STROKE:g}"/>'
            ),
            _label(
                _LEGEND_CIRCLE_CX + _LEGEND_R + 3.0,
                _LEGEND_Y,
                "Node",
                anchor="start",
                size=_LEGEND_SIZE,
            ),
            _z(
                68.0,
                _LEGEND_Y,
                "a",
                anchor="start",
                size=_LEGEND_SIZE,
            ),
            _label(
                78.0,
                _LEGEND_Y,
                "Activation from node A",
                anchor="start",
                size=_LEGEND_SIZE,
            ),
            "</svg>",
            "",
        ]
    )


def write_figure(out_path: Path | None = None) -> Path:
    """Write the skip-memory schematic SVG."""
    path = _OUT_PATH if out_path is None else out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_text())
    return path


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
