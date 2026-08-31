"""Generate the layered-versus-adjacency layout schematic.

Writes ``docs/figures/layered_vs_adjacency.svg``, 108 mm by 47 mm.

The same three-node DAG (A -> H -> C plus skip A -> C) is parsed
two ways. Panel (a) ranks it into hops. Panel (b) puts every node
in one alphabetical state vector behind a square mask.
"""

from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parents[1]
_OUT_PATH = _DOCS_DIR / "figures" / "layered_vs_adjacency.svg"
_FONT = "Liberation Sans, sans-serif"
_FONTSIZE = "6"

_MM_TO_PT = 72.0 / 25.4
_WIDTH_MM = 108.0
_HEIGHT_MM = 47.0
_WIDTH_PT = _WIDTH_MM * _MM_TO_PT
_HEIGHT_PT = _HEIGHT_MM * _MM_TO_PT

_STROKE = 0.6
_DASH = "2 1.2"
_ARROW = 3.5
_CELL = 12.0
_ROW_LABEL = 10.0
_TITLE_H = 10.0
_COL_H = 8.0

# Node boxes: (x, y, width, height).
_LA = (12.0, 28.0, 26.0, 13.0)
_LH = (54.0, 28.0, 26.0, 13.0)
_LC = (96.0, 28.0, 26.0, 13.0)
_RA = (168.0, 28.0, 26.0, 13.0)
_RC = (210.0, 28.0, 26.0, 13.0)
_RH = (252.0, 28.0, 26.0, 13.0)

_DIVIDER_X = 148.0
_SKIP_DEPTH = 56.0


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


def _skip(
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


def _matrix(
    origin: tuple[float, float],
    title: str,
    col_names: tuple[str, ...],
    row_names: tuple[str, ...],
    values: tuple[tuple[int, ...], ...],
) -> str:
    ox, oy = origin
    grid_x = ox + _ROW_LABEL
    grid_y = oy + _TITLE_H + _COL_H
    grid_w = _CELL * len(col_names)
    parts = [
        _label(
            grid_x + grid_w / 2.0,
            oy,
            title,
            weight="700",
        )
    ]
    col_y = oy + _TITLE_H
    for index, name in enumerate(col_names):
        cx = grid_x + (index + 0.5) * _CELL
        parts.append(
            _label(
                cx,
                col_y,
                name,
            )
        )
    for row_index, row_name in enumerate(row_names):
        cell_top = grid_y + row_index * _CELL
        baseline = cell_top + _CELL / 2.0 + 2.2
        parts.append(
            _label(
                grid_x - 2.0,
                baseline,
                row_name,
                anchor="end",
            )
        )
        for col_index, value in enumerate(values[row_index]):
            x = grid_x + col_index * _CELL
            parts.append(
                f'  <rect x="{x:g}" y="{cell_top:g}" '
                f'width="{_CELL:g}" height="{_CELL:g}" '
                f'fill="#fff" stroke="#000" '
                f'stroke-width="{_STROKE:g}"/>'
            )
            weight = "700" if value else "400"
            parts.append(
                _label(
                    x + _CELL / 2.0,
                    baseline,
                    str(value),
                    weight=weight,
                )
            )
    return "\n".join(parts)


def svg_text() -> str:
    a_mid = _center(_LA)
    h_mid = _center(_LH)
    c_mid = _center(_LC)
    ra_mid = _center(_RA)
    rh_mid = _center(_RH)
    nodes_mid = (ra_mid[0] + rh_mid[0]) / 2.0
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
            f'  <line x1="{_DIVIDER_X:g}" y1="8" '
            f'x2="{_DIVIDER_X:g}" y2="{_HEIGHT_PT - 8.0:.3f}"',
            '        stroke="#000" stroke-width="0.4"/>',
            "",
            _label(
                8,
                12,
                "(a) parse_layered",
                anchor="start",
                weight="700",
            ),
            _label(
                a_mid[0],
                22,
                "layer 0",
            ),
            _label(
                h_mid[0],
                22,
                "layer 1",
            ),
            _label(
                c_mid[0],
                22,
                "layer 2",
            ),
            _box(_LA, "A"),
            _box(_LH, "H"),
            _box(_LC, "C"),
            _line(
                _right_center(_LA),
                _left_center(_LH),
            ),
            _line(
                _right_center(_LH),
                _left_center(_LC),
            ),
            _skip(
                _bottom_center(_LA),
                _bottom_center(_LC),
                _SKIP_DEPTH,
            ),
            _matrix(
                (29.0, 78.0),
                "hops[0]",
                ("A",),
                ("H",),
                ((1,),),
            ),
            _matrix(
                (66.0, 78.0),
                "hops[1]",
                ("A", "H"),
                ("C",),
                ((1, 1),),
            ),
            "",
            _label(
                156,
                12,
                "(b) parse_adjacency",
                anchor="start",
                weight="700",
            ),
            _label(
                nodes_mid,
                22,
                "spec.nodes",
            ),
            _box(_RA, "A"),
            _box(_RC, "C"),
            _box(_RH, "H"),
            _matrix(
                (195.0, 70.0),
                "spec.mask",
                ("A", "C", "H"),
                ("A", "C", "H"),
                (
                    (0, 0, 0),
                    (1, 0, 1),
                    (1, 0, 0),
                ),
            ),
            "</svg>",
            "",
        ]
    )


def write_figure(out_path: Path | None = None) -> Path:
    """Write the layered-versus-adjacency schematic SVG."""
    path = _OUT_PATH if out_path is None else out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_text())
    return path


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
