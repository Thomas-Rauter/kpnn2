"""Generate the Skip edges SkipAdd schematic.

Writes ``docs/figures/skip_add.svg``.
"""

from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parents[1]
_OUT_PATH = _DOCS_DIR / "figures" / "skip_add.svg"
_FONT = "Liberation Sans, sans-serif"
_FONTSIZE = "6"

# Node boxes: (x, y, width, height).
_A = (12.0, 18.0, 36.0, 16.0)
_B = (12.0, 46.0, 36.0, 16.0)
_H1 = (92.0, 32.0, 36.0, 16.0)
_H2 = (178.0, 32.0, 36.0, 16.0)
_C = (264.0, 32.0, 36.0, 16.0)


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, width, height = box
    return (
        x + width / 2.0,
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


def _box(
    box: tuple[float, float, float, float],
    label: str,
) -> str:
    x, y, width, height = box
    cx, cy = _center(box)
    return (
        f'  <rect x="{x:g}" y="{y:g}" width="{width:g}" '
        f'height="{height:g}" stroke="#000" '
        f'stroke-width="0.6"/>\n'
        f'  <text x="{cx:g}" y="{cy + 2.5:g}" '
        f'text-anchor="middle" fill="#000">{label}</text>'
    )


def _line(
    start: tuple[float, float],
    end: tuple[float, float],
) -> str:
    x1, y1 = start
    x2, y2 = end
    return (
        f'  <line x1="{x1:g}" y1="{y1:g}" '
        f'x2="{x2:g}" y2="{y2:g}" '
        f'stroke="#000" stroke-width="0.6"\n'
        f'        marker-end="url(#arrow)"/>'
    )


def _skip(
    start: tuple[float, float],
    end: tuple[float, float],
    depth: float,
) -> str:
    x1, y1 = start
    x2, y2 = end
    return (
        f'  <path d="M{x1:g} {y1:g} '
        f"C{x1:g} {depth:g}, {x2:g} {depth:g}, "
        f'{x2:g} {y2:g}"\n'
        f'        stroke="#000" stroke-width="0.6" '
        f'stroke-dasharray="2 1.2"\n'
        f'        marker-end="url(#arrow)"/>'
    )


def svg_text() -> str:
    a_mid = _center(_A)
    h1_mid = _center(_H1)
    h2_mid = _center(_H2)
    c_mid = _center(_C)
    a_bottom = _bottom_center(_A)
    h1_bottom = _bottom_center(_H1)
    h2_bottom = _bottom_center(_H2)
    c_bottom = _bottom_center(_C)
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg"',
            '     viewBox="0 0 320 108"',
            f'     font-family="{_FONT}"',
            f'     font-size="{_FONTSIZE}"',
            '     fill="none">',
            "  <defs>",
            '    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"',
            '            markerWidth="5" markerHeight="5" '
            'orient="auto-start-reverse">',
            '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#000"/>',
            "    </marker>",
            "  </defs>",
            f'  <text x="{a_mid[0]:g}" y="12" '
            f'text-anchor="middle" fill="#000">'
            f"layer 0</text>",
            f'  <text x="{h1_mid[0]:g}" y="12" '
            f'text-anchor="middle" fill="#000">'
            f"layer 1</text>",
            f'  <text x="{h2_mid[0]:g}" y="12" '
            f'text-anchor="middle" fill="#000">'
            f"layer 2</text>",
            f'  <text x="{c_mid[0]:g}" y="12" '
            f'text-anchor="middle" fill="#000">'
            f"layer 3</text>",
            "",
            _box(_A, "A"),
            _box(_B, "B"),
            "",
            _box(_H1, "H1"),
            _box(_H2, "H2"),
            _box(_C, "C"),
            "",
            _line(_right_center(_A), _left_center(_H1)),
            _line(_right_center(_B), _left_center(_H1)),
            f'  <text x="{(a_mid[0] + h1_mid[0]) / 2:g}" '
            f'y="24" text-anchor="middle" fill="#000">'
            f"mask[0]</text>",
            "",
            _line(_right_center(_H1), _left_center(_H2)),
            f'  <text x="{(h1_mid[0] + h2_mid[0]) / 2:g}" '
            f'y="36" text-anchor="middle" fill="#000">'
            f"mask[1]</text>",
            "",
            _line(_right_center(_H2), _left_center(_C)),
            f'  <text x="{(h2_mid[0] + c_mid[0]) / 2:g}" '
            f'y="36" text-anchor="middle" fill="#000">'
            f"mask[2]</text>",
            "",
            # A -> H2 and A -> C share A's bottom-center.
            _skip(a_bottom, h2_bottom, 72.0),
            _skip(h1_bottom, c_bottom, 80.0),
            _skip(a_bottom, c_bottom, 96.0),
            '  <text x="160" y="106" text-anchor="middle" fill="#000">',
            (
                "    Dashed: SkipAdd into the target "
                "pre-activation (not in masks)"
            ),
            "  </text>",
            "</svg>",
            "",
        ]
    )


def write_figure(out_path: Path | None = None) -> Path:
    """Write the SkipAdd schematic SVG."""
    path = _OUT_PATH if out_path is None else out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_text())
    return path


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
