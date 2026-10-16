"""Shared SVG helpers for correctness icons.

Icons write to ``docs/figures/correctness/``. ViewBox units are
points. There is no canvas fill: the SVG background is
transparent. Live versus decoy uses brick red and grey; ink
and any text stay black.
"""

from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parents[2]
_OUT_DIR = _DOCS_DIR / "figures" / "correctness"

LIVE = "#C44536"
DEAD = "#A0A0A0"
INK = "#000000"
WATER = "#B7C4CC"
STROKE = 1.4
SIZE_PT = 80.0
FONT = "Liberation Sans, sans-serif"
FONTSIZE = "6"
DASH = "3.2 2.2"


def wrap(parts: list[str]) -> str:
    """Join icon drawing commands into a square SVG."""
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg"',
        f'     viewBox="0 0 {SIZE_PT:g} {SIZE_PT:g}"',
        f'     font-family="{FONT}"',
        f'     font-size="{FONTSIZE}"',
        '     fill="none">',
        *parts,
        "</svg>",
        "",
    ]
    return "\n".join(lines)


def write_svg(
    name: str,
    parts: list[str],
    *,
    out_path: Path | None = None,
) -> Path:
    """Write a named icon SVG under figures/correctness."""
    path = _OUT_DIR / name if out_path is None else out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(wrap(parts))
    return path


def circle(
    cx: float,
    cy: float,
    radius: float,
    *,
    fill: str = "none",
    stroke: str = INK,
    stroke_width: float = STROKE,
) -> str:
    return (
        f'  <circle cx="{cx:g}" cy="{cy:g}" r="{radius:g}" '
        f'fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:g}"/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = INK,
    stroke_width: float = STROKE,
    dash: str | None = None,
) -> str:
    dash_attr = ""
    if dash is not None:
        dash_attr = f' stroke-dasharray="{dash}"'
    return (
        f'  <line x1="{x1:g}" y1="{y1:g}" '
        f'x2="{x2:g}" y2="{y2:g}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:g}"'
        f'{dash_attr} stroke-linecap="round"/>'
    )


def rounded_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = "none",
    stroke: str = INK,
    radius: float = 2.0,
    stroke_width: float = STROKE,
) -> str:
    return (
        f'  <rect x="{x:g}" y="{y:g}" width="{width:g}" '
        f'height="{height:g}" rx="{radius:g}" '
        f'fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:g}"/>'
    )


def polygon(
    points: list[tuple[float, float]],
    *,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 0.0,
) -> str:
    joined = " ".join(f"{x:g},{y:g}" for x, y in points)
    stroke_attr = ""
    if stroke != "none":
        stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width:g}"'
    return f'  <polygon points="{joined}" fill="{fill}"{stroke_attr}/>'


def path(
    d: str,
    *,
    fill: str = "none",
    stroke: str = INK,
    stroke_width: float = STROKE,
    dash: str | None = None,
    linejoin: str = "round",
    linecap: str = "round",
) -> str:
    dash_attr = ""
    if dash is not None:
        dash_attr = f' stroke-dasharray="{dash}"'
    return (
        f'  <path d="{d}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:g}"'
        f'{dash_attr} stroke-linejoin="{linejoin}" '
        f'stroke-linecap="{linecap}"/>'
    )


def pin(
    x: float,
    y: float,
    *,
    toward_x: float,
    toward_y: float,
    color: str,
    head_r: float = 2.1,
    shaft: float = 5.5,
) -> str:
    """Pin head at (x, y); shaft points toward the given point."""
    dx = toward_x - x
    dy = toward_y - y
    length = (dx * dx + dy * dy) ** 0.5
    ux = dx / length
    uy = dy / length
    return "\n".join(
        [
            circle(
                x,
                y,
                head_r,
                fill=color,
                stroke=color,
                stroke_width=0.6,
            ),
            line(
                x,
                y,
                x + ux * shaft,
                y + uy * shaft,
                stroke=color,
                stroke_width=1.2,
            ),
        ]
    )


def cross(
    cx: float,
    cy: float,
    size: float,
    *,
    stroke: str = INK,
    stroke_width: float = STROKE,
) -> str:
    half = size / 2.0
    return "\n".join(
        [
            line(
                cx - half,
                cy - half,
                cx + half,
                cy + half,
                stroke=stroke,
                stroke_width=stroke_width,
            ),
            line(
                cx - half,
                cy + half,
                cx + half,
                cy - half,
                stroke=stroke,
                stroke_width=stroke_width,
            ),
        ]
    )
