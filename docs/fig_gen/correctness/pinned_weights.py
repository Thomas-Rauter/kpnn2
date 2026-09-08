"""Generate the pinned-weight correctness icon.

Writes ``docs/figures/correctness/pinned_weights.svg``.
Live edges are thick and pinned; the dead edge is dashed
and pinned at zero.
"""

from pathlib import Path

from _style import (
    DASH,
    DEAD,
    INK,
    LIVE,
    STROKE,
    circle,
    line,
    pin,
    write_svg,
)

_R = 5.0
_LIVE_W = 2.4
_DEAD_W = 1.4


def svg_body() -> list[str]:
    a = (40.0, 16.0)
    b = (16.0, 64.0)
    c = (64.0, 64.0)
    center = (40.0, 46.0)
    mid_ab = (
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
    )
    mid_ac = (
        (a[0] + c[0]) / 2.0,
        (a[1] + c[1]) / 2.0,
    )
    mid_bc = (
        (b[0] + c[0]) / 2.0,
        (b[1] + c[1]) / 2.0,
    )
    return [
        line(
            a[0],
            a[1],
            b[0],
            b[1],
            stroke=LIVE,
            stroke_width=_LIVE_W,
        ),
        line(
            a[0],
            a[1],
            c[0],
            c[1],
            stroke=LIVE,
            stroke_width=_LIVE_W,
        ),
        line(
            b[0],
            b[1],
            c[0],
            c[1],
            stroke=DEAD,
            stroke_width=_DEAD_W,
            dash=DASH,
        ),
        circle(a[0], a[1], _R, stroke=INK, stroke_width=STROKE),
        circle(b[0], b[1], _R, stroke=INK, stroke_width=STROKE),
        circle(c[0], c[1], _R, stroke=INK, stroke_width=STROKE),
        pin(
            mid_ab[0] - 3.2,
            mid_ab[1] - 3.2,
            toward_x=center[0],
            toward_y=center[1],
            color=LIVE,
        ),
        pin(
            mid_ac[0] + 3.2,
            mid_ac[1] - 3.2,
            toward_x=center[0],
            toward_y=center[1],
            color=LIVE,
        ),
        pin(
            mid_bc[0],
            mid_bc[1] + 5.0,
            toward_x=center[0],
            toward_y=center[1],
            color=DEAD,
        ),
    ]


def write_figure(out_path: Path | None = None) -> Path:
    """Write the pinned-weight icon SVG."""
    return write_svg(
        "pinned_weights.svg",
        svg_body(),
        out_path=out_path,
    )


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
