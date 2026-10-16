"""Generate the live-path correctness icon.

Writes ``docs/figures/correctness/live_path.svg``.
One complete path is red; the other branch is grey and stops.
"""

from pathlib import Path

from _style import (
    DEAD,
    INK,
    LIVE,
    STROKE,
    circle,
    line,
    write_svg,
)

_R = 5.0
_LIVE_W = 2.4
_DEAD_W = 1.4


def svg_body() -> list[str]:
    n1 = (20.0, 24.0)
    n2 = (44.0, 24.0)
    n3 = (20.0, 56.0)
    n4 = (44.0, 56.0)
    n5 = (68.0, 56.0)
    return [
        line(
            n1[0] + _R,
            n1[1],
            n2[0] - _R,
            n2[1],
            stroke=DEAD,
            stroke_width=_DEAD_W,
        ),
        line(
            n3[0] + _R,
            n3[1],
            n4[0] - _R,
            n4[1],
            stroke=LIVE,
            stroke_width=_LIVE_W,
        ),
        line(
            n4[0] + _R,
            n4[1],
            n5[0] - _R,
            n5[1],
            stroke=LIVE,
            stroke_width=_LIVE_W,
        ),
        circle(n1[0], n1[1], _R, stroke=INK, stroke_width=STROKE),
        circle(n2[0], n2[1], _R, stroke=INK, stroke_width=STROKE),
        circle(n3[0], n3[1], _R, stroke=INK, stroke_width=STROKE),
        circle(n4[0], n4[1], _R, stroke=INK, stroke_width=STROKE),
        circle(n5[0], n5[1], _R, stroke=INK, stroke_width=STROKE),
    ]


def write_figure(out_path: Path | None = None) -> Path:
    """Write the live-path icon SVG."""
    return write_svg(
        "live_path.svg",
        svg_body(),
        out_path=out_path,
    )


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
