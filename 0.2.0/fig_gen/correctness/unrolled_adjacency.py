"""Generate the unrolled-adjacency correctness icon.

Writes ``docs/figures/correctness/unrolled_adjacency.svg``.
The first hop around the cycle is grey and does not reach the
output; the wrap that does is red.
"""

from pathlib import Path

from _style import (
    DEAD,
    INK,
    LIVE,
    STROKE,
    circle,
    line,
    path,
    write_svg,
)

_R = 5.0
_LIVE_W = 2.4
_DEAD_W = 1.4


def svg_body() -> list[str]:
    inp = (16.0, 40.0)
    upper = (38.0, 20.0)
    lower = (38.0, 60.0)
    pred = (66.0, 40.0)
    return [
        line(
            inp[0] + _R,
            inp[1],
            upper[0] - _R * 0.4,
            upper[1] + _R * 0.7,
            stroke=DEAD,
            stroke_width=_DEAD_W,
        ),
        line(
            upper[0],
            upper[1] + _R,
            lower[0],
            lower[1] - _R,
            stroke=DEAD,
            stroke_width=_DEAD_W,
        ),
        path(
            f"M {lower[0] + _R * 0.6:g},{lower[1] - _R * 0.7:g} "
            f"Q 52,40 {upper[0] + _R * 0.6:g},"
            f"{upper[1] + _R * 0.7:g}",
            stroke=LIVE,
            stroke_width=_LIVE_W,
        ),
        line(
            lower[0] + _R * 0.7,
            lower[1] - _R * 0.4,
            pred[0] - _R,
            pred[1] + _R * 0.3,
            stroke=LIVE,
            stroke_width=_LIVE_W,
        ),
        circle(inp[0], inp[1], _R, stroke=INK, stroke_width=STROKE),
        circle(
            upper[0],
            upper[1],
            _R,
            stroke=INK,
            stroke_width=STROKE,
        ),
        circle(
            lower[0],
            lower[1],
            _R,
            stroke=INK,
            stroke_width=STROKE,
        ),
        circle(
            pred[0],
            pred[1],
            _R,
            stroke=INK,
            stroke_width=STROKE,
        ),
    ]


def write_figure(out_path: Path | None = None) -> Path:
    """Write the unrolled-adjacency icon SVG."""
    return write_svg(
        "unrolled_adjacency.svg",
        svg_body(),
        out_path=out_path,
    )


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
