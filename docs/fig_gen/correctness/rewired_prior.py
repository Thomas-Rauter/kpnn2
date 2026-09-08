"""Generate the rewired-prior correctness icon.

Writes ``docs/figures/correctness/rewired_prior.svg``.
The red stack is plugged into the grey decoy; the path to
prediction is an empty dashed socket.
"""

from pathlib import Path

from _style import (
    DASH,
    DEAD,
    INK,
    LIVE,
    circle,
    line,
    path,
    rounded_rect,
    write_svg,
)

_R = 5.0


def svg_body() -> list[str]:
    top = (18.0, 24.0)
    bot = (18.0, 44.0)
    decoy = (64.0, 22.0)
    pred = (64.0, 58.0)
    plug = (46.0, 22.0)
    return [
        line(
            top[0],
            top[1] + _R,
            bot[0],
            bot[1] - _R,
            stroke=LIVE,
            stroke_width=2.2,
        ),
        circle(
            top[0],
            top[1],
            _R,
            fill=LIVE,
            stroke=INK,
        ),
        circle(
            bot[0],
            bot[1],
            _R,
            fill=LIVE,
            stroke=INK,
        ),
        path(
            (
                f"M{bot[0] + _R:g} {bot[1]:g} "
                f"C32 44, 34 22, {plug[0]:g} {plug[1]:g}"
            ),
            stroke=LIVE,
            stroke_width=2.0,
        ),
        rounded_rect(
            plug[0] - 3.0,
            plug[1] - 3.5,
            8.0,
            7.0,
            fill=LIVE,
            stroke=INK,
            radius=1.2,
        ),
        line(
            plug[0] + 5.0,
            plug[1],
            decoy[0] - _R,
            decoy[1],
            stroke=LIVE,
            stroke_width=2.0,
        ),
        circle(
            decoy[0],
            decoy[1],
            _R + 1.0,
            fill=DEAD,
            stroke=INK,
        ),
        line(
            38.0,
            34.0,
            pred[0] - _R - 1.0,
            pred[1],
            stroke=DEAD,
            stroke_width=1.4,
            dash=DASH,
        ),
        circle(
            pred[0],
            pred[1],
            _R + 1.0,
            fill="none",
            stroke=INK,
        ),
    ]


def write_figure(out_path: Path | None = None) -> Path:
    """Write the rewired-prior icon SVG."""
    return write_svg(
        "rewired_prior.svg",
        svg_body(),
        out_path=out_path,
    )


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
