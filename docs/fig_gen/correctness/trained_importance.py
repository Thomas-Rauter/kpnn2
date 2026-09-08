"""Generate the trained-importance correctness icon.

Writes ``docs/figures/correctness/trained_importance.svg``.
Two matched towers share one prediction node: red is the
data-generating tower, grey is the decoy.
"""

from pathlib import Path

from _style import (
    DEAD,
    INK,
    LIVE,
    STROKE,
    circle,
    line,
    rounded_rect,
    write_svg,
)

_BOX = 12.0
_GAP = 4.0
_LEFT_X = 16.0
_RIGHT_X = 52.0
_TOP_Y = 10.0


def _tower(
    x: float,
    *,
    fill: str,
) -> list[str]:
    parts: list[str] = []
    centers: list[tuple[float, float]] = []
    for index in range(3):
        y = _TOP_Y + index * (_BOX + _GAP)
        parts.append(
            rounded_rect(
                x,
                y,
                _BOX,
                _BOX,
                fill=fill,
                stroke=INK,
                radius=1.8,
            )
        )
        centers.append(
            (
                x + _BOX / 2.0,
                y + _BOX / 2.0,
            )
        )
    for start, end in zip(
        centers,
        centers[1:],
    ):
        parts.append(
            line(
                start[0],
                start[1] + _BOX / 2.0,
                end[0],
                end[1] - _BOX / 2.0,
                stroke=INK,
                stroke_width=STROKE,
            )
        )
    return parts


def svg_body() -> list[str]:
    pred = (40.0, 70.0)
    left_bottom = (
        _LEFT_X + _BOX / 2.0,
        _TOP_Y + 2 * (_BOX + _GAP) + _BOX,
    )
    right_bottom = (
        _RIGHT_X + _BOX / 2.0,
        left_bottom[1],
    )
    return [
        *_tower(_LEFT_X, fill=LIVE),
        *_tower(_RIGHT_X, fill=DEAD),
        line(
            left_bottom[0],
            left_bottom[1],
            pred[0],
            pred[1] - 5.0,
            stroke=INK,
            stroke_width=STROKE,
        ),
        line(
            right_bottom[0],
            right_bottom[1],
            pred[0],
            pred[1] - 5.0,
            stroke=INK,
            stroke_width=STROKE,
        ),
        circle(
            pred[0],
            pred[1],
            5.0,
            stroke=INK,
            stroke_width=STROKE,
        ),
    ]


def write_figure(out_path: Path | None = None) -> Path:
    """Write the trained-importance icon SVG."""
    return write_svg(
        "trained_importance.svg",
        svg_body(),
        out_path=out_path,
    )


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
