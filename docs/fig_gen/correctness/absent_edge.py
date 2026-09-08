"""Generate the absent-edge correctness icon.

Writes ``docs/figures/correctness/absent_edge.svg``.
A complete red path above; the grey path has a gap and an X
where the edge is missing.
"""

from pathlib import Path

from _style import (
    DASH,
    DEAD,
    INK,
    LIVE,
    STROKE,
    circle,
    cross,
    line,
    write_svg,
)

_R = 5.0
_XS = (16.0, 40.0, 64.0)
_TOP_Y = 26.0
_BOT_Y = 56.0


def _row(
    y: float,
    *,
    fill: str,
    edge: str,
    connected: tuple[bool, bool],
) -> list[str]:
    parts: list[str] = []
    for index, linked in enumerate(connected):
        x1 = _XS[index]
        x2 = _XS[index + 1]
        if linked:
            parts.append(
                line(
                    x1 + _R,
                    y,
                    x2 - _R,
                    y,
                    stroke=edge,
                    stroke_width=2.0 if fill == LIVE else 1.4,
                )
            )
        else:
            parts.append(
                line(
                    x1 + _R,
                    y,
                    x2 - _R,
                    y,
                    stroke=DEAD,
                    stroke_width=1.4,
                    dash=DASH,
                )
            )
            parts.append(
                cross(
                    (x1 + x2) / 2.0,
                    y,
                    7.0,
                    stroke=INK,
                    stroke_width=STROKE,
                )
            )
    for x in _XS:
        parts.append(
            circle(
                x,
                y,
                _R,
                fill=fill,
                stroke=INK,
                stroke_width=STROKE,
            )
        )
    return parts


def svg_body() -> list[str]:
    return [
        *_row(
            _TOP_Y,
            fill=LIVE,
            edge=LIVE,
            connected=(True, True),
        ),
        *_row(
            _BOT_Y,
            fill=DEAD,
            edge=DEAD,
            connected=(True, False),
        ),
    ]


def write_figure(out_path: Path | None = None) -> Path:
    """Write the absent-edge icon SVG."""
    return write_svg(
        "absent_edge.svg",
        svg_body(),
        out_path=out_path,
    )


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
