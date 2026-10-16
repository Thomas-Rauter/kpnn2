"""Generate the name-mapping correctness icon.

Writes ``docs/figures/correctness/name_mapping.svg``.
Anonymous units with name tags clipped onto each column.
"""

from pathlib import Path

from _style import (
    INK,
    LIVE,
    STROKE,
    line,
    rounded_rect,
    write_svg,
)

_XS = (16.0, 34.0, 52.0)
_SQUARE = 12.0
_SQUARE_Y = 18.0
_TAG_Y = 48.0
_TAG_W = 14.0
_TAG_H = 10.0


def svg_body() -> list[str]:
    parts: list[str] = []
    for x in _XS:
        cx = x + _SQUARE / 2.0
        tag_x = cx - _TAG_W / 2.0
        parts.append(
            rounded_rect(
                x,
                _SQUARE_Y,
                _SQUARE,
                _SQUARE,
                fill="none",
                stroke=INK,
                radius=1.2,
            )
        )
        parts.append(
            line(
                cx,
                _SQUARE_Y + _SQUARE,
                cx,
                _TAG_Y,
                stroke=INK,
                stroke_width=STROKE,
            )
        )
        parts.append(
            rounded_rect(
                tag_x,
                _TAG_Y,
                _TAG_W,
                _TAG_H,
                fill="none",
                stroke=LIVE,
                radius=1.6,
                stroke_width=STROKE,
            )
        )
    return parts


def write_figure(out_path: Path | None = None) -> Path:
    """Write the name-mapping icon SVG."""
    return write_svg(
        "name_mapping.svg",
        svg_body(),
        out_path=out_path,
    )


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
