"""Generate the negative-controls correctness icon.

Writes ``docs/figures/correctness/negative_controls.svg``.
A glass of water (placebo) is not a drug bottle.
"""

from pathlib import Path

from _style import (
    INK,
    LIVE,
    STROKE,
    WATER,
    line,
    path,
    polygon,
    rounded_rect,
    write_svg,
)


def svg_body() -> list[str]:
    water = polygon(
        [
            (17.5, 34.0),
            (28.5, 34.0),
            (27.6, 56.0),
            (18.4, 56.0),
        ],
        fill=WATER,
    )
    glass = path(
        "M15 20 L31 20 L28.2 58 L17.8 58 Z",
        stroke=INK,
        stroke_width=STROKE,
        linejoin="round",
        linecap="round",
    )
    ne_bar_1 = line(
        35.0,
        37.0,
        45.0,
        37.0,
        stroke=INK,
        stroke_width=1.8,
    )
    ne_bar_2 = line(
        35.0,
        43.0,
        45.0,
        43.0,
        stroke=INK,
        stroke_width=1.8,
    )
    ne_slash = line(
        36.0,
        48.0,
        44.0,
        32.0,
        stroke=INK,
        stroke_width=1.8,
    )
    cap = rounded_rect(
        54.0,
        16.0,
        10.0,
        10.0,
        fill="none",
        stroke=INK,
        radius=1.2,
    )
    body = rounded_rect(
        50.0,
        24.0,
        18.0,
        38.0,
        fill=LIVE,
        stroke=INK,
        radius=3.0,
    )
    return [
        water,
        glass,
        ne_bar_1,
        ne_bar_2,
        ne_slash,
        body,
        cap,
    ]


def write_figure(out_path: Path | None = None) -> Path:
    """Write the negative-controls icon SVG."""
    return write_svg(
        "negative_controls.svg",
        svg_body(),
        out_path=out_path,
    )


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
