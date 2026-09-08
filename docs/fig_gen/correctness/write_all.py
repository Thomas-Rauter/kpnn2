"""Write every scientific-correctness icon SVG.

Run from the repository root:

    python docs/fig_gen/correctness/write_all.py
"""

import absent_edge
import live_path
import name_mapping
import negative_controls
import pinned_weights
import rewired_prior
import trained_importance


def write_all() -> list[str]:
    """Write all seven icons; return the output paths."""
    writers = (
        live_path.write_figure,
        pinned_weights.write_figure,
        trained_importance.write_figure,
        negative_controls.write_figure,
        rewired_prior.write_figure,
        absent_edge.write_figure,
        name_mapping.write_figure,
    )
    paths = [str(write()) for write in writers]
    return paths


def main() -> None:
    for path in write_all():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
