"""Generate the Why not custom PyTorch pathway figure.

Writes ``docs/figures/custom_pytorch_pathway.svg``.
The edge list matches the homepage comparison snippets.
Dashed edges skip a layer.
"""

from collections import defaultdict
from pathlib import Path

from graphviz import Digraph

_DOCS_DIR = Path(__file__).resolve().parents[1]
_OUT_PATH = _DOCS_DIR / "figures" / "custom_pytorch_pathway.svg"
_FONT = "Liberation Sans"
_FONTSIZE = "6"

PAIRS: list[tuple[str, str]] = [
    (f"gene_s{index}", tf)
    for index in range(1, 5)
    for tf in ("tf_myc", "tf_stat")
]
PAIRS += [(f"gene_n{index}", "tf_nfkb") for index in range(1, 4)]
PAIRS += [
    ("tf_myc", "kin_mapk"),
    ("tf_stat", "kin_mapk"),
    ("tf_stat", "kin_pi3k"),
    ("tf_nfkb", "kin_ikk"),
    ("kin_mapk", "proc_prolif"),
    ("kin_pi3k", "proc_prolif"),
    ("kin_pi3k", "proc_apop"),
    ("kin_ikk", "proc_apop"),
    ("proc_prolif", "phenotype"),
    ("proc_apop", "phenotype"),
    ("gene_s1", "kin_mapk"),
    ("tf_myc", "proc_prolif"),
    ("tf_nfkb", "phenotype"),
]


def _depths(
    pairs: list[tuple[str, str]],
) -> dict[str, int]:
    parents: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for source, target in pairs:
        parents[target].add(source)
        nodes.add(source)
        nodes.add(target)
    memo: dict[str, int] = {}

    def depth(name: str) -> int:
        if name in memo:
            return memo[name]
        if not parents[name]:
            memo[name] = 0
            return 0
        memo[name] = 1 + max(depth(parent) for parent in parents[name])
        return memo[name]

    return {name: depth(name) for name in nodes}


def _layers(
    pairs: list[tuple[str, str]],
) -> list[list[str]]:
    depths = _depths(pairs)
    by_depth: dict[int, list[str]] = defaultdict(list)
    for name, depth in depths.items():
        by_depth[depth].append(name)
    return [sorted(by_depth[depth]) for depth in range(max(by_depth) + 1)]


def _pathway_graph() -> Digraph:
    graph = Digraph(format="svg")
    graph.attr(
        rankdir="LR",
        bgcolor="transparent",
        fontname=_FONT,
        fontsize=_FONTSIZE,
        pad="0.12",
        nodesep="0.18",
        ranksep="0.55",
        fontcolor="black",
        splines="true",
    )
    graph.attr(
        "graph",
        fontname=_FONT,
        fontsize=_FONTSIZE,
        fontcolor="black",
    )
    graph.attr(
        "node",
        shape="ellipse",
        fontname=_FONT,
        fontsize=_FONTSIZE,
        color="black",
        fontcolor="black",
        fillcolor="none",
        margin="0.04,0.03",
        height="0.28",
        width="0.48",
    )
    graph.attr(
        "edge",
        color="black",
        arrowsize="0.5",
        penwidth="0.8",
    )
    for layer in _layers(PAIRS):
        with graph.subgraph() as subgraph:
            subgraph.attr(rank="same")
            for name in layer:
                subgraph.node(name)
            for source, target in zip(
                layer,
                layer[1:],
            ):
                subgraph.edge(
                    source,
                    target,
                    style="invis",
                    weight="100",
                )
    depths = _depths(PAIRS)
    for source, target in PAIRS:
        is_skip = depths[target] - depths[source] > 1
        if is_skip:
            graph.edge(
                source,
                target,
                style="dashed",
            )
        else:
            graph.edge(
                source,
                target,
            )
    return graph


def write_figure(out_path: Path | None = None) -> Path:
    """Render the pathway prior and write the SVG."""
    path = _OUT_PATH if out_path is None else out_path
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = _pathway_graph().pipe().decode("utf-8")
    path.write_text(svg)
    return path


def main() -> None:
    path = write_figure()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
