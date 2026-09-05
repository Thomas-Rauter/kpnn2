from pathlib import Path

import kpnn2

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE = _REPO_ROOT / "docs" / "reference"

_PUBLIC_NAMES = [
    "parse_layered",
    "parse_adjacency",
    "LayeredSpec",
    "Hop",
    "Skip",
    "AdjacencySpec",
    "MaskedLinear",
    "PackedLinear",
    "PackedMultiheadAttention",
    "gather_hop_inputs",
    "align_inputs",
    "map_node_attributions",
    "Kpnn2Error",
    "__version__",
]

_COMPILER_LEFTOVERS = [
    "compile_graph",
    "customize_model",
    "interpret_model",
    "align_features_to_input_nodes",
    "edge_weights",
    "CompileArtifact",
    "ConstrainedMaskedLinear",
    "SkipAdd",
]


def test_public_api_exports_match_frozen_set():
    assert kpnn2.__all__ == _PUBLIC_NAMES
    for name in _PUBLIC_NAMES:
        assert hasattr(
            kpnn2,
            name,
        )


def test_compiler_symbols_are_not_exported():
    public = set(kpnn2.__all__)
    for name in _COMPILER_LEFTOVERS:
        assert name not in public
        assert not hasattr(
            kpnn2,
            name,
        )


def _reference_filename(name: str) -> str:
    if name == "__version__":
        return "version.md"
    return f"{name}.md"


def test_each_public_name_has_a_reference_page():
    leftover = {path.name for path in _REFERENCE.glob("*.md")}
    leftover.discard("index.md")
    for name in _PUBLIC_NAMES:
        filename = _reference_filename(name)
        path = _REFERENCE / filename
        leftover.discard(filename)
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert f"::: kpnn2.{name}" in text
    assert leftover == set()
