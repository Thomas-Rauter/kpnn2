import kpnn2

_PUBLIC_NAMES = [
    "parse_layered",
    "LayeredSpec",
    "Skip",
    "MaskedLinear",
    "SkipAdd",
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
