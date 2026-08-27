import ast
import hashlib
import inspect
import json

import pandas as pd
import pytest

import kpnn2
import kpnn2._adjacency_spec as adjacency_spec_mod
import kpnn2._serialize as serialize_mod
import kpnn2._spec as spec_mod
from kpnn2 import (
    AdjacencySpec,
    Kpnn2Error,
    LayeredSpec,
    parse_adjacency,
    parse_layered,
)
from kpnn2._serialize import canonical_edges
from tests.module.test_spec_edgelist import (
    _assert_adjacency_structure,
    _assert_layered_structure,
    _cycle_edgelist,
    _skip_edgelist,
    _unsorted_skip_edgelist,
)

_PRIVATE_NAMES = (
    "to_dict",
    "from_dict",
    "fingerprint",
    "spec_from_dict",
)


def _chain_edgelist():
    return pd.DataFrame(
        {
            "source": ["A", "H"],
            "target": ["H", "C"],
        }
    )


def _sorted_skip_edgelist():
    return (
        _unsorted_skip_edgelist()
        .sort_values(
            ["source", "target"],
        )
        .reset_index(drop=True)
    )


def _top_level_imported_modules(module):
    tree = ast.parse(inspect.getsource(module))
    imported = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    return imported


def _expected_fingerprint(spec):
    payload = json.dumps(
        spec.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_dict_helpers_are_not_public_names():
    for name in _PRIVATE_NAMES:
        assert name not in kpnn2.__all__
        assert not hasattr(
            kpnn2,
            name,
        )
    for spec_type in (LayeredSpec, AdjacencySpec):
        assert hasattr(
            spec_type,
            "to_dict",
        )
        assert hasattr(
            spec_type,
            "from_dict",
        )
        assert hasattr(
            spec_type,
            "fingerprint",
        )


def test_layered_round_trip_through_dict():
    spec = parse_layered(_skip_edgelist())
    payload = spec.to_dict()

    assert set(payload) == {
        "kpnn2_spec",
        "layout",
        "edges",
    }
    assert payload["kpnn2_spec"] == 1
    assert type(payload["kpnn2_spec"]) is int
    assert payload["layout"] == "layered"
    assert payload["edges"] == [list(pair) for pair in canonical_edges(spec)]
    assert all(isinstance(pair, list) for pair in payload["edges"])

    roundtrip = LayeredSpec.from_dict(payload)
    _assert_layered_structure(
        spec,
        roundtrip,
    )


def test_adjacency_round_trip_through_dict():
    spec = parse_adjacency(_cycle_edgelist())
    payload = spec.to_dict()

    assert payload["layout"] == "adjacency"
    assert payload["kpnn2_spec"] == 1
    assert payload["edges"] == [list(pair) for pair in canonical_edges(spec)]
    assert all(isinstance(pair, list) for pair in payload["edges"])

    roundtrip = AdjacencySpec.from_dict(payload)
    _assert_adjacency_structure(
        spec,
        roundtrip,
    )


def test_to_dict_returns_a_new_dict():
    spec = parse_layered(_chain_edgelist())
    first = spec.to_dict()
    second = spec.to_dict()

    assert first is not second
    first["layout"] = "adjacency"
    first["edges"].clear()
    assert spec.to_dict()["layout"] == "layered"
    assert spec.to_dict()["edges"] == [
        ["A", "H"],
        ["H", "C"],
    ]


def test_fingerprint_stable_across_unsorted_vs_sorted_rows():
    unsorted = parse_layered(_unsorted_skip_edgelist())
    sorted_copy = parse_layered(_sorted_skip_edgelist())

    assert unsorted.fingerprint == sorted_copy.fingerprint


def test_same_dag_layered_vs_adjacency_fingerprints_differ():
    edgelist = _skip_edgelist()
    layered = parse_layered(edgelist)
    adjacency = parse_adjacency(edgelist)

    assert layered.to_dict()["edges"] == adjacency.to_dict()["edges"]
    assert layered.to_dict()["layout"] != adjacency.to_dict()["layout"]
    assert layered.fingerprint != adjacency.fingerprint


def test_fingerprint_matches_canonical_json_sha256():
    spec = parse_layered(_chain_edgelist())
    digest = spec.fingerprint

    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(char in "0123456789abcdef" for char in digest)
    assert digest == _expected_fingerprint(spec)


def test_fingerprint_equals_reparse_of_edgelist():
    spec = parse_layered(_skip_edgelist())

    assert spec.fingerprint == parse_layered(spec.to_edgelist()).fingerprint


def test_one_edge_rewire_changes_fingerprint():
    original = parse_layered(_chain_edgelist())
    rewired = parse_layered(
        pd.DataFrame(
            {
                "source": ["A", "H"],
                "target": ["H", "D"],
            }
        )
    )

    assert original.fingerprint != rewired.fingerprint


def test_add_remove_rename_changes_fingerprint():
    base = parse_layered(_chain_edgelist())
    added = parse_layered(_skip_edgelist())
    removed = parse_layered(
        pd.DataFrame(
            {
                "source": ["A"],
                "target": ["H"],
            }
        )
    )
    renamed = parse_layered(
        pd.DataFrame(
            {
                "source": ["A", "H"],
                "target": ["H", "D"],
            }
        )
    )

    digests = {
        base.fingerprint,
        added.fingerprint,
        removed.fingerprint,
        renamed.fingerprint,
    }
    assert len(digests) == 4


def test_layered_from_dict_rejects_adjacency_layout():
    payload = parse_adjacency(_skip_edgelist()).to_dict()

    with pytest.raises(
        Kpnn2Error,
        match="LayeredSpec.from_dict received layout 'adjacency'",
    ) as caught:
        LayeredSpec.from_dict(payload)

    message = str(caught.value)
    assert "adjacency" in message
    assert "layered" in message


def test_adjacency_from_dict_rejects_layered_layout():
    payload = parse_layered(_skip_edgelist()).to_dict()

    with pytest.raises(
        Kpnn2Error,
        match="AdjacencySpec.from_dict received layout 'layered'",
    ) as caught:
        AdjacencySpec.from_dict(payload)

    message = str(caught.value)
    assert "layered" in message
    assert "adjacency" in message


def test_from_dict_rejects_bad_version():
    payload = parse_layered(_chain_edgelist()).to_dict()
    payload["kpnn2_spec"] = 2

    with pytest.raises(
        Kpnn2Error,
        match="kpnn2_spec",
    ):
        LayeredSpec.from_dict(payload)


def test_from_dict_rejects_missing_edges():
    payload = parse_layered(_chain_edgelist()).to_dict()
    del payload["edges"]

    with pytest.raises(
        Kpnn2Error,
        match="edges",
    ):
        LayeredSpec.from_dict(payload)


def test_from_dict_rejects_non_dict_payload():
    with pytest.raises(
        Kpnn2Error,
        match="dict",
    ):
        LayeredSpec.from_dict(["A", "H"])


def test_from_dict_ignores_unknown_extra_key():
    spec = parse_layered(_skip_edgelist())
    payload = spec.to_dict()
    payload["future_field"] = {"note": "ignored"}

    roundtrip = LayeredSpec.from_dict(payload)
    _assert_layered_structure(
        spec,
        roundtrip,
    )


def test_from_dict_accepts_tuple_pairs():
    spec = parse_layered(_chain_edgelist())
    payload = spec.to_dict()
    payload["edges"] = [tuple(pair) for pair in payload["edges"]]

    roundtrip = LayeredSpec.from_dict(payload)
    _assert_layered_structure(
        spec,
        roundtrip,
    )


def test_from_dict_rejects_edges_that_are_not_pairs():
    payload = parse_layered(_chain_edgelist()).to_dict()
    payload["edges"] = "not-pairs"

    with pytest.raises(
        Kpnn2Error,
        match="sequence of",
    ):
        LayeredSpec.from_dict(payload)

    payload["edges"] = [["A"]]
    with pytest.raises(
        Kpnn2Error,
        match="pair of two nonempty names",
    ):
        LayeredSpec.from_dict(payload)

    payload["edges"] = [["A", ""]]
    with pytest.raises(
        Kpnn2Error,
        match="pair of two nonempty names",
    ):
        LayeredSpec.from_dict(payload)


def test_from_dict_rejects_missing_version_and_layout():
    payload = parse_layered(_chain_edgelist()).to_dict()
    del payload["kpnn2_spec"]
    with pytest.raises(
        Kpnn2Error,
        match="kpnn2_spec",
    ):
        LayeredSpec.from_dict(payload)

    payload = parse_layered(_chain_edgelist()).to_dict()
    del payload["layout"]
    with pytest.raises(
        Kpnn2Error,
        match="layout",
    ):
        LayeredSpec.from_dict(payload)

    payload = parse_layered(_chain_edgelist()).to_dict()
    payload["layout"] = "other"
    with pytest.raises(
        Kpnn2Error,
        match="layout",
    ):
        LayeredSpec.from_dict(payload)


def test_spec_modules_lazy_import_serialize_and_not_parse():
    assert "_serialize" not in _top_level_imported_modules(spec_mod)
    assert "_serialize" not in _top_level_imported_modules(adjacency_spec_mod)
    assert "_parse" not in _top_level_imported_modules(spec_mod)
    assert "_parse" not in _top_level_imported_modules(adjacency_spec_mod)
    assert "_parse_adjacency" not in _top_level_imported_modules(spec_mod)
    assert "_parse_adjacency" not in _top_level_imported_modules(
        adjacency_spec_mod
    )

    layered_from = inspect.getsource(LayeredSpec.from_dict)
    adjacency_from = inspect.getsource(AdjacencySpec.from_dict)
    assert "from ._serialize import layered_spec_from_dict" in layered_from
    assert "from ._serialize import adjacency_spec_from_dict" in adjacency_from
    assert "from ._parse import parse_layered" not in layered_from
    assert "from ._parse_adjacency import parse_adjacency" not in adjacency_from
    assert "Hop(" not in layered_from

    layered_to = inspect.getsource(LayeredSpec.to_dict)
    assert "from ._serialize import spec_to_dict" in layered_to
    layered_fp = inspect.getsource(LayeredSpec.fingerprint.fget)
    assert "from ._serialize import spec_fingerprint" in layered_fp


def test_from_dict_helpers_call_parsers():
    layered_src = inspect.getsource(serialize_mod.layered_spec_from_dict)
    adjacency_src = inspect.getsource(serialize_mod.adjacency_spec_from_dict)
    fingerprint_src = inspect.getsource(serialize_mod.spec_fingerprint)

    assert "from ._parse import parse_layered" in layered_src
    assert "parse_layered(table)" in layered_src
    assert "from ._parse_adjacency import parse_adjacency" in adjacency_src
    assert "parse_adjacency(table)" in adjacency_src
    assert "hashlib.sha256" in fingerprint_src
