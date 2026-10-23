"""
Edgelist parsing for kpnn2.
"""

from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet

import numpy as np
import pandas as pd

from ._adjacency_spec import AdjacencySpec
from ._errors import (
    Kpnn2Error,
    caller_error,
    close_matches,
    internal_error,
    listed,
    warn_at_caller,
)
from ._layout import (
    Layout,
    NodeSlot,
    build_layout,
    concat_layouts,
    iter_block_pairs,
    unknown_name_hints,
)
from ._serialize import canonical_edges
from ._spec import Hop, LayeredSpec, Skip
from ._validate import (
    as_name_list,
    describe,
    is_integer,
    require_node_mapping,
)

_SOURCE = "source"
_TARGET = "target"
# How a source / target value becomes a node name.
_NAME = "name"
_CONVERTED = "converted"
_REJECTED = "rejected"
# Column types pandas infers when every value is already a name.
_NAME_COLUMN_TYPES = (
    "string",
    "integer",
)


def _unknown_names_error(
    argument: str,
    unknown: list[str],
    n_given: int,
    nodes: Iterable[str],
) -> Kpnn2Error:
    """
    Build the error for names in ``argument`` that are no node.

    The hints are those of ``unknown_name_hints``.
    """
    candidates, fix = unknown_name_hints(
        unknown,
        n_given,
        nodes,
        argument,
    )
    return caller_error(
        f"Unknown node name(s) in '{argument}'.",
        got=listed(repr(name) for name in unknown),
        candidates=candidates,
        fix=fix,
    )


def _flagged_rows(
    flags: Mapping[str, pd.Series],
    index: pd.Index,
) -> str:
    """
    Name, per column, how many rows are flagged and where.

    Rows are named by their label in ``index``, as the caller sees
    them, not by position. Columns with no flagged row are left
    out.
    """
    parts = []
    for column, flagged in flags.items():
        labels = index[flagged.to_numpy()].tolist()
        if labels:
            labels_str = listed(repr(label) for label in labels)
            parts.append(
                f"'{column}': {len(labels)} row(s) at index {labels_str}"
            )
    return "; ".join(parts)


def _bad_entries(bad_values: Mapping[str, object]) -> str:
    """
    Name each ``widths`` or ``ranks`` entry with a bad value.

    Entries are sorted by node name and separated by ``"; "``,
    since a described value can hold commas.
    """
    return listed(
        (
            f"{name!r}: {describe(bad_values[name])}"
            for name in sorted(bad_values)
        ),
        separator="; ",
    )


def _name_kind(value: object) -> str:
    """
    Classify one ``source`` / ``target`` value as a node name.

    A ``str`` is a name, and so is an integer, whose text is its
    ID. A float or ``bool`` is converted with ``str``, but that
    text may not be the one meant (``1.0`` becomes ``'1.0'``).
    Any other value has no text that names a node.
    """
    if isinstance(value, str) or is_integer(value):
        return _NAME
    if isinstance(value, (bool, np.bool_, float, np.floating)):
        return _CONVERTED
    return _REJECTED


def _column_name_kinds(column: pd.Series) -> pd.Series:
    """
    Return the ``_name_kind`` of every value in ``column``.

    A column pandas infers as all strings or all integers skips
    the per-value pass.
    """
    inferred = pd.api.types.infer_dtype(
        column,
        skipna=False,
    )
    if inferred in _NAME_COLUMN_TYPES:
        return pd.Series(
            _NAME,
            index=column.index,
        )
    return pd.Series(
        [_name_kind(value) for value in column.tolist()],
        index=column.index,
    )


def _first_flagged(
    edgelist: pd.DataFrame,
    flags: Mapping[str, pd.Series],
) -> tuple[str, object, object]:
    """
    Return the column, index label, and value of the first flag.

    Columns are searched in the order of ``flags``, rows in table
    order. At least one flag must be set.
    """
    for column, flagged in flags.items():
        positions = np.flatnonzero(flagged.to_numpy())
        if len(positions) > 0:
            position = int(positions[0])
            return (
                column,
                edgelist.index[position],
                edgelist[column].iloc[position],
            )
    raise internal_error("_first_flagged found no flagged row")


def _check_name_types(edgelist: pd.DataFrame) -> None:
    """
    Reject values that are not node names; warn on float or bool.

    Runs after the missing-value check, so NaN never reaches it.

    Raises
    ------
    Kpnn2Error
        If a value is not a ``str``, an integer, a float, or a
        ``bool``: a list, a tensor, bytes, or any other object.
        The message counts the rows per column, names the first
        few by index label, and describes the first value.

    Warns
    -----
    UserWarning
        If a value is a float or a ``bool``, which ``str`` turns
        into ``'1.0'`` or ``'True'``.
    """
    kinds = {
        column: _column_name_kinds(edgelist[column])
        for column in (
            _SOURCE,
            _TARGET,
        )
    }
    rejected = {column: kind == _REJECTED for column, kind in kinds.items()}
    if any(bool(flagged.any()) for flagged in rejected.values()):
        column, label, value = _first_flagged(
            edgelist,
            rejected,
        )
        rows_str = _flagged_rows(
            rejected,
            edgelist.index,
        )
        raise caller_error(
            "Edgelist 'source' and 'target' must hold node names: "
            "strings, or integers that are read through str().",
            got=f"{describe(value)} in '{column}' at index {label!r}; "
            f"{rows_str}",
            fix="Replace each with the name it stands for.",
        )
    converted = {column: kind == _CONVERTED for column, kind in kinds.items()}
    if any(bool(flagged.any()) for flagged in converted.values()):
        _, _, value = _first_flagged(
            edgelist,
            converted,
        )
        text = str(value)
        rows_str = _flagged_rows(
            converted,
            edgelist.index,
        )
        warn_at_caller(
            "Edgelist holds float or bool values in 'source' or "
            "'target', which become node names through str(): node "
            "names are text, so 1.0 and 1 name different nodes.",
            UserWarning,
            got=f"{text} became {text!r}; {rows_str}",
            fix="Convert the column to the intended strings first, for "
            "example with .astype(int).astype(str) for float IDs; the "
            "warning then stops.",
        )


def _validate_edgelist(edgelist: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a source/target edgelist and return a normalized copy.

    Node names are converted to strings: integers silently, floats
    and ``bool`` with a warning; any other value raises. Only
    ``source`` and ``target`` are kept. Extra columns are ignored.

    Parameters
    ----------
    edgelist
        Edge table with required columns ``source`` and ``target``.

    Returns
    -------
    pd.DataFrame
        Copy with string ``source`` and ``target`` columns.

    Raises
    ------
    Kpnn2Error
        If ``edgelist`` is not a DataFrame, required columns are
        missing (message lists the present columns), values are
        missing, empty, or not names (a list, a tensor, any value
        but a ``str``, integer, float, or ``bool``; message counts
        the rows per column and names the first few by index
        label), the table has no rows, two or more names differ
        only by leading or trailing whitespace (message names the
        groups, sorted), or ``(source, target)`` pairs are
        duplicated (message names the unique pairs, sorted).

    Warns
    -----
    UserWarning
        If ``source`` or ``target`` holds a float or a ``bool``,
        which becomes a name such as ``'1.0'`` or ``'True'``.
    """
    if not isinstance(edgelist, pd.DataFrame):
        raise caller_error(
            "'edgelist' must be a pandas DataFrame.",
            got=describe(edgelist),
        )

    missing_columns = [
        name for name in (_SOURCE, _TARGET) if name not in edgelist.columns
    ]
    if missing_columns:
        present = edgelist.columns.tolist()
        if present:
            present_str = listed(repr(column) for column in present)
        else:
            present_str = "none"
        raise caller_error(
            "Edgelist must contain columns 'source' and 'target'; "
            f"missing: {listed(repr(name) for name in missing_columns)}.",
            got=f"columns {present_str}",
            candidates=close_matches(
                [
                    column
                    for column in present
                    if isinstance(column, str)
                    and column not in (_SOURCE, _TARGET)
                ],
                missing_columns,
            ),
        )

    if len(edgelist) == 0:
        raise caller_error("Edgelist must contain at least one edge.")

    source = edgelist[_SOURCE]
    target = edgelist[_TARGET]
    if source.isna().any() or target.isna().any():
        rows_str = _flagged_rows(
            {
                _SOURCE: source.isna(),
                _TARGET: target.isna(),
            },
            edgelist.index,
        )
        raise caller_error(
            "Edgelist contains missing values in 'source' or 'target'.",
            got=rows_str,
        )
    _check_name_types(edgelist)

    normalized = pd.DataFrame(
        {
            _SOURCE: source.map(str),
            _TARGET: target.map(str),
        }
    )

    empty_source = normalized[_SOURCE] == ""
    empty_target = normalized[_TARGET] == ""
    if empty_source.any() or empty_target.any():
        rows_str = _flagged_rows(
            {
                _SOURCE: empty_source,
                _TARGET: empty_target,
            },
            edgelist.index,
        )
        raise caller_error(
            "Edgelist contains empty node names in 'source' or 'target'.",
            got=rows_str,
        )

    names = set(normalized[_SOURCE]) | set(normalized[_TARGET])
    names_by_stripped: dict[str, list[str]] = {}
    for name in names:
        names_by_stripped.setdefault(
            name.strip(),
            [],
        ).append(name)
    collisions = [
        sorted(names_by_stripped[stripped])
        for stripped in sorted(names_by_stripped)
        if len(names_by_stripped[stripped]) > 1
    ]
    if collisions:
        groups_str = listed(
            f"({', '.join(repr(name) for name in group)})"
            for group in collisions
        )
        raise caller_error(
            "Edgelist contains node names that differ only by leading "
            "or trailing whitespace.",
            got=groups_str,
            fix="Strip the names in both 'source' and 'target' before "
            "parsing (for example edgelist[column].astype(str).str.strip() "
            "for each column) or give them distinct names.",
        )

    n_duplicates = int(normalized.duplicated().sum())
    if n_duplicates > 0:
        duplicated_rows = normalized[normalized.duplicated(keep=False)]
        unique_pairs = duplicated_rows.drop_duplicates().sort_values(
            by=[_SOURCE, _TARGET],
        )
        pair_labels = [
            f"{source} -> {target}"
            for source, target in zip(
                unique_pairs[_SOURCE],
                unique_pairs[_TARGET],
            )
        ]
        raise caller_error(
            f"Edgelist contains {n_duplicates} duplicate edge(s): at most "
            "one connection is allowed for each source-target pair.",
            got=listed(pair_labels),
        )

    return pd.DataFrame(
        {
            _SOURCE: normalized[_SOURCE].tolist(),
            _TARGET: normalized[_TARGET].tolist(),
        }
    )


def _reject_self_loops(edgelist: pd.DataFrame) -> None:
    """
    Raise if any edge has the same source and target.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.

    Raises
    ------
    Kpnn2Error
        If one or more self-loops are present. The message names
        the unique self-loop nodes, sorted alphabetically.
    """
    self_loops = edgelist[_SOURCE] == edgelist[_TARGET]
    n_self_loops = int(self_loops.sum())
    if n_self_loops > 0:
        loop_nodes = sorted(set(edgelist[_SOURCE][self_loops].tolist()))
        raise caller_error(
            f"Edgelist contains {n_self_loops} self-loop(s), and "
            "parse_layered() supports only DAGs.",
            got=f"self-loop(s) on {listed(loop_nodes)}",
            fix="Use parse_adjacency(), which allows self-loops and "
            "cycles, or drop those rows.",
        )


def _build_adjacency(
    edgelist: pd.DataFrame,
) -> tuple[
    set[str],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, int],
    dict[str, int],
]:
    """
    Build parent/child adjacency and degree maps.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.

    Returns
    -------
    tuple
        ``nodes``, ``children``, ``parents``, ``in_degree``,
        ``out_degree``.
    """
    sources = edgelist[_SOURCE].tolist()
    targets = edgelist[_TARGET].tolist()
    nodes = set(sources) | set(targets)
    children: dict[str, list[str]] = {node: [] for node in nodes}
    parents: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in zip(
        sources,
        targets,
    ):
        children[source].append(target)
        parents[target].append(source)
    in_degree = {node: len(parents[node]) for node in nodes}
    out_degree = {node: len(children[node]) for node in nodes}
    return (
        nodes,
        children,
        parents,
        in_degree,
        out_degree,
    )


def _node_sets(
    nodes: set[str],
    in_degree: dict[str, int],
    out_degree: dict[str, int],
) -> tuple[list[str], list[str], list[str]]:
    """
    Infer sorted input, output, and hidden node lists.

    Parameters
    ----------
    nodes
        All node names in the edgelist.
    in_degree
        Incoming edge counts.
    out_degree
        Outgoing edge counts.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``input_nodes``, ``output_nodes``, ``hidden_nodes``.

    Raises
    ------
    Kpnn2Error
        If there is no in-degree-0 node or no out-degree-0 node.
    """
    input_nodes = sorted(node for node in nodes if in_degree[node] == 0)
    output_nodes = sorted(node for node in nodes if out_degree[node] == 0)
    if not input_nodes and not output_nodes:
        raise caller_error(
            "Edgelist must contain at least one input node (in-degree 0) "
            "and at least one output node (out-degree 0): every node has "
            "an incoming and an outgoing edge, so the graph has no entry "
            "or exit point.",
            fix="Add input nodes that appear only in 'source', and output "
            "nodes that appear only in 'target'.",
        )
    if not input_nodes:
        raise caller_error(
            "Edgelist must contain at least one input node (in-degree 0): "
            "every node has an incoming edge, so the graph has no entry "
            "point.",
            fix="Add input nodes that appear only in 'source'.",
        )
    if not output_nodes:
        raise caller_error(
            "Edgelist must contain at least one output node (out-degree "
            "0): every node has an outgoing edge, so the graph has no "
            "exit point.",
            fix="Add output nodes that appear only in 'target'.",
        )
    input_set = set(input_nodes)
    output_set = set(output_nodes)
    hidden_nodes = sorted(
        node
        for node in nodes
        if node not in input_set and node not in output_set
    )
    return (
        input_nodes,
        output_nodes,
        hidden_nodes,
    )


def _rank_layers(
    nodes: set[str],
    children: dict[str, list[str]],
    parents: dict[str, list[str]],
    in_degree: dict[str, int],
) -> list[list[str]]:
    """
    Assign Kahn depths and return alphabetically sorted layers.

    ``depth(input) = 0``. For every other node,
    ``depth = 1 + max(parent depths)``.

    Parameters
    ----------
    nodes
        All node names.
    children
        Adjacency list from source to targets.
    parents
        Adjacency list from target to sources.
    in_degree
        Incoming edge counts (not mutated).

    Returns
    -------
    list[list[str]]
        ``layer_nodes[d]`` is sorted names with depth ``d``.

    Raises
    ------
    Kpnn2Error
        If a cycle leaves some nodes unranked. The message names
        the leftover nodes (``nodes`` minus keys of ``depths``),
        sorted alphabetically.
    """
    remaining = dict(in_degree)
    depths: dict[str, int] = {}
    ready: deque[str] = deque(node for node in nodes if remaining[node] == 0)
    while ready:
        node = ready.popleft()
        if not parents[node]:
            depths[node] = 0
        else:
            depths[node] = 1 + max(depths[parent] for parent in parents[node])
        for child in children[node]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)

    if len(depths) != len(nodes):
        unranked = sorted(nodes - depths.keys())
        raise caller_error(
            "Edgelist contains a cycle, and parse_layered() supports "
            "only DAGs.",
            got=f"nodes on or downstream of a cycle: {listed(unranked)}",
            fix="Use parse_adjacency(), which allows cycles, or remove "
            "an edge from each cycle.",
        )

    return _layers_from_depths(depths)


def _layers_from_depths(depths: dict[str, int]) -> list[list[str]]:
    """
    Group nodes into alphabetically sorted layers by depth.

    ``layer_nodes[d]`` is every name with ``depth == d``. Depths
    must be a compact ``0 .. L-1`` range; compaction of user
    ranks happens before this helper is called.
    """
    n_layers = max(depths.values()) + 1
    layer_nodes: list[list[str]] = [[] for _ in range(n_layers)]
    for node, depth in depths.items():
        layer_nodes[depth].append(node)
    for layer in layer_nodes:
        layer.sort()
    return layer_nodes


def _layers_from_user_ranks(
    ranks: Mapping[str, int],
    nodes: set[str],
    input_nodes: list[str],
    edgelist: pd.DataFrame,
) -> list[list[str]]:
    """
    Compact user ranks into dense layers ``0 .. L-1``.

    Every graph node must be present. Unknown names, ``bool``
    values, negatives, and non-integers raise; numpy integers
    are kept as ``int``. Inputs must share the minimum user
    rank, and no non-input may sit there. Every named edge must
    be strictly forward after compacting.
    """
    require_node_mapping(
        ranks,
        "ranks",
    )
    bad_values = {
        str(key): value
        for key, value in ranks.items()
        if not is_integer(value) or value < 0
    }
    if bad_values:
        raise caller_error(
            "'ranks' values must be non-negative ints.",
            got=_bad_entries(bad_values),
        )
    requested = {str(key): int(value) for key, value in ranks.items()}
    unknown = sorted(set(requested) - nodes)
    if unknown:
        raise _unknown_names_error(
            "ranks",
            unknown,
            len(requested),
            nodes,
        )
    missing = sorted(nodes - set(requested))
    if missing:
        raise caller_error(
            "Missing node name(s) in 'ranks': every node needs a rank.",
            got=f"no rank for {listed(missing)}",
        )

    used = sorted(set(requested.values()))
    rank_to_layer = {rank: index for index, rank in enumerate(used)}
    depths = {node: rank_to_layer[requested[node]] for node in nodes}

    min_rank = used[0]
    input_set = set(input_nodes)
    inputs_off_min = sorted(
        node for node in input_nodes if requested[node] != min_rank
    )
    non_inputs_at_min = sorted(
        node
        for node in nodes
        if node not in input_set and requested[node] == min_rank
    )
    if inputs_off_min or non_inputs_at_min:
        offenders = []
        if inputs_off_min:
            offenders.append(
                "input node(s) not at the minimum rank: "
                f"{listed(inputs_off_min)}"
            )
        if non_inputs_at_min:
            offenders.append(
                "non-input node(s) at the minimum rank: "
                f"{listed(non_inputs_at_min)}"
            )
        raise caller_error(
            "Input nodes (in-degree 0) must share the minimum rank in "
            "'ranks', and no other node may use it.",
            got="; ".join(offenders),
        )

    bad_pairs: list[tuple[str, str]] = []
    for source, target in zip(
        edgelist[_SOURCE].tolist(),
        edgelist[_TARGET].tolist(),
        strict=True,
    ):
        if depths[source] >= depths[target]:
            bad_pairs.append(
                (
                    source,
                    target,
                )
            )
    if bad_pairs:
        unique_pairs = sorted(set(bad_pairs))
        # The ranks the caller passed, not the compacted layers.
        pair_labels = [
            f"{source} -> {target} "
            f"(rank {requested[source]} -> {requested[target]})"
            for source, target in unique_pairs
        ]
        raise caller_error(
            "Every named edge must go strictly forward under 'ranks'.",
            got=f"non-forward edge(s) {listed(pair_labels)}",
        )

    return _layers_from_depths(depths)


def _normalize_widths(
    widths: Mapping[str, int] | None,
    nodes: set[str],
    input_nodes: list[str],
) -> dict[str, int]:
    """
    Map each graph node to a positive unit count.

    ``None`` or omitted keys are width 1. Keys are matched after
    ``str(...)``. Unknown names and non-positive or non-integer
    values raise ``Kpnn2Error``. ``bool`` is rejected. numpy
    integers are kept as ``int``. An input node wider than 1
    raises: each input reads one feature column, so its extra
    units would only hold copies.
    """
    if widths is None:
        return {node: 1 for node in nodes}
    require_node_mapping(
        widths,
        "widths",
    )
    bad_values = {
        str(key): value
        for key, value in widths.items()
        if not is_integer(value) or value < 1
    }
    if bad_values:
        raise caller_error(
            "'widths' values must be positive ints.",
            got=_bad_entries(bad_values),
        )
    requested = {str(key): int(value) for key, value in widths.items()}
    unknown = sorted(set(requested) - nodes)
    if unknown:
        raise _unknown_names_error(
            "widths",
            unknown,
            len(requested),
            nodes,
        )
    wide_inputs = sorted(
        name for name in input_nodes if requested.get(name, 1) != 1
    )
    if wide_inputs:
        raise caller_error(
            "Input nodes (in-degree 0) must have width 1 in 'widths', "
            "since each reads one feature column.",
            got=listed(f"{name}={requested[name]}" for name in wide_inputs),
            fix="Leave input nodes out of 'widths'; to feed several "
            "measurements, give each its own input node.",
        )
    return {node: requested.get(node, 1) for node in nodes}


def _layer_layouts(
    layer_nodes: list[list[str]],
    width_of: dict[str, int],
) -> list[Layout]:
    """
    Place each layer's nodes on that layer's unit axis.

    One layout per depth. ``layout.n_units`` is the sum of the
    node widths at that depth.
    """
    return [
        build_layout(
            names,
            [width_of[name] for name in names],
        )
        for names in layer_nodes
    ]


def _node_placement(
    layouts: list[Layout],
) -> dict[str, tuple[int, NodeSlot]]:
    """
    Map each node name to its ``(depth, slot)``.
    """
    placement: dict[str, tuple[int, NodeSlot]] = {}
    for depth, layout in enumerate(layouts):
        for slot in layout.slots:
            placement[slot.name] = (depth, slot)
    return placement


def _parent_layers(
    edgelist: pd.DataFrame,
    placement: dict[str, tuple[int, NodeSlot]],
    n_layers: int,
) -> list[set[int]]:
    """
    Collect, per depth, the depths that feed it.

    Entry ``d`` is every depth with at least one edge into depth
    ``d``. Under longest-path ranking ``d - 1`` is in it for
    every ``d > 0``. With user ranks a hop may omit ``d - 1``
    when every parent is a skip. Entry 0 is empty.
    """
    parents: list[set[int]] = [set() for _ in range(n_layers)]
    for source, target in zip(
        edgelist[_SOURCE].tolist(),
        edgelist[_TARGET].tolist(),
    ):
        source_layer, _ = placement[source]
        target_layer, _ = placement[target]
        parents[target_layer].add(source_layer)
    return parents


def _build_hops(
    edgelist: pd.DataFrame,
    layouts: list[Layout],
    placement: dict[str, tuple[int, NodeSlot]],
) -> list[Hop]:
    """
    Build one incoming packed hop per depth after the first.

    ``hops[i]`` targets depth ``i + 1`` and its packed indices
    hold every edge entering that depth, adjacent or skip. The
    column axis is the source depths concatenated in ascending
    order. Each named edge expands into every unit pair of the
    ``(target.width, source.width)`` block. At width 1 that is
    one pair per original edge.

    Only depths that really feed the target become columns, so a
    graph without skips gives exactly one source depth per hop.
    Does not allocate an ``(out, in)`` tensor.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.
    layouts
        Unit placement per depth.
    placement
        Node name to ``(depth, slot)``.

    Returns
    -------
    list[Hop]
        One hop per depth from 1 upwards, in depth order.
    """
    n_layers = len(layouts)
    parents = _parent_layers(
        edgelist,
        placement,
        n_layers,
    )

    source_layers: dict[int, tuple[int, ...]] = {}
    source_layouts: dict[int, Layout] = {}
    packed: dict[int, list[tuple[str, str, NodeSlot, NodeSlot]]] = {}
    for target_layer in range(1, n_layers):
        ordered = tuple(sorted(parents[target_layer]))
        source_layout = concat_layouts(
            [layouts[layer] for layer in ordered],
        )
        source_layers[target_layer] = ordered
        source_layouts[target_layer] = source_layout
        packed[target_layer] = []

    for source, target in zip(
        edgelist[_SOURCE].tolist(),
        edgelist[_TARGET].tolist(),
        strict=True,
    ):
        target_layer, target_slot = placement[target]
        source_slot = source_layouts[target_layer].slot(source)
        packed[target_layer].append(
            (
                source,
                target,
                source_slot,
                target_slot,
            )
        )

    hops: list[Hop] = []
    for target_layer in range(1, n_layers):
        ordered = source_layers[target_layer]
        rows = sorted(
            packed[target_layer],
            key=lambda row: (row[0], row[1]),
        )
        source_index: list[int] = []
        target_index: list[int] = []
        for _, _, source_slot, target_slot in rows:
            for source_unit, target_unit in iter_block_pairs(
                source_slot,
                target_slot,
            ):
                source_index.append(source_unit)
                target_index.append(target_unit)
        hops.append(
            Hop(
                target_layer=target_layer,
                source_layers=ordered,
                source_dims=tuple(layouts[layer].n_units for layer in ordered),
                source_nodes=source_layouts[target_layer].names,
                target_dim=layouts[target_layer].n_units,
                source_index=tuple(source_index),
                target_index=tuple(target_index),
            )
        )
    return hops


def _build_skips(
    edgelist: pd.DataFrame,
    placement: dict[str, tuple[int, NodeSlot]],
) -> list[Skip]:
    """
    Collect original edges with depth gap greater than 1.

    These records are metadata: the named edges themselves are
    already unit-pair blocks in the target depth's hop. Adjacent
    edges (gap exactly 1) are omitted, since nothing
    distinguishes them. Each recorded index is the first unit
    its node owns, which equals the node's ordinal in
    ``layer_nodes`` while every node is one unit wide.

    Parameters
    ----------
    edgelist
        Normalized edgelist with string ``source`` and ``target``.
    placement
        Node name to ``(depth, slot)``.

    Returns
    -------
    list[Skip]
        One record per skip edge, in edgelist order.
    """
    skips: list[Skip] = []
    sources = edgelist[_SOURCE].tolist()
    targets = edgelist[_TARGET].tolist()
    for source, target in zip(
        sources,
        targets,
    ):
        source_layer, source_slot = placement[source]
        target_layer, target_slot = placement[target]
        gap = target_layer - source_layer
        if gap > 1:
            skips.append(
                Skip(
                    source=source,
                    target=target,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    source_in_layer=source_slot.start,
                    target_in_layer=target_slot.start,
                )
            )
    return skips


def _check_edges_conserved(
    spec: LayeredSpec | AdjacencySpec,
    edgelist: pd.DataFrame,
    width_of: Mapping[str, int],
    packed: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
) -> None:
    """
    Raise unless every named edge owns its full unit block once.

    Three conditions together pin the packed pairs down: the
    named edges read back from the spec equal the edgelist, no
    unit pair repeats on one packed axis, and the pair count is
    the sum of ``k_source * k_target``. A block holds at most
    ``k_source * k_target`` distinct pairs, so every block is
    exactly full. Named edges are rebuilt by
    ``canonical_edges``, independently of the code that packed
    them.

    Parameters
    ----------
    spec
        Spec the parser is about to return.
    edgelist
        Normalized edgelist with string ``source`` and ``target``.
    width_of
        Width of every node.
    packed
        ``(source_index, target_index)`` per packed axis: one
        per hop on a ``LayeredSpec``, one on an
        ``AdjacencySpec``.

    Raises
    ------
    AssertionError
        From ``internal_error``, if any condition fails.
    """
    parser = (
        "parse_layered" if isinstance(spec, LayeredSpec) else "parse_adjacency"
    )
    expected = tuple(
        sorted(
            zip(
                edgelist[_SOURCE].tolist(),
                edgelist[_TARGET].tolist(),
                strict=True,
            )
        )
    )
    found = canonical_edges(spec)
    if found != expected:
        n_missing = len(set(expected) - set(found))
        n_unexpected = len(set(found) - set(expected))
        raise internal_error(
            f"{parser} packed pairs encode {len(found)} named edges, "
            f"not the {len(expected)} of the edgelist "
            f"({n_missing} missing, {n_unexpected} unexpected)"
        )
    n_expected = sum(
        width_of[source] * width_of[target] for source, target in expected
    )
    n_pairs = 0
    for source_index, target_index in packed:
        unique = set(
            zip(
                source_index,
                target_index,
                strict=True,
            )
        )
        n_repeats = len(source_index) - len(unique)
        if n_repeats:
            raise internal_error(
                f"{parser} packed {n_repeats} unit pair(s) more than once"
            )
        n_pairs += len(unique)
    if n_pairs != n_expected:
        raise internal_error(
            f"{parser} packed {n_pairs} unit pairs, expected "
            f"{n_expected} from the edgelist and widths"
        )


def _check_layer_axes(spec: LayeredSpec) -> None:
    """
    Raise unless the fields that describe each axis agree.

    ``align_inputs``, ``map_node_attributions``, and
    ``gather_hop_inputs`` build an axis from ``layer_nodes`` and
    ``layer_widths`` and trust ``input_nodes``, ``layer_dims``,
    and each hop's ``source_nodes`` and ``source_dims`` to match
    it. The parser stores those fields separately, so they are
    checked once here, where every spec is made, instead of on
    each call that reads them.

    Raises
    ------
    AssertionError
        From ``internal_error``, if a field disagrees.
    """
    if spec.input_nodes != spec.layer_nodes[0]:
        raise internal_error(
            f"parse_layered input_nodes ({len(spec.input_nodes)} "
            "names) differ from layer_nodes[0] "
            f"({len(spec.layer_nodes[0])} names) in names or order"
        )
    n_layers = len(spec.layer_nodes)
    if len(spec.layer_widths) != n_layers or len(spec.layer_dims) != n_layers:
        raise internal_error(
            f"parse_layered stored {n_layers} layer_nodes, "
            f"{len(spec.layer_widths)} layer_widths, and "
            f"{len(spec.layer_dims)} layer_dims"
        )
    for layer in range(n_layers):
        n_units = build_layout(
            spec.layer_nodes[layer],
            spec.layer_widths[layer],
        ).n_units
        if n_units != spec.layer_dims[layer]:
            raise internal_error(
                f"parse_layered layer {layer} holds {n_units} unit(s) "
                f"by layer_widths, but layer_dims[{layer}] is "
                f"{spec.layer_dims[layer]}"
            )
    for hop in spec.hops:
        source_nodes = tuple(
            name
            for layer in hop.source_layers
            for name in spec.layer_nodes[layer]
        )
        if hop.source_nodes != source_nodes:
            raise internal_error(
                f"parse_layered hop into layer {hop.target_layer} lists "
                f"{len(hop.source_nodes)} source_nodes that differ from "
                f"the {len(source_nodes)} names of its source layers in "
                "names or order"
            )
        source_dims = tuple(
            spec.layer_dims[layer] for layer in hop.source_layers
        )
        if hop.source_dims != source_dims:
            raise internal_error(
                f"parse_layered hop into layer {hop.target_layer} has "
                f"source_dims {hop.source_dims}, but its source layers "
                f"are {source_dims} units wide"
            )


def _declared_outputs(outputs: object) -> list[str] | None:
    """
    Return ``outputs`` as a list of names, or ``None`` if omitted.

    The names are compared as a set, so a set is accepted and
    order does not matter. Each name goes through ``str(...)``,
    as edgelist node names do. A string, a mapping, an empty
    collection, and a repeated name raise ``Kpnn2Error``.
    """
    if outputs is None:
        return None
    if isinstance(outputs, Mapping):
        raise caller_error(
            "'outputs' must be a collection of output node names, not a "
            "mapping.",
            got=describe(outputs),
            fix="Pass the names as a list.",
        )
    names = as_name_list(
        list(outputs) if isinstance(outputs, AbstractSet) else outputs,
        "outputs",
        "output node names",
        container="collection",
    )
    if not names:
        raise caller_error(
            "'outputs' must name at least one node.",
            got=describe(outputs),
            fix="Pass the output nodes you read, or omit 'outputs'.",
        )
    counts = Counter(names)
    repeated = sorted(name for name, count in counts.items() if count > 1)
    if repeated:
        raise caller_error(
            "'outputs' names a node more than once.",
            got=listed(repeated),
        )
    return names


def _without_path_to(
    targets: Sequence[str],
    edgelist: pd.DataFrame,
) -> list[str]:
    """
    Return the nodes with no directed path to any of ``targets``.

    A node in ``targets`` reaches itself. The walk follows parents
    back from ``targets`` and visits each node once, so a cycle
    ends it. Names are sorted.
    """
    nodes, _, parents, _, _ = _build_adjacency(edgelist)
    reached = set(targets)
    queue = deque(targets)
    while queue:
        for parent in parents[queue.popleft()]:
            if parent not in reached:
                reached.add(parent)
                queue.append(parent)
    return sorted(nodes - reached)


def _confirm_outputs(
    spec: LayeredSpec | AdjacencySpec,
    declared: list[str] | None,
    *,
    widths: Mapping[str, int] | None,
    ranks: Mapping[str, int] | None,
) -> None:
    """
    Check the output nodes against ``declared``, or warn without it.

    Output nodes are inferred from out-degree 0, which a node
    whose onward rows are missing also has. ``declared`` must
    be exactly ``spec.output_nodes``. Unknown names and names of
    nodes with an outgoing edge raise first, in one message.
    Output nodes left out of ``declared`` raise next, naming
    every node with no path to a declared one: in the rows whose
    target is none of those, exactly ``declared`` have no
    outgoing edge. The ``Fix`` also says to leave those nodes
    out of ``widths`` or ``ranks`` when either names one, since
    either raises on a name that is not in the edgelist. Without
    ``declared``, more than one output node warns.
    """
    found = spec.output_nodes
    if declared is None:
        if len(found) > 1:
            warn_at_caller(
                f"The edgelist has {len(found)} output nodes, nodes with "
                "no outgoing edge; a node whose onward rows are missing "
                "from the edgelist is an output node too.",
                UserWarning,
                got=listed(found),
                fix="Pass the output nodes you read as 'outputs' to "
                "confirm them; the warning then stops, and parsing raises "
                "on any other node with no outgoing edge.",
            )
        return
    role = {name: "input" for name in spec.input_nodes}
    role.update({name: "hidden" for name in spec.hidden_nodes})
    output_set = set(found)
    unknown = sorted(
        name for name in declared if name not in role and name not in output_set
    )
    not_outputs = sorted(name for name in declared if name in role)
    if unknown or not_outputs:
        raise _wrong_outputs_error(
            unknown,
            [f"{name} ({role[name]})" for name in not_outputs],
            declared,
            found,
        )
    extra = sorted(output_set - set(declared))
    if extra:
        dead = _without_path_to(
            declared,
            spec.to_edgelist(),
        )
        dead_set = set(dead)
        naming = [
            f"'{argument}'"
            for argument, mapping in (("widths", widths), ("ranks", ranks))
            if mapping is not None
            and any(str(key) in dead_set for key in mapping)
        ]
        cut = (
            f", leave those nodes out of {' and '.join(naming)},"
            if naming
            else ""
        )
        # Every name, not listed(): the fix drops the rows of each.
        raise caller_error(
            "Node(s) with no outgoing edge that are not in 'outputs': "
            "the edgelist has no row leaving them, so they are output "
            "nodes too.",
            got=listed(extra),
            fix="Drop the edgelist rows whose target is one of the "
            "nodes with no path to a node in 'outputs' "
            f"({', '.join(dead)}){cut} and parse again, or add a node to "
            "'outputs' if you read it.",
        )


def _wrong_outputs_error(
    unknown: list[str],
    not_outputs: list[str],
    declared: list[str],
    found: Sequence[str],
) -> Kpnn2Error:
    """
    Build the error for names in ``outputs`` that are no output node.

    Both kinds of wrong name go into one message: names that are
    not in the edgelist, and nodes with an outgoing edge, given as
    ``"name (role)"``.
    """
    kinds = []
    found_parts = []
    candidates = None
    fixes = []
    if unknown:
        kinds.append("node(s) that are not in the edgelist")
        found_parts.append(
            (
                "not in the edgelist",
                listed(repr(name) for name in unknown),
            )
        )
        candidates, fix = unknown_name_hints(
            unknown,
            len(declared),
            [name for name in found if name not in declared],
            "outputs",
        )
        if fix is not None:
            fixes.append(fix)
    if not_outputs:
        kinds.append(
            "node(s) with an outgoing edge, which are not output nodes"
        )
        found_parts.append(("with an outgoing edge", listed(not_outputs)))
        fixes.append(
            "Leave nodes with an outgoing edge out of 'outputs', and read "
            "them with spec.unit_index()."
        )
    if len(found_parts) == 1:
        got = found_parts[0][1]
    else:
        got = "; ".join(f"{label}: {names}" for label, names in found_parts)
    return caller_error(
        f"'outputs' names {', and '.join(kinds)}.",
        got=got,
        candidates=candidates,
        fix=" ".join(fixes) or None,
    )


def _layered_from_edgelist(
    edgelist: pd.DataFrame,
    *,
    widths: Mapping[str, int] | None,
    ranks: Mapping[str, int] | None,
) -> LayeredSpec:
    """
    Parse ``edgelist`` into a ``LayeredSpec`` without ``outputs``.

    Runs every check of ``parse_layered`` except the output
    confirmation, which neither raises nor warns here.
    ``LayeredSpec.from_dict`` uses it: it rebuilds a spec that
    was parsed before, so several outputs are not news.
    """
    normalized = _validate_edgelist(edgelist)
    _reject_self_loops(normalized)
    (
        nodes,
        children,
        parents,
        in_degree,
        out_degree,
    ) = _build_adjacency(normalized)
    input_nodes, output_nodes, hidden_nodes = _node_sets(
        nodes,
        in_degree,
        out_degree,
    )
    layer_nodes = _rank_layers(
        nodes,
        children,
        parents,
        in_degree,
    )
    if ranks is not None:
        layer_nodes = _layers_from_user_ranks(
            ranks,
            nodes,
            input_nodes,
            normalized,
        )
    width_of = _normalize_widths(
        widths,
        nodes,
        input_nodes,
    )
    layouts = _layer_layouts(
        layer_nodes,
        width_of,
    )
    placement = _node_placement(layouts)
    layer_dims = [layout.n_units for layout in layouts]
    hops = _build_hops(
        normalized,
        layouts,
        placement,
    )
    skips = _build_skips(
        normalized,
        placement,
    )
    spec = LayeredSpec(
        input_nodes=tuple(input_nodes),
        output_nodes=tuple(output_nodes),
        hidden_nodes=tuple(hidden_nodes),
        layer_nodes=tuple(tuple(layer) for layer in layer_nodes),
        layer_dims=tuple(layer_dims),
        layer_widths=tuple(layout.widths() for layout in layouts),
        hops=tuple(hops),
        skips=tuple(skips),
    )
    _check_edges_conserved(
        spec,
        normalized,
        width_of,
        [(hop.source_index, hop.target_index) for hop in spec.hops],
    )
    _check_layer_axes(spec)
    return spec


def parse_layered(  # numpydoc ignore=PR06
    edgelist: pd.DataFrame,
    *,
    widths: Mapping[str, int] | None = None,
    ranks: Mapping[str, int] | None = None,
    outputs: Iterable[str] | None = None,
) -> LayeredSpec:
    """
    Parse a source/target edgelist into a ``LayeredSpec``.

    The parser sorts the nodes of a DAG into layers and turns the
    edges entering each layer into one packed hop, ready for one
    ``PackedLinear`` per hop. Reach for it when prior-knowledge
    edges, such as genes into pathways, should run as a
    layer-by-layer network; ``parse_adjacency`` is the
    shared-state packed alternative, which also allows cycles.
    Names sort alphabetically within a layer, and no hop mask is
    allocated.

    Two defaults shape the result, and one keyword changes each.
    Every node is a single unit, one neuron, unless ``widths``
    makes it a block of several. Every node sits one layer above
    its deepest parent, so its layer is the length of its longest
    path from the inputs, unless ``ranks`` places it at a level
    you choose.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edge table with required columns ``source`` and ``target``,
        one row per directed edge in the direction of computation.
        Node names are text: a ``str`` is kept, and an integer is
        converted with ``str(...)``, so ``1`` and ``"1"`` name the
        same node. A float or ``bool`` is converted too but warns,
        since ``1.0`` becomes ``"1.0"``; any other value raises.
        Extra columns are ignored. The frame is read, never
        modified.
    widths : mapping of str to int or numpy integer, optional
        Number of units per named node. A unit is one neuron: it
        holds one number per sample. A node of width ``k`` is a
        block of ``k`` units, so it can carry ``k`` numbers
        instead of one. An edge ``A -> B`` then connects every
        unit of ``A`` to every unit of ``B``, which is
        ``k_B * k_A`` weights instead of one. Widths change how
        many units a layer has, never which layer a node sits in.

        Unlike ``ranks``, widths change what the network can
        compute. A node of width 1 squeezes everything its
        parents send into one number, and every node it feeds
        sees that same number. So wider nodes matter where one
        number per node is too little:

        - a hidden node that combines many parents and should
          pass on more than one summary of them
        - capacity that grows with the node, such as a width set
          by its number of parents
        - an output that is several numbers, such as class
          scores or the parameters of a distribution

        The price is ``k_B * k_A`` weights per edge, and a score
        per node has to combine the scores of its units.

        Omitted names, ``None``, and an empty mapping are width 1.
        Keys are matched after ``str(...)``. Unknown names raise
        ``Kpnn2Error``. Values must be positive ints or numpy
        integers, stored as ``int``; ``bool``, ``numpy.bool_``,
        floats, ``0``, and negatives are rejected. Input nodes
        must stay width 1: each reads one feature column, so
        separate measurements belong on separate input nodes.
    ranks : mapping of str to int or numpy integer, optional
        Level per named node, in place of longest-path ranking.
        By default a node sits one layer above its deepest
        parent, so two nodes that belong on the same level can
        land in different layers only because one has a longer
        chain of parents below it. ``ranks`` says where each
        node belongs: nodes of equal rank share a layer, and a
        higher rank is a later layer. Only the order of the
        values matters. They are renumbered to layers
        ``0 .. L-1`` with no empty layers, so ranks 0, 10, 20
        give the same layers as 0, 1, 2. An edge that ends up
        spanning more than one layer is a skip edge: an ordinary
        weight in the hop into its target's layer.

        Ranks change how nodes are grouped into layers, never
        what the network computes: every node keeps the same
        parents through the same weights. So they matter only
        where something treats a layer as a group:

        - normalization across a layer's units
        - a prediction head per layer
        - a dropout rate or activation that differs by layer
        - reading all outputs from one layer tensor
        - summing attributions per layer
        - matching a layer structure given elsewhere, such as a
          published model

        ``None`` or omitted keeps longest-path ranking. Keys are
        matched after ``str(...)``. Every graph node must be
        present; unknown names raise ``Kpnn2Error``. Values are
        non-negative ints or numpy integers; ``bool``,
        ``numpy.bool_``, and floats are rejected. All
        in-degree-0 nodes must share the minimum rank, and no
        other node may use it. Every edge must go from a lower
        rank to a strictly higher one, so an edge between two
        nodes of equal rank is illegal. A node may have only
        skip parents, with none in the layer just below it.
    outputs : iterable of str, optional
        The output nodes you read, to confirm the ones the parser
        infers. Every node with no outgoing edge is an output
        node, so a node whose onward rows are missing from the
        edgelist is one too, and a model that reads every output
        node would read it as a prediction. With ``outputs``, any
        other node with no outgoing edge raises, and the message
        names every node with no path to a name in ``outputs``:
        drop the rows whose target is one of those and parse
        again. ``outputs`` checks and never assigns. It does not
        make a node an output, and the spec is the same as
        without it.

        ``None`` or omitted skips the check, and the parse warns
        when there is more than one output node. Names are
        matched after ``str(...)`` and compared as a set, so a
        list, tuple, set, one-dimensional numpy array, or
        ``pandas.Index`` works in any order. A string, a mapping,
        an empty collection, a repeated name, an unknown name, or
        a node with an outgoing edge raises ``Kpnn2Error``. To
        read a node that feeds others, use
        ``LayeredSpec.unit_index``.

    Returns
    -------
    LayeredSpec
        Frozen structure: layers, one ``Hop`` per layer after the
        first, and ``skips`` metadata. ``skips`` is empty when no
        edge spans more than one layer; ``hops`` never is, because
        a valid edgelist always yields at least two layers.

    Raises
    ------
    Kpnn2Error
        If ``edgelist`` is not a DataFrame; ``source`` or ``target``
        is absent, missing, or an empty name, or holds a value that
        is not a ``str``, integer, float, or ``bool`` (a list, a
        tensor, bytes, any other object); the table has no rows;
        two node names differ only by leading or trailing whitespace
        (``'A'`` and ``'A '``); a ``(source, target)`` pair is
        duplicated; any edge is a self-loop; the graph has a cycle;
        there is no in-degree-0 node or no out-degree-0 node;
        ``widths`` names an unknown node, is not a mapping of
        positive integers, or gives an input node a width other
        than 1; or ``ranks`` is incomplete, names an
        unknown node, is not a mapping of non-negative integers,
        places inputs off the minimum rank, places a non-input at
        that minimum, or contains a non-forward edge; or
        ``outputs`` is a string, a mapping, or empty, repeats a
        name, names a node that is not an output node, or leaves
        out an output node. Each message names the offending
        pairs or nodes, sorted.

    Warns
    -----
    UserWarning
        If ``source`` or ``target`` holds a float or a ``bool``,
        converted to a name such as ``"1.0"`` or ``"True"``.
        Convert the column to the intended strings first to
        silence it. Also if ``outputs`` is omitted and the graph
        has more than one output node; the message names them.
        Pass the output nodes you read as ``outputs`` to silence
        it.

    See Also
    --------
    parse_adjacency : Pack the same table into one state vector;
        allows cycles and self-loops. Has no ``ranks`` argument.
    PackedLinear.from_hop : Build the layer for one hop.
    gather_hop_inputs : Build one hop's input from the saved layer
        tensors.

    Notes
    -----
    Every named edge belongs to exactly one hop, the hop into its
    target's layer, however many layers it spans. A hop whose
    target layer has parents further back reads several layers:
    its source columns are those whole layers concatenated in
    ascending order, including nodes that feed nothing in this
    hop. A skip edge is therefore an ordinary weight rather than
    a dummy neuron or a second mechanism; ``skips`` only reports
    it. Under longest-path ranking a hop always reads the
    previous layer; with ``ranks`` it may omit that layer when
    every parent is a skip.

    Packed indices count units, not nodes: named edge ``A -> B``
    expands into a ``(k_B, k_A)`` block of live pairs, which is
    one pair per named edge at the default width 1.
    ``layer_dims`` counts units too, while ``layer_nodes`` and
    ``Hop.source_nodes`` hold one name per node, so their
    lengths differ once any node is wider than 1.
    ``LayeredSpec.node_units`` and ``LayeredSpec.hop_units``
    find the units of one node.

    Terminals below maximum depth (early outputs) are allowed,
    and isolated nodes cannot appear, since the node set is the
    union of ``source`` and ``target``. ``to_dict()`` stores the
    renumbered ranks only when they put some node in a different
    layer than longest-path ranking does on the same edges.

    Examples
    --------
    A chain ``A -> H -> C`` plus the skip ``A -> C``. The hop into
    ``C`` reads both earlier layers, so the skip is a packed pair
    of that hop:

    >>> import pandas as pd
    >>> import kpnn2
    >>> edgelist = pd.DataFrame(
    ...     {"source": ["A", "H", "A"], "target": ["H", "C", "C"]}
    ... )
    >>> spec = kpnn2.parse_layered(edgelist)
    >>> spec.layer_nodes
    (('A',), ('H',), ('C',))
    >>> spec.hops[1].source_nodes
    ('A', 'H')
    >>> spec.hops[1].source_index, spec.hops[1].target_index
    ((0, 1), (0, 0))
    >>> spec.hops[1].to_mask().tolist()
    [[1.0, 1.0]]
    >>> spec.skips[0].source, spec.skips[0].target
    ('A', 'C')

    With ``widths``, a hidden node of width 2 owns two units. Each
    layer keeps the same nodes, only its unit count grows, and the
    named edge ``A -> H`` becomes a 2-by-1 block:

    >>> spec = kpnn2.parse_layered(
    ...     edgelist,
    ...     widths={"H": 2},
    ... )
    >>> spec.layer_nodes
    (('A',), ('H',), ('C',))
    >>> spec.layer_dims
    (1, 2, 1)
    >>> spec.layer_widths
    ((1,), (2,), (1,))
    >>> spec.hops[0].to_mask().tolist()
    [[1.0], [1.0]]

    ``C`` now reads two numbers from ``H`` instead of one, next to
    the one from ``A``:

    >>> spec.hops[1].to_mask().tolist()
    [[1.0, 1.0, 1.0]]

    With ``ranks``, two nodes can share a layer even when their
    chains of parents differ in length. By default ``Short`` sits
    in layer 1, one above its only parent, while ``Long`` sits in
    layer 2 behind ``Mid``. Both are output nodes, so ``outputs``
    confirms them; without it, the parse warns:

    >>> siblings = pd.DataFrame(
    ...     {
    ...         "source": ["A", "A", "Mid"],
    ...         "target": ["Short", "Mid", "Long"],
    ...     }
    ... )
    >>> kpnn2.parse_layered(
    ...     siblings,
    ...     outputs=["Long", "Short"],
    ... ).layer_nodes
    (('A',), ('Mid', 'Short'), ('Long',))

    Giving both the same rank puts them in one layer, and
    ``A -> Short`` becomes a skip edge. The edges and their
    weights stay the same; only the grouping moves, so a head or
    a normalization on the last layer now sees both nodes:

    >>> spec = kpnn2.parse_layered(
    ...     siblings,
    ...     ranks={
    ...         "A": 0,
    ...         "Mid": 1,
    ...         "Short": 2,
    ...         "Long": 2,
    ...     },
    ...     outputs=["Long", "Short"],
    ... )
    >>> spec.layer_nodes
    (('A',), ('Mid',), ('Long', 'Short'))
    >>> spec.skips[0].source, spec.skips[0].target
    ('A', 'Short')

    ``outputs`` catches a node whose onward rows are missing. This
    edgelist has no row out of ``D``, so ``D`` is an output node
    too, and the parse raises:

    >>> dangling = pd.DataFrame(
    ...     {
    ...         "source": ["A", "H", "B", "A"],
    ...         "target": ["H", "C", "D", "D"],
    ...     }
    ... )
    >>> kpnn2.parse_layered(
    ...     dangling,
    ...     outputs=["C"],
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    Kpnn2Error: Node(s) with no outgoing edge that are not in ...

    The message names ``B`` and ``D``, the nodes with no path to
    ``C``. Dropping the rows whose target is one of them leaves
    ``C`` as the only output node:

    >>> kept = dangling[~dangling["target"].isin(["B", "D"])]
    >>> kpnn2.parse_layered(
    ...     kept,
    ...     outputs=["C"],
    ... ).output_nodes
    ('C',)
    """
    declared = _declared_outputs(outputs)
    spec = _layered_from_edgelist(
        edgelist,
        widths=widths,
        ranks=ranks,
    )
    _confirm_outputs(
        spec,
        declared,
        widths=widths,
        ranks=ranks,
    )
    return spec
