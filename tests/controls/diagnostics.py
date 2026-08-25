"""
Failure diagnostics for control tests.

These helpers must not raise: a crash while formatting a message
would hide the real assertion failure.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def score_report(
    *,
    scenario_id: str,
    label: str,
    scores: pd.Series,
    important: Sequence[str],
    unimportant: Sequence[str],
) -> str:
    """
    Render ranked |score| rows for CI logs.
    """
    records = []
    for group, names in (
        ("important", important),
        ("unimportant", unimportant),
    ):
        for name in names:
            value = float("nan")
            if name in scores.index:
                value = float(scores[name])
            records.append(
                {
                    "name": name,
                    "group": group,
                    "score": value,
                }
            )
    header = f"[{scenario_id}] {label}: |score| by ground-truth group"
    if not records:
        return f"{header}\n(no names were declared)"
    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(
        "score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return (
        header
        + "\n"
        + frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )
