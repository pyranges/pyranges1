#!/usr/bin/env python3
"""Regression tests for gh-166.

``prepare_by_binary`` took a copy of ``other`` in order to flip its strand, and
that copy was also a projection down to ``RANGE_COLS + by``. Every other column
of ``other`` was discarded, so the methods that report columns from ``other``
lost them — but only for ``strand_behavior="opposite"``, since the other
branches pass ``other`` through whole.
"""

import pandas as pd

import pyranges1 as pr

LOC_SUFFIXED = {"Chromosome_b", "Start_b", "End_b", "Strand_b"}


def _left() -> pr.PyRanges:
    return pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": ["chr1"],
                "Start": [10],
                "End": [20],
                "Strand": ["+"],
                "ID": ["a1"],
                "Score": [1],
            }
        )
    )


def _right() -> pr.PyRanges:
    return pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": ["chr1"],
                "Start": [12],
                "End": [30],
                "Strand": ["-"],
                "ID": ["b1"],
                "Score": [7],
            }
        )
    )


def _carried_metadata(gr: pr.PyRanges) -> list[str]:
    """Columns taken from ``other`` that are not location columns."""
    return [c for c in gr.columns if c.endswith("_b") and c not in LOC_SUFFIXED]


def test_join_overlaps_keeps_other_metadata_on_opposite_strand() -> None:
    result = _left().join_overlaps(_right(), strand_behavior="opposite")
    assert _carried_metadata(result) == ["ID_b", "Score_b"]
    assert result["ID_b"].tolist() == ["b1"]
    assert result["Score_b"].tolist() == [7]


def test_nearest_ranges_keeps_other_metadata_on_opposite_strand() -> None:
    result = _left().nearest_ranges(_right(), strand_behavior="opposite")
    assert _carried_metadata(result) == ["ID_b", "Score_b"]
    assert result["ID_b"].tolist() == ["b1"]


def test_opposite_carries_the_same_metadata_as_same() -> None:
    """The strand option decides which rows match, not which columns survive."""
    for method in ("join_overlaps", "nearest_ranges"):
        same = getattr(_left(), method)(_left(), strand_behavior="same")
        opposite = getattr(_left(), method)(_right(), strand_behavior="opposite")
        assert _carried_metadata(same) == _carried_metadata(opposite), method


def test_flipping_the_strand_does_not_mutate_the_callers_frame() -> None:
    other = _right()
    _left().join_overlaps(other, strand_behavior="opposite")
    assert other["Strand"].tolist() == ["-"]
