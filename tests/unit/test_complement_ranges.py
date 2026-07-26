#!/usr/bin/env python3
"""Regression tests for gh-157.

``complement_ranges`` used to label each gap with the group columns of an
arbitrary row, so the reported ``Chromosome``/``Strand`` depended on the order
of the input rows.
"""

import itertools

import pandas as pd

import pyranges1 as pr


def _rows(gr, columns) -> list[tuple]:
    return sorted(tuple(row) for row in gr[columns].to_numpy().tolist())


def test_gaps_keep_their_own_chromosome_when_groups_are_not_in_ascending_order() -> None:
    # ``chr2`` rows come first, so the group ids do not ascend with first
    # appearance. Both gaps used to be reported on ``chr2``.
    gr = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": ["chr2", "chr2", "chr1", "chr1"],
                "Start": [0, 20, 0, 10],
                "End": [1, 21, 1, 11],
            }
        )
    )

    assert _rows(gr.complement_ranges(), ["Chromosome", "Start", "End"]) == [
        ("chr1", 1, 10),
        ("chr2", 1, 20),
    ]


def test_gap_is_labelled_with_the_strand_that_produced_it() -> None:
    # The '-' strand holds a single interval and therefore has no internal gap;
    # the only gap belongs to the '+' strand.
    gr = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": ["chr01", "chr01", "chr01"],
                "Start": [0, 2, 0],
                "End": [1, 3, 1],
                "Strand": ["-", "+", "+"],
            }
        )
    )

    assert _rows(gr.complement_ranges(use_strand=True), ["Chromosome", "Start", "End", "Strand"]) == [
        ("chr01", 1, 2, "+")
    ]


def test_result_does_not_depend_on_input_row_order() -> None:
    base = [("chr01", 0, 1, "-"), ("chr01", 2, 3, "+"), ("chr01", 0, 1, "+")]

    results = set()
    for permutation in itertools.permutations(range(len(base))):
        rows = [base[index] for index in permutation]
        gr = pr.PyRanges(
            pd.DataFrame(
                {
                    "Chromosome": [row[0] for row in rows],
                    "Start": [row[1] for row in rows],
                    "End": [row[2] for row in rows],
                    "Strand": [row[3] for row in rows],
                }
            )
        )
        results.add(tuple(_rows(gr.complement_ranges(use_strand=True), ["Start", "End", "Strand"])))

    assert results == {((1, 2, "+"),)}


def test_categorical_group_columns_keep_their_dtype() -> None:
    gr = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": pd.Categorical(["chr2", "chr2", "chr1", "chr1"]),
                "Start": [0, 20, 0, 10],
                "End": [1, 21, 1, 11],
                "Strand": pd.Categorical(["+", "+", "+", "+"]),
            }
        )
    )

    result = gr.complement_ranges(use_strand=True)

    assert result["Chromosome"].dtype == "category"
    assert result["Strand"].dtype == "category"
    assert _rows(result, ["Chromosome", "Start", "End", "Strand"]) == [
        ("chr1", 1, 10, "+"),
        ("chr2", 1, 20, "+"),
    ]


def test_grouped_complement_with_chromsizes_labels_each_group() -> None:
    gr = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": ["chr1"] * 4,
                "Start": [2, 10, 20, 40],
                "End": [5, 18, 30, 46],
                "ID": ["a", "a", "b", "b"],
            }
        )
    )

    result = gr.complement_ranges(
        group_by="ID",
        group_sizes_col="ID",
        chromsizes={"a": 22, "b": 100},
        include_first_interval=True,
    )

    assert _rows(result, ["Chromosome", "Start", "End", "ID"]) == [
        ("chr1", 0, 2, "a"),
        ("chr1", 0, 20, "b"),
        ("chr1", 5, 10, "a"),
        ("chr1", 18, 22, "a"),
        ("chr1", 30, 40, "b"),
        ("chr1", 46, 100, "b"),
    ]
