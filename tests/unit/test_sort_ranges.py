"""Contract tests for ``sort_ranges`` on both ``RangeFrame`` and ``PyRanges``.

The interesting cases are the ones the doctests cannot show: a frame large enough
that a transposed kernel argument is visible, keys relocated out of their implicit
positions, and descending keys.
"""

import numpy as np
import pandas as pd
import pytest

import pyranges1 as pr


def _realistic_frame(n: int = 100_000, seed: int = 0) -> pd.DataFrame:
    """Coordinates spread widely enough that Start-then-End and End-then-Start
    orderings disagree on nearly every row."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, 10_000_000, n)
    return pd.DataFrame({"Start": starts, "End": starts + rng.integers(1, 10_000, n)})


def test_range_frame_sort_ranges_orders_by_start_before_end() -> None:
    """The kernel takes (starts, ends, groups) but was called with the group ids
    first, so the primary key became ``End``. On a frame with realistic
    coordinates that put 97,964 of 100,000 rows in the wrong place."""
    frame = _realistic_frame()
    sorted_frame = pr.RangeFrame(frame).sort_ranges().reset_index(drop=True)
    expected = frame.sort_values(["Start", "End"], kind="stable").reset_index(drop=True)
    assert sorted_frame["Start"].tolist() == expected["Start"].tolist()
    assert sorted_frame["End"].tolist() == expected["End"].tolist()


def test_range_frame_sort_ranges_orders_string_keys_lexically() -> None:
    """No ``natsort`` knob on the generic namespace: it exists for chromosome
    names, and a RangeFrame has no Chromosome column."""
    frame = pr.RangeFrame({"Start": [1, 2, 3], "End": [2, 3, 4], "Name": ["t9", "t10", "t1"]})
    assert frame.sort_ranges(by="Name")["Name"].tolist() == ["t1", "t10", "t9"]


def test_range_frame_sort_ranges_rejects_natsort_argument() -> None:
    frame = pr.RangeFrame({"Start": [1], "End": [2]})
    with pytest.raises(TypeError):
        frame.sort_ranges(natsort=True)  # type: ignore[call-arg]


def test_sort_by_position_warns_and_still_sorts() -> None:
    frame = pr.RangeFrame({"Start": [10, 1], "End": [12, 4]})
    with pytest.warns(DeprecationWarning, match="sort_ranges"):
        result = frame.sort_by_position()
    assert result["Start"].tolist() == [1, 10]


@pytest.fixture
def genomic() -> pr.PyRanges:
    return pr.PyRanges(
        {
            "Chromosome": ["chr2", "chr1", "chr2", "chr1", "chr10"],
            "Strand": ["-", "+", "+", "-", "+"],
            "Start": [1, 2, 3, 4, 5],
            "End": [2, 3, 4, 5, 6],
            "Name": ["c2m", "c1p", "c2p", "c1m", "c10p"],
        }
    )


def test_pyranges_sort_ranges_natsort_default(genomic: pr.PyRanges) -> None:
    assert genomic.sort_ranges()["Chromosome"].tolist() == [
        "chr1",
        "chr1",
        "chr2",
        "chr2",
        "chr10",
    ]
    assert genomic.sort_ranges(natsort=False)["Chromosome"].tolist() == [
        "chr1",
        "chr1",
        "chr10",
        "chr2",
        "chr2",
    ]


def test_pyranges_sort_ranges_by_relocates_implicit_keys(genomic: pr.PyRanges) -> None:
    """Issue #94: sorting by Strand before Chromosome, and by a key placed after
    the coordinates."""
    swapped = genomic.sort_ranges(by=["Strand", "Chromosome"])
    assert list(zip(swapped["Strand"], swapped["Chromosome"], strict=True)) == [
        ("+", "chr1"),
        ("+", "chr2"),
        ("+", "chr10"),
        ("-", "chr1"),
        ("-", "chr2"),
    ]

    tied = pr.PyRanges(
        {
            "Chromosome": ["chr1"] * 4,
            "Strand": ["+"] * 4,
            "Start": [10, 10, 5, 5],
            "End": [20, 20, 8, 8],
            "Name": ["late-b", "late-a", "early-b", "early-a"],
        }
    )
    trailing = tied.sort_ranges(by=["Start", "End", "Name"])
    assert trailing["Name"].tolist() == ["early-a", "early-b", "late-a", "late-b"]


def test_pyranges_sort_ranges_sort_descending(genomic: pr.PyRanges) -> None:
    assert genomic.sort_ranges(sort_descending="Chromosome")["Chromosome"].tolist() == [
        "chr10",
        "chr2",
        "chr2",
        "chr1",
        "chr1",
    ]
    # Start is an inner key, so descending applies within each Chromosome/Strand
    # group. One group, so the whole frame reverses.
    one_group = pr.PyRanges(
        {
            "Chromosome": ["chr1"] * 4,
            "Strand": ["+"] * 4,
            "Start": [30, 10, 40, 20],
            "End": [31, 11, 41, 21],
        }
    )
    assert one_group.sort_ranges(sort_descending="Start")["Start"].tolist() == [40, 30, 20, 10]


def test_pyranges_sort_ranges_sort_descending_rejects_non_keys(genomic: pr.PyRanges) -> None:
    with pytest.raises(ValueError, match="sort_descending"):
        genomic.sort_ranges(sort_descending="Name")


def test_pyranges_sort_descending_composes_with_use_strand() -> None:
    """``use_strand`` reverses coordinates per row; ``sort_descending`` reverses a
    key globally. On the coordinates the two compose by XOR, so a minus-strand
    frame sorted descending comes back ascending."""
    minus = pr.PyRanges(
        {
            "Chromosome": ["chr1"] * 3,
            "Strand": ["-"] * 3,
            "Start": [1, 5, 10],
            "End": [2, 6, 11],
        }
    )
    assert minus.sort_ranges(use_strand=True)["Start"].tolist() == [10, 5, 1]
    assert minus.sort_ranges(use_strand=True, sort_descending=["Start", "End"])["Start"].tolist() == [1, 5, 10]


def test_pyranges_sort_ranges_use_strand_false_still_groups_by_strand() -> None:
    """Documented, not changed: ``use_strand=False`` only stops the 5'->3'
    coordinate reversal. ``Strand`` stays a sort key."""
    frame = pr.PyRanges(
        {
            "Chromosome": ["chr1"] * 4,
            "Strand": ["-", "+", "-", "+"],
            "Start": [1, 2, 3, 4],
            "End": [2, 3, 4, 5],
            "Name": ["m1", "p2", "m3", "p4"],
        }
    )
    assert frame.sort_ranges(use_strand=False)["Name"].tolist() == ["p2", "p4", "m1", "m3"]


def test_high_cardinality_by_matches_a_plain_pandas_sort() -> None:
    """Per-column ranking must give the same answer as ranking key tuples. The
    keys here are zero-padded so natural and lexical order agree, which lets
    pandas act as the oracle."""
    rng = np.random.default_rng(1)
    n = 20_000
    frame = pd.DataFrame(
        {
            "Chromosome": [f"chr{i:02d}" for i in rng.integers(1, 23, n)],
            "Strand": rng.choice(["+", "-"], n),
            "Start": rng.integers(0, 1_000_000, n),
            "transcript_id": [f"t{i:06d}" for i in rng.integers(0, n // 10, n)],
        }
    )
    frame["End"] = frame["Start"] + rng.integers(1, 1000, n)
    got = (
        pr.PyRanges(frame)
        .sort_ranges(by="transcript_id", use_strand=False)
        .reset_index(drop=True)
    )
    want = frame.sort_values(
        ["Chromosome", "Strand", "transcript_id", "Start", "End"], kind="stable"
    ).reset_index(drop=True)
    for column in ("Chromosome", "Strand", "transcript_id", "Start", "End"):
        assert got[column].tolist() == want[column].tolist(), column


def test_empty_frame_sorts_without_error() -> None:
    empty = pr.PyRanges({"Chromosome": [], "Start": [], "End": []})
    assert len(empty.sort_ranges()) == 0
    assert len(pr.RangeFrame({"Start": [], "End": []}).sort_ranges()) == 0
