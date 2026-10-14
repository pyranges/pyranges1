"""A missing value in a match_by or group_by column is a group of its own.

So every operation must give what it gives when the missing values are replaced by a
value that occurs nowhere else; in particular, rows never pair across chromosomes.
"""

import numpy as np
import pandas as pd
import pytest

import pyranges1 as pr

# Sorts after every real value, as a missing value does, so group numbers line up too.
PLACEHOLDER = "~missing"


def _frames(seed: int) -> tuple[pr.PyRanges, pr.PyRanges]:
    rng = np.random.default_rng(seed)

    def one(n: int) -> pr.PyRanges:
        starts = rng.integers(0, 400, n)
        gene = rng.choice(np.array(["a", "b", None], dtype=object), n)
        return pr.PyRanges(
            {
                "Chromosome": rng.choice(["chr1", "chr2"], n),
                "Start": starts,
                "End": starts + rng.integers(1, 60, n),
                "Strand": rng.choice(["+", "-"], n),
                "Gene": gene,
            }
        )

    return one(40), one(30)


def _filled(gr: pr.PyRanges) -> pr.PyRanges:
    return pr.PyRanges(gr.assign(Gene=gr["Gene"].fillna(PLACEHOLDER)))


def _same(with_missing: pd.DataFrame, with_placeholder: pd.DataFrame) -> None:
    got = pd.DataFrame(with_missing).copy()
    want = pd.DataFrame(with_placeholder).copy()
    for frame in (got, want):
        for column in [c for c in frame.columns if str(c).startswith("Gene")]:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), PLACEHOLDER)
        frame.reset_index(drop=True, inplace=True)
    sort_by = list(got.columns)
    got = got.sort_values(sort_by, kind="stable").reset_index(drop=True)
    want = want.sort_values(sort_by, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False, check_categorical=False)


TWO_FRAME = [
    lambda a, b: a.overlap(b, match_by="Gene"),
    lambda a, b: a.count_overlaps(b, match_by="Gene"),
    lambda a, b: a.join_overlaps(b, match_by="Gene"),
    lambda a, b: a.nearest_ranges(b, match_by="Gene"),
    lambda a, b: a.subtract_overlaps(b, match_by="Gene"),
    lambda a, b: a.intersect_overlaps(b, match_by="Gene"),
    lambda a, b: a.set_union_overlaps(b, match_by="Gene"),
    lambda a, b: a.set_intersect_overlaps(b, match_by="Gene"),
]
ONE_FRAME = [
    lambda a: a.cluster_overlaps(match_by="Gene"),
    lambda a: a.merge_overlaps(match_by="Gene"),
    lambda a: a.split_overlaps(match_by="Gene"),
    lambda a: a.max_disjoint_overlaps(match_by="Gene"),
    lambda a: a.coverage_ranges(match_by="Gene"),
    lambda a: a.complement_ranges(group_by="Gene"),
    lambda a: a.extend_ranges(ext=5, group_by="Gene"),
    lambda a: a.five_end(group_by="Gene"),
    lambda a: a.outer_ranges(group_by="Gene"),
    lambda a: a.window_ranges(10, group_by="Gene"),
    lambda a: a.slice_ranges(0, 5, group_by="Gene"),
]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("op", TWO_FRAME)
def test_two_frame_operations_treat_missing_as_its_own_group(seed, op) -> None:
    a, b = _frames(seed)
    _same(op(a, b), op(_filled(a), _filled(b)))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("op", ONE_FRAME)
def test_one_frame_operations_treat_missing_as_its_own_group(seed, op) -> None:
    a, _ = _frames(seed)
    _same(op(a), op(_filled(a)))


def test_missing_values_never_pair_across_chromosomes() -> None:
    a = pr.PyRanges({"Chromosome": ["chr2"], "Start": [100], "End": [200], "gene": [None]})
    b = pr.PyRanges({"Chromosome": ["chr1", "chr2"], "Start": [100, 5000], "End": [200, 5100], "gene": ["y", "x"]})
    assert len(a.overlap(b, match_by="gene")) == 0
    assert len(a.join_overlaps(b, match_by="gene")) == 0


def test_window_ranges_group_by_windows_each_row_by_its_own_strand() -> None:
    # Sorting by group_by reorders these rows; each must still be windowed by its own strand.
    gr = pr.PyRanges(
        {"Chromosome": ["chr1", "chr1"], "Start": [100, 0], "End": [130, 30], "Strand": ["-", "+"], "Gene": ["b", "a"]}
    )
    got = gr.window_ranges(20, group_by="Gene")
    assert list(zip(got["Gene"], got["Start"], got["End"], strict=True)) == [
        ("a", 0, 20),
        ("a", 20, 30),
        ("b", 110, 130),
        ("b", 100, 110),
    ]
