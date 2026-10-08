import numpy as np
import pandas as pd

import pyranges1 as pr


def _per_base(gr: pr.PyRanges, size: int, value_col: str | None = None) -> np.ndarray:
    depth = np.zeros(size)
    for row in gr.itertuples():
        depth[row.Start : row.End] += 1 if value_col is None else getattr(row, value_col)
    return depth


def test_coverage_matches_a_per_base_count_and_runs_are_maximal() -> None:
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = 80
        starts = rng.integers(0, 300, n)
        gr = pr.PyRanges({"Chromosome": "chr1", "Start": starts, "End": starts + rng.integers(0, 50, n), "W": rng.integers(1, 4, n) / 2})
        for value_col in (None, "W"):
            cov = gr.coverage_ranges(value_col)
            got = np.zeros(400)
            for row in cov.itertuples():
                got[row.Start : row.End] = row.Coverage
            np.testing.assert_allclose(got, _per_base(gr, 400, value_col))
            touching = cov["Start"].to_numpy()[1:] == cov["End"].to_numpy()[:-1]
            assert not (touching & (cov["Coverage"].to_numpy()[1:] == cov["Coverage"].to_numpy()[:-1])).any()
            assert (cov["Coverage"] != 0).all()


def test_coverage_by_strand_and_match_by() -> None:
    gr = pr.PyRanges({"Chromosome": "chr1", "Start": [0, 0, 0], "End": [10, 10, 10], "Strand": ["+", "+", "-"], "Tag": ["a", "b", "a"]})
    assert gr.coverage_ranges()["Coverage"].tolist() == [2, 1]  # + then -
    assert gr.coverage_ranges(use_strand=False)["Coverage"].tolist() == [3]
    by_tag = gr.coverage_ranges(use_strand=False, match_by="Tag")
    assert list(zip(by_tag["Tag"], by_tag["Coverage"], strict=True)) == [("a", 2), ("b", 1)]


def test_weights_that_cancel_leave_no_residue() -> None:
    gr = pr.PyRanges({"Chromosome": "chr1", "Start": [0, 0, 20], "End": [10, 10, 30], "W": [0.1, 0.2, 0.3]})
    cov = gr.coverage_ranges("W")
    assert list(zip(cov["Start"], cov["End"], strict=True)) == [(0, 10), (20, 30)]


def test_coverage_of_nothing() -> None:
    empty = pr.PyRanges({"Chromosome": pd.Series([], dtype=str), "Start": pd.Series([], dtype=int), "End": pd.Series([], dtype=int)})
    assert len(empty.coverage_ranges()) == 0
