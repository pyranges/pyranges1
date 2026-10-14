import numpy as np
import pandas as pd
import pytest

import pyranges1 as pr

pytest.importorskip("runlength")


def dense_coverage(df: pd.DataFrame, value_col: str | None = None) -> np.ndarray:
    """Coverage computed position by position, for comparison."""
    out = np.zeros(int(df.End.max()) if len(df) else 0)
    for row in df.itertuples():
        out[row.Start : row.End] += getattr(row, value_col) if value_col else 1
    return out


def expand(rle) -> np.ndarray:
    return np.repeat(np.asarray(rle.values, dtype=float), np.asarray(rle.runs))


@pytest.fixture
def ranges() -> pr.PyRanges:
    rng = np.random.default_rng(42)
    n = 200
    starts = rng.integers(0, 1000, n)
    return pr.PyRanges(
        {
            "Chromosome": rng.choice(["chr1", "chr2", "chrX"], n),
            "Start": starts,
            "End": starts + rng.integers(1, 100, n),
            "Strand": rng.choice(["+", "-"], n),
            "Score": rng.choice([0.25, 1.0, 3.5, -2.0], n),
        },
    )


@pytest.mark.parametrize("value_col", [None, "Score"])
@pytest.mark.parametrize("strand", [True, False])
def test_to_rle_matches_dense_coverage(ranges, value_col, strand) -> None:
    rles = ranges.to_rle(value_col, strand=strand)
    groups = ranges.groupby(["Chromosome", "Strand"] if strand else ["Chromosome"])
    assert len(rles) == groups.ngroups
    for key, group in groups:
        rle = rles.get(key)
        np.testing.assert_allclose(expand(rle), dense_coverage(group, value_col), atol=1e-12)


def test_to_rle_rpm_scales_by_number_of_intervals(ranges) -> None:
    plain = ranges.to_rle(strand=False)
    scaled = ranges.to_rle(strand=False, rpm=True)
    for (key, rle), (_, scaled_rle) in zip(plain.items(), scaled.items(), strict=True):
        np.testing.assert_allclose(expand(scaled_rle), expand(rle) * 1e6 / len(ranges), err_msg=str(key))


def test_to_rle_leaves_no_residue_where_uncovered() -> None:
    gr = pr.PyRanges({"Chromosome": ["chr1"] * 2, "Start": [0, 5], "End": [3, 8], "Score": [0.1, 0.2]})
    rle = gr.to_rle("Score", strand=False).get("chr1")
    assert np.asarray(rle.values).tolist() == [0.1, 0.0, 0.2]
    assert np.asarray(rle.runs).tolist() == [3, 2, 3]


def test_to_rle_empty() -> None:
    gr = pr.PyRanges({"Chromosome": [], "Start": [], "End": []})
    assert len(gr.to_rle(rpm=True)) == 0
