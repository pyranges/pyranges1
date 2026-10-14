import numpy as np
import pandas as pd
import pytest

import pyranges1 as pr


def _covered_fraction(a: pr.PyRanges, b: pr.PyRanges, strand_behavior: str) -> list[float]:
    """Per row of a, the covered fraction by brute force over positions."""
    out = []
    for r in a.itertuples():
        covered: set[int] = set()
        for q in b.itertuples():
            same = q.Strand == r.Strand
            if q.Chromosome == r.Chromosome and {"same": same, "opposite": not same, "ignore": True}[strand_behavior]:
                covered |= set(range(max(r.Start, q.Start), min(r.End, q.End)))
        out.append(len(covered) / (r.End - r.Start) if r.End > r.Start else 0.0)
    return out


@pytest.mark.parametrize("strand_behavior", ["same", "opposite", "ignore"])
def test_coverage_counts_each_covered_position_once(strand_behavior: str) -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        frames = []
        for n in (int(rng.integers(0, 10)), int(rng.integers(0, 10))):
            starts = rng.integers(0, 60, n)
            frames.append(
                pr.PyRanges(
                    pd.DataFrame(
                        {
                            "Chromosome": rng.choice(["c1", "c2"], n),
                            "Start": starts,
                            "End": starts + rng.integers(0, 15, n),
                            "Strand": rng.choice(["+", "-"], n),
                        },
                        index=rng.integers(0, 3, n),  # a non-unique index
                    )
                )
            )
        a, b = frames
        got = a.count_overlaps(b, strand_behavior=strand_behavior, calculate_coverage=True)["CoverageOverlaps"]
        np.testing.assert_allclose(got.to_numpy(), _covered_fraction(a, b, strand_behavior))


def test_coverage_arguments_are_checked() -> None:
    f1, f2 = pr.example_data.f1, pr.example_data.f2
    with pytest.raises(ValueError, match="slack=0"):
        f1.count_overlaps(f2, slack=1, calculate_coverage=True)
    with pytest.raises(ValueError, match="calculate_coverage=True"):
        f1.count_overlaps(f2, coverage_col="F")
