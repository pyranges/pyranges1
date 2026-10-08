from collections import Counter

import numpy as np
import pandas as pd
import pytest

import pyranges1 as pr


def test_every_placement_is_equally_likely() -> None:
    # chr a is 10 bp with [3, 5) excluded, b is 5 bp. A 2 bp interval fits at
    # a:0-1, a:5-8 and b:0-3 -- ten starts, each with probability 1/10.
    n = 200_000
    gr = pr.PyRanges({"Chromosome": ["a"] * n, "Start": [0] * n, "End": [2] * n})
    exclude = pr.PyRanges({"Chromosome": ["a"], "Start": [3], "End": [5]})
    shuffled = gr.shuffle_ranges({"a": 10, "b": 5}, exclude=exclude, seed=0)
    counts = Counter(zip(shuffled["Chromosome"], shuffled["Start"], strict=True))
    expected = {("a", 0), ("a", 1), ("a", 5), ("a", 6), ("a", 7), ("a", 8), ("b", 0), ("b", 1), ("b", 2), ("b", 3)}
    assert set(counts) == expected
    chi2 = sum((c - n / 10) ** 2 / (n / 10) for c in counts.values())
    assert chi2 < 30  # 9 degrees of freedom; p < 0.0005 above this


def test_placements_are_valid_and_the_rest_is_kept() -> None:
    rng = np.random.default_rng(1)
    sizes = {"chr1": 5_000, "chr2": 800, "chr3": 2_000}
    n = 5_000
    starts = rng.integers(0, 700, n)
    gr = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": rng.choice(list(sizes), n),
                "Start": starts,
                "End": starts + rng.integers(0, 90, n),
                "Strand": rng.choice(["+", "-"], n),
                "Name": [f"r{i}" for i in range(n)],
            },
            index=rng.permutation(n),
        )
    )
    exclude = pr.PyRanges({"Chromosome": ["chr1", "chr1", "chr3"], "Start": [100, 150, 0], "End": [400, 3_000, 1_500]})
    for within in (False, True):
        out = gr.shuffle_ranges(sizes, within_chromosomes=within, exclude=exclude, seed=2)
        assert out.index.equals(gr.index)
        assert (out["End"] - out["Start"]).tolist() == (gr["End"] - gr["Start"]).tolist()
        assert out[["Strand", "Name"]].equals(gr[["Strand", "Name"]])
        assert (out["Start"] >= 0).all()
        assert (out["End"] <= out["Chromosome"].map(sizes)).all()
        assert len(out.overlap(exclude, strand_behavior="ignore")) == 0
        if within:
            assert out["Chromosome"].tolist() == gr["Chromosome"].tolist()


def test_shuffle_is_reproducible_with_a_seed() -> None:
    gr = pr.PyRanges({"Chromosome": ["chr1"] * 50, "Start": range(50), "End": range(10, 60)})
    a = gr.shuffle_ranges({"chr1": 10_000}, seed=7)
    b = gr.shuffle_ranges({"chr1": 10_000}, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_shuffle_refuses_what_cannot_be_placed() -> None:
    gr = pr.PyRanges({"Chromosome": ["chr1"], "Start": [0], "End": [50]})
    with pytest.raises(ValueError, match="fits nowhere"):
        gr.shuffle_ranges({"chr1": 40})
    with pytest.raises(ValueError, match="fits nowhere"):
        # Room on chr2, but not on chr1 outside [30, 80).
        blocked = pr.PyRanges({"Chromosome": ["chr1"], "Start": [30], "End": [80]})
        gr.shuffle_ranges({"chr1": 100, "chr2": 60}, within_chromosomes=True, exclude=blocked)
    gr.shuffle_ranges({"chr1": 100, "chr2": 60}, exclude=pr.PyRanges({"Chromosome": ["chr1"], "Start": [30], "End": [80]}))
    with pytest.raises(ValueError, match="chr1"):
        gr.shuffle_ranges({"chr2": 100})
