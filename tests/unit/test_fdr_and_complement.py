import numpy as np
import pandas as pd
import pytest

import pyranges1 as pr


def test_fdr_is_benjamini_hochberg() -> None:
    # Sorted, p * m / rank is 0.03, 0.045, 0.031. The running minimum from the
    # largest p down lowers 0.045 to 0.031, so the adjusted values stay monotone in p.
    p = pd.Series([0.031, 0.01, 0.03], index=[10, 11, 12])
    adjusted = pr.stats.fdr(p)
    assert adjusted.index.equals(p.index)
    assert adjusted.round(6).tolist() == [0.031, 0.03, 0.031]


def test_fdr_is_monotone_and_skips_missing_p_values() -> None:
    p = pd.Series([0.0039591368855297175, 0.0037600512992788937, np.nan, 0.0075061166500909205])
    adjusted = pr.stats.fdr(p)
    assert np.isnan(adjusted[2])
    # m counts the three tested p-values only; the smallest p no longer gets the largest q.
    assert adjusted.round(8).tolist()[:2] == [0.00593871, 0.00593871]
    assert adjusted[3] == pytest.approx(0.0075061166500909205)


def test_fdr_matches_scipy() -> None:
    bh = pytest.importorskip("scipy.stats").false_discovery_control
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = np.round(rng.uniform(0, 1, int(rng.integers(1, 60))) ** 3, 2)  # rounding gives ties
        np.testing.assert_allclose(pr.stats.fdr(p), bh(p), rtol=0, atol=1e-12)


def test_reverse_complement_iupac_codes() -> None:
    assert pr.seqs.reverse_complement("ACGTRYKMBVDHSWN") == "NWSDHBVKMRYACGT"
    assert pr.seqs.reverse_complement("acgtrykmbvdhswn") == "nwsdhbvkmryacgt"
