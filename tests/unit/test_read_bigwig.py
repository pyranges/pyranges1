"""Reproducible round-trip test for :func:`pyranges1.read_bigwig`.

Rather than depend on the committed ``bigwig.bw`` fixture (and a repr-based doctest
that silently tracks pandas' default string dtype), this builds a bigWig from a
**fixed random seed**, reads it back, and asserts the data round-trips exactly and
the ``Chromosome`` column is categorical — the convention every reader follows.

The categorical assertion is the regression guard: pandas 2.x would otherwise hand
back ``object`` and pandas 3.0 the new ``str`` dtype, both of which diverge from
read_bed / read_gtf / read_pairs.
"""

import numpy as np
import pandas as pd
import pytest

import pyranges1 as pr

pyBigWig = pytest.importorskip("pyBigWig")

SEED = 0
CHROM_SIZES = {"chr1": 2000, "chr2": 1000, "chr10": 1500}


def _seeded_intervals(rng: np.random.Generator) -> dict[str, list[tuple[int, int, float]]]:
    """Non-overlapping (start, end, value) intervals per chromosome from ``rng``.

    A >=1 bp gap between intervals guarantees bigWig never merges adjacent equal-value
    runs, so what we write is exactly what read_bigwig must return.
    """
    out: dict[str, list[tuple[int, int, float]]] = {}
    for chrom, size in CHROM_SIZES.items():
        rows: list[tuple[int, int, float]] = []
        pos = int(rng.integers(0, 5))
        while pos < size - 12:
            width = int(rng.integers(1, 10))
            end = min(pos + width, size)
            value = float(np.float32(rng.uniform(0.0, 5.0)))  # float32: bigWig's storage dtype
            rows.append((pos, end, value))
            pos = end + 1 + int(rng.integers(0, 5))  # >=1 bp gap -> no merging
        out[chrom] = rows
    return out


def _write_bigwig(path: str, intervals: dict[str, list[tuple[int, int, float]]]) -> None:
    bw = pyBigWig.open(path, "w")
    bw.addHeader([(c, CHROM_SIZES[c]) for c in CHROM_SIZES])  # header order = write order
    for chrom, rows in intervals.items():
        if not rows:
            continue
        starts = [s for s, _, _ in rows]
        ends = [e for _, e, _ in rows]
        values = [v for _, _, v in rows]
        bw.addEntries([chrom] * len(rows), starts, ends=ends, values=values)
    bw.close()


def test_read_bigwig_roundtrip_seeded(tmp_path):
    rng = np.random.default_rng(SEED)
    intervals = _seeded_intervals(rng)
    path = str(tmp_path / "seeded.bw")
    _write_bigwig(path, intervals)

    gr = pr.read_bigwig(path)

    # Chromosome is categorical, like every other reader (the regression guard).
    assert isinstance(gr["Chromosome"].dtype, pd.CategoricalDtype)
    assert set(gr.columns) >= {"Chromosome", "Start", "End", "Value"}

    expected = sorted((c, s, e, v) for c, rows in intervals.items() for s, e, v in rows)
    got = sorted(
        zip(
            gr["Chromosome"].astype(str).tolist(),
            gr["Start"].tolist(),
            gr["End"].tolist(),
            gr["Value"].tolist(),
            strict=True,
        )
    )
    assert len(got) == len(expected)
    for (gc, gs, ge, gv), (ec, es, ee, ev) in zip(got, expected, strict=True):
        assert (gc, gs, ge) == (ec, es, ee)
        assert gv == pytest.approx(ev, abs=1e-4)  # float32 storage tolerance


def test_read_bigwig_seeded_is_deterministic(tmp_path):
    """Same seed -> identical file -> identical frame (true reproducibility)."""
    out = []
    for name in ("a.bw", "b.bw"):
        rng = np.random.default_rng(SEED)
        path = str(tmp_path / name)
        _write_bigwig(path, _seeded_intervals(rng))
        out.append(pr.read_bigwig(path))
    pd.testing.assert_frame_equal(out[0], out[1])  # PyRanges is a pd.DataFrame subclass
