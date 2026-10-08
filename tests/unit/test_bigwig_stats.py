import numpy as np
import pandas as pd
import pytest

import pyranges1 as pr

pyBigWig = pytest.importorskip("pyBigWig")

SIZES = {"chr1": 300, "chr2": 120}


def _write_track(path, rng) -> dict[str, np.ndarray]:
    """A random bigWig, and the same signal as one value per base (NaN where there is none)."""
    signal = {}
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader(list(SIZES.items()))
    for chrom, size in SIZES.items():
        per_base = np.full(size, np.nan)
        pos = int(rng.integers(0, 5))
        starts, ends, values = [], [], []
        while pos < size - 2:
            end = min(pos + int(rng.integers(1, 12)), size)
            value = float(np.float32(rng.uniform(-2, 5)))
            starts.append(pos), ends.append(end), values.append(value)
            per_base[pos:end] = value
            pos = end + int(rng.integers(0, 6))
        bw.addEntries([chrom] * len(starts), starts, ends=ends, values=values)
        signal[chrom] = per_base
    bw.close()
    return signal


def _expected(values: np.ndarray, stat: str, n_bins: int) -> np.ndarray:
    edges = np.linspace(0, len(values), n_bins + 1).astype(int)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        chunk = values[lo:hi][~np.isnan(values[lo:hi])]
        if stat == "sum":
            out.append(chunk.sum() if len(chunk) else np.nan)
        elif stat == "std":
            out.append(chunk.std(ddof=1) if len(chunk) > 1 else np.nan)  # sample std, as pyBigWig
        else:
            out.append(getattr(np, stat)(chunk) if len(chunk) else np.nan)
    return np.array(out)


@pytest.mark.parametrize("missing", [None, 0.0])
@pytest.mark.parametrize("bins", [None, 1, 3, 7])
def test_bigwig_stats_matches_the_signal_base_by_base(tmp_path, missing, bins) -> None:
    rng = np.random.default_rng(bins or 0)
    signal = _write_track(tmp_path / "a.bw", rng)
    n = 60
    starts = rng.integers(0, 320, n)
    gr = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": rng.choice(["chr1", "chr2", "chrX"], n),  # chrX is not in the file
                "Start": starts,
                "End": starts + rng.integers(1, 40, n),
                "Strand": rng.choice(["+", "-"], n),
            },
            index=rng.integers(0, 5, n),
        )
    )
    stats = ["mean", "max", "min", "sum", "std"]
    pad = 3
    got = gr.bigwig_stats(tmp_path / "a.bw", stats, missing=missing, pad=pad, bins=bins)
    assert got.index.equals(gr.index)
    for k, row in enumerate(gr.itertuples()):
        per_base = signal.get(row.Chromosome)
        lo = max(row.Start - pad, 0)
        hi = min(row.End + pad, len(per_base) if per_base is not None else 0)
        for stat in stats:
            if per_base is None or hi <= lo:
                want = np.full(bins or 1, np.nan)
            else:
                window = per_base[lo:hi].copy()
                if missing is not None:
                    window[np.isnan(window)] = missing
                want = _expected(window, stat, bins or 1)
                if bins is not None and row.Strand == "-":
                    want = want[::-1]
            names = [stat] if bins is None else [f"{stat}_{i}" for i in range(bins)]
            np.testing.assert_allclose(got[names].iloc[k].to_numpy(dtype=float), want, atol=1e-9, err_msg=f"{stat} row {k}")


def test_bigwig_stats_agrees_with_pybigwig(tmp_path) -> None:
    rng = np.random.default_rng(1)
    _write_track(tmp_path / "a.bw", rng)
    bw = pyBigWig.open(str(tmp_path / "a.bw"))
    gr = pr.PyRanges({"Chromosome": ["chr1"] * 4 + ["chr2"], "Start": [0, 10, 40, 150, 5], "End": [300, 90, 41, 260, 100]})
    got = gr.bigwig_stats(tmp_path / "a.bw", ["mean", "max", "min", "sum", "std"])
    for k, row in enumerate(gr.itertuples()):
        for stat in ("mean", "max", "min", "sum", "std"):
            (want,) = bw.stats(row.Chromosome, row.Start, row.End, type=stat, exact=True)
            assert got[stat].iloc[k] == pytest.approx(np.nan if want is None else want, nan_ok=True)


def test_bigwig_stats_several_tracks_and_prefix(tmp_path) -> None:
    rng = np.random.default_rng(2)
    _write_track(tmp_path / "a.bw", rng)
    _write_track(tmp_path / "b.bw", rng)
    gr = pr.PyRanges({"Chromosome": ["chr1", "chr2"], "Start": [0, 10], "End": [50, 60]})
    both = gr.bigwig_stats({"A": tmp_path / "a.bw", "B": tmp_path / "b.bw"}, ["mean", "max"], prefix="x_")
    assert list(both.columns[-4:]) == ["x_A_mean", "x_A_max", "x_B_mean", "x_B_max"]
    alone = gr.bigwig_stats(tmp_path / "b.bw", ["mean", "max"])
    np.testing.assert_allclose(both[["x_B_mean", "x_B_max"]].to_numpy(), alone[["mean", "max"]].to_numpy())


def test_bigwig_stats_rejects_unknown_stats(tmp_path) -> None:
    _write_track(tmp_path / "a.bw", np.random.default_rng(3))
    gr = pr.PyRanges({"Chromosome": ["chr1"], "Start": [0], "End": [10]})
    with pytest.raises(ValueError, match="median"):
        gr.bigwig_stats(tmp_path / "a.bw", "median")
    with pytest.raises(ValueError, match="bins"):
        gr.bigwig_stats(tmp_path / "a.bw", bins=0)
