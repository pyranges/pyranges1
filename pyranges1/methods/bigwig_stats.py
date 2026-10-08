from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyranges1.core.names import CHROM_COL, END_COL, START_COL, STRAND_COL
from pyranges1.core.pyranges_helpers import ensure_pyranges

if TYPE_CHECKING:
    from pathlib import Path

    from pyranges1 import PyRanges

STATS = ("mean", "max", "min", "sum", "std")


def _reduce(values: np.ndarray, n_bins: int, stat: str) -> np.ndarray:
    """Reduce one region's per-base values into `n_bins` equal-width bins, ignoring NaN.

    All bins at once with `ufunc.reduceat`. A bin with no values is NaN for every
    statistic, as pyBigWig reports it. `std` is the sample standard deviation, also
    as pyBigWig reports it, so a bin with one value is NaN too. Bins differ in width
    by at most 1 bp.
    """
    edges = np.linspace(0, len(values), n_bins + 1).astype(np.int64)
    starts = edges[:-1]
    # reduceat returns the element at a zero-width segment rather than the identity.
    empty = np.diff(edges) == 0
    present = ~np.isnan(values)
    counts = np.where(empty, 0, np.add.reduceat(present.astype(np.int64), starts))
    sums = np.where(empty, 0.0, np.add.reduceat(np.where(present, values, 0.0), starts))
    if stat == "sum":
        return np.where(counts > 0, sums, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        if stat == "mean":
            return means
        if stat == "std":
            squares = np.where(empty, 0.0, np.add.reduceat(np.where(present, values * values, 0.0), starts))
            variance = (squares - counts * means**2) / np.maximum(counts - 1, 1)
            return np.where(counts > 1, np.sqrt(np.maximum(variance, 0.0)), np.nan)
    ufunc, fill = (np.maximum, -np.inf) if stat == "max" else (np.minimum, np.inf)
    return np.where(counts > 0, ufunc.reduceat(np.where(present, values, fill), starts), np.nan)


def _read_track(
    bw: object,
    regions: tuple[np.ndarray, np.ndarray, np.ndarray],
    stat_list: list[str],
    n_bins: int,
    missing: float | None,
    reverse: np.ndarray,
) -> dict[str, np.ndarray]:
    """One open bigWig summarised over every region: a rows x bins array per statistic."""
    chromosomes, starts, ends = regions
    sizes = bw.chroms()  # type: ignore[attr-defined]
    columns = {stat: np.full((len(chromosomes), n_bins), np.nan) for stat in stat_list}
    for row, (chromosome, start, end) in enumerate(zip(chromosomes, starts, ends, strict=True)):
        # Clamped to the chromosome; a region off it, or on a chromosome the file
        # does not have, stays NaN.
        lo, hi = max(int(start), 0), min(int(end), sizes.get(chromosome, 0))
        if hi <= lo:
            continue
        values = np.asarray(bw.values(chromosome, lo, hi, numpy=True), dtype=float)  # type: ignore[attr-defined]
        if missing is not None:
            values[np.isnan(values)] = missing
        for stat in stat_list:
            reduced = _reduce(values, n_bins, stat)
            columns[stat][row] = reduced[::-1] if reverse[row] else reduced
    return columns


def _bigwig_stats(
    self: "PyRanges",
    path: "str | Path | Mapping[str, str | Path]",
    stats: str | Iterable[str],
    missing: float | None,
    pad: int,
    bins: int | None,
    reverse: np.ndarray,
    prefix: str,
) -> "PyRanges":
    try:
        import pyBigWig  # type: ignore[import]
    except ImportError:
        msg = "pyBigWig must be installed to read bigwigs. Use `pip install pyBigWig` to install it."
        raise ImportError(msg) from None

    stat_list = [stats] if isinstance(stats, str) else list(stats)
    unknown = [s for s in stat_list if s not in STATS]
    if unknown:
        msg = f"Unknown stats {unknown}; choose from {list(STATS)}."
        raise ValueError(msg)
    if bins is not None and bins < 1:
        msg = f"bins must be at least 1, got {bins}."
        raise ValueError(msg)

    tracks = list(path.items()) if isinstance(path, Mapping) else [("", path)]
    n_bins = bins or 1
    width = len(str(n_bins - 1))
    regions = (
        self[CHROM_COL].astype(str).to_numpy(),
        (self[START_COL] - pad).to_numpy(),
        (self[END_COL] + pad).to_numpy(),
    )

    new_columns: dict[str, np.ndarray] = {}
    for label, track in tracks:
        bw = pyBigWig.open(str(track))
        columns = _read_track(bw, regions, stat_list, n_bins, missing, reverse)
        bw.close()
        name = f"{prefix}{label}_" if label else prefix
        for stat in stat_list:
            names = [f"{name}{stat}"] if bins is None else [f"{name}{stat}_{i:0{width}d}" for i in range(n_bins)]
            new_columns.update(zip(names, columns[stat].T, strict=True))
    # One concat, not a column at a time: a regions x bins matrix is hundreds of columns.
    return ensure_pyranges(pd.concat([self, pd.DataFrame(new_columns, index=self.index)], axis=1))


def _reverse_rows(self: "PyRanges", *, use_strand: bool) -> np.ndarray:
    """Rows whose bins run 3' to 5' in genome order, so bin 0 should be read from the end."""
    if not use_strand:
        return np.zeros(len(self), dtype=bool)
    return (self[STRAND_COL] == "-").to_numpy()
