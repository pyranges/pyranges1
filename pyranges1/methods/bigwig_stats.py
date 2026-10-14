from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pyranges1.core.names import CHROM_COL, END_COL, START_COL, STRAND_COL
from pyranges1.core.pyranges_helpers import ensure_pyranges

if TYPE_CHECKING:
    from pathlib import Path

    from pyranges1 import PyRanges

STATS = ("mean", "max", "min", "sum", "std")

# Up to this many separate stretches of a chromosome are read one by one through
# the bigWig index; beyond it, reading the whole chromosome at once is faster.
MAX_REGION_READS = 2000


def _entries(bw: Any, chromosome: str, starts: np.ndarray, ends: np.ndarray, size: int) -> np.ndarray:
    """Read the bigWig entries on one chromosome under the intervals, as an (n, 3) start/end/value array."""
    order = np.argsort(starts, kind="stable")
    starts, ends = np.clip(starts[order], 0, size), np.clip(ends[order], 0, size)
    # Merge the intervals into the stretches to read; touching ones are merged too.
    new_block = np.r_[True, starts[1:] > np.maximum.accumulate(ends)[:-1]]
    block_starts = starts[new_block]
    block_ends = np.maximum.reduceat(ends, np.flatnonzero(new_block)) if len(ends) else ends
    if len(block_starts) > MAX_REGION_READS:
        intervals = bw.intervals(chromosome)  # type: ignore[attr-defined]
        return np.array(intervals or [], dtype=float).reshape(-1, 3)
    parts = [
        bw.intervals(chromosome, int(s), int(e))  # type: ignore[attr-defined]
        for s, e in zip(block_starts, block_ends, strict=True)
        if e > s
    ]
    entries = np.array([entry for part in parts if part for entry in part], dtype=float).reshape(-1, 3)
    # An entry that crosses from one stretch into the next is read with both.
    _, first = np.unique(entries[:, 0], return_index=True)
    return entries[first]


def _read_track(
    path: "str | Path",
    chromosomes: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    missing: float | None,
) -> tuple[Any, dict[str, int]]:
    """Read the signal under the intervals into one Rle per chromosome, null where the bigWig has no data."""
    import pyarrow as pa
    import pyBigWig  # type: ignore[import]
    import runlength as rl  # type: ignore[import-not-found]

    bw = pyBigWig.open(str(path))
    sizes = bw.chroms()
    names, rles = [], []
    for chromosome in dict.fromkeys(chromosomes.tolist()):
        if chromosome not in sizes:
            continue
        on_chromosome = chromosomes == chromosome
        size = sizes[chromosome]
        entries = _entries(bw, chromosome, starts[on_chromosome], ends[on_chromosome], size)
        if len(entries):
            rle = rl.coverage(
                entries[:, 0].astype(np.int64), entries[:, 1].astype(np.int64), entries[:, 2], length=size, fill=None
            )
        else:
            rle = rl.Rle([size], pa.array([None], pa.float64()))
        names.append(chromosome)
        rles.append(rle if missing is None else rle.fill_null(missing))
    bw.close()
    return rl.RleBatch.from_rles(pa.table({CHROM_COL: pa.array(names, pa.string())}), rles), sizes


def _slice_agg(signal: Any, bins: tuple[object, np.ndarray, np.ndarray], agg: str) -> np.ndarray:
    """Compute one statistic over every bin, as floats with NaN for null."""
    import pyarrow as pa

    result = signal.slice_agg(*bins, agg)  # type: ignore[attr-defined]
    return pa.array(result).to_numpy(zero_copy_only=False).astype(float)


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
        import pyBigWig  # type: ignore[import]  # noqa: F401
    except ImportError:
        msg = "pyBigWig must be installed to read bigwigs. Use `pip install pyBigWig` to install it."
        raise ImportError(msg) from None
    try:
        import pyarrow as pa
        import runlength  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        msg = "bigwig_stats requires runlength and pyarrow: pip install runlength pyarrow"
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
    chromosomes = self[CHROM_COL].astype(str).to_numpy()
    starts = (self[START_COL] - pad).to_numpy(dtype=np.int64)
    ends = (self[END_COL] + pad).to_numpy(dtype=np.int64)

    new_columns: dict[str, np.ndarray] = {}
    for label, track in tracks:
        signal, sizes = _read_track(track, chromosomes, starts, ends, missing)
        # Clamped to the chromosome; a region off it, or on a chromosome the file
        # does not have, stays NaN.
        lo = np.maximum(starts, 0)
        hi = np.maximum(np.minimum(ends, np.array([sizes.get(c, 0) for c in chromosomes], dtype=np.int64)), lo)
        # Equal-width bins, differing in width by at most 1 bp, all queried at once.
        edges = lo[:, None] + ((hi - lo)[:, None] * np.arange(n_bins + 1)) // n_bins
        keys = pa.table({CHROM_COL: pa.array(np.repeat(chromosomes, n_bins), pa.string())})
        bins_of = (keys, edges[:, :-1].ravel(), edges[:, 1:].ravel())
        counts = _slice_agg(signal, bins_of, "count")
        name = f"{prefix}{label}_" if label else prefix
        for stat in stat_list:
            values = _slice_agg(signal, bins_of, stat)
            # A bin with no values is NaN for every statistic, and std, the sample
            # standard deviation, needs two; as pyBigWig reports them.
            values[counts < (2 if stat == "std" else 1)] = np.nan
            matrix = values.reshape(-1, n_bins)
            matrix[reverse] = matrix[reverse, ::-1]
            names = [f"{name}{stat}"] if bins is None else [f"{name}{stat}_{i:0{width}d}" for i in range(n_bins)]
            new_columns.update(zip(names, matrix.T, strict=True))
    # One concat, not a column at a time: a regions x bins matrix is hundreds of columns.
    return ensure_pyranges(pd.concat([self, pd.DataFrame(new_columns, index=self.index)], axis=1))


def _reverse_rows(self: "PyRanges", *, use_strand: bool) -> np.ndarray:
    """Rows whose bins run 3' to 5' in genome order, so bin 0 should be read from the end."""
    if not use_strand:
        return np.zeros(len(self), dtype=bool)
    return (self[STRAND_COL] == "-").to_numpy()
