from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from pyranges1.core.names import CHROM_COL, END_COL, START_COL

if TYPE_CHECKING:
    from pyranges1 import PyRanges


def _chromsizes_dict(chromsizes: Any) -> dict:
    """Chromosome sizes as a dict, from a dict, a DataFrame/PyRanges or a pyfaidx.Fasta, as clip_ranges takes them."""
    if isinstance(chromsizes, pd.DataFrame):
        return dict(zip(chromsizes[CHROM_COL], chromsizes[END_COL], strict=True))
    if isinstance(chromsizes, dict):
        return chromsizes
    faidx = cast("dict[str | int, list]", chromsizes)
    return {k: len(faidx[k]) for k in faidx.keys()}  # noqa: SIM118 -- pyfaidx.Fasta iterates records, not keys


def _allowed_segments(sizes: dict, exclude: "PyRanges | None") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the (chromosome, start, end) stretches of the genome outside `exclude`."""
    chromosomes, starts, ends = [], [], []
    merged = None if exclude is None else exclude.merge_overlaps(use_strand=False)
    for chromosome, size in sizes.items():
        if merged is None:
            cuts_start, cuts_end = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        else:
            on = merged[merged[CHROM_COL] == chromosome]
            cuts_start, cuts_end = on[START_COL].to_numpy(np.int64), on[END_COL].to_numpy(np.int64)
        gap_starts = np.concatenate([[0], np.minimum(cuts_end, size)])
        gap_ends = np.concatenate([np.clip(cuts_start, 0, size), [size]])
        keep = gap_ends > gap_starts
        chromosomes += [chromosome] * int(keep.sum())
        starts.append(gap_starts[keep])
        ends.append(gap_ends[keep])
    return (
        np.array(chromosomes, dtype=object),
        np.concatenate(starts) if starts else np.array([], dtype=np.int64),
        np.concatenate(ends) if ends else np.array([], dtype=np.int64),
    )


def _place(
    lengths: np.ndarray, seg_starts: np.ndarray, seg_ends: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a segment and a start for each length, uniformly over all placements that fit.

    A segment of length S holds S - L + 1 placements of an interval of length L. With
    the segments sorted longest first, the k that fit form a prefix, and the
    placements in the first j of them number prefix[j] - j * (L - 1): one uniform draw
    per interval and a binary search over j give the segment and the offset in it.
    Returns (segment index, start).
    """
    order = np.argsort(seg_ends - seg_starts, kind="stable")[::-1]
    seg_lengths = (seg_ends - seg_starts)[order]
    prefix = np.concatenate([[0], np.cumsum(seg_lengths)])
    fits = np.searchsorted(-seg_lengths, -lengths, side="right")  # segments with length >= L
    total = prefix[fits] - fits * (lengths - 1)
    if (total <= 0).any():
        longest = int(lengths[total <= 0].max())
        msg = f"An interval of length {longest} fits nowhere in the allowed part of the genome."
        raise ValueError(msg)
    draw = (rng.random(len(lengths)) * total).astype(np.int64)
    lo, hi = np.zeros(len(lengths), dtype=np.int64), fits.astype(np.int64)
    while (lo < hi).any():
        mid = (lo + hi) // 2
        nxt = np.minimum(mid + 1, len(prefix) - 1)  # rows already settled may sit at k
        beyond = prefix[nxt] - nxt * (lengths - 1) > draw
        hi = np.where(beyond & (lo < hi), mid, hi)
        lo = np.where(~beyond & (lo < hi), mid + 1, lo)
    offset = draw - (prefix[lo] - lo * (lengths - 1))
    return order[lo], seg_starts[order[lo]] + offset


def _shuffle(
    self: "PyRanges",
    chromsizes: Any,
    *,
    within_chromosomes: bool,
    exclude: "PyRanges | None",
    seed: "int | np.random.Generator | None",
) -> "PyRanges":
    rng = np.random.default_rng(seed)
    sizes = _chromsizes_dict(chromsizes)
    if missing := set(self[CHROM_COL]) - sizes.keys():
        msg = f"chromsizes has no size for {sorted(missing, key=str)}."
        raise ValueError(msg)
    seg_chromosomes, seg_starts, seg_ends = _allowed_segments(sizes, exclude)
    lengths = (self[END_COL] - self[START_COL]).to_numpy(np.int64)

    chromosomes = np.empty(len(self), dtype=object)
    starts = np.empty(len(self), dtype=np.int64)
    pools = (
        [(self[CHROM_COL].to_numpy() == c, seg_chromosomes == c) for c in pd.unique(self[CHROM_COL])]
        if within_chromosomes
        else [(np.ones(len(self), dtype=bool), np.ones(len(seg_starts), dtype=bool))]
    )
    for rows, segments in pools:
        chosen, placed = _place(lengths[rows], seg_starts[segments], seg_ends[segments], rng)
        chromosomes[rows] = seg_chromosomes[segments][chosen]
        starts[rows] = placed

    result = self.copy()
    original = self[CHROM_COL]
    if isinstance(original.dtype, pd.CategoricalDtype):
        categories = list(dict.fromkeys([*original.cat.categories, *sizes]))
        result[CHROM_COL] = pd.Categorical(chromosomes, categories=categories)
    else:
        result[CHROM_COL] = pd.Series(chromosomes, index=self.index).astype(original.dtype)
    result[START_COL] = pd.Series(starts, index=self.index).astype(self[START_COL].dtype)
    result[END_COL] = pd.Series(starts + lengths, index=self.index).astype(self[END_COL].dtype)
    return result
