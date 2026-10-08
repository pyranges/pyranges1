from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pyranges1.core.names import CHROM_COL, END_COL, START_COL, STRAND_COL
from pyranges1.core.pyranges_helpers import ensure_rangeframe

if TYPE_CHECKING:
    from pyranges1.range_frame.range_frame import RangeFrame


def _complement(
    df: "RangeFrame",
    *,
    by: list[str],
    slack: int = 0,
    chromsizes_col: str | None = None,
    chromsizes: "dict[str | int, int] | None" = None,
    include_first_interval: bool = False,
) -> "RangeFrame":
    from pyranges1._ruranges import require_ruranges

    ruranges = require_ruranges()

    from pyranges1.range_frame.range_frame import RangeFrame

    if chromsizes and chromsizes_col:
        missing = sorted({*df[chromsizes_col].astype(object)} - set(chromsizes), key=str)
        if missing:
            msg = f"chromsizes has no size for {missing}."
            raise ValueError(msg)

    # The complement of the whole genome also contains every chromosome the
    # intervals leave untouched.
    whole_genome = bool(chromsizes) and include_first_interval and chromsizes_col == CHROM_COL
    whole_genome = whole_genome and set(by) <= {CHROM_COL, STRAND_COL}

    if df.empty:
        return _with_uncovered_chromosomes(df, df, by, chromsizes) if whole_genome and chromsizes else df

    if include_first_interval:
        below_origin = int((df[START_COL].to_numpy(copy=False) < 0).sum())
        if below_origin:
            msg = (
                f"complement_ranges(include_first_interval=True) found {below_origin} "
                f"interval{'s' if below_origin != 1 else ''} starting below zero. "
                "The first complement interval runs from coordinate 0 up to the first "
                "interval, which is not a valid range when that interval starts below "
                "the origin."
            )
            raise ValueError(msg)

    col_order = [col for col in df if col in [*by, START_COL, END_COL]]

    factorized = pd.Series(np.zeros(len(df), dtype=np.uint32)) if not by else df.groupby(by).ngroup().astype(np.uint32)
    # The Rust kernel must see the *same* dtype that we pass for starts/ends
    pos_dtype: np.dtype[Any] = df[START_COL].to_numpy(copy=False).dtype
    grp_dtype: np.dtype[Any] = factorized.to_numpy(copy=False).dtype

    if chromsizes and chromsizes_col:
        the_chromsizes_col = df[chromsizes_col]
        if isinstance(the_chromsizes_col.dtype, pd.CategoricalDtype):
            # drop the categorical metadata; cheap view, no copy
            the_chromsizes_col = the_chromsizes_col.astype(object)

        #  vectorised lookup with no FutureWarning
        lengths = the_chromsizes_col.map(chromsizes).astype(pos_dtype)  # faster/cleaner than replace

        group_to_len = pd.DataFrame(
            {
                "group_id": factorized,
                "length": lengths,
            }
        ).drop_duplicates()
        chrom_len_ids = group_to_len["group_id"].to_numpy(grp_dtype)
        chrom_lens = group_to_len["length"].to_numpy(pos_dtype)

    else:
        chrom_len_ids = np.array([], dtype=grp_dtype)
        chrom_lens = np.array([], dtype=pos_dtype)

    chrs, start, end, idxs = ruranges.numpy.complement(
        groups=factorized.to_numpy(),
        starts=df.Start.to_numpy(),
        ends=df.End.to_numpy(),
        slack=slack,
        chrom_len_ids=chrom_len_ids,  # type: ignore[arg-type]
        chrom_lens=chrom_lens,
        include_first_interval=include_first_interval,
    )

    # An interval of non-positive length is not a valid range. The kernel emits one
    # when the last interval ends exactly on the chromosome size: a terminal gap of
    # zero length at Start == End == size.
    keep = end > start
    if not keep.all():
        chrs, start, end, idxs = chrs[keep], start[keep], end[keep], idxs[keep]

    ids = df.take(idxs)  # type: ignore[arg-type]

    result = RangeFrame({CHROM_COL: chrs, START_COL: start, END_COL: end} | {_by: ids[_by] for _by in by})[col_order]

    if whole_genome and chromsizes:
        result = _with_uncovered_chromosomes(result, df, by, chromsizes)

    return ensure_rangeframe(result.reset_index(drop=True))


def _with_uncovered_chromosomes(
    result: "RangeFrame",
    df: "RangeFrame",
    by: list[str],
    chromsizes: "dict[str | int, int]",
) -> "RangeFrame":
    """Append [0, size) for each chromosome in chromsizes that df has no interval on.

    With strands, a chromosome is uncovered on each strand the intervals use but
    do not reach it on.
    """
    from pyranges1.range_frame.range_frame import RangeFrame

    strands = list(dict.fromkeys(df[STRAND_COL])) if STRAND_COL in by else [None]
    present = set(df[by].astype(object).itertuples(index=False, name=None))
    rows = [
        {CHROM_COL: chromosome, START_COL: 0, END_COL: size} | ({} if strand is None else {STRAND_COL: strand})
        for chromosome, size in chromsizes.items()
        for strand in strands
        if ((chromosome,) if strand is None else (chromosome, strand)) not in present
    ]
    if not rows:
        return result
    uncovered = pd.DataFrame(rows).astype({START_COL: result[START_COL].dtype, END_COL: result[END_COL].dtype})
    return RangeFrame(pd.concat([result, uncovered[list(result.columns)]], ignore_index=True))
