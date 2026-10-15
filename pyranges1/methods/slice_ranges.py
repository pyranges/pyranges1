from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyranges1.core.names import (
    END_COL,
    FORWARD_STRAND,
    START_COL,
    STRAND_COL,
)
from pyranges1.core.pyranges_helpers import factorize

if TYPE_CHECKING:
    import pyranges1 as pr


def _spliced_subseq(
    df: "pr.PyRanges",
    *,
    by: list[str],
    force_plus_strand: bool = False,
    start: int | Sequence[int] | np.ndarray = 0,
    end: int | Sequence[int] | np.ndarray | None = None,
    preserve_input_order: bool = True,
) -> pd.DataFrame:
    from pyranges1._ruranges import require_ruranges

    ruranges = require_ruranges()

    if df.empty:
        return df

    chrs = factorize(df, by) if by else np.arange(len(df), dtype=np.uint32)

    outidx, outstarts, outends = ruranges.numpy.spliced_subsequence(
        groups=chrs,
        starts=df[START_COL].to_numpy(),
        ends=df[END_COL].to_numpy(),
        strand_flags=(df[STRAND_COL] == FORWARD_STRAND).to_numpy()
        if (STRAND_COL in df and not force_plus_strand)
        else np.ones(len(df), dtype=bool),
        start=start,
        end=end,
        force_plus_strand=force_plus_strand,
        sort_output=preserve_input_order,
    )

    outdf = df.take(outidx)
    outdf.loc[:, START_COL] = outstarts
    outdf.loc[:, END_COL] = outends

    return outdf


# deprecated. Due to unexpected behavior, this is not used anymore (see slice_ranges with count_introns=False)
def _subseq(
    df: "pr.PyRanges",
    *,
    by: list[str],
    start: int = 0,
    end: int | None = None,
    force_plus_strand: bool = False,
) -> pd.DataFrame:
    from pyranges1._ruranges import require_ruranges

    ruranges = require_ruranges()

    if df.empty:
        return df

    chrs = factorize(df, by)

    outidx, outstarts, outends = ruranges.numpy.subsequence_numpy(
        chrs,
        df[START_COL].to_numpy(),
        df[END_COL].to_numpy(),
        (df[STRAND_COL] == FORWARD_STRAND).to_numpy()
        if (STRAND_COL in df and not force_plus_strand)
        else np.ones(len(df), dtype=bool),
        start=start,
        end=end,
        force_plus_strand=force_plus_strand,
    )

    outdf = df.take(outidx)
    outdf.loc[:, START_COL] = outstarts
    outdf.loc[:, END_COL] = outends

    return outdf
