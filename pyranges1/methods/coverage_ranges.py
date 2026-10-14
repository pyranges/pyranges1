from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyranges1.core.names import END_COL, START_COL
from pyranges1.core.pyranges_helpers import ensure_pyranges

if TYPE_CHECKING:
    from pyranges1 import PyRanges


def _coverage_ranges(self: "PyRanges", by: list[str], value_col: str | None, coverage_col: str) -> "PyRanges":
    """Maximal runs of constant, non-zero depth within each group of `by`, as bedtools genomecov -bg.

    The coverage is computed by runlength, as for to_rle and to_bigwig; its runs are
    already maximal, so only the uncovered ones are left out.
    """
    try:
        import runlength as rl  # type: ignore[import-not-found]
    except ImportError as e:
        msg = "coverage_ranges requires that runlength is installed: pip install runlength"
        raise ImportError(msg) from e

    columns = [*by, START_COL, END_COL, coverage_col]
    if self.empty:
        return ensure_pyranges(pd.DataFrame({c: self[c] if c in self.columns else [] for c in columns}))

    values = self[value_col] if value_col is not None else None
    # Positions no interval covers are exactly 0, also for floating-point values.
    coverage = rl.coverage_by_key(self[by], self[START_COL], self[END_COL], values)
    runs = coverage.to_pandas().rename(columns={"start": START_COL, "end": END_COL, "value": coverage_col})
    # Each group's runs come as one block, in position order; put the blocks in the
    # order of their keys, which is far cheaper than sorting every run by them.
    sizes = np.array([rle.num_runs for _, rle in coverage.items()], dtype=np.int64)
    firsts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    order = runs.iloc[firsts][by].reset_index(drop=True).sort_values(by, kind="stable").index.to_numpy()
    # Row numbers of the blocks in that order: each block's first row, counted up through it.
    ordered_sizes = sizes[order]
    block_offsets = np.concatenate([[0], np.cumsum(ordered_sizes)[:-1]])
    rows = np.repeat(firsts[order] - block_offsets, ordered_sizes) + np.arange(ordered_sizes.sum())
    runs = runs.iloc[rows]
    return ensure_pyranges(runs[runs[coverage_col] != 0][columns].reset_index(drop=True))
