from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyranges1.core.names import END_COL, START_COL
from pyranges1.core.pyranges_helpers import ensure_pyranges

if TYPE_CHECKING:
    from pyranges1 import PyRanges


def _coverage_ranges(self: "PyRanges", by: list[str], value_col: str | None, coverage_col: str) -> "PyRanges":
    """Maximal runs of constant, non-zero depth within each group of `by`, as bedtools genomecov -bg.

    One sorted sweep over every group at once: each interval adds its weight at
    Start and removes it at End, the running sum after the last event at a position
    is the depth up to the next one, and neighbouring runs of equal depth are merged.
    """
    weighted = value_col is not None
    columns = [*by, START_COL, END_COL, coverage_col]
    if self.empty:
        return ensure_pyranges(pd.DataFrame({c: self[c] if c in self.columns else [] for c in columns}))

    groups = self.groupby(by, sort=True, observed=True).ngroup().to_numpy()
    starts, ends = self[START_COL].to_numpy(np.int64), self[END_COL].to_numpy(np.int64)
    weight = self[value_col].to_numpy(np.float64) if weighted else np.ones(len(self), np.int64)

    group = np.concatenate([groups, groups])
    position = np.concatenate([starts, ends])
    change = np.concatenate([weight, -weight])
    opens = np.concatenate([np.ones(len(self), np.int64), -np.ones(len(self), np.int64)])
    order = np.lexsort((position, group))
    group, position, change, opens = group[order], position[order], change[order], opens[order]

    # Every group's events sum to zero, so one running sum serves them all.
    depth, open_count = np.cumsum(change), np.cumsum(opens)
    if weighted:  # no rounding residue where nothing is open
        depth = np.where(open_count == 0, 0.0, depth)
    last = np.r_[(group[1:] != group[:-1]) | (position[1:] != position[:-1]), True]
    group, position, depth, open_count = group[last], position[last], depth[last], open_count[last]

    within = group[:-1] == group[1:]
    run_group, run_start, run_end = group[:-1][within], position[:-1][within], position[1:][within]
    run_depth, run_open = depth[:-1][within], open_count[:-1][within]
    keep = (run_open > 0) & (run_depth != 0)
    run_group, run_start, run_end, run_depth = run_group[keep], run_start[keep], run_end[keep], run_depth[keep]

    joins = np.r_[
        False,
        (run_group[1:] == run_group[:-1]) & (run_start[1:] == run_end[:-1]) & (run_depth[1:] == run_depth[:-1]),
    ]
    first, final = ~joins, np.r_[~joins[1:], True]
    keys = self[by].iloc[np.unique(groups, return_index=True)[1]].reset_index(drop=True)
    result = keys.iloc[run_group[first]].reset_index(drop=True)
    result[START_COL] = run_start[first]
    result[END_COL] = run_end[final]
    result[coverage_col] = run_depth[first]
    return ensure_pyranges(result[columns])
