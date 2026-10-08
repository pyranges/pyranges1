from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pyranges1.core.names import END_COL, START_COL, STRAND_COL

if TYPE_CHECKING:
    from pyranges1 import PyRanges


def _minus(self: "PyRanges", *, use_strand: bool) -> np.ndarray:
    """Rows on the minus strand; with use_strand False, none."""
    if not use_strand:
        return np.zeros(len(self), dtype=bool)
    return (self[STRAND_COL] == "-").to_numpy()


def _with_coordinates(self: "PyRanges", starts: np.ndarray, ends: np.ndarray) -> "PyRanges":
    result = self.copy()
    result[START_COL] = pd.Series(starts, index=self.index).astype(self[START_COL].dtype)
    result[END_COL] = pd.Series(ends, index=self.index).astype(self[END_COL].dtype)
    return result


def _resize(self: "PyRanges", width: int, fix: Literal["start", "end", "center"], *, use_strand: bool) -> "PyRanges":
    if width < 1:
        msg = f"width must be at least 1, got {width}."
        raise ValueError(msg)
    starts, ends = self[START_COL].to_numpy(), self[END_COL].to_numpy()
    minus = _minus(self, use_strand=use_strand)
    if fix == "start":
        new_starts = np.where(minus, ends - width, starts)
    elif fix == "end":
        new_starts = np.where(minus, starts, ends - width)
    elif fix == "center":
        # As GenomicRanges::resize: the same on both strands, the odd base to the right.
        new_starts = starts + (ends - starts - width) // 2
    else:
        msg = f"fix must be 'start', 'end' or 'center', got {fix!r}."
        raise ValueError(msg)
    return _with_coordinates(self, new_starts, new_starts + width)


def _promoters(self: "PyRanges", upstream: int, downstream: int, *, use_strand: bool) -> "PyRanges":
    if upstream < 0 or downstream < 0 or upstream + downstream == 0:
        msg = f"upstream and downstream must be non-negative, and not both 0; got {upstream} and {downstream}."
        raise ValueError(msg)
    starts, ends = self[START_COL].to_numpy(), self[END_COL].to_numpy()
    minus = _minus(self, use_strand=use_strand)
    return _with_coordinates(
        self,
        np.where(minus, ends - downstream, starts - upstream),
        np.where(minus, ends + upstream, starts + downstream),
    )
