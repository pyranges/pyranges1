from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyranges1.core.names import END_COL, START_COL, STRAND_COL
from pyranges1.core.pyranges_helpers import ensure_pyranges

if TYPE_CHECKING:
    from pyranges1 import PyRanges

_BLOCK_COLUMNS = ["BlockCount", "BlockSizes", "BlockStarts"]


def _block_list(values: "object") -> list[list[int]]:
    """Parse BED12 block lists ("10,20," with or without the trailing comma)."""
    return [[int(v) for v in str(text).rstrip(",").split(",") if v] for text in values]  # type: ignore[attr-defined]


def _explode_blocks(
    self: "PyRanges", *, block_index_col: str | None, use_strand: bool, drop_block_columns: bool
) -> "PyRanges":
    missing = [c for c in ("BlockSizes", "BlockStarts") if c not in self.columns]
    if missing:
        msg = f"explode_blocks needs the BED12 columns {missing}."
        raise ValueError(msg)
    sizes, offsets = _block_list(self["BlockSizes"]), _block_list(self["BlockStarts"])
    counts = np.array([len(s) for s in sizes], dtype=np.int64)
    if any(len(s) != len(o) for s, o in zip(sizes, offsets, strict=True)):
        msg = "BlockSizes and BlockStarts list different numbers of blocks."
        raise ValueError(msg)
    if "BlockCount" in self.columns and (self["BlockCount"].to_numpy() != counts).any():
        msg = "BlockCount does not match the number of blocks in BlockSizes."
        raise ValueError(msg)

    flat_sizes = np.fromiter((v for s in sizes for v in s), dtype=np.int64, count=int(counts.sum()))
    flat_offsets = np.fromiter((v for o in offsets for v in o), dtype=np.int64, count=int(counts.sum()))
    parent = np.repeat(np.arange(len(self)), counts)
    starts = self[START_COL].to_numpy(np.int64)[parent] + flat_offsets
    ends = starts + flat_sizes
    last = np.cumsum(counts) - 1
    has_blocks = counts > 0
    if (ends[last[has_blocks]] != self[END_COL].to_numpy(np.int64)[has_blocks]).any():
        msg = "The last block of an interval must end at its End, as BED12 requires."
        raise ValueError(msg)

    result = self.iloc[parent].copy()
    result[START_COL] = pd.Series(starts, index=result.index).astype(self[START_COL].dtype)
    result[END_COL] = pd.Series(ends, index=result.index).astype(self[END_COL].dtype)
    if block_index_col is not None:
        index = np.arange(len(parent)) - np.repeat(np.cumsum(counts) - counts, counts)
        if use_strand:
            minus = (self[STRAND_COL] == "-").to_numpy()[parent]
            index = np.where(minus, counts[parent] - 1 - index, index)
        result[block_index_col] = index
    if drop_block_columns:
        result = result.loc[:, [c for c in result.columns if c not in _BLOCK_COLUMNS]]
    return ensure_pyranges(result)
