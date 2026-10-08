from typing import TYPE_CHECKING, Literal

import numpy as np

from pyranges1.core.names import END_COL, START_COL, STRAND_COL

if TYPE_CHECKING:
    from pyranges1 import PyRanges


def _sign_distance(result: "PyRanges", dist_col: str, suffix: str, signed: Literal["self", "other"]) -> "PyRanges":
    """Give the distances of a nearest_ranges result the sign bedtools closest -D a / -D b gives them.

    Negative means upstream. With "self" the sign follows each interval's strand: the
    match lies upstream of it. With "other" it follows the match's strand: the
    interval lies upstream of its match. Unstranded intervals count as "+".
    """
    starts, ends = result[START_COL].to_numpy(), result[END_COL].to_numpy()
    other_starts, other_ends = result[START_COL + suffix].to_numpy(), result[END_COL + suffix].to_numpy()
    # -1 when the match lies left of the interval, +1 right of it, 0 overlapping.
    side = np.where(other_ends <= starts, -1, np.where(other_starts >= ends, 1, 0))
    strand_col = STRAND_COL if signed == "self" else STRAND_COL + suffix
    minus = (result[strand_col] == "-").to_numpy() if strand_col in result.columns else np.zeros(len(result), bool)
    sign = np.where(minus, -side, side) if signed == "self" else np.where(minus, side, -side)
    signed_result = result.copy()
    signed_result[dist_col] = result[dist_col] * sign
    return signed_result
