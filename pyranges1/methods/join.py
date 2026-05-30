from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyranges1.core.names import VALID_BY_TYPES, VALID_JOIN_TYPE, VALID_OVERLAP_TYPE
from pyranges1.methods.overlap import _both_idxs

if TYPE_CHECKING:
    from pyranges1.range_frame.range_frame import RangeFrame


def _both_dfs(
    df: "RangeFrame",
    df2: "RangeFrame",
    *,
    by: VALID_BY_TYPES = None,
    multiple: VALID_OVERLAP_TYPE = "all",
    contained: bool = False,
    slack: int = 0,
    join_type: VALID_JOIN_TYPE,
    suffix: str,
    preserve_input_order: bool = True,
) -> pd.DataFrame:
    _self_indexes, _other_indexes = _both_idxs(
        df=df,
        df2=df2,
        by=by,
        multiple=multiple,
        contained=contained,
        slack=slack,
        preserve_input_order=preserve_input_order,
    )
    expected_columns = [*df.head(0).join(df2.head(0), how="inner", rsuffix=suffix).columns]
    df2.columns = expected_columns[df.shape[1] :]
    # _self_indexes / _other_indexes are positional (from the overlap engine). Take the
    # matching rows, then temporarily reset both indexes to a RangeIndex so the column-wise
    # join pairs them by position rather than by label.
    _df = df.take(_self_indexes)  # type: ignore[arg-type]
    _df.index = pd.RangeIndex(len(_df))
    _df2 = pd.DataFrame(df2).take(_other_indexes)  # type: ignore[arg-type]
    _df2.index = pd.RangeIndex(len(_df2))
    j = _df.join(_df2, how="inner")
    # Restore the original input index, like overlap()/nearest_ranges()/etc. The primary
    # side of the join keeps its index: self for inner/left/outer, other for right.
    j.index = df2.index[_other_indexes] if join_type == "right" else df.index[_self_indexes]

    if join_type == "inner":
        return j

    if join_type == "left":
        return pd.concat([j, _missing_rows_left(_self_indexes, df)])

    if join_type == "right":
        return pd.concat([j, _missing_rows_right(_other_indexes, df2)])

    if join_type == "outer":
        return pd.concat([j, _missing_rows_left(_self_indexes, df), _missing_rows_right(_other_indexes, df2)])

    msg = f"Invalid join type: {join_type}"
    raise ValueError(msg)


def _missing_rows_left(_self_indexes, df) -> pd.DataFrame:
    # Rows of self without an overlap are the positions not present in _self_indexes.
    # df.take keeps their original index labels, preserving the input index.
    missing_positions = np.setdiff1d(np.arange(len(df)), _self_indexes)
    return df.take(missing_positions)


def _missing_rows_right(_other_indexes, df2) -> pd.DataFrame:
    missing_positions = np.setdiff1d(np.arange(len(df2)), _other_indexes)
    return pd.DataFrame(df2).take(missing_positions)
