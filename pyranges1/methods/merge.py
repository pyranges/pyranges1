from typing import TYPE_CHECKING

from pyranges1.core.names import END_COL, START_COL
from pyranges1.core.pyranges_helpers import ensure_rangeframe, factorize

if TYPE_CHECKING:
    from pyranges1.range_frame.range_frame import RangeFrame


def _merge(
    df: "RangeFrame",
    by: list[str],
    count_col: str | None = None,
    slack: int | None = None,
) -> "RangeFrame":
    from pyranges1._ruranges import require_ruranges

    ruranges = require_ruranges()

    from pyranges1.range_frame.range_frame import RangeFrame

    if df.empty:
        return df

    col_order = [col for col in df if col in [*by, START_COL, END_COL]]

    factorized = factorize(df, by)

    indices, start, end, counts = ruranges.numpy.merge(
        groups=factorized,
        starts=df.Start.to_numpy(),
        ends=df.End.to_numpy(),
        slack=slack or 0,
    )

    by_subset = df[col_order].take(indices)
    by_subset.loc[:, START_COL] = start
    by_subset.loc[:, END_COL] = end

    outpr = RangeFrame(by_subset)

    if count_col:
        outpr.insert(outpr.shape[1], count_col, counts)

    return ensure_rangeframe(outpr.reset_index(drop=True))
