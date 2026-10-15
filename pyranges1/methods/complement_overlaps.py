from typing import TYPE_CHECKING

from pyranges1.core.pyranges_helpers import ensure_rangeframe, factorize_binary

if TYPE_CHECKING:
    from pyranges1.range_frame.range_frame import RangeFrame


def _complement_overlaps(
    df: "RangeFrame",
    df2: "RangeFrame",
    by: list[str],
    slack: int | None = None,
    *,
    preserve_input_order: bool = True,
) -> "RangeFrame":
    from pyranges1._ruranges import require_ruranges

    ruranges = require_ruranges()

    if df.empty:
        return df

    factorized, factorized2 = factorize_binary(df, df2, by)

    indices = ruranges.numpy.complement_overlaps(
        groups=factorized,
        starts=df.Start.to_numpy(),
        ends=df.End.to_numpy(),
        groups2=factorized2,
        starts2=df2.Start.to_numpy(),
        ends2=df2.End.to_numpy(),
        slack=slack or 0,
        sort_output=preserve_input_order,
    )

    return ensure_rangeframe(df.take(indices))
