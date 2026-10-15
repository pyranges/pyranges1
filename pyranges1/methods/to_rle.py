from typing import TYPE_CHECKING

from pyranges1.core.names import END_COL, START_COL

if TYPE_CHECKING:
    import contextlib

    from pyranges1 import PyRanges

    with contextlib.suppress(ImportError):
        from runlength import RleBatch


def _to_rle(
    ranges: "PyRanges",
    value_col: str | None = None,
    *,
    strand: bool = True,
    rpm: bool = False,
) -> "RleBatch":
    try:
        import runlength as rl
    except ImportError as e:
        msg = "Using the coverage method requires that runlength is installed: pip install runlength"
        raise ImportError(msg) from e

    ranges = ranges.remove_strand() if not strand else ranges
    keys = ranges[ranges.loc_columns]
    values = ranges[value_col] if value_col is not None else None
    # Positions no interval covers are exactly 0, also for floating-point values.
    coverage = rl.coverage_by_key(keys, ranges[START_COL], ranges[END_COL], values)

    if rpm and len(ranges) > 0:
        coverage = coverage * (1e6 / len(ranges))

    return coverage
