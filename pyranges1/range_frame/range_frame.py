import inspect
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from pyranges1.core.names import (
    BY_ENTRY_IN_KWARGS,
    END_COL,
    JOIN_SUFFIX,
    RANGE_COLS,
    SKIP_IF_DF_EMPTY_TYPE,
    SKIP_IF_EMPTY_ANY,
    SKIP_IF_EMPTY_BOTH,
    SKIP_IF_EMPTY_LEFT,
    SKIP_IF_EMPTY_RIGHT,
    START_COL,
    VALID_BY_TYPES,
    VALID_COMBINE_OPTIONS,
    VALID_DIRECTION_TYPE,
    VALID_JOIN_TYPE,
    VALID_OVERLAP_TYPE,
    CombineIntervalColumnsOperation,
)
from pyranges1.core.pyranges_helpers import arg_to_list, factorize, factorize_binary
from pyranges1.core.tostring import tohtml, tostring
from pyranges1.methods.complement_overlaps import _complement_overlaps
from pyranges1.methods.join import _both_dfs
from pyranges1.methods.merge import _merge
from pyranges1.methods.sort import sort_factorize_dict
from pyranges1.range_frame.range_frame_validator import InvalidRangesReason


def should_skip_operation(df: pd.DataFrame, *, df2: pd.DataFrame, skip_if_empty: SKIP_IF_DF_EMPTY_TYPE) -> bool:
    """Whether to skip operation because one or more dfs are empty."""
    if df.empty and df2.empty:
        return skip_if_empty in {SKIP_IF_EMPTY_BOTH, SKIP_IF_EMPTY_ANY, SKIP_IF_EMPTY_LEFT, SKIP_IF_EMPTY_RIGHT}
    if df.empty:
        return skip_if_empty in {SKIP_IF_EMPTY_LEFT, SKIP_IF_EMPTY_ANY}
    if df2.empty:
        return skip_if_empty in {SKIP_IF_EMPTY_RIGHT, SKIP_IF_EMPTY_ANY}
    return False


TRangeFrame = TypeVar("TRangeFrame", bound="RangeFrame")


class RangeFrame(pd.DataFrame):
    """Class for range based operations.

    A table with Start and End columns. Parent class of PyRanges. Subclass of pandas DataFrame.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(
        self,
        **kwargs: int | None,
    ) -> str:  # , max_col_width: int | None = None, max_total_width: int | None = None) -> str:
        """Return string representation."""
        str_repr = tostring(
            self,
            max_col_width=kwargs.get("max_col_width"),
            max_total_width=kwargs.get("max_total_width"),
        )
        if reasons := InvalidRangesReason.formatted_reasons_list(self):
            return f"{str_repr}\nInvalid ranges:\n{reasons}"
        return str_repr

    def __repr__(self, max_col_width: int | None = None, max_total_width: int | None = None) -> str:
        return self.__str__(max_col_width=max_col_width, max_total_width=max_total_width)

    def _repr_html_(self) -> str:
        return tohtml(self)

    def merge_overlaps(
        self,
        *,
        count_col: str | None = None,
        match_by: VALID_BY_TYPES = None,
        slack: int = 0,
    ) -> "RangeFrame":
        """Merge overlapping intervals into one.

        Merge overlapping intervals into a single superinterval by uniting intervals that overlap,
        optionally allowing a small gap (specified by ``slack``) between intervals to be merged. The resulting
        RangeFrame will contain the merged intervals, and if ``count_col`` is provided, a column with the counts
        of merged intervals will be included.

        Parameters
        ----------
        count_col : str or None, default None
            Name of the column to store the count of intervals merged into each superinterval.
            If None, no count column is added.
        match_by : str or list, default None
            Column(s) to group intervals by before merging. Only intervals with equal values in the specified
            column(s) will be considered as overlapping.
        slack : int, default 0
            Allow this many nucleotides between intervals to still consider them overlapping.

        Returns
        -------
        RangeFrame
            A RangeFrame with merged (super) intervals. Metadata columns, index, and order are not necessarily preserved.

        """
        match_by = arg_to_list(match_by)
        return _merge(self, by=match_by, count_col=count_col, slack=slack)

    def count_overlaps(
        self,
        other: "RangeFrame",
        *,
        match_by: str | list[str] | None = None,
        slack: int = 0,
    ) -> "pd.Series":
        """Count the number of overlaps per interval.

        For each interval in self, count how many intervals in ``other`` overlap with it.
        The overlap computation is based on the start and end coordinates, with an optional
        ``slack`` parameter to adjust the overlap threshold by temporarily extending the intervals.

        Parameters
        ----------
        other : RangeFrame
            The RangeFrame whose intervals are compared against those in self for overlap counting.
        match_by : str or list, default None
            Column(s) to group intervals by when determining overlaps. Only intervals with equal values in the specified
            column(s) will be considered as overlapping.
        slack : int, default 0
            Temporarily extend intervals in self by this many nucleotides before checking for overlaps,
            thereby adjusting the overlap threshold.

        Returns
        -------
        pd.Series
            A pandas Series where each element corresponds to the number of overlapping intervals in ``other``
            for the corresponding interval in self.

        """
        from pyranges1._ruranges import require_ruranges

        ruranges = require_ruranges()

        f1, f2 = factorize_binary(self, other, match_by)

        return ruranges.numpy.count_overlaps(  # type: ignore[attr-defined]
            self[START_COL].to_numpy(),
            self[END_COL].to_numpy(),
            other[START_COL].to_numpy(),
            other[END_COL].to_numpy(),
            f1,
            f2,
            slack=slack,
        )

    def combine_interval_columns(
        self,
        function: VALID_COMBINE_OPTIONS | CombineIntervalColumnsOperation = "intersect",
        *,
        start: str = START_COL,
        end: str = END_COL,
        start2: str = START_COL + JOIN_SUFFIX,
        end2: str = END_COL + JOIN_SUFFIX,
        drop_old_columns: bool = True,
    ) -> "RangeFrame":
        """Use two pairs of columns representing intervals to create a new start and end column.

        The function is designed as post-processing after join_overlaps to aggregate the coordinates of the two intervals.
        By default, the new start and end columns will be the intersection of the intervals.

        Parameters
        ----------
        function : {"intersect", "union", "swap"} or Callable, default "intersect"
            How to combine the self and other intervals: "intersect", "union", or "swap"
            If a callable is passed, it should take four Series arguments: start1, end1, start2, end2;
            and return a tuple of two integers: (new_starts, new_ends).

        start : str, default "Start"
            Column name for Start of first interval
        end : str, default "End"
            Column name for End of first interval
        start2 : str, default "Start_b"
            Column name for Start of second interval
        end2 : str, default "End_b"
            Column name for End of second interval
        drop_old_columns : bool, default True
            Whether to drop the above mentioned columns.

        """
        from pyranges1.methods.combine_positions import (
            _intersect_interval_columns,
            _swap_interval_columns,
            _union_interval_columns,
        )

        if function == "intersect":
            function = _intersect_interval_columns
        elif function == "union":
            function = _union_interval_columns
        elif function == "swap":
            function = _swap_interval_columns

        new_starts, new_ends = function(self[start], self[end], self[start2], self[end2])

        z = self.copy()
        z[START_COL] = new_starts
        z[END_COL] = new_ends

        cols_to_drop = list({start, end, start2, end2}.difference(RANGE_COLS) if drop_old_columns else {})

        return z.drop_and_return(labels=cols_to_drop, axis="columns")

    def cluster_overlaps(
        self,
        *,
        match_by: VALID_BY_TYPES = None,
        cluster_column: str = "Cluster",
        slack: int = 0,
    ) -> "RangeFrame":
        """Give overlapping intervals a common id.

        Parameters
        ----------
        match_by : str or list, default None
            If provided, only intervals with an equal value in column(s) `match_by` may be considered as overlapping.

        slack : int, default 0
            Length by which the criteria of overlap are loosened.
            A value of 1 clusters also bookended intervals.
            Higher slack values cluster more distant intervals (with a maximum distance of slack-1 between them).

        cluster_column:
            Name the cluster column added in output. Default: "Cluster"

        Returns
        -------
        RangeFrame
            RangeFrame with an ID-column "Cluster" added.

        See Also
        --------
        RangeFrame.merge: combine overlapping intervals into one

        """
        from pyranges1._ruranges import require_ruranges

        ruranges = require_ruranges()

        match_by = arg_to_list(match_by)

        factorized = factorize(self, match_by)
        cluster, idx = ruranges.numpy.cluster(  # type: ignore[attr-defined]
            self[START_COL].to_numpy(),
            self[END_COL].to_numpy(),
            factorized,
            slack,
        )

        res = self.take(idx).copy()  # type: ignore[arg-type]
        res.insert(res.shape[1], cluster_column, cluster)
        return _mypy_ensure_rangeframe(res)

    # complement_overlaps: unexpected behavior. Not sure what to do with it. Right now, not exposed in PyRanges API.
    def _complement_overlaps(
        self: "RangeFrame",
        other: "RangeFrame",
        *,
        group_by: VALID_BY_TYPES = None,
        slack: int = 0,
        preserve_input_order: bool = True,
    ) -> "RangeFrame":
        """Return the non-overlaps of self with other.

        Parameters
        ----------
        other : PyRanges
           PyRanges to find non-overlaps with.

        group_by : str or list, default None
            Column(s) to group by intervals (i.e. exons). If provided, the complement will be calculated for each group.

        slack : int, default 0
            An integer offset that adjusts the overlap threshold when computing the complement intervals.
            Negative values reduce the required gap between intervals (effectively "shrinking" them), while positive values
            increase the gap threshold.

        preserve_input_order : bool, default True
            Whether to preserve the original input order in the result.

            If False, rows may be returned in algorithm/output order instead, which can
            be faster for large results.

        Returns
        -------
        PyRanges
            Non-overlapping intervals

        Notes
        -----
        * To ensure non-overlap among the input intervals, merge_overlaps is run before the complement is calculated.
        * Bookended intervals will result in no complement intervals returned since they would be of length 0.

        See Also
        --------
        PyRanges.subtract_overlaps : report non-overlapping subintervals
        PyRanges.outer_ranges : report the boundaries of groups of intervals (e.g. transcripts/genes)

        """
        group_by = arg_to_list(group_by)
        return _complement_overlaps(self, other, by=group_by, slack=slack, preserve_input_order=preserve_input_order)

    def join_overlaps(
        self,
        other: "RangeFrame",
        *,
        join_type: VALID_JOIN_TYPE = "inner",
        multiple: VALID_OVERLAP_TYPE = "all",
        match_by: VALID_BY_TYPES = None,
        slack: int = 0,
        suffix: str = JOIN_SUFFIX,
        contained_intervals_only: bool = False,
        report_overlap_column: str | None = None,
        preserve_input_order: bool = True,
    ) -> "RangeFrame":
        """Join RangeFrame objects based on overlapping intervals.

        Find pairs of overlapping intervals between self and other and combine their attributes.
        Each row in the output contains columns from both intervals, including their start and end positions.
        By default, only overlapping intervals are included, but the join_type parameter controls how intervals
        without overlaps are handled.

        Parameters
        ----------
        other : RangeFrame
            The RangeFrame to join with.
        join_type : {"inner", "left", "right", "outer"}, default "inner"
            Specifies how to handle intervals that do not overlap. "inner" returns only overlapping intervals,
            "left" returns all intervals from self (with missing values for non-overlapping intervals from other),
            "right" returns all intervals from other, and "outer" returns all intervals from both.
        multiple : {"all", "first", "last"}, default "all"
            Determines which overlapping interval(s) to report when multiple intervals in other overlap the same interval in self.
            "all" reports all overlaps (which may lead to duplicate rows), "first" reports only the overlapping interval with
            the smallest start in other, and "last" reports only the overlapping interval with the largest end in other.
        match_by : str or list, default None
            If provided, only intervals with matching values in the specified column(s) will be joined.
        slack : int, default 0
            Temporarily extend intervals in self by this many units on both ends before checking for overlaps.
        suffix : str, default JOIN_SUFFIX
            Suffix to append to columns from the other RangeFrame in the output.
        contained_intervals_only : bool, default False
            If True, only join intervals from self that are entirely contained within an interval from other.
        report_overlap_column : str or None, default None
            If provided, add a column with this name reporting the amount of overlap between joined intervals.
            The overlap is computed as the minimum of the end positions minus the maximum of the start positions.

        preserve_input_order : bool, default True
            Whether to preserve the original input order in the result.

            If False, rows may be returned in algorithm/output order instead, which can
            be faster for large results.

        Returns
        -------
        RangeFrame
            A new RangeFrame containing the joined intervals with columns from both input RangeFrames.
            The output keeps the index of the primary side of the join (``self`` for inner/left/outer
            joins, ``other`` for right joins), so it may contain duplicates.

        Notes
        -----
        Attributes from the other RangeFrame may have their column names modified by appending the specified suffix.

        """
        res = _both_dfs(
            self,
            other,
            by=match_by,
            slack=slack,
            multiple=multiple,
            contained=contained_intervals_only,
            join_type=join_type,
            suffix=suffix,
            preserve_input_order=preserve_input_order,
        )

        if report_overlap_column:
            res[report_overlap_column] = res[["End", "End" + suffix]].min(axis=1) - res[
                ["Start", "Start" + suffix]
            ].max(axis=1)

        res.index = res.index.astype(np.int64)

        return _mypy_ensure_rangeframe(res)

    def max_disjoint_overlaps(
        self,
        *,
        slack: int = 0,
        match_by: VALID_BY_TYPES = None,
        preserve_input_order: bool = True,
    ) -> "RangeFrame":
        """Find the maximal disjoint set of intervals.

        Returns a subset of the rows in self so that no two intervals overlap, choosing those that
        maximize the number of intervals in the result.

        Parameters
        ----------
        slack : int, default 0
            Length by which the criteria of overlap are loosened.
            A value of 1 implies that bookended intervals are considered overlapping.
            Higher slack values allow more distant intervals (with a maximum distance of slack-1 between them).

        match_by : str or list, default None
            If provided, only intervals with an equal value in column(s) `match_by` may be considered as overlapping.

        preserve_input_order : bool, default True
            Whether to preserve the original input order in the result.

            If False, rows may be returned in algorithm/output order instead, which can
            be faster for large results.

        Returns
        -------
        RangeFrame
            RangeFrame with maximal disjoint set of intervals.

        See Also
        --------
        RangeFrame.merge_overlaps : merge intervals into non-overlapping superintervals
        RangeFrame.cluster : annotate overlapping intervals with common ID

        """
        from pyranges1._ruranges import require_ruranges

        ruranges = require_ruranges()

        factorized = factorize(self, match_by)

        idx = ruranges.numpy.max_disjoint(  # type: ignore[attr-defined]
            starts=self[START_COL].to_numpy(),
            ends=self[END_COL].to_numpy(),
            groups=factorized,
            slack=slack,
            sort_output=preserve_input_order,
        )
        return _mypy_ensure_rangeframe(self.take(idx))  # type: ignore[arg-type]

    def nearest_ranges(
        self,
        other: "RangeFrame",
        *,
        match_by: VALID_BY_TYPES = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        k: int = 1,
        dist_col: str | None = "Distance",
        direction: VALID_DIRECTION_TYPE = "any",
        preserve_input_order: bool = True,
    ) -> "RangeFrame":
        """Find closest interval.

        For each interval in self RangeFrame, the columns of the nearest interval in other RangeFrame are appended.

        Parameters
        ----------
        other : RangeFrame
            RangeFrame to find nearest interval in.

        exclude_overlaps : bool, default True
            Whether to not report intervals of others that overlap with self as the nearest ones.

        direction : {"any", "forward", "backward"}, default "any", i.e. both directions
            Whether to only look for nearest in one direction.

        match_by : str or list, default None
            If provided, only intervals with an equal value in column(s) `match_by` may be matched.

        k : int, default 1
            Number of nearest intervals to fetch.

        suffix : str, default "_b"
            Suffix to give columns with shared name in other.

        dist_col : str or None
            Optional column to store the distance in.

        preserve_input_order : bool, default True
            Whether to preserve the original input order in the result.

            If False, rows may be returned in algorithm/output order instead, which can
            be faster for large results.

        Returns
        -------
        RangeFrame

            A RangeFrame with columns representing nearest interval horizontally appended.

        See Also
        --------
        RangeFrame.join_overlaps : Has a slack argument to find intervals within a distance.

        """
        from pyranges1._ruranges import require_ruranges

        ruranges = require_ruranges()

        f1, f2 = factorize_binary(self, other, match_by)
        idx1, idx2, dist = ruranges.numpy.nearest(  # type: ignore[attr-defined]
            groups=f1,
            starts=self[START_COL].to_numpy(),
            ends=self[END_COL].to_numpy(),
            groups2=f2,
            starts2=other[START_COL].to_numpy(),
            ends2=other[END_COL].to_numpy(),
            k=k,
            slack=0,
            include_overlaps=not exclude_overlaps,
            direction=direction,
            sort_output=preserve_input_order,
        )

        left = self.take(idx1)  # type: ignore[arg-type]
        to_concat: list[pd.DataFrame | pd.Series] = [
            left.reset_index(drop=True),
            pd.DataFrame(other).take(idx2).add_suffix(suffix).reset_index(drop=True),  # type: ignore[arg-type]
        ]
        if dist_col is not None:
            to_concat.append(pd.Series(dist, name=dist_col))

        res = pd.concat(to_concat, axis=1)
        res.index = pd.Index(left.index.to_numpy())

        return _mypy_ensure_rangeframe(res)

    def overlap(
        self,
        other: "RangeFrame",
        multiple: VALID_OVERLAP_TYPE = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: VALID_BY_TYPES = None,
        preserve_input_order: bool = True,
    ) -> "RangeFrame":
        """Return overlapping intervals.

        Returns the intervals in self which overlap with those in other.

        Parameters
        ----------
        other : RangeFrame
            RangeFrame to find overlaps with.

        multiple : {"all", "first", "last"}, default "all"
            What intervals to report when multiple intervals in 'other' overlap with the same interval in self.
            The default "all" reports all overlapping subintervals, which will have duplicate indices.
            "first" reports only, for each interval in self, the overlapping subinterval with smallest Start in 'other'
            "last" reports only the overlapping subinterval with the biggest End in 'other'

        slack : int, default 0
            Intervals in self are temporarily extended by slack on both ends before overlap is calculated, so that
            we allow non-overlapping intervals to be considered overlapping if they are within less than slack distance
            e.g. slack=1 reports bookended intervals.

        contained_intervals_only : bool, default False
            Whether to report only intervals that are entirely contained in an interval of 'other'.

        match_by : str or list, default None
            If provided, only overlapping intervals with an equal value in column(s) `match_by` are reported.

        preserve_input_order : bool, default True
            Whether to preserve the original input order in the result.

            If False, rows may be returned in algorithm/output order instead, which can
            be faster for large results.

        Returns
        -------
        RangeFrame

            A RangeFrame with overlapping intervals.

        See Also
        --------
        RangeFrame.intersect : report overlapping subintervals
        RangeFrame.set_intersect : set-intersect RangeFrame (e.g. merge then intersect)

        """
        from pyranges1.methods.overlap import _overlap

        by = arg_to_list(match_by)

        result = _overlap(
            self,
            other,
            by=by,
            slack=slack,
            multiple=multiple,
            contained=contained_intervals_only,
            preserve_input_order=preserve_input_order,
        )

        return _mypy_ensure_rangeframe(result)

    def sort_ranges(
        self: "RangeFrame",
        by: VALID_BY_TYPES = None,
        *,
        natsort: bool = True,
        sort_rows_reverse_order: Sequence[bool] | None = None,
    ) -> "RangeFrame":
        """Sort RangeFrame according to Start, End, and any other columns given.

        For uses not covered by this function, use  DataFrame.sort_values().

        Parameters
        ----------
        by : str or list of str, default None
            in the desired order as part of the 'by' argument.

        natsort : bool, default False
            Whether to use natural sorting for the columns in match_by.

        sort_rows_reverse_order : sequence of bools or None
            Whether to sort these rows in the reverse order for the starts and ends.

        Returns
        -------
        RangeFrame

            Sorted RangeFrame. The index is preserved. Use .reset_index(drop=True) to reset the index.

        """
        from pyranges1._ruranges import require_ruranges

        ruranges = require_ruranges()

        by = arg_to_list(by)
        by_sort_order_as_int = sort_factorize_dict(self, by, use_natsort=natsort)
        idxs = ruranges.numpy.sort_intervals(  # type: ignore[attr-defined]
            by_sort_order_as_int,
            self[START_COL].to_numpy(),
            self[END_COL].to_numpy(),
            sort_reverse_direction=np.array(sort_rows_reverse_order, dtype=bool) if sort_rows_reverse_order else None,
        )
        return _mypy_ensure_rangeframe(self.take(idxs))  # type: ignore[arg-type]

    def subtract_overlaps(
        self: "RangeFrame",
        other: "RangeFrame",
        match_by: VALID_BY_TYPES = None,
        *,
        preserve_input_order: bool = True,
    ) -> "RangeFrame":
        """Subtract intervals, i.e. return non-overlapping subintervals.

        Identify intervals in other that overlap with intervals in self; return self with the overlapping parts removed.

        Parameters
        ----------
        other:
            RangeFrame to subtract.

        match_by : str or list, default None
            If provided, only intervals with an equal value in column(s) `match_by` may be considered as overlapping.

        preserve_input_order : bool, default True
            Whether to preserve the original input order in the result.

            If False, rows may be returned in algorithm/output order instead, which can
            be faster for large results.

        Returns
        -------
        RangeFrame
            RangeFrame with subintervals from self that do not overlap with any interval in other.
            Columns and index are preserved.

        Warning
        -------
        The returned Pyranges may have index duplicates. Call .reset_index(drop=True) to fix it.

        See Also
        --------
        RangeFrame.overlap : use with invert=True to return all intervals without overlap
        RangeFrame.complement_ranges : return the internal complement_ranges of intervals, i.e. its introns.

        """
        from pyranges1._ruranges import require_ruranges

        ruranges = require_ruranges()

        f1, f2 = factorize_binary(self, other, match_by)

        idx, start, end = ruranges.numpy.subtract(  # type: ignore[attr-defined]
            self[START_COL].to_numpy(),
            self[END_COL].to_numpy(),
            other[START_COL].to_numpy(),
            other[END_COL].to_numpy(),
            f1,
            f2,
            sort_output=preserve_input_order,
        )

        output = self.take(idx).copy()  # type: ignore[arg-type]
        output[START_COL], output[END_COL] = start, end
        return _mypy_ensure_rangeframe(output)

    def sort_by_position(self) -> "RangeFrame":
        """Sort by Start and End columns."""
        return _mypy_ensure_rangeframe(self.sort_values(RANGE_COLS))

    def reasons_why_frame_is_invalid(self) -> list[InvalidRangesReason] | None:  # noqa: D102
        __doc__ = InvalidRangesReason.is_invalid_ranges_reasons.__doc__  # noqa: F841

        return InvalidRangesReason.is_invalid_ranges_reasons(self)

    def copy(self, *args, **kwargs) -> "RangeFrame":  # pyright: ignore[reportIncompatibleMethodOverride]  # noqa: D102
        return _mypy_ensure_rangeframe(super().copy(*args, **kwargs))

    def drop(self, *args, **kwargs) -> "RangeFrame | None":  # type: ignore[override]  # noqa: D102
        return self.__class__(super().drop(*args, **kwargs))

    def drop_and_return[T: "RangeFrame"](self: T, *args: Any, **kwargs: Any) -> T:  # noqa: PYI019, D102
        kwargs["inplace"] = False
        return self.__class__(super().drop(*args, **kwargs))

    def reindex(self, *args, **kwargs) -> "RangeFrame":  # noqa: D102
        return self.__class__(super().reindex(*args, **kwargs))


def _mypy_ensure_rangeframe(r: pd.DataFrame) -> "RangeFrame":
    result = RangeFrame(r)
    if not isinstance(result, RangeFrame):
        msg = f"Expected RangeFrame, got {type(result)}"
        raise TypeError(msg)
    return result


def assert_valid_ranges(function: Callable, *args: "RangeFrame") -> None:
    """Raise ValueError because function is called on invalid ranges."""
    if any(r.reasons_why_frame_is_invalid() for r in args):
        is_not_lambda = function.__name__ != "<lambda>"
        function_repr = function.__name__ if is_not_lambda else inspect.getsource(function).strip()
        msg = f"Cannot perform function on invalid ranges (function was {function_repr})."
        raise ValueError(msg)


def _with_group_keys_to_kwargs(by: list[str]) -> Callable:
    def decorator(func) -> Callable:
        def wrapper(group, **kwargs) -> Any:
            # Extract the group keys and add them to kwargs
            names = group.name if isinstance(group.name, Iterable) and not isinstance(group.name, str) else [group.name]
            kwargs[BY_ENTRY_IN_KWARGS] = dict(zip(by, names, strict=True))
            return func(group, **kwargs)

        return wrapper

    return decorator
