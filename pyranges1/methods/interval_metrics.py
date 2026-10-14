from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from pyranges1.core.names import END_COL, JOIN_SUFFIX, START_COL

if TYPE_CHECKING:
    from pyranges1.range_frame.range_frame import RangeFrame

# ------------------------------------------------------------------
# Built-in metric identifiers (all single-column outputs)
# ------------------------------------------------------------------
VALID_METRICS: set[str] = {
    "overlap_length",  # raw overlap in bp
    "fraction",  # overlap / chosen denominator
    "jaccard",  # overlap / union length
    "distance",  # unsigned gap distance
    "overlap",  # boolean flag
    "signed_distance",  # negative = second interval to the left, positive = to the right (strand ignored)
    "midpoint_distance",  # absolute distance between midpoints
    "symmetric_coverage",  # 2 * overlap / (len1 + len2)
    "relative_direction",  # same / opposite / unknown (strand aware)
}


def compute_interval_metrics(
    df: RangeFrame,
    metrics: str | Iterable[str] | Mapping[str, str] = "fraction",
    *,
    start: str = START_COL,
    end: str = END_COL,
    start2: str = START_COL + JOIN_SUFFIX,
    end2: str = END_COL + JOIN_SUFFIX,
    denom: str = "first",
) -> RangeFrame:
    """Attach per-row interval-relationship metrics as new columns.

    Parameters
    ----------
    df : RangeFrame
        Frame in which both intervals reside (e.g. the result of a join).
    metrics : str | Iterable[str] | Mapping[str, str], default "fraction"
        Metrics to compute - a name, a list of names, or a mapping
        {metric_name -> output_column_name}.  Valid names are in
        VALID_METRICS.
    start, end : str, default START_COL / END_COL
        Column names holding the first interval coordinates.
    start2, end2 : str, default START_COL + "_b" / END_COL + "_b"
        Column names holding the second interval coordinates.
    denom : {"first", "second", "union"}, default "first"
        Denominator used by the *fraction* metric.

    Returns
    -------
    RangeFrame
        Copy of *df* with one extra column per requested metric.

    """
    # --------------------------------------------------------------
    # 0. Resolve requested operations -> output column names
    # --------------------------------------------------------------
    if isinstance(metrics, Mapping):
        op_to_name = dict(metrics)
    elif isinstance(metrics, str):
        op_to_name = {metrics: metrics}
    else:
        op_to_name = {m: m for m in metrics}

    unknown = set(op_to_name) - VALID_METRICS
    if unknown:
        raise ValueError("Unknown metric(s): " + ", ".join(sorted(unknown)))

    # --------------------------------------------------------------
    # 1. Shared NumPy arrays (vectorised)
    # --------------------------------------------------------------
    s1 = df[start].to_numpy()
    e1 = df[end].to_numpy()
    s2 = df[start2].to_numpy()
    e2 = df[end2].to_numpy()

    len1 = e1 - s1
    len2 = e2 - s2
    overlap = np.maximum(0, np.minimum(e1, e2) - np.maximum(s1, s2))
    union_len = len1 + len2 - overlap
    gap_unsigned = np.where(overlap > 0, 0, np.maximum(s1, s2) - np.minimum(e1, e2))
    signed_gap = np.where(overlap > 0, 0, np.where(e1 <= s2, s2 - e1, -(s1 - e2)))
    midpoint_dist = np.abs(((s1 + e1) / 2) - ((s2 + e2) / 2))
    symmetric_cov = 2 * overlap / np.where(len1 + len2 == 0, np.nan, len1 + len2)
    denom_len = {"first": len1, "second": len2, "union": union_len}[denom]

    # --------------------------------------------------------------
    # 2. Helper for strand-aware metric
    # --------------------------------------------------------------
    def _relative_direction() -> np.ndarray:
        if "Strand" not in df.columns or "Strand" + JOIN_SUFFIX not in df.columns:
            msg = "relative_direction requires 'Strand' and 'Strand_b' columns."
            raise KeyError(msg)
        str1 = df["Strand"].to_numpy(dtype="U1")
        str2 = df["Strand" + JOIN_SUFFIX].to_numpy(dtype="U1")
        return np.where(
            (str1 == ".") | (str2 == "."),
            "unknown",
            np.where(str1 == str2, "same", "opposite"),
        )

    # --------------------------------------------------------------
    # 3. Map each metric name to a no-argument callable
    # --------------------------------------------------------------
    metric_funcs = {
        "overlap_length": lambda: overlap,
        "fraction": lambda: overlap / np.where(denom_len == 0, np.nan, denom_len),
        "jaccard": lambda: overlap / np.where(union_len == 0, np.nan, union_len),
        "distance": lambda: gap_unsigned,
        "overlap": lambda: overlap > 0,
        "signed_distance": lambda: signed_gap,
        "midpoint_distance": lambda: midpoint_dist,
        "symmetric_coverage": lambda: symmetric_cov,
        "relative_direction": _relative_direction,
    }

    # --------------------------------------------------------------
    # 4. Assemble output
    # --------------------------------------------------------------
    result = df.copy()
    for op, col_name in op_to_name.items():
        result[col_name] = metric_funcs[op]()

    return result
