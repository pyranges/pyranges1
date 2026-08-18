"""Sort-key machinery shared by ``RangeFrame.sort_ranges`` and ``PyRanges.sort_ranges``.

A sort is four steps:

1. Resolve the key list -- ``Chromosome, Strand, *by, Start, End`` on ``PyRanges``,
   ``*by, Start, End`` on ``RangeFrame`` -- honouring any key the caller moved by
   naming it in ``by`` (:func:`resolve_sort_keys`).
2. Code every key column to an ascending integer, one column at a time
   (:func:`group_ids`).
3. One ``ruranges`` kernel call for the row order.
4. One ``take``.

The ordering of a column's distinct string values comes from
``ruranges.numpy.natural_rank`` / ``lexical_rank``, which ``polaranges`` also
calls, so the two libraries cannot order the same chromosome names differently.
Doing it in Python cost 24.9 s for 10 million distinct values against 0.7 s in
Rust, and it was the single largest line item in every large sort.

Coding *per column* rather than per key tuple is what makes a high-cardinality
``by`` affordable: the cost is the number of distinct values in each column, not
the number of distinct combinations, which approaches the row count. It changes
no answer -- lexicographic order on key tuples is lexicographic order on
per-column order-preserving codes.
"""

import numpy as np
import pandas as pd

# The kernel takes exactly this many innermost coordinate-like keys (its
# `starts` and `ends` arguments).
_KERNEL_INNER_KEY_COUNT = 2


def sort_one_by_one(d: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Equivalent to pd.sort_values(by=[col1, col2]), but faster."""
    d = d.sort_values(by=[col2])
    return d.sort_values(by=[col1], kind="mergesort")


def resolve_sort_keys(
    columns: "pd.Index | list[str]",
    head: list[str],
    by: list[str],
    tail: list[str],
    sort_descending: list[str],
) -> tuple[list[str], list[bool]]:
    """Return the sort keys, outermost first, and which of them run backwards.

    The rule, in one sentence: **the key list is** ``head``, **then** ``by``,
    **then** ``tail``, **and any column named in** ``by`` **is taken out of its
    implicit position and used where the caller put it.**

    ``head`` is ``["Chromosome", "Strand"]`` on ``PyRanges`` and empty on
    ``RangeFrame``; ``tail`` is ``["Start", "End"]`` on both.

    ============================================  ================================================
    call                                          key list
    ============================================  ================================================
    ``sort_ranges()``                             ``Chromosome, Strand, Start, End``
    ``sort_ranges(by="transcript_id")``           ``Chromosome, Strand, transcript_id, Start, End``
    ``sort_ranges(by=["Strand", "Chromosome"])``  ``Strand, Chromosome, Start, End``
    ``sort_ranges(by=["Start", "End", "score"])`` ``Chromosome, Strand, Start, End, score``
    ============================================  ================================================

    The rule applies per column, so ``by=["score", "Chromosome"]`` yields
    ``Strand, score, Chromosome, Start, End``: only ``Chromosome`` was pulled out
    of the implicit head. Name every key explicitly when that matters.

    Raises
    ------
    ValueError
        If a key is not a column of the frame, or if ``sort_descending`` names
        something that is not a sort key. The latter is rejected rather than
        ignored, because a silently dropped name looks like a sort that worked.

    """
    keys: list[str] = []
    for name in [c for c in head if c not in by] + list(by) + [c for c in tail if c not in by]:
        if name not in keys:
            keys.append(name)

    if missing := [c for c in keys if c not in columns]:
        msg = f"sort keys are not columns of the frame: {missing}"
        raise ValueError(msg)

    if unknown := [c for c in sort_descending if c not in keys]:
        msg = f"sort_descending names columns that are not sort keys: {unknown}; sort keys are {keys}"
        raise ValueError(msg)

    return keys, [c in sort_descending for c in keys]


def _column_ranks(values: pd.Series, *, use_natsort: bool, descending: bool) -> tuple[np.ndarray, int]:
    """Code one key column to dense ascending integers.

    Returns one code per row in ``0..cardinality``, ordered by the column's
    values, and the cardinality. The work is proportional to the number of
    *distinct* values, which is why 25 chromosome names are free however many
    rows there are.
    """
    codes, uniques = pd.factorize(values, sort=False)
    if len(uniques) == 0:
        return np.zeros(len(values), dtype=np.uint32), 0

    if pd.api.types.is_numeric_dtype(uniques) or pd.api.types.is_bool_dtype(uniques):
        # Nothing about ordering numbers needs Rust.
        order = np.argsort(np.asarray(uniques), kind="stable")
        rank = np.empty(len(uniques), dtype=np.uint32)
        rank[order] = np.arange(len(uniques), dtype=np.uint32)
    else:
        from pyranges1._ruranges import require_ruranges

        ruranges = require_ruranges()
        as_text = [str(value) for value in uniques]
        rank = (
            ruranges.numpy.natural_rank(as_text)  # type: ignore[attr-defined]
            if use_natsort
            else ruranges.numpy.lexical_rank(as_text)  # type: ignore[attr-defined]
        )

    if descending:
        rank = (len(uniques) - 1 - rank).astype(np.uint32)

    # `codes` is -1 where the value was missing. pandas has always sorted those
    # first, so give them a rank of their own below every real value.
    if (codes < 0).any():
        rank = (rank + 1).astype(np.uint32)
        out = np.where(codes < 0, np.uint32(0), rank[np.maximum(codes, 0)])
        return out.astype(np.uint32), len(uniques) + 1

    return rank[codes].astype(np.uint32), len(uniques)


def group_ids(df: pd.DataFrame, keys: list[str], descending: list[bool], *, use_natsort: bool) -> np.ndarray:
    """Fold every key column into one ascending group id, one per row.

    The result is what ``ruranges.numpy.sort_intervals`` takes as ``groups``: two
    rows compare in group-id order exactly as they compare on the key columns.
    """
    if not keys:
        # No keys means one group. The old implementation still walked every row
        # twice in Python to produce this array of zeros, which was 97 % of the
        # cost of an unkeyed sort.
        return np.zeros(len(df), dtype=np.uint32)

    from pyranges1._ruranges import require_ruranges

    ruranges = require_ruranges()

    group = np.zeros(len(df), dtype=np.uint32)
    cardinality = 1
    for name, reverse in zip(keys, descending, strict=True):
        codes, count = _column_ranks(df[name], use_natsort=use_natsort, descending=reverse)
        group, cardinality = ruranges.numpy.fold_ranks(group, cardinality, codes, count)  # type: ignore[attr-defined]
    return group


def _inner_key_values(
    df: pd.DataFrame,
    name: str,
    *,
    coordinate_columns: tuple[str, str],
    use_natsort: bool,
    descending: bool,
) -> np.ndarray:
    """Values for one of the two innermost keys, as the integers the kernel sorts on.

    A coordinate column keeps its own values; anything else is ranked. Descending
    is applied by negation, so that the kernel's per-row 5'->3' reversal composes
    with it by XOR instead of fighting it: the kernel negates a reversed row's
    coordinates again, undoing this one.
    """
    if name in coordinate_columns:
        values = df[name].to_numpy().astype(np.int64, copy=False)
        return -values if descending else values
    codes, _ = _column_ranks(df[name], use_natsort=use_natsort, descending=descending)
    return codes.astype(np.int64, copy=False)


def sort_order(
    df: pd.DataFrame,
    keys: list[str],
    descending: list[bool],
    *,
    use_natsort: bool,
    reverse_rows: np.ndarray | None = None,
    coordinate_columns: tuple[str, str] = ("Start", "End"),
) -> np.ndarray:
    """Return the permutation that sorts ``df`` by ``keys``.

    All keys but the innermost two are folded into the kernel's ``groups``
    argument; the innermost two become its ``starts`` and ``ends``. In the
    overwhelmingly common case those are the frame's own coordinate columns,
    ascending, and are handed over untouched.
    """
    from pyranges1._ruranges import require_ruranges

    ruranges = require_ruranges()

    split = max(len(keys) - _KERNEL_INNER_KEY_COUNT, 0)
    outer, inner = keys[:split], keys[split:]
    outer_descending, inner_descending = descending[:split], descending[split:]

    group = group_ids(df, outer, outer_descending, use_natsort=use_natsort)

    start_col, end_col = coordinate_columns
    if inner == [start_col, end_col] and not any(inner_descending):
        starts = df[start_col].to_numpy()
        ends = df[end_col].to_numpy()
    else:
        values = [
            _inner_key_values(
                df,
                name,
                coordinate_columns=coordinate_columns,
                use_natsort=use_natsort,
                descending=reverse,
            )
            for name, reverse in zip(inner, inner_descending, strict=True)
        ]
        while len(values) < _KERNEL_INNER_KEY_COUNT:
            values.insert(0, np.zeros(len(df), dtype=np.int64))
        starts, ends = values

    return ruranges.numpy.sort_intervals(  # type: ignore[attr-defined]
        starts,
        ends,
        groups=group,
        sort_reverse_direction=reverse_rows,
    )


def sort_factorize_dict(df: pd.DataFrame, by: list[str], *, use_natsort: bool = True) -> np.ndarray:
    """Return an ascending group id per row, ordered by the ``by`` columns.

    Superseded by :func:`group_ids`, which also handles per-key descending order.
    Kept because the name has been importable for several releases.
    """
    return group_ids(df, list(by), [False] * len(by), use_natsort=use_natsort)
