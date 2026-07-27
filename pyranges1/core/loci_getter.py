import typing
from dataclasses import dataclass

import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    from pyranges1 import PyRanges
from pyranges1.core.names import CHROM_COL, END_COL, START_COL, STRAND_COL, VALID_GENOMIC_STRAND_INFO
from pyranges1.core.pyranges_helpers import ensure_pyranges

# Three types of accessors:
# 1. Columns or columns, rows
# 2. Genomic location


LociKeyType = str | int | slice | tuple[str | int, str] | tuple[str | int, slice] | tuple[str | int, str, slice]


@dataclass
class LociGetter:
    pr: "PyRanges"

    def __getitem__(
        self,
        key: LociKeyType,
    ) -> "PyRanges":
        return ensure_pyranges(self.pr.loc[self._matching_rows(key)])

    def _matching_rows(
        self,
        key: LociKeyType,
    ) -> "pd.Series[bool]":
        if isinstance(key, tuple):
            if is_chrom_and_strand(key):
                rows = chrom_and_strand(self.pr, key)
            elif is_chrom_or_strand_with_slice(key):
                rows = chrom_or_strand_with_slice(self.pr, key)
            elif is_3_tuple(key):
                rows = get_chrom_strand_and_range(self.pr, key)
            else:
                msg = f"Indexing tuple for loci must be of length 2 or 3, but was {len(key)}."
                raise ValueError(msg)
        elif isinstance(key, slice):
            rows = _rows_matching_range(self.pr, key)
        elif not isinstance(key, list):
            rows = get_chrom_and_strand(self.pr, key)
        else:
            msg = (
                "The loci accessor does not accept a list. If you meant to retrieve columns, use "
                "get_with_loc_columns instead."
            )
            raise TypeError(msg)
        return rows

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        rows = self._matching_rows(key)
        self.pr.loc[rows] = value


def _rows_matching_chrom(gr: "PyRanges", chrom: str | int | None) -> "pd.Series[bool]":
    return (gr[CHROM_COL].astype(type(chrom)) == chrom) if chrom is not None else pd.Series(data=True, index=gr.index)


def _rows_matching_strand(gr: "PyRanges", strand: str | None) -> "pd.Series[bool]":
    return (gr[STRAND_COL] == strand) if strand is not None else pd.Series(data=True, index=gr.index)


def _rows_matching_range(gr: "PyRanges", _range: slice | None) -> "pd.Series[bool]":
    if _range is None:
        return pd.Series(data=True, index=gr.index)
    # An omitted bound leaves that side of the window open, so both use an
    # infinite limit. A finite lower default such as -1 would silently drop
    # intervals lying entirely below it.
    start_in_range = gr[START_COL] < (_range.stop if _range.stop is not None else np.inf)
    end_in_range = gr[END_COL] > (_range.start if _range.start is not None else -np.inf)
    return start_in_range & end_in_range


def _rows_matching_chrom_and_strand(gr: "PyRanges", chrom: str | int | None, strand: str | None) -> "pd.Series[bool]":
    return _rows_matching_chrom(gr, chrom) & _rows_matching_strand(gr, strand)


def _rows_matching_chrom_and_strand_and_range(
    gr: "PyRanges",
    chrom: str | None | int,
    strand: str | None,
    _range: slice | None,
) -> pd.Series:
    return _rows_matching_chrom(gr, chrom) & _rows_matching_strand(gr, strand) & _rows_matching_range(gr, _range)


def is_3_tuple(key: tuple) -> bool:
    """Check if key is a 3-tuple."""
    return len(key) == 3  # noqa: PLR2004


def is_2_tuple(key: tuple) -> bool:
    """Check if key is a 2-tuple."""
    return len(key) == 2  # noqa: PLR2004


def is_chrom_or_strand_with_slice(key: tuple) -> bool:
    """Check if key is a chromosome or strand and slice."""
    return is_2_tuple(key) and isinstance(key[1], slice)


def is_chrom_and_strand(key: tuple) -> bool:
    """Check if key is a chromosome and strand."""
    return is_2_tuple(key) and isinstance(key[1], str)


def chrom_and_strand(pr: "PyRanges", key: tuple) -> "pd.Series[bool]":
    """Get rows matching chromosome and strand."""
    chrom, strand = key
    return _rows_matching_chrom_and_strand(pr, chrom, strand)


def chrom_or_strand_with_slice(pr: "PyRanges", key: tuple) -> "pd.Series[bool]":
    """Get rows matching chromosome or strand and slice."""
    chrom_or_strand, loc = key

    # We can get a view with right chromosome, then match range only there, and reindex to original.
    # Or we can interrogate the whole PyRanges for chromosome and range separately, then combine the boolean Series
    # I timeit tested random pyranges up to 10^8 rows.
    # The second was consistently faster, so here it is:

    # in ambyguous cases, we test is strand is + or -, otherwise we assume it's a chromosome
    if chrom_or_strand in VALID_GENOMIC_STRAND_INFO and pr.has_strand:
        rows = _rows_matching_strand(pr, chrom_or_strand) & (_rows_matching_range(pr, loc))
    else:
        rows = _rows_matching_chrom(pr, chrom_or_strand) & (_rows_matching_range(pr, loc))
    return rows


def get_chrom_strand_and_range(pr: "PyRanges", key: tuple) -> "pd.Series":
    """Get rows matching chromosome, strand and range."""
    chrom, strand, _range = key
    return _rows_matching_chrom_and_strand_and_range(pr, chrom, strand, _range)


def get_chrom_and_strand(pr: "PyRanges", key: str | int) -> "pd.Series[bool]":
    """Get rows matching chromosome or strand.

    We do not know whether the key is a chromosome or a strand.
    We determine it's strand only if + or -, chromosome otherwise
    """
    if key in VALID_GENOMIC_STRAND_INFO and pr.has_strand:
        rows = _rows_matching_strand(pr, key)
    else:
        rows = _rows_matching_chrom(pr, key)
    return rows
