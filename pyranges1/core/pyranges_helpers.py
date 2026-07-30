import inspect
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy import ndarray

from pyranges1.core.names import (
    CHROM_AND_STRAND_COLS,
    CHROM_COL,
    FORWARD_STRAND,
    OVERLAP_ALL,
    OVERLAP_FIRST,
    RANGE_COLS,
    REMOVED_OVERLAP_CONTAINED,
    REVERSE_STRAND,
    STRAND_BEHAVIOR_AUTO,
    STRAND_BEHAVIOR_IGNORE,
    STRAND_BEHAVIOR_OPPOSITE,
    STRAND_BEHAVIOR_SAME,
    STRAND_COL,
    STRICT_STRAND_BEHAVIOR_TYPE,
    USE_STRAND_AUTO,
    VALID_BY_OPTIONS,
    VALID_BY_TYPES,
    VALID_OVERLAP_OPTIONS,
    VALID_OVERLAP_TYPE,
    VALID_STRAND_BEHAVIOR_OPTIONS,
    VALID_STRAND_BEHAVIOR_TYPE,
    VALID_USE_STRAND_OPTIONS,
    VALID_USE_STRAND_TYPE,
)

if TYPE_CHECKING:
    from pyranges1 import PyRanges
    from pyranges1.range_frame.range_frame import RangeFrame


def factorize(
    df: "pd.DataFrame",
    by: VALID_BY_TYPES,
) -> ndarray:
    """Factorize a DataFrame based on specified grouping columns.

    This function assigns a unique integer (of type uint32) to each group in the DataFrame,
    where groups are defined by the columns specified in ``by``. If ``by`` is empty or evaluates
    to False, an array of zeros is returned.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be factorized.
    by : VALID_BY_TYPES
        Column name(s) or criteria used to group the DataFrame. If empty, no grouping is performed.

    Returns
    -------
    ndarray
        A NumPy array of type uint32 containing group labels for each row in the DataFrame.

    """
    if not by:
        return np.zeros(len(df), dtype=np.uint8)
    _by = arg_to_list(by)
    return df.groupby(_by).ngroup().to_numpy().astype(np.uint32)


def factorize_arange(
    df: "pd.DataFrame",
    by: VALID_BY_TYPES,
) -> ndarray:
    """Factorize a DataFrame based on specified grouping columns.

    This function assigns a unique integer (of type uint32) to each group in the DataFrame,
    where groups are defined by the columns specified in ``by``. If ``by`` is empty or evaluates
    to False, an array of zeros is returned.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be factorized.
    by : VALID_BY_TYPES
        Column name(s) or criteria used to group the DataFrame. If empty, no grouping is performed.

    Returns
    -------
    ndarray
        A NumPy array of type uint32 containing group labels for each row in the DataFrame.

    """
    if not by:
        return np.arange(len(df), dtype=np.uint)
    _by = arg_to_list(by)
    return df.groupby(_by).ngroup().to_numpy().astype(np.uint32)


def factorize_binary(
    df: "pd.DataFrame",
    df2: "pd.DataFrame",
    by: VALID_BY_TYPES,
) -> tuple[ndarray, ndarray]:
    """Factorize two DataFrames simultaneously based on specified grouping columns.

    This function concatenates the specified columns from both DataFrames, assigns consistent group labels
    across the combined data, and then splits the labels back into two arrays corresponding to each DataFrame.
    If ``by`` is empty or evaluates to False, arrays of zeros are returned for both DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        The first DataFrame to be factorized.
    df2 : pd.DataFrame
        The second DataFrame to be factorized.
    by : VALID_BY_TYPES
        Column name(s) or criteria used to group both DataFrames.

    Returns
    -------
    tuple of ndarray
        A tuple containing two NumPy arrays of type uint32. The first array corresponds to group labels for
        ``df`` and the second array corresponds to group labels for ``df2``.

    """
    if not by:
        return np.zeros(len(df), dtype=np.uint8), np.zeros(len(df2), dtype=np.uint8)
    _by = arg_to_list(by)
    factorized = (
        pd.concat([df[_by], df2[_by]], ignore_index=True)
        .groupby(_by, observed=False)
        .ngroup()
        .astype(np.uint32)
        .to_numpy()
    )
    return factorized[: len(df)], factorized[len(df) :]


def split_on_strand(gr: "PyRanges") -> tuple["PyRanges", "PyRanges"]:
    """Split a PyRanges object into forward and reverse strand subsets.

    This function filters the input PyRanges object based on strand information, returning two PyRanges objects:
    one for the forward strand and another for the reverse strand.

    Parameters
    ----------
    gr : PyRanges
        A PyRanges object containing genomic ranges with strand information.

    Returns
    -------
    tuple of PyRanges
        A tuple containing two PyRanges objects: the first for ranges on the forward strand and the second for ranges
        on the reverse strand.

    """
    return (
        ensure_pyranges(gr.query(f"{STRAND_COL} == '{FORWARD_STRAND}'")),
        ensure_pyranges(gr.query(f"{STRAND_COL} == '{REVERSE_STRAND}'")),
    )


def prepare_by_single(
    self: "PyRanges",
    use_strand: VALID_USE_STRAND_TYPE,
    match_by: VALID_BY_OPTIONS,
) -> list[str]:
    """Prepare a list of column names for grouping or matching within a PyRanges object.

    This function determines which columns to use for operations by including the chromosome column by default
    and adding strand information if indicated by the ``use_strand`` parameter. Additional columns specified in
    ``match_by`` are appended to this list.

    Parameters
    ----------
    self : PyRanges
        The PyRanges object on which the operation is performed.
    use_strand : VALID_USE_STRAND_TYPE
        Indicator determining whether strand information should be included.
    match_by : VALID_BY_OPTIONS
        Additional column name(s) to use for grouping or matching.

    Returns
    -------
    list of str
        A list of column names used for grouping or matching.

    """
    use_strand = validate_and_convert_use_strand(self, use_strand)
    by = arg_to_list(match_by)
    return [*([CHROM_COL] if not use_strand else CHROM_AND_STRAND_COLS), *by]


def prepare_by_binary(
    self: "PyRanges",
    other: "PyRanges",
    strand_behavior: VALID_STRAND_BEHAVIOR_TYPE = "auto",
    match_by: VALID_BY_TYPES = None,
) -> tuple["PyRanges", list[str]]:
    """Prepare a secondary PyRanges object and determine the matching columns for binary operations.

    This function constructs a list of columns for matching (including chromosome and, optionally, strand columns)
    and may modify the secondary PyRanges object based on the specified strand behavior. For example, if the
    strand behavior is set to 'opposite', the strand information in the secondary object is inverted.

    Parameters
    ----------
    self : PyRanges
        The primary PyRanges object.
    other : PyRanges
        The secondary PyRanges object to be prepared.
    strand_behavior : VALID_STRAND_BEHAVIOR_TYPE, default "auto"
        Specifies how strand information should be handled. Options include 'auto', 'ignore', or 'opposite'.
    match_by : VALID_BY_TYPES, optional
        Additional column name(s) to include for matching.

    Returns
    -------
    tuple
        A tuple where the first element is the (possibly modified) secondary PyRanges object and the second element
        is a list of column names used for matching.

    """
    strand_behavior = validate_and_convert_strand_behavior(self, other, strand_behavior)
    default_cols = [CHROM_COL] if strand_behavior == STRAND_BEHAVIOR_IGNORE else CHROM_AND_STRAND_COLS
    by = [*default_cols, *arg_to_list(match_by)]

    if strand_behavior == STRAND_BEHAVIOR_OPPOSITE:
        _other = other.loc[:, [*RANGE_COLS, *by]].copy()
        _other.loc[:, STRAND_COL] = other[STRAND_COL].replace({"+": "-", "-": "+"})
    else:
        _other = other
    return ensure_pyranges(_other), by


def validate_and_convert_use_strand(self: "PyRanges", use_strand: VALID_USE_STRAND_TYPE) -> bool:
    """Validate and convert strand option.

    Used inside various methods accepting a PyRanges as input. It processes use_strand option provided by the user upon
    calling the function. This function validates the option and converts it to a boolean value.

    If use_strand is True, an error will be raised if the Strand column is missing or invalid.
    If use_strand is 'auto', it will be converted to True if the Strand column is valid, otherwise False.
    Note that if use_strand is 'auto' and the Strand column is invalid, a warning will be shown.

    Parameters
    ----------
    self : PyRanges
        PyRanges object to be validated.

    use_strand : {'auto', True, False}
        The use_strand option provided by the user.


    Returns
    -------
    bool
        The converted boolean value of use_strand.

    """
    if use_strand not in VALID_USE_STRAND_OPTIONS:
        msg = f"Invalid use_strand option: {use_strand}"
        raise ValueError(msg)
    if use_strand is True and not self.has_strand:
        msg = "Cannot have use_strand=True when the Strand column is missing."
        raise ValueError(msg)
    if use_strand == USE_STRAND_AUTO:
        use_strand = self.strand_valid

        # If Strand column is invalid, set use_strand to False
        # but if there are intervals on the negative strand, warn user
        if not use_strand and contains_negative_strand(self):
            fn_name = stack[2].function if len(stack := inspect.stack()) > 2 else "unknown_function"  # noqa: PLR2004
            fn_name = stack[1].function if fn_name == "<module>" else fn_name
            msg = f"{fn_name}: '{USE_STRAND_AUTO}' use_strand treated as False due to invalid Strand values. Please use use_strand=False"
            warnings.warn(msg, stacklevel=4)
    return use_strand


def validate_and_convert_multiple(multiple: VALID_OVERLAP_TYPE) -> VALID_OVERLAP_TYPE:
    """Validate the multiplicity option of the operations that select a right interval.

    ``intersect_overlaps``, ``join_overlaps`` and ``set_intersect_overlaps`` return
    information taken from ``other``, so 'first' and 'last' pick genuinely different
    output. ``overlap`` does not, and spells the option as a bool instead; see
    :func:`validate_and_convert_overlap_multiple`.

    Parameters
    ----------
    multiple : {'first', 'all', 'last'}
        The multiple option provided by the user.

    Returns
    -------
    str
        One of VALID_OVERLAP_OPTIONS.

    Examples
    --------
    >>> validate_and_convert_multiple("last")
    'last'
    >>> validate_and_convert_multiple("frist")
    Traceback (most recent call last):
    ...
    ValueError: Invalid multiple option: 'frist'. Use one of ['first', 'all', 'last'].

    'contained' was removed in 1.3.12; the argument that replaces it is named:

    >>> validate_and_convert_multiple("contained")
    Traceback (most recent call last):
    ...
    ValueError: multiple='contained' was removed; use contained_intervals_only=True instead.

    """
    if multiple == REMOVED_OVERLAP_CONTAINED:
        msg = f"multiple={REMOVED_OVERLAP_CONTAINED!r} was removed; use contained_intervals_only=True instead."
        raise ValueError(msg)
    if not isinstance(multiple, str) or multiple not in VALID_OVERLAP_OPTIONS:
        msg = f"Invalid multiple option: {multiple!r}. Use one of {VALID_OVERLAP_OPTIONS}."
        raise ValueError(msg)
    return multiple


def validate_and_convert_overlap_multiple(multiple: bool) -> VALID_OVERLAP_TYPE:  # noqa: FBT001
    """Convert ``overlap``'s multiplicity option to the string the kernel takes.

    ``overlap`` returns rows of ``self`` and nothing from ``other``, so the only thing
    this option can change is how many times a row appears: 'first' and 'last' would
    select the same rows and differ in output order alone. Both ``RangeFrame.overlap``
    and ``PyRanges.overlap`` therefore spell it as a bool.

    Parameters
    ----------
    multiple : bool
        True reports a row once per overlap, False reports it at most once.

    Returns
    -------
    str
        'all' for True and 'first' for False.

    Examples
    --------
    >>> validate_and_convert_overlap_multiple(True)
    'all'
    >>> validate_and_convert_overlap_multiple(False)
    'first'
    >>> validate_and_convert_overlap_multiple("all")
    Traceback (most recent call last):
    ...
    TypeError: overlap takes multiple as a bool: use True for 'all' or False for 'first'.

    """
    if not isinstance(multiple, bool):
        msg = "overlap takes multiple as a bool: use True for 'all' or False for 'first'."
        raise TypeError(msg)
    return OVERLAP_ALL if multiple else OVERLAP_FIRST


def validate_and_convert_strand_behavior(
    self: "PyRanges",
    other: "PyRanges",
    strand_behavior: VALID_STRAND_BEHAVIOR_TYPE,
) -> STRICT_STRAND_BEHAVIOR_TYPE:
    """Validate and convert strand behavior.

    Used inside various methods accepting two PyRanges as input. It processes strand_behavior option provided
    by the user upon calling the function. This function validates the option and converts it to a valid unambiguous
    value.
    """
    if strand_behavior not in VALID_STRAND_BEHAVIOR_OPTIONS:
        msg = f"{VALID_STRAND_BEHAVIOR_OPTIONS} are the only valid values for strand_behavior. Was: {strand_behavior}"
        raise ValueError(msg)

    if not (self.strand_valid and other.strand_valid):
        if strand_behavior == STRAND_BEHAVIOR_AUTO:
            # If any Strand column is invalid, set strand_behavior to STRAND_BEHAVIOR_IGNORE
            # but warn user if both objects are Stranded and there are intervals on the negative strand
            strand_behavior = STRAND_BEHAVIOR_IGNORE
            if (
                self.has_strand
                and other.has_strand
                and (contains_negative_strand(self) or contains_negative_strand(other))
            ):
                fn_name = stack[2].function if len(stack := inspect.stack()) > 2 else "unknown_function"  # noqa: PLR2004
                fn_name = stack[1].function if fn_name == "<module>" else fn_name
                msg = (
                    f"{fn_name}: '{STRAND_BEHAVIOR_AUTO}' strand_behavior treated as {STRAND_BEHAVIOR_IGNORE} due to invalid Strand values. "
                    f"Please use strand_behavior={STRAND_BEHAVIOR_IGNORE}"
                )
                warnings.warn(msg, stacklevel=4)

        elif strand_behavior in (STRAND_BEHAVIOR_SAME, STRAND_BEHAVIOR_OPPOSITE):
            msg = f"Can only do {strand_behavior} strand operations when both PyRanges contain valid strand info."
            raise ValueError(msg)
    elif strand_behavior == STRAND_BEHAVIOR_AUTO:
        strand_behavior = STRAND_BEHAVIOR_SAME

    return strand_behavior


def group_keys_from_validated_strand_behavior(
    strand_behavior: VALID_STRAND_BEHAVIOR_TYPE,
    by: VALID_BY_OPTIONS = None,
) -> list[str]:
    """Return group keys based on strand behavior.

    If strand_behavior is 'same', return [CHROM_COL, STRAND_COL].
    If strand_behavior is 'opposite', return [CHROM_COL, STRAND_COL].
    If strand_behavior is 'ignore', return [CHROM_COL].

    In each case, if a by argument is provided, it is appended/extended to the list of group keys.

    """
    if strand_behavior == STRAND_BEHAVIOR_AUTO:
        msg = "this function must be called with a validated strand_behavior"
        raise ValueError(msg)

    strand_cols = [STRAND_COL] if strand_behavior in [STRAND_BEHAVIOR_SAME, STRAND_BEHAVIOR_OPPOSITE] else []

    return [CHROM_COL, *strand_cols, *arg_to_list(by)]


def strand_behavior_from_validated_use_strand(
    df: "PyRanges",
    use_strand: VALID_USE_STRAND_TYPE,
) -> VALID_STRAND_BEHAVIOR_TYPE:
    """Return strand_behavior based on use_strand bool.

    If strand is True, returns 'same' strand behavior, otherwise 'ignore' strand behavior.

    This function must be used with use_strand validated and converted to a boolean value.
    """
    return STRAND_BEHAVIOR_SAME if validate_and_convert_use_strand(df, use_strand) else STRAND_BEHAVIOR_IGNORE


def use_strand_from_validated_strand_behavior(
    self: "PyRanges",
    other: "PyRanges",
    strand_behavior: VALID_STRAND_BEHAVIOR_TYPE,
) -> bool:
    """Return use_strand based on strand_behavior.

    If strand_behavior is 'ignore', returns False, otherwise True if both PyRanges contain valid strand info.

    This function must be used with strand_behavior validated and converted to either 'same', 'opposite' or 'ignore'.
    """
    strand_behavior = validate_and_convert_strand_behavior(self, other, strand_behavior)
    return strand_behavior != STRAND_BEHAVIOR_IGNORE


def ensure_pyranges(df: "pd.DataFrame") -> "PyRanges":
    """Ensure df is a PyRanges.

    Helps pyright.
    """
    from pyranges1 import PyRanges

    if not isinstance(ret := PyRanges(df, copy=False), PyRanges):
        msg = "Not a PyRanges"
        raise TypeError(msg)
    return ret


def ensure_rangeframe(df: "pd.DataFrame") -> "RangeFrame":
    """Ensure df is a rangeframe.

    Helps pyright.
    """
    from pyranges1.range_frame.range_frame import RangeFrame

    if not isinstance(ret := RangeFrame(df, copy=False), RangeFrame):
        msg = "Not a RangeFrame"
        raise TypeError(msg)
    return ret


def arg_to_list(by: VALID_BY_OPTIONS) -> list[str]:
    """Convert a by or column_key argument to a list.

    If by is a string, it will be converted to a list.
    If by is None, an empty list will be returned.
    If by is already a list, it will be returned as is.
    """
    return [by] if isinstance(by, str) else ([*by] if by is not None else [])


def contains_negative_strand(self: "PyRanges") -> bool:
    """Return True if negative strand is present."""
    return self.has_strand and REVERSE_STRAND in self[STRAND_COL].to_numpy()
