from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, get_args

import pandas as pd

if TYPE_CHECKING:
    from pyranges1 import PyRanges, RangeFrame


def return_pyranges_if_possible(
    method: Callable,
) -> Callable:
    """Return a PyRanges object if possible."""

    def wrapper(*args, **kwargs) -> "PyRanges | pd.DataFrame | pd.Series":
        # Call the original groupby method
        result = method(*args, **kwargs)

        if isinstance(result, pd.DataFrame) and set(GENOME_LOC_COLS).issubset(result.columns):
            import pyranges1 as pr

            return pr.PyRanges(result)

        return result

    return wrapper


# Define the Literal type
VALID_OVERLAP_TYPE = Literal["first", "all", "last"]

# Extract the options from the Literal type
VALID_OVERLAP_OPTIONS = list(get_args(VALID_OVERLAP_TYPE))
OVERLAP_FIRST, OVERLAP_ALL, OVERLAP_LAST = VALID_OVERLAP_OPTIONS

# Removed in 1.3.12: 'contained' was declared here but rejected by ruranges, and it only
# ever duplicated contained_intervals_only=True. Kept as a name so the migration error
# below can point at the argument that replaces it.
REMOVED_OVERLAP_CONTAINED = "contained"

BY_ENTRY_IN_KWARGS = "__by__"

PRESERVE_INDEX_COLUMN = "__old_index__"


CHROM_COL: Final = "Chromosome"
START_COL = "Start"
END_COL = "End"
STRAND_COL = "Strand"
RANGE_COLS = [START_COL, END_COL]
CHROM_AND_STRAND_COLS = [CHROM_COL, STRAND_COL]
GENOME_LOC_COLS = [CHROM_COL, *RANGE_COLS]
GENOME_LOC_COLS_WITH_STRAND = [*GENOME_LOC_COLS, STRAND_COL]

BIGWIG_SCORE_COL = "Score"
FRAME_COL = "Frame"

FORWARD_STRAND: Final = "+"
REVERSE_STRAND: Final = "-"
VALID_GENOMIC_STRAND_TYPE = Literal["+", "-"]
VALID_GENOMIC_STRAND_INFO = [FORWARD_STRAND, REVERSE_STRAND]

VALID_BY_OPTIONS = str | Iterable[str] | None

USE_STRAND_AUTO: Final = "auto"
USE_STRAND_DEFAULT: Final = USE_STRAND_AUTO
VALID_USE_STRAND_TYPE = Literal["auto"] | bool
VALID_USE_STRAND_OPTIONS = [USE_STRAND_AUTO, True, False]

STRAND_BEHAVIOR_AUTO: Final = "auto"
STRAND_BEHAVIOR_SAME: Final = "same"
STRAND_BEHAVIOR_OPPOSITE: Final = "opposite"
STRAND_BEHAVIOR_IGNORE: Final = "ignore"
STRAND_BEHAVIOR_DEFAULT: Final = "auto"
VALID_STRAND_BEHAVIOR_TYPE = Literal["auto", "same", "opposite", "ignore"]
STRICT_STRAND_BEHAVIOR_TYPE = Literal["same", "opposite", "ignore"]
VALID_STRAND_BEHAVIOR_OPTIONS = [
    STRAND_BEHAVIOR_SAME,
    STRAND_BEHAVIOR_OPPOSITE,
    STRAND_BEHAVIOR_AUTO,
    STRAND_BEHAVIOR_IGNORE,
]


JOIN_OUTER: Final = "outer"
JOIN_INNER: Final = "inner"
JOIN_RIGHT: Final = "right"
JOIN_LEFT: Final = "left"
VALID_JOIN_TYPE = Literal["inner", "left", "outer", "right"]
VALID_JOIN_OPTIONS = [JOIN_INNER, JOIN_LEFT, JOIN_OUTER, JOIN_RIGHT]

JOIN_SUFFIX = "_b"
VALID_COMBINE_OPTIONS = Literal["intersect", "union", "swap"]

# Two direction vocabularies, deliberately kept apart. The genomic one is
# strand-aware: on the reverse strand "upstream" is the higher coordinate. The
# coordinate one knows nothing of strand: "forward" is always the higher
# coordinate. Sharing one word between them would let a coordinate-space intent
# pass silently where a biological one was meant.
NEAREST_ANY_DIRECTION: Final = "any"
NEAREST_UPSTREAM: Final = "upstream"
NEAREST_DOWNSTREAM: Final = "downstream"
VALID_GENOMIC_DIRECTION_TYPE = Literal["any", "upstream", "downstream"]
VALID_GENOMIC_DIRECTION_OPTIONS = [
    NEAREST_ANY_DIRECTION,
    NEAREST_UPSTREAM,
    NEAREST_DOWNSTREAM,
]

FORWARD_DIRECTION: Final = "forward"
BACKWARD_DIRECTION: Final = "backward"
VALID_COORDINATE_DIRECTION_TYPE = Literal["any", "forward", "backward"]
VALID_COORDINATE_DIRECTION_OPTIONS = [
    NEAREST_ANY_DIRECTION,
    FORWARD_DIRECTION,
    BACKWARD_DIRECTION,
]

TEMP_INDEX_COL = "__temp_index__"
TEMP_TYPE_COL = "__temp_type__"
TEMP_START_COL = "__temp_start__"
TEMP_END_COL = "__temp_end__"
TEMP_STRAND_COL = "__temp_strand__"
TEMP_NAME_COL = "__temp_name__"
TEMP_CUMSUM_COL = "__temp_cumsum__"
TEMP_NUM_COL = "__temp_num__"
TEMP_ID_COL = "__temp_id__"
TEMP_MIN_COL = "__temp_min__"
TEMP_MAX_COL = "__temp_max__"
TEMP_CLUSTER_COL = "__temp_cluster__"
TEMP_LENGTH_COL = "__temp_length__"
TEMP_START_SLACK_COL = "__temp_start_slack__"
TEMP_END_SLACK_COL = "__temp_end_slack__"
TEMP_TRANSCRIPT_ID_COL = "__temp_transcript_id__"

DEFAULT_CLUSTER_ID_COL = "Cluster"

VALID_BY_TYPES = str | Iterable[str] | None

PANDAS_COMPRESSION_TYPE = Literal["infer", "gzip", "bz2", "zip", "xz", "zstd"] | dict[str, Any] | None

SKIP_IF_DF_EMPTY_TYPE = Literal["left", "right", "any", "both", False]
SKIP_IF_EMPTY_LEFT: Final = "left"
SKIP_IF_EMPTY_RIGHT: Final = "right"
SKIP_IF_EMPTY_ANY: Final = "any"
SKIP_IF_EMPTY_BOTH: Final = "both"
SKIP_IF_DF_EMPTY_OPTIONS = [False, SKIP_IF_EMPTY_ANY, SKIP_IF_EMPTY_BOTH, SKIP_IF_EMPTY_LEFT, SKIP_IF_EMPTY_RIGHT]
SKIP_IF_DF_EMPTY_DEFAULT = "any"


class UnaryOperation[T: "RangeFrame"](Protocol):
    """A protocol for unary operations on RangeFrames."""

    def __call__(self, df: T, **kwargs: Any) -> "pd.DataFrame":
        """Perform the operation on the RangeFrame.

        Examples: cluster, merge, split, etc.
        """
        ...


class BinaryOperation[T: "RangeFrame"](Protocol):
    """A protocol for binary operations on RangeFrames."""

    def __call__(self, df: T, *, df2: T, **kwargs: Any) -> "pd.DataFrame":
        """Perform the operation on the pair of RangeFrames.

        Examples: overlap, nearest, join, etc.
        """
        ...


class CombineIntervalColumnsOperation(Protocol):
    """A protocol for functions passed to combine_interval_columns.

    A protocol indicating that the operation combines interval columns from a Pyranges object,
    expecting four pd.Series as inputs (starts, ends, starts2, ends2), and returns a tuple of pd.Series (new_starts, new_ends).
    """

    def __call__(
        self,
        starts: pd.Series,
        ends: pd.Series,
        starts2: pd.Series,
        ends2: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """Perform the operation on starts, starts2, ends, ends2 columns and return new start and end columns."""
        ...
