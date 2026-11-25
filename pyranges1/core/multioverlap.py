from typing import TYPE_CHECKING, Optional

from pyranges1.core.names import VALID_STRAND_BEHAVIOR_TYPE
from pyranges1.core.pyranges_helpers import ensure_pyranges
from pyranges1.methods import concat

if TYPE_CHECKING:
    from pyranges1.core.pyranges_main import PyRanges


def count_overlaps(
    grs: dict[str, "PyRanges"],
    features: Optional["PyRanges"] = None,
    strand_behavior: VALID_STRAND_BEHAVIOR_TYPE = "auto",
    by: list[str] | None = None,
) -> "PyRanges":
    """Count overlaps in multiple pyranges.

    Parameters
    ----------
    grs : dict of PyRanges
        The PyRanges to use as queries.

    features : PyRanges, default None
        The PyRanges to use as subject in the query. If None, the PyRanges themselves are used as a query.

    strand_behavior : {None, "same", "opposite", False}, default None, i.e. auto
        Whether to compare PyRanges on the same strand, the opposite or ignore strand
        information. The default, None, means use "same" if both PyRanges are stranded,
        otherwise ignore the strand information.

     how : {None, "all", "containment", "first"}, default None, i.e. all
        What intervals to report. By default reports all overlapping intervals. "containment"
        reports intervals where the overlapping is contained within it.

    by : list of str, default None
        Columns that must match between features and queries to count an overlap.
        If features is None, the inferred features are split within each group.

    Examples
    --------
    >>> import pyranges1 as pr
    >>> a = '''Chromosome Start End
    ... chr1    6    12
    ... chr1    10    20
    ... chr1    22    27
    ... chr1    24    30'''

    >>> b = '''Chromosome Start End
    ... chr1    12    32
    ... chr1    14    30'''

    >>> c = '''Chromosome Start End
    ... chr1    8    15
    ... chr1    10    14
    ... chr1    32    34'''

    >>> grs = {n: pr.from_string(s) for n, s in zip(["a", "b", "c"], [a, b, c])}
    >>> for k, v in grs.items():
    ...     print("Name: " + k)
    ...     print(v)
    Name: a
      index  |    Chromosome      Start      End
      int64  |    str             int64    int64
    -------  ---  ------------  -------  -------
          0  |    chr1                6       12
          1  |    chr1               10       20
          2  |    chr1               22       27
          3  |    chr1               24       30
    PyRanges with 4 rows, 3 columns, and 1 index columns.
    Contains 1 chromosomes.
    Name: b
      index  |    Chromosome      Start      End
      int64  |    str             int64    int64
    -------  ---  ------------  -------  -------
          0  |    chr1               12       32
          1  |    chr1               14       30
    PyRanges with 2 rows, 3 columns, and 1 index columns.
    Contains 1 chromosomes.
    Name: c
      index  |    Chromosome      Start      End
      int64  |    str             int64    int64
    -------  ---  ------------  -------  -------
          0  |    chr1                8       15
          1  |    chr1               10       14
          2  |    chr1               32       34
    PyRanges with 3 rows, 3 columns, and 1 index columns.
    Contains 1 chromosomes.

    >>> pr.count_overlaps(grs)
    index    |    Chromosome    Start    End      a        b        c
    int64    |    str           int64    int64    int64    int64    int64
    -------  ---  ------------  -------  -------  -------  -------  -------
    0        |    chr1          6        8        1        0        0
    1        |    chr1          8        10       1        0        1
    2        |    chr1          10       12       2        0        2
    3        |    chr1          12       14       1        1        2
    ...      |    ...           ...      ...      ...      ...      ...
    8        |    chr1          24       27       2        2        0
    9        |    chr1          27       30       1        2        0
    10       |    chr1          30       32       0        1        0
    11       |    chr1          32       34       0        0        1
    PyRanges with 12 rows, 6 columns, and 1 index columns.
    Contains 1 chromosomes.

    >>> gr = pr.PyRanges({"Chromosome": ["chr1"] * 2, "Start": [0, 25], "End": [40, 35]}).tile_ranges(10)
    >>> gr
      index  |    Chromosome      Start      End
      int64  |    str             int64    int64
    -------  ---  ------------  -------  -------
          0  |    chr1                0       10
          0  |    chr1               10       20
          0  |    chr1               20       30
          0  |    chr1               30       40
          1  |    chr1               20       30
          1  |    chr1               30       40
    PyRanges with 6 rows, 3 columns, and 1 index columns (with 4 index duplicates).
    Contains 1 chromosomes.

    >>> pr.count_overlaps(grs, gr)
      index  |    Chromosome      Start      End        a        b        c
      int64  |    str             int64    int64    int64    int64    int64
    -------  ---  ------------  -------  -------  -------  -------  -------
          0  |    chr1                0       10        1        0        1
          0  |    chr1               10       20        2        2        2
          0  |    chr1               20       30        2        2        0
          0  |    chr1               30       40        0        1        1
          1  |    chr1               20       30        2        2        0
          1  |    chr1               30       40        0        1        1
    PyRanges with 6 rows, 6 columns, and 1 index columns (with 4 index duplicates).
    Contains 1 chromosomes.

    """
    concated = concat.concat(grs.values())
    _features = concated.split_overlaps(between=True, match_by=by) if features is None else features.copy()

    for name, gr in grs.items():
        counts = _features._count_overlaps(  # noqa: SLF001
            gr,
            match_by=by,
            strand_behavior=strand_behavior,
        )
        _features.insert(_features.shape[1], name, counts)
    return ensure_pyranges(_features.astype(dict.fromkeys(grs, int)))
