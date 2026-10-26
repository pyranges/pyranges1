"""Operations accept columns that are strided views of a 2-D block.

pandas can hand out such columns from `Series.values` (a frame built from a 2-D
array without copying, and since pandas 3.1 some join and concat results). The
Rust kernels need contiguous input; coordinates past int32 are passed through
uncast, so nothing on the way copies them.
"""

import numpy as np
import pandas as pd
import pytest

import pyranges1 as pr

BIG = 2**40


def ranges(rows: list[list[int]], *, strided: bool) -> pr.PyRanges:
    block = np.array(rows, dtype=np.int64) + BIG
    if strided:
        df = pd.DataFrame(block, columns=["Start", "End"], copy=False)
    else:
        df = pd.DataFrame({"Start": block[:, 0].copy(), "End": block[:, 1].copy()})
    assert df["Start"].to_numpy().flags.c_contiguous is not strided
    df.insert(0, "Chromosome", "chr1")
    df["Strand"] = ["+", "-", "+"]
    return pr.PyRanges(df)


A = [[1, 5], [3, 9], [20, 30]]
B = [[2, 4], [25, 26], [40, 41]]


@pytest.mark.parametrize(
    "operation",
    [
        lambda a, b: a.overlap(b),
        lambda a, b: a.join_overlaps(b),
        lambda a, b: a.nearest_ranges(b),
        lambda a, b: a.subtract_overlaps(b),
        lambda a, b: a.count_overlaps(b),
        lambda a, b: a.merge_overlaps(),
        lambda a, b: a.cluster_overlaps(),
        lambda a, b: a.extend_ranges(ext=1),
        lambda a, b: a.window_ranges(2),
        lambda a, b: a.tile_ranges(2),
        lambda a, b: a.sort_ranges(),
        lambda a, b: a.max_disjoint_overlaps(),
        lambda a, b: a.complement_ranges(),
    ],
)
def test_operations_accept_strided_columns(operation) -> None:
    contiguous = operation(ranges(A, strided=False), ranges(B, strided=False))
    result = operation(ranges(A, strided=True), ranges(B, strided=True))
    pd.testing.assert_frame_equal(pd.DataFrame(result), pd.DataFrame(contiguous))
