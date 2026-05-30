import pandas as pd
import pytest
from numpy import nan

import pyranges1 as pr


@pytest.mark.parametrize("index", [[0, 1], [1, 0], [1, 2], [2, 1]])
def test_join_left_nonzero_index(index) -> None:
    """A left join must not depend on the labels of self's index (issue: rows duplicated
    when the left index did not start at 0). The original self index is also preserved,
    consistent with overlap()/nearest_ranges()."""
    left = pd.DataFrame(
        {"Chromosome": ["1", "2"], "Start": [100, 300], "End": [200, 400]},
        index=index,
    )
    right = pd.DataFrame({"Chromosome": ["1"], "Start": [150], "End": [160]})

    j = pr.PyRanges(left).join_overlaps(pr.PyRanges(right), join_type="left")

    expected = pr.PyRanges(
        {
            "Chromosome": ["1", "2"],
            "Start": [100, 300],
            "End": [200, 400],
            "Start_b": [150.0, nan],
            "End_b": [160.0, nan],
        },
        index=index,
    )
    assert j.equals(expected)


def test_join_preserves_input_index() -> None:
    """join_overlaps preserves the original input index (self for inner/left/outer,
    other for right), like the rest of the overlap-family methods."""
    left = pd.DataFrame(
        {"Chromosome": ["1", "2"], "Start": [100, 300], "End": [200, 400]},
        index=[10, 20],
    )
    right = pd.DataFrame({"Chromosome": ["1"], "Start": [150], "End": [160]}, index=[77])
    gl, gr = pr.PyRanges(left), pr.PyRanges(right)

    assert gl.join_overlaps(gr, join_type="inner").index.tolist() == [10]
    assert gl.join_overlaps(gr, join_type="left").index.tolist() == [10, 20]
    assert gl.join_overlaps(gr, join_type="right").index.tolist() == [77]
    assert gl.join_overlaps(gr, join_type="outer").index.tolist() == [10, 20]
    # inputs must not be mutated
    assert left.index.tolist() == [10, 20]
    assert right.index.tolist() == [77]


def test_join_issue_4_right() -> None:
    import numpy as np

    chromsizes = pr.example_data.chromsizes
    query_regions = pr.tile_genome(chromsizes, int(1e6))
    signal_data = pr.example_data.chipseq
    signal_data["Score"] = np.random.randint(0, 100, len(signal_data))

    query_regions.join_overlaps(signal_data)


def test_join_issue_8() -> None:
    gd = {
        "Chromosome": ["chr1", "chr1", "chr1", "chr1"],
        "Start": [157, 584, 731, 821],
        "End": [257, 684, 831, 921],
        "Strand": ["-", "-", "-", "-"],
    }
    md = {
        "Chromosome": ["chr1", "chr1", "chr1", "chr1"],
        "Start": [316, 793, 889, 795],
        "End": [416, 893, 989, 895],
        "Strand": ["+", "+", "+", "-"],
    }

    g = pr.PyRanges(gd)
    m = pr.PyRanges(md)

    j = m.join_overlaps(g)
    expected_result = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [795, 795],
            "End": [895, 895],
            "Strand": ["-", "-"],
            "Start_b": [731, 821],
            "End_b": [831, 921],
        },
        index=[0, 1],
    )

    assert j.reset_index(drop=True).equals(expected_result)


def test_join_issue_8_right() -> None:
    gd = {
        "Chromosome": ["chr1", "chr1", "chr1", "chr1"],
        "Start": [157, 584, 731, 821],
        "End": [257, 684, 831, 921],
        "Strand": ["-", "-", "-", "-"],
    }
    md = {
        "Chromosome": ["chr1", "chr1", "chr1", "chr1"],
        "Start": [316, 793, 889, 795],
        "End": [416, 893, 989, 895],
        "Strand": ["+", "+", "+", "-"],
    }

    g = pr.PyRanges(gd)
    m = pr.PyRanges(md)

    j = m.join_overlaps(g, join_type="right")

    expected_result = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1", nan, nan],
            "Start": [795.0, 795.0, nan, nan],
            "End": [895.0, 895.0, nan, nan],
            "Strand": ["-", "-", nan, nan],
            "Start_b": [731, 821, 157, 584],
            "End_b": [831, 921, 257, 684],
        },
        index=[0, 1, 2, 3],
    )
    assert j.reset_index(drop=True).equals(expected_result)
