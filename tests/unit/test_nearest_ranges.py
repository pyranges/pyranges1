import pyranges1 as pr


def test_nearest_ranges_direction_respects_minus_strand() -> None:
    """nearest_ranges(direction="upstream"/"downstream") must flip coordinate direction
    on the minus strand, exactly like PyRanges.upstream()/downstream() do.

    For a minus-strand query, "upstream" is the higher-coordinate neighbor and
    "downstream" is the lower-coordinate neighbor. Regression test for a bug where the
    reverse-strand half of nearest_ranges queried ruranges with the same coordinate
    direction as the forward-strand half, so minus-strand results were backwards.
    """
    other = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [10, 200],
            "End": [20, 210],
            "Strand": ["-", "-"],
        }
    )

    # Minus-strand query: upstream must pick the higher-coordinate neighbor (200-210).
    minus = pr.PyRanges({"Chromosome": ["chr1"], "Start": [100], "End": [120], "Strand": ["-"]})
    assert minus.nearest_ranges(other, direction="upstream")["Start_b"].tolist() == [200]
    # ... and downstream must pick the lower-coordinate neighbor (10-20).
    assert minus.nearest_ranges(other, direction="downstream")["Start_b"].tolist() == [10]

    # Control: plus-strand query behavior is unaffected (upstream = lower coordinates).
    other_plus = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [10, 200],
            "End": [20, 210],
            "Strand": ["+", "+"],
        }
    )
    plus = pr.PyRanges({"Chromosome": ["chr1"], "Start": [100], "End": [120], "Strand": ["+"]})
    assert plus.nearest_ranges(other_plus, direction="upstream")["Start_b"].tolist() == [10]
    assert plus.nearest_ranges(other_plus, direction="downstream")["Start_b"].tolist() == [200]
