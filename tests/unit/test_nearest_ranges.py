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


def test_rangeframe_rejects_the_genomic_direction_vocabulary() -> None:
    """A bad direction must raise, not abort the interpreter.

    RangeFrame passed `direction` straight to ruranges, which panics with
    "Invalid direction string". PanicException is not an Exception, so
    `except Exception` could not catch it. Same defect 1.4.0 fixed for
    `multiple`.
    """
    query = pr.RangeFrame({"Start": [100], "End": [120]})
    other = pr.RangeFrame({"Start": [10, 200], "End": [20, 210]})

    # "upstream"/"downstream" are strand-aware and belong to PyRanges.
    for direction in ("upstream", "downstream", "nonsense"):
        try:
            query.nearest_ranges(other, direction=direction)
        except ValueError as error:
            assert "direction must be one of" in str(error)
        else:
            msg = f"{direction!r} should have raised ValueError"
            raise AssertionError(msg)

    # The coordinate vocabulary still works: forward is the higher coordinate.
    assert query.nearest_ranges(other, direction="forward")["Start_b"].tolist() == [200]
    assert query.nearest_ranges(other, direction="backward")["Start_b"].tolist() == [10]


def test_pyranges_rejects_the_coordinate_direction_vocabulary() -> None:
    """The two vocabularies stay apart: PyRanges speaks upstream/downstream only."""
    query = pr.PyRanges({"Chromosome": ["chr1"], "Start": [100], "End": [120], "Strand": ["+"]})
    other = pr.PyRanges({"Chromosome": ["chr1"], "Start": [200], "End": [210], "Strand": ["+"]})
    for direction in ("forward", "backward"):
        try:
            query.nearest_ranges(other, direction=direction)
        except ValueError as error:
            assert "direction must be one of" in str(error)
        else:
            msg = f"{direction!r} should have raised ValueError"
            raise AssertionError(msg)


def test_directional_query_without_strand_says_so() -> None:
    """Unstranded input used to fail with "name 'Strand' is not defined"."""
    query = pr.PyRanges({"Chromosome": ["chr1"], "Start": [100], "End": [120]})
    other = pr.PyRanges({"Chromosome": ["chr1", "chr1"], "Start": [10, 200], "End": [20, 210]})
    try:
        query.nearest_ranges(other, direction="upstream")
    except ValueError as error:
        assert "strand-aware" in str(error)
    else:
        raise AssertionError("an unstranded directional query should have raised")

    # Undirected queries are unaffected.
    assert sorted(query.nearest_ranges(other, direction="any")["Start_b"].tolist()) == [10, 200]
