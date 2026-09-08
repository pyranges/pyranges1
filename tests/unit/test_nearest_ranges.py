import numpy as np

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


def test_ties_first_answers_the_same_queries_with_one_row_each() -> None:
    """The one property that catches nearly everything.

    ties="first" must answer exactly the queries ties="all" answers, at the
    same distance, with exactly one row each. Dense pseudo-random input, so
    most queries overlap several intervals of other and so tie at distance 0.
    """
    rng = np.random.default_rng(42)
    starts = rng.integers(0, 5_000, size=400)
    starts_b = rng.integers(0, 5_000, size=400)
    query = pr.PyRanges(
        {
            "Chromosome": rng.choice(["chr1", "chr2"], size=400),
            "Start": starts,
            "End": starts + rng.integers(1, 200, size=400),
        }
    )
    other = pr.PyRanges(
        {
            "Chromosome": rng.choice(["chr1", "chr2"], size=400),
            "Start": starts_b,
            "End": starts_b + rng.integers(1, 200, size=400),
        }
    )

    for exclude_overlaps in (False, True):
        every = query.nearest_ranges(other, exclude_overlaps=exclude_overlaps)
        one = query.nearest_ranges(other, exclude_overlaps=exclude_overlaps, ties="first")

        assert not one.index.duplicated().any()
        assert sorted(one.index) == sorted(set(every.index))
        # k=1 leaves a single distance bucket, so every row of a query shares
        # its distance and comparing per index is well defined.
        winning = dict(zip(every.index, every["Distance"], strict=True))
        assert {i: d for i, d in zip(one.index, one["Distance"], strict=True)} == winning
        # A fixture without ties would pass all of the above and prove nothing.
        assert len(one) < len(every)


def test_ties_first_reports_one_of_the_overlapping_intervals() -> None:
    """Overlaps are all at distance 0, which is where the row count explodes."""
    query = pr.PyRanges({"Chromosome": ["chr1"], "Start": [100], "End": [200]})
    other = pr.PyRanges({"Chromosome": ["chr1"] * 3, "Start": [90, 120, 150], "End": [110, 130, 160]})

    every = query.nearest_ranges(other)
    assert len(every) == 3
    assert every["Distance"].tolist() == [0, 0, 0]

    one = query.nearest_ranges(other, ties="first")
    assert len(one) == 1
    assert one["Distance"].tolist() == [0]
    assert one["Start_b"].tolist()[0] in (90, 120, 150)


def test_ties_first_breaks_a_two_sided_tie_and_keeps_the_overlap() -> None:
    """A neighbour that merely touches is at distance 1, not 0, so the
    overlapping one must still win when only one row comes back."""
    query = pr.PyRanges({"Chromosome": ["chr1"], "Start": [100], "End": [110]})

    # 85-95 and 115-125 are equidistant, one on each side.
    both_sides = pr.PyRanges({"Chromosome": ["chr1", "chr1"], "Start": [85, 115], "End": [95, 125]})
    assert query.nearest_ranges(both_sides)["Distance"].tolist() == [6, 6]
    assert query.nearest_ranges(both_sides, ties="first")["Distance"].tolist() == [6]

    # 95-105 overlaps, 110-120 only touches.
    overlap_and_touch = pr.PyRanges({"Chromosome": ["chr1", "chr1"], "Start": [95, 110], "End": [105, 120]})
    picked = query.nearest_ranges(overlap_and_touch, ties="first")
    assert picked["Distance"].tolist() == [0]
    assert picked["Start_b"].tolist() == [95]


def test_ties_first_reports_one_row_per_distance_when_k_is_two() -> None:
    """k counts distinct distances, so ties="first" gives at most k rows."""
    query = pr.PyRanges({"Chromosome": ["chr1"], "Start": [100], "End": [110]})
    other = pr.PyRanges({"Chromosome": ["chr1"] * 4, "Start": [85, 115, 75, 125], "End": [95, 125, 85, 135]})

    assert query.nearest_ranges(other, k=2)["Distance"].tolist() == [6, 6, 16, 16]
    assert query.nearest_ranges(other, k=2, ties="first")["Distance"].tolist() == [6, 16]


def test_ties_first_applies_per_strand_for_a_directional_query() -> None:
    """The directional path splits self by strand and queries each half, so the
    option has to survive both halves rather than only the "any" shortcut."""
    query = pr.PyRanges({"Chromosome": ["chr1", "chr1"], "Start": [100, 100], "End": [110, 110], "Strand": ["+", "-"]})
    other = pr.PyRanges(
        {
            "Chromosome": ["chr1"] * 4,
            "Start": [50, 50, 200, 200],
            "End": [60, 60, 210, 210],
            "Strand": ["+", "-", "+", "-"],
        }
    )

    assert len(query.nearest_ranges(other, direction="upstream", strand_behavior="ignore")) == 4
    upstream = query.nearest_ranges(other, direction="upstream", strand_behavior="ignore", ties="first")
    assert len(upstream) == 2
    # Upstream is the lower coordinate on +, the higher one on -.
    assert dict(zip(upstream["Strand"], upstream["Start_b"], strict=True)) == {"+": 50, "-": 200}


def test_directional_nearest_preserves_input_order_across_strands() -> None:
    """preserve_input_order must interleave the strand halves back into input order.

    A directional query (upstream/downstream) splits self by strand, searches each
    half on its own, and concatenates the results. That concat groups the rows by
    strand -- every + row, then every - row -- so on an interleaved-strand frame the
    output no longer matches the input and preserve_input_order had no effect at all
    (issue #169). direction="any" never splits and always honoured the option, so it
    is the oracle here.
    """
    query = pr.PyRanges(
        {
            "Chromosome": ["chr1"] * 4,
            "Start": [100, 300, 500, 700],
            "End": [120, 320, 520, 720],
            "Strand": ["+", "-", "+", "-"],
            "Id": ["a", "b", "c", "d"],
        }
    )
    other = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [0, 900],
            "End": [10, 910],
            "Strand": ["+", "+"],
        }
    )

    # direction="any" is unaffected: it returns rows in input order (the oracle).
    for preserve in (True, False):
        result = query.nearest_ranges(
            other, direction="any", strand_behavior="ignore", preserve_input_order=preserve
        )
        assert result["Id"].tolist() == ["a", "b", "c", "d"]
        assert list(result.index) == [0, 1, 2, 3]

    # Directional queries must honour it too: input order, original index.
    for direction in ("upstream", "downstream"):
        result = query.nearest_ranges(
            other, direction=direction, strand_behavior="ignore", preserve_input_order=True
        )
        assert result["Id"].tolist() == ["a", "b", "c", "d"]
        assert list(result.index) == [0, 1, 2, 3]

    # preserve_input_order=False is unchanged: rows stay grouped by strand, every
    # forward row before every reverse row, never the interleaved input order.
    for direction in ("upstream", "downstream"):
        grouped = query.nearest_ranges(
            other, direction=direction, strand_behavior="ignore", preserve_input_order=False
        )
        ids = grouped["Id"].tolist()
        assert set(ids) == {"a", "b", "c", "d"}
        assert set(ids[:2]) == {"a", "c"}
        assert set(ids[2:]) == {"b", "d"}


def test_nearest_ranges_rejects_an_unknown_ties() -> None:
    """An unknown value must raise, not reach the kernel: a Rust panic is not
    an Exception, so `except Exception` cannot catch it."""
    query = pr.PyRanges({"Chromosome": ["chr1"], "Start": [100], "End": [120]})
    other = pr.PyRanges({"Chromosome": ["chr1"], "Start": [200], "End": [210]})

    for frames in ((query, other), (pr.RangeFrame(query), pr.RangeFrame(other))):
        try:
            frames[0].nearest_ranges(frames[1], ties="nonsense")
        except ValueError as error:
            assert "ties must be one of" in str(error)
        else:
            raise AssertionError("an unknown ties should have raised ValueError")
