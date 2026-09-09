#!/usr/bin/env python3
"""Tests for open-ended ``loci`` slice bounds.

An omitted bound in ``gr.loci[start:end]`` leaves that side of the window
open. The upper bound already used an infinite limit, but the lower bound
defaulted to ``-1``, which silently dropped intervals lying entirely below
that coordinate.
"""

import numpy as np
import pandas as pd

import pyranges1 as pr


def _spans(gr) -> list[tuple[int, int]]:
    return sorted(zip(gr.Start.tolist(), gr.End.tolist(), strict=True))


def _negative_ranges() -> "pr.PyRanges":
    """Intervals below, across, and above the origin."""
    return pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": ["chr1"] * 3,
                "Start": [-100, -20, 5],
                "End": [-50, 10, 30],
                "Strand": ["+", "-", "+"],
            },
        ),
    )


def test_omitted_start_selects_intervals_below_the_origin() -> None:
    gr = _negative_ranges()
    # Every interval starting before 1, including the wholly negative one.
    assert _spans(gr.loci[:1]) == [(-100, -50), (-20, 10)]


def test_omitted_start_reaches_a_negative_upper_bound() -> None:
    gr = _negative_ranges()
    # A window lying entirely in negative coordinates.
    assert _spans(gr.loci[:-60]) == [(-100, -50)]


def test_omitted_end_is_unchanged_by_the_lower_bound_default() -> None:
    gr = _negative_ranges()
    assert _spans(gr.loci[-60:]) == [(-100, -50), (-20, 10), (5, 30)]


def test_both_bounds_omitted_selects_everything() -> None:
    gr = _negative_ranges()
    assert _spans(gr.loci[:]) == [(-100, -50), (-20, 10), (5, 30)]


def test_open_bounds_agree_with_an_explicit_infinite_window() -> None:
    gr = _negative_ranges()
    for key, explicit in (
        (slice(None, 1), slice(-np.inf, 1)),
        (slice(-60, None), slice(-60, np.inf)),
        (slice(None, None), slice(-np.inf, np.inf)),
    ):
        assert _spans(gr.loci[key]) == _spans(gr.loci[explicit]), key


def test_non_negative_coordinates_are_unaffected() -> None:
    """The old lower default only differed for coordinates at or below -1."""
    gr = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": ["chr1"] * 3,
                "Start": [0, 10, 20],
                "End": [5, 15, 25],
                "Strand": ["+", "-", "+"],
            },
        ),
    )
    assert _spans(gr.loci[:12]) == [(0, 5), (10, 15)]
    assert _spans(gr.loci[12:]) == [(10, 15), (20, 25)]
    assert _spans(gr.loci[:]) == [(0, 5), (10, 15), (20, 25)]


def test_open_bounds_compose_with_chromosome_and_strand() -> None:
    gr = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr1", "chr2"],
                "Start": [-100, -20, -100],
                "End": [-50, 10, -50],
                "Strand": ["+", "-", "+"],
            },
        ),
    )
    assert _spans(gr.loci["chr1", :1]) == [(-100, -50), (-20, 10)]
    assert _spans(gr.loci["chr1", "+", slice(None, 1)]) == [(-100, -50)]
    assert _spans(gr.loci["+", :1]) == [(-100, -50), (-100, -50)]
