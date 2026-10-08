"""nearest_ranges(signed=...), against bedtools closest -D a / -D b.

The expected signs were checked with bedtools 2.31.1.
"""

import pytest

import pyranges1 as pr

# One feature 10 bp left of each query and one 20 bp right of it, so the nearest is on the left.
QUERY = pr.PyRanges({"Chromosome": "chr1", "Start": [100, 100], "End": [110, 110], "Strand": ["+", "-"]})


def _distances(query: pr.PyRanges, features: pr.PyRanges, signed: str) -> list[int]:
    result = query.nearest_ranges(features, strand_behavior="ignore", signed=signed)
    return result.sort_values("Strand")["Distance"].tolist()


@pytest.mark.parametrize("feature_strand", ["+", "-"])
def test_signed_self_follows_the_query_strand(feature_strand) -> None:
    left = pr.PyRanges({"Chromosome": "chr1", "Start": [80], "End": [90], "Strand": [feature_strand]})
    # The feature is to the left: upstream of a "+" query (-), downstream of a "-" one (+).
    assert _distances(QUERY, left, "self") == [-11, 11]


def test_signed_other_follows_the_feature_strand() -> None:
    plus = pr.PyRanges({"Chromosome": "chr1", "Start": [80], "End": [90], "Strand": ["+"]})
    minus = pr.PyRanges({"Chromosome": "chr1", "Start": [80], "End": [90], "Strand": ["-"]})
    # The query lies right of the feature: downstream of a "+" feature, upstream of a "-" one.
    assert _distances(QUERY, plus, "other") == [11, 11]
    assert _distances(QUERY, minus, "other") == [-11, -11]


def test_overlaps_stay_zero_and_unsigned_is_unchanged() -> None:
    overlapping = pr.PyRanges({"Chromosome": "chr1", "Start": [105], "End": [120], "Strand": ["-"]})
    assert _distances(QUERY, overlapping, "self") == [0, 0]
    plain = QUERY.nearest_ranges(overlapping, strand_behavior="ignore")
    assert (plain["Distance"] >= 0).all()


def test_signed_needs_a_distance_column() -> None:
    with pytest.raises(ValueError, match="dist_col"):
        QUERY.nearest_ranges(QUERY, signed="self", dist_col=None)
