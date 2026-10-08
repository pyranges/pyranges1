"""resize_ranges and promoters, against GenomicRanges::resize and ::promoters.

The expected values were computed with GenomicRanges 1.64 (1-based coordinates
converted to 0-based, half-open).
"""

import pytest

import pyranges1 as pr

# GenomicRanges strands "+", "-" and "*"; pyranges1 keeps strands valid within a frame,
# so the unstranded row is a frame of its own.
STRANDED = pr.PyRanges({"Chromosome": "chr1", "Start": [10, 10], "End": [20, 20], "Strand": ["+", "-"]})
UNSTRANDED = pr.PyRanges({"Chromosome": "chr1", "Start": [10], "End": [14]})


def _coords(gr: pr.PyRanges) -> list[tuple[int, int]]:
    return list(zip(gr["Start"].tolist(), gr["End"].tolist(), strict=True))


def _both(method: str, *args, **kwargs) -> list[tuple[int, int]]:
    return _coords(getattr(STRANDED, method)(*args, **kwargs)) + _coords(getattr(UNSTRANDED, method)(*args, **kwargs))


@pytest.mark.parametrize(
    ("width", "fix", "expected"),
    [
        (3, "center", [(13, 16), (13, 16), (10, 13)]),
        (4, "start", [(10, 14), (16, 20), (10, 14)]),
        (4, "end", [(16, 20), (10, 14), (10, 14)]),
        (16, "center", [(7, 23), (7, 23), (4, 20)]),
    ],
)
def test_resize_ranges_matches_genomicranges(width, fix, expected) -> None:
    assert _both("resize_ranges", width, fix) == expected


def test_promoters_matches_genomicranges() -> None:
    assert _both("promoters", upstream=5, downstream=2) == [(5, 12), (18, 25), (5, 12)]


def test_promoters_and_resize_ignore_strand_when_asked() -> None:
    assert _coords(STRANDED.promoters(upstream=5, downstream=2, use_strand=False)) == [(5, 12), (5, 12)]
    assert _coords(STRANDED.resize_ranges(4, "start", use_strand=False)) == [(10, 14), (10, 14)]


def test_arguments_are_checked() -> None:
    with pytest.raises(ValueError, match="width"):
        STRANDED.resize_ranges(0)
    with pytest.raises(ValueError, match="fix"):
        STRANDED.resize_ranges(3, "middle")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="upstream"):
        STRANDED.promoters(upstream=0, downstream=0)
