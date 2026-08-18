"""`multiple` says one thing on overlap and another on the operations that select.

`overlap` returns rows of self and nothing from other, so the only thing the option can
change is how many times a row appears -- hence a bool on both `RangeFrame.overlap` and
`PyRanges.overlap`. `intersect_overlaps`, `join_overlaps` and `set_intersect_overlaps`
return information taken from other, so "first" and "last" pick different output there
and keep the string vocabulary.

`PyRanges.overlap` used to derive the string the kernel wants with a bare truthiness
test, so every non-empty string -- including "first" -- silently meant "all".
"""

import pytest

import pyranges1 as pr
from pyranges1.core.names import VALID_OVERLAP_OPTIONS

# One annotation overlapped by two reads: enough to tell "one row per overlapping
# annotation" apart from "one row per overlapping pair".
ANNOTATION = pr.PyRanges({"Chromosome": ["chr1"], "Start": [100], "End": [200], "ID": ["ann"]})
READS = pr.PyRanges({"Chromosome": ["chr1", "chr1"], "Start": [110, 150], "End": [120, 160]})

PLAIN = pr.RangeFrame({"Start": [100], "End": [200], "ID": ["ann"]})
PLAIN_READS = pr.RangeFrame({"Start": [110, 150], "End": [120, 160]})

SELECTORS = ["intersect_overlaps", "join_overlaps", "set_intersect_overlaps"]


@pytest.mark.parametrize(("multiple", "expected_rows"), [(False, 1), (True, 2)])
def test_overlap_multiple_is_a_row_multiplicity_switch(multiple: bool, expected_rows: int) -> None:
    assert len(ANNOTATION.overlap(READS, multiple=multiple)) == expected_rows
    assert len(PLAIN.overlap(PLAIN_READS, multiple=multiple)) == expected_rows


def test_both_overlaps_default_to_filtering() -> None:
    """A bare overlap() is a filter, on the genomic class and the generic one alike.

    RangeFrame.overlap used to default to reporting every pair while PyRanges.overlap
    defaulted to filtering, which made the same bare call mean two different things.
    """
    assert len(ANNOTATION.overlap(READS)) == 1
    assert len(PLAIN.overlap(PLAIN_READS)) == 1


def test_both_overlaps_take_the_same_argument() -> None:
    """The override and the method it overrides now read `multiple` identically."""
    for multiple in (True, False):
        assert len(ANNOTATION.overlap(READS, multiple=multiple)) == len(
            PLAIN.overlap(PLAIN_READS, multiple=multiple)
        )


@pytest.mark.parametrize("multiple", ["all", "first", "last", "frist", ""])
def test_overlap_rejects_strings(multiple: str) -> None:
    """A string used to be truthy and silently mean "all"."""
    for frame, other in ((ANNOTATION, READS), (PLAIN, PLAIN_READS)):
        with pytest.raises(TypeError, match="takes multiple as a bool"):
            frame.overlap(other, multiple=multiple)


@pytest.mark.parametrize("method", SELECTORS)
@pytest.mark.parametrize("multiple", ["all", "first", "last"])
def test_selecting_operations_keep_the_vocabulary(method: str, multiple: str) -> None:
    assert len(getattr(ANNOTATION, method)(READS, multiple=multiple)) in (1, 2)


@pytest.mark.parametrize("method", SELECTORS)
def test_first_and_last_select_different_output(method: str) -> None:
    """Unlike overlap, these carry information from other, so the choice is visible."""
    first = getattr(ANNOTATION, method)(READS, multiple="first")
    last = getattr(ANNOTATION, method)(READS, multiple="last")
    assert not first.reset_index(drop=True).equals(last.reset_index(drop=True))


@pytest.mark.parametrize("method", SELECTORS)
def test_selecting_operations_reject_unknown_values(method: str) -> None:
    """ruranges panics on an unknown overlap type; the Python layer rejects it first."""
    with pytest.raises(ValueError, match="Invalid multiple option"):
        getattr(ANNOTATION, method)(READS, multiple="frist")


@pytest.mark.parametrize("method", SELECTORS)
def test_contained_is_rejected_with_a_pointer(method: str) -> None:
    """'contained' duplicated contained_intervals_only=True and only ever panicked."""
    with pytest.raises(ValueError, match="use contained_intervals_only=True"):
        getattr(ANNOTATION, method)(READS, multiple="contained")


def test_vocabulary_constant_is_guarded() -> None:
    """So a future option cannot be added without revisiting these call sites."""
    assert VALID_OVERLAP_OPTIONS == ["first", "all", "last"]
