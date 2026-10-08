"""complement_ranges over the whole genome, given chromsizes."""

import pytest

import pyranges1 as pr


def test_whole_genome_complement_includes_chromosomes_without_intervals() -> None:
    sizes = {"chr1": 1000, "chr2": 500}
    gr = pr.PyRanges({"Chromosome": ["chr1"], "Start": [50], "End": [60]})
    got = gr.complement_ranges(chromsizes=sizes, include_first_interval=True)
    assert list(zip(got["Chromosome"], got["Start"], got["End"], strict=True)) == [
        ("chr1", 0, 50),
        ("chr1", 60, 1000),
        ("chr2", 0, 500),
    ]

    stranded = pr.PyRanges({"Chromosome": ["chr1", "chr1"], "Start": [50, 70], "End": [60, 80], "Strand": ["+", "-"]})
    got = stranded.complement_ranges(chromsizes=sizes, include_first_interval=True)
    assert sorted(zip(got["Chromosome"], got["Strand"], got["Start"], got["End"], strict=True))[-2:] == [
        ("chr2", "+", 0, 500),
        ("chr2", "-", 0, 500),
    ]

    # Without the first interval, the result is the internal-plus-trailing complement, as before.
    assert len(gr.complement_ranges(chromsizes=sizes)) == 1


def test_complement_names_chromosomes_missing_from_chromsizes() -> None:
    gr = pr.PyRanges({"Chromosome": ["chr1", "chr3"], "Start": [50, 5], "End": [60, 9]})
    with pytest.raises(ValueError, match="chr3"):
        gr.complement_ranges(chromsizes={"chr1": 1000}, include_first_interval=True)
