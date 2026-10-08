"""Tests for the additional readers: read_sam, read_cram, read_bigbed, read_pairs.

read_pairs is pure pandas and always runs. read_sam/read_cram need pysam and
read_bigbed's round-trip fixture needs pybigtools to *write* a BigBed (read_bigbed
itself uses pyBigWig); those tests skip when the optional dependency is absent, so
the suite stays green in environments without them.
"""

import tempfile
from pathlib import Path

import pytest

import pyranges1 as pr


def _write_pairs(contents: str) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".pairs", delete=False)
    tmp.write(contents)
    tmp.flush()
    tmp.close()
    return tmp.name


PAIRS = (
    "## pairs format v1.0\n"
    "#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n"
    "r1\tchr1\t100\tchr2\t200\t+\t-\n"
    "r2\tchr1\t150\tchr1\t900\t-\t+\n"
)


def test_read_pairs_anchor1() -> None:
    gr = pr.read_pairs(_write_pairs(PAIRS))
    assert list(gr.columns) == [
        "Chromosome",
        "Start",
        "End",
        "Strand",
        "OtherChromosome",
        "OtherStart",
        "OtherEnd",
        "OtherStrand",
        "ReadID",
    ]
    # 1-based pos -> 0-based Start; anchor=1 -> mate 1 on the genomic columns.
    assert gr["Start"].tolist() == [99, 149]
    assert gr["End"].tolist() == [100, 150]
    assert gr["OtherStart"].tolist() == [199, 899]
    assert [str(c) for c in gr["Chromosome"]] == ["chr1", "chr1"]
    assert [str(c) for c in gr["OtherChromosome"]] == ["chr2", "chr1"]
    assert gr["ReadID"].tolist() == ["r1", "r2"]


def test_read_pairs_anchor2_swaps_sides() -> None:
    gr = pr.read_pairs(_write_pairs(PAIRS), anchor="2")
    # anchor=2 promotes mate 2 to the genomic columns.
    assert gr["Start"].tolist() == [199, 899]
    assert gr["OtherStart"].tolist() == [99, 149]
    assert [str(c) for c in gr["Chromosome"]] == ["chr2", "chr1"]


def test_read_pairs_nrows() -> None:
    gr = pr.read_pairs(_write_pairs(PAIRS), nrows=1)
    assert len(gr) == 1
    assert gr["ReadID"].tolist() == ["r1"]


def test_read_pairs_bad_anchor() -> None:
    with pytest.raises(ValueError, match="anchor must be"):
        pr.read_pairs(_write_pairs(PAIRS), anchor="3")


def test_read_sam_matches_known_values() -> None:
    pysam = pytest.importorskip("pysam")
    bam = str(pr.example_data.files["smaller.bam"])
    sam = tempfile.mktemp(suffix=".sam")
    with pysam.AlignmentFile(bam, "rb") as b, pysam.AlignmentFile(sam, "w", header=b.header) as s:
        for r in b:
            s.write(r)

    gr = pr.read_sam(sam)
    assert gr.shape == (100, 5)
    assert list(gr.columns) == ["Chromosome", "Start", "End", "Strand", "Flag"]
    assert str(gr["Chromosome"].dtype) == "category"
    assert str(gr["Flag"].dtype) == "uint16"
    # First/third rows as documented for read_bam on the same fixture.
    assert (str(gr["Chromosome"][0]), int(gr["Start"][0]), int(gr["End"][0])) == ("chr1", 887771, 887796)
    assert (str(gr["Strand"][0]), int(gr["Flag"][0])) == ("-", 16)
    assert (str(gr["Strand"][2]), int(gr["Flag"][2])) == ("+", 0)

    full = pr.read_sam(sam, sparse=False)
    assert list(full.columns) == [
        "Chromosome",
        "Start",
        "End",
        "Strand",
        "Flag",
        "QueryStart",
        "QueryEnd",
        "QuerySequence",
        "Name",
        "Cigar",
        "Quality",
    ]
    Path(sam).unlink()


def test_read_cram_roundtrip() -> None:
    pysam = pytest.importorskip("pysam")
    d = Path(tempfile.mkdtemp())
    ref = d / "ref.fa"
    ref.write_text(">chrT\n" + "ACGT" * 100 + "\n")
    pysam.faidx(str(ref))

    cram = d / "reads.cram"
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chrT", "LN": 400}]}
    with pysam.AlignmentFile(str(cram), "wc", header=header, reference_filename=str(ref)) as out:
        for i, (start, rev) in enumerate([(10, False), (50, True), (100, False)]):
            a = pysam.AlignedSegment()
            a.query_name = f"r{i}"
            a.query_sequence = "ACGTACGTAC"
            a.flag = 16 if rev else 0
            a.reference_id = 0
            a.reference_start = start
            a.mapping_quality = 60
            a.cigarstring = "10M"
            a.query_qualities = pysam.qualitystring_to_array("IIIIIIIIII")
            out.write(a)

    gr = pr.read_cram(str(cram), reference_filename=str(ref))
    assert gr.shape == (3, 5)
    assert [int(s) for s in gr["Start"]] == [10, 50, 100]
    assert [int(e) for e in gr["End"]] == [20, 60, 110]
    assert [str(s) for s in gr["Strand"]] == ["+", "-", "+"]
    assert [int(f) for f in gr["Flag"]] == [0, 16, 0]


def test_read_bigbed_roundtrip() -> None:
    pybigtools = pytest.importorskip("pybigtools")
    pytest.importorskip("pyBigWig")

    bb = tempfile.mktemp(suffix=".bb")
    autosql = (
        'table bed6\n"BED6"\n(\nstring chrom;\nuint chromStart;\nuint chromEnd;\n'
        "string name;\nuint score;\nchar[1] strand;\n)"
    )
    w = pybigtools.open(bb, "w")
    w.write(
        {"chr1": 1000},
        iter([("chr1", 10, 20, "gene1\t100\t+"), ("chr1", 30, 45, "gene2\t200\t-")]),
        autosql=autosql,
    )

    gr = pr.read_bigbed(bb)
    assert list(gr.columns) == ["Chromosome", "Start", "End", "Name", "Score", "Strand"]
    assert [int(s) for s in gr["Start"]] == [10, 30]
    assert [int(e) for e in gr["End"]] == [20, 45]
    assert [str(n) for n in gr["Name"]] == ["gene1", "gene2"]
    assert [str(s) for s in gr["Strand"]] == ["+", "-"]
    Path(bb).unlink()
