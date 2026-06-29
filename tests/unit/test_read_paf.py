"""Tests for read_paf (minimap2 PAF). Pure pandas, no optional dependency, so all run."""

import gzip
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import pyranges1 as pr

# Two alignments + a malformed short line. Tags: NM (int), dv (float), tp (char).
# Record 2 deliberately lacks NM/tp to exercise ragged tag fill.
PAF = (
    "q1\t100\t10\t90\t+\tchr1\t1000\t200\t280\t75\t80\t60\tNM:i:5\tdv:f:0.02\ttp:A:P\n"
    "q2\t50\t0\t40\t-\tchr2\t2000\t500\t540\t38\t40\t30\tdv:f:0.10\n"
    "short\tline\tskip\n"
)


def _write(contents: str, suffix: str = ".paf") -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    tmp.write(contents)
    tmp.flush()
    tmp.close()
    return tmp.name


def test_columns_and_target_anchor() -> None:
    gr = pr.read_paf(_write(PAF))
    assert list(gr.columns) == [
        "Chromosome",
        "Start",
        "End",
        "Strand",
        "OtherChromosome",
        "OtherStart",
        "OtherEnd",
        "QueryLength",
        "TargetLength",
        "Matches",
        "BlockLength",
        "MapQ",
        "NM",
        "dv",
        "tp",
    ]
    assert len(gr) == 2  # the <12-field line is skipped
    # default anchor=target -> reference side is the genomic column; PAF coords kept verbatim (0-based)
    assert [str(c) for c in gr["Chromosome"]] == ["chr1", "chr2"]
    assert gr["Start"].tolist() == [200, 500]
    assert gr["End"].tolist() == [280, 540]
    assert [str(c) for c in gr["OtherChromosome"]] == ["q1", "q2"]
    assert gr["OtherStart"].tolist() == [10, 0]
    assert gr["OtherEnd"].tolist() == [90, 40]
    assert [str(s) for s in gr["Strand"]] == ["+", "-"]
    assert str(gr["Chromosome"].dtype) == "category"
    assert str(gr["Strand"].dtype) == "category"
    # mandatory extras
    assert gr["QueryLength"].tolist() == [100, 50]
    assert gr["TargetLength"].tolist() == [1000, 2000]
    assert gr["Matches"].tolist() == [75, 38]
    assert gr["BlockLength"].tolist() == [80, 40]
    assert gr["MapQ"].tolist() == [60, 30]


def test_query_anchor_swaps_sides() -> None:
    gr = pr.read_paf(_write(PAF), anchor="query")
    assert [str(c) for c in gr["Chromosome"]] == ["q1", "q2"]
    assert gr["Start"].tolist() == [10, 0]
    assert gr["End"].tolist() == [90, 40]
    assert [str(c) for c in gr["OtherChromosome"]] == ["chr1", "chr2"]
    assert gr["OtherStart"].tolist() == [200, 500]


def test_tag_dtypes_and_ragged_fill() -> None:
    gr = pr.read_paf(_write(PAF))
    # i -> nullable Int64, with <NA> for the record lacking NM
    assert str(gr["NM"].dtype) == "Int64"
    assert gr["NM"][0] == 5
    assert pd.isna(gr["NM"][1])
    # f -> float
    assert str(gr["dv"].dtype) == "float64"
    assert gr["dv"].tolist() == [0.02, 0.10]
    # A (char) -> object/str, missing where absent. The missing scalar is version-dependent
    # (None on pandas 2.x, NaN under pandas 3.0's string dtype), so assert via isna.
    assert gr["tp"][0] == "P"
    assert pd.isna(gr["tp"][1])


def test_nrows() -> None:
    gr = pr.read_paf(_write(PAF), nrows=1)
    assert len(gr) == 1
    assert [str(c) for c in gr["Chromosome"]] == ["chr1"]


def test_gzip_roundtrip() -> None:
    gz = tempfile.mktemp(suffix=".paf.gz")
    with gzip.open(gz, "wt") as fh:
        fh.write(PAF)
    gr = pr.read_paf(gz)
    assert len(gr) == 2
    assert [str(c) for c in gr["Chromosome"]] == ["chr1", "chr2"]
    Path(gz).unlink()


def test_empty_file() -> None:
    gr = pr.read_paf(_write(""))
    assert len(gr) == 0
    assert list(gr.columns) == ["Chromosome", "Start", "End"]


def test_bad_anchor() -> None:
    with pytest.raises(ValueError, match="anchor must be"):
        pr.read_paf(_write(PAF), anchor="reference")
