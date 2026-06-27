"""Tests for the writers: to_parquet, to_narrowPeak, to_pairs, to_bigbed.

to_narrowPeak and to_pairs are pure pandas/text and always run. to_parquet needs
a Parquet engine (pyarrow) and to_bigbed needs pybigtools (to write) + pyBigWig
(to read back); those tests skip when the optional dependency is absent.

The matching readers live in separate PRs, so these tests verify the written
output directly rather than via a round-trip through the readers.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

import pyranges1 as pr


def _gr_narrowpeak() -> "pr.PyRanges":
    return pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr2"],
            "Start": [100, 300],
            "End": [200, 400],
            "Name": ["peak1", "peak2"],
            "Score": [500, 800],
            "Strand": ["+", "-"],
            "SignalValue": [5.5, 9.1],
            "PValue": [3.2, 4.0],
            "QValue": [2.1, 3.0],
            "Peak": [50, 60],
        },
    )


def _gr_pairs() -> "pr.PyRanges":
    return pr.PyRanges(
        {
            "Chromosome": ["chr1"],
            "Start": [99],
            "End": [100],
            "Strand": ["+"],
            "OtherChromosome": ["chr2"],
            "OtherStart": [199],
            "OtherEnd": [200],
            "OtherStrand": ["-"],
            "ReadID": ["r1"],
        },
    )


def test_to_narrowpeak_string() -> None:
    assert _gr_narrowpeak().to_narrowPeak().splitlines()[0] == "chr1\t100\t200\tpeak1\t500\t+\t5.5\t3.2\t2.1\t50"


def test_to_narrowpeak_file() -> None:
    f = tempfile.mktemp(suffix=".narrowPeak")
    assert _gr_narrowpeak().to_narrowPeak(f) is None
    back = pd.read_csv(f, sep="\t", header=None)
    assert back.shape == (2, 10)
    assert back.iloc[0, 1] == 100
    assert back.iloc[1, 3] == "peak2"
    Path(f).unlink()


def test_to_pairs_string() -> None:
    txt = _gr_pairs().to_pairs()
    lines = txt.splitlines()
    assert lines[0] == "## pairs format v1.0"
    assert lines[1].startswith("#columns:")
    # 0-based Start 99/199 written back as 1-based pos 100/200.
    assert lines[2] == "r1\tchr1\t100\tchr2\t200\t+\t-"


def test_to_pairs_requires_paired_columns() -> None:
    gr = pr.PyRanges({"Chromosome": ["chr1"], "Start": [1], "End": [2], "Strand": ["+"]})
    with pytest.raises(ValueError, match="paired columns"):
        gr.to_pairs()


def test_to_parquet_roundtrip() -> None:
    pytest.importorskip("pyarrow")
    gr = pr.PyRanges({"Chromosome": ["chr1", "chr2"], "Start": [10, 50], "End": [20, 80], "Strand": ["+", "-"]})
    pq = tempfile.mktemp(suffix=".parquet")
    assert gr.to_parquet(pq) is None
    back = pd.read_parquet(pq)
    assert list(back.columns) == ["Chromosome", "Start", "End", "Strand"]
    assert "__index_level_0__" not in back.columns  # index=False default
    assert back["Start"].tolist() == [10, 50]
    # path=None returns bytes
    assert isinstance(gr.to_parquet(), bytes)
    Path(pq).unlink()


def test_to_bigbed_roundtrip() -> None:
    pytest.importorskip("pybigtools")
    pyBigWig = pytest.importorskip("pyBigWig")
    gr = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [10, 30],
            "End": [20, 45],
            "Name": ["g1", "g2"],
            "Score": [100, 200],
            "Strand": ["+", "-"],
        },
    )
    bb = tempfile.mktemp(suffix=".bb")
    gr.to_bigbed(bb)

    f = pyBigWig.open(bb)
    assert f.chroms() == {"chr1": 45}
    entries = f.entries("chr1", 0, 45)
    assert entries == [(10, 20, "g1\t100\t+"), (30, 45, "g2\t200\t-")]
    sql = f.SQL() or b""
    if isinstance(sql, str):
        sql = sql.encode()
    assert b"name" in sql
    assert b"strand" in sql
    Path(bb).unlink()
