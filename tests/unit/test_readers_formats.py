"""Tests for read_narrowPeak, read_parquet, read_vcf.

read_narrowPeak is pure pandas and always runs. read_parquet needs a Parquet
engine (pyarrow) and read_vcf needs pysam; those tests skip when the optional
dependency is absent, so the suite stays green without them.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

import pyranges1 as pr

NARROWPEAK = "#track\nchr1\t100\t200\tpeak1\t500\t+\t5.5\t3.2\t2.1\t50\nchr2\t300\t400\tpeak2\t800\t-\t9.1\t4.0\t3.0\t60\n"


def test_read_narrowpeak() -> None:
    f = tempfile.mktemp(suffix=".narrowPeak")
    Path(f).write_text(NARROWPEAK)
    gr = pr.read_narrowPeak(f)
    assert list(gr.columns) == [
        "Chromosome",
        "Start",
        "End",
        "Name",
        "Score",
        "Strand",
        "SignalValue",
        "PValue",
        "QValue",
        "Peak",
    ]
    assert gr["Start"].tolist() == [100, 300]
    assert gr["End"].tolist() == [200, 400]
    assert gr["SignalValue"].tolist() == [5.5, 9.1]
    assert gr["Peak"].tolist() == [50, 60]
    assert str(gr["Chromosome"].dtype) == "category"
    assert str(gr["Strand"].dtype) == "category"
    Path(f).unlink()


def test_read_narrowpeak_nrows() -> None:
    f = tempfile.mktemp(suffix=".narrowPeak")
    Path(f).write_text(NARROWPEAK)
    assert len(pr.read_narrowPeak(f, nrows=1)) == 1
    Path(f).unlink()


def test_read_parquet_roundtrip() -> None:
    pytest.importorskip("pyarrow")
    pq = tempfile.mktemp(suffix=".parquet")
    pd.DataFrame(
        {"Chromosome": ["chr1", "chr2"], "Start": [10, 50], "End": [20, 80], "Strand": ["+", "-"], "Name": ["a", "b"]},
    ).to_parquet(pq)

    gr = pr.read_parquet(pq)
    assert list(gr.columns) == ["Chromosome", "Start", "End", "Strand", "Name"]
    assert str(gr["Chromosome"].dtype) == "category"
    assert str(gr["Strand"].dtype) == "category"
    assert gr["Start"].tolist() == [10, 50]

    # column projection
    sub = pr.read_parquet(pq, columns=["Chromosome", "Start", "End"])
    assert list(sub.columns) == ["Chromosome", "Start", "End"]
    Path(pq).unlink()


def test_read_vcf() -> None:
    pytest.importorskip("pysam")
    vcf = tempfile.mktemp(suffix=".vcf")
    Path(vcf).write_text(
        "##fileformat=VCFv4.2\n"
        '##FILTER=<ID=q10,Description="q">\n'
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="depth">\n'
        "##contig=<ID=chr1>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t100\trs1\tA\tG\t50\tPASS\tDP=30\n"
        "chr1\t200\t.\tAC\tA\t.\tq10\tDP=12\n"
        "chr1\t300\trs3\tT\tC,G\t99\tPASS\tDP=40\n",
    )

    gr = pr.read_vcf(vcf)
    assert list(gr.columns)[:8] == ["Chromosome", "Start", "End", "ID", "REF", "ALT", "QUAL", "FILTER"]
    # 1-based POS -> 0-based Start; End spans the REF allele.
    assert gr["Start"].tolist() == [99, 199, 299]
    assert gr["End"].tolist() == [100, 201, 300]  # SNV len1, AC len2, SNV len1
    # missing ID (".") -> NA; its scalar repr is version-dependent (None on pandas 2.x,
    # NaN under pandas 3.0's string dtype), so assert missingness rather than identity.
    assert gr["ID"].isna().tolist() == [False, True, False]
    assert gr["ID"].dropna().tolist() == ["rs1", "rs3"]
    assert list(gr["REF"]) == ["A", "AC", "T"]
    assert list(gr["ALT"]) == ["G", "A", "C,G"]  # multi-allelic comma-joined
    assert [str(x) for x in gr["FILTER"]] == ["PASS", "q10", "PASS"]
    assert gr["DP"].tolist() == [30, 12, 40]

    # nrows + INFO selection
    assert len(pr.read_vcf(vcf, nrows=2)) == 2
    assert "DP" not in pr.read_vcf(vcf, info_fields=[]).columns
    Path(vcf).unlink()
