import gzip
from pathlib import Path

import pyranges1 as pr

BED6 = "chr1\t10\t20\tp1\t5\t+\nchr1\t30\t40\tp2\t6\t-\n"


def _rows(gr: pr.PyRanges) -> list[tuple]:
    return list(zip(gr["Chromosome"].astype(str), gr["Start"], gr["End"], strict=True))


def test_read_bed_skips_track_and_browser_lines(tmp_path: Path) -> None:
    path = tmp_path / "peaks.bed"
    path.write_text('track name="peaks" description="x"\nbrowser position chr1:1-100\n# a comment\n' + BED6)
    gr = pr.read_bed(path)
    assert _rows(gr) == [("chr1", 10, 20), ("chr1", 30, 40)]
    assert gr["Strand"].tolist() == ["+", "-"]


def test_read_bed_reads_bgzip_by_content(tmp_path: Path) -> None:
    path = tmp_path / "peaks.bed.bgz"
    path.write_bytes(gzip.compress(BED6.encode()))
    assert _rows(pr.read_bed(path)) == [("chr1", 10, 20), ("chr1", 30, 40)]


def test_read_gff_reads_gff3_attributes(tmp_path: Path) -> None:
    gff3 = tmp_path / "a.gff"
    gff3.write_text("##gff-version 3\nchr1\tsrc\tgene\t11\t20\t.\t+\t.\tID=g1;Name=A\n")
    gr = pr.read_gff(gff3)
    assert gr["ID"].tolist() == ["g1"]
    assert gr["Name"].tolist() == ["A"]

    gtf = tmp_path / "a.gff2"
    gtf.write_text('chr1\tsrc\tgene\t11\t20\t.\t+\t.\tgene_id "g1"; gene_name "A";\n')
    assert pr.read_gff(gtf)["gene_id"].tolist() == ["g1"]
