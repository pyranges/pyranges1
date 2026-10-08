import shutil

import pytest

import pyranges1 as pr


def _span(gr: pr.PyRanges) -> list[tuple[int, int]]:
    return sorted(zip(gr["Start"].tolist(), gr["End"].tolist(), strict=True))


def test_read_bigwig_region() -> None:
    pytest.importorskip("pyBigWig")
    path = pr.example_data.files["bigwig.bw"]
    full = pr.read_bigwig(path)
    assert _span(pr.read_bigwig(path, region="1:121-130")) == [(100, 150)]  # overlapping, not clipped
    assert _span(pr.read_bigwig(path, region="10:1-200")) == []  # 1-based inclusive: base 200 is [199, 200)
    assert _span(pr.read_bigwig(path, region="10:1-201")) == [(200, 300)]
    assert _span(pr.read_bigwig(path, region="1")) == _span(full[full["Chromosome"] == "1"])
    empty = pr.read_bigwig(path, region="chrX")
    assert len(empty) == 0
    assert list(empty.dtypes.astype(str)) == list(full.dtypes.astype(str))


def test_read_bigbed_region(tmp_path) -> None:
    pytest.importorskip("pybigtools")
    pytest.importorskip("pyBigWig")
    gr = pr.PyRanges({"Chromosome": ["chr1", "chr1", "chr2"], "Start": [10, 50, 5], "End": [20, 60, 9], "Name": list("abc")})
    path = str(tmp_path / "a.bb")
    gr.to_bigbed(path)
    assert pr.read_bigbed(path, region="chr1:15-55")["Name"].tolist() == ["a", "b"]
    assert pr.read_bigbed(path, region="chr2")["Name"].tolist() == ["c"]
    assert len(pr.read_bigbed(path, region="chr3")) == 0


def test_read_bam_region_matches_the_full_read(tmp_path) -> None:
    pysam = pytest.importorskip("pysam")
    bam = tmp_path / "smaller.bam"
    shutil.copy(pr.example_data.files["smaller.bam"], bam)
    pysam.index(str(bam))
    full = pr.read_bam(bam)
    lo, hi = 4_871_145, 12_742_424
    within = full[(full["Chromosome"] == "chr1") & (full["Start"] < hi) & (full["End"] > lo)]
    region = pr.read_bam(bam, region=f"chr1:{lo + 1:,}-{hi:,}")
    assert len(region) == len(within) > 0
    assert sorted(zip(region["Start"], region["End"], region["Flag"], strict=True)) == sorted(
        zip(within["Start"], within["End"], within["Flag"], strict=True)
    )


def test_read_bed_region(tmp_path) -> None:
    pysam = pytest.importorskip("pysam")
    shutil.copy(pr.example_data.files["aorta.bed"], tmp_path / "a.bed")
    pysam.tabix_index(str(tmp_path / "a.bed"), preset="bed")
    path = tmp_path / "a.bed.gz"
    full = pr.read_bed(path)
    region = pr.read_bed(path, region="chr1:9930-9960")
    assert _span(region) == _span(full[(full["Start"] < 9960) & (full["End"] > 9929)])
    assert list(region.columns) == list(full.columns)
    assert len(pr.read_bed(path, region="chrZ:1-10")) == 0
    with pytest.raises(ValueError, match="tabix"):
        pr.read_bed(pr.example_data.files["aorta.bed"], region="chr1:1-100")
