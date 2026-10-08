import gzip

import pytest

import pyranges1 as pr


def test_bedgraph_round_trip(tmp_path) -> None:
    gr = pr.PyRanges({"Chromosome": ["chr1", "chr2"], "Start": [0, 5], "End": [10, 9], "Value": [1.5, -2.0]})
    path = tmp_path / "s.bedGraph.gz"
    gr.to_bedgraph(str(path))
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    back = pr.read_bedgraph(path)
    assert back["Value"].tolist() == [1.5, -2.0]
    assert back["Start"].tolist() == [0, 5]


def test_read_bedgraph_skips_track_lines_and_refuses_other_files(tmp_path) -> None:
    path = tmp_path / "s.bedGraph"
    path.write_text("track type=bedGraph\n# x\nchr1\t0\t10\t1\n")
    assert len(pr.read_bedgraph(path)) == 1
    path.write_text("chr1\t0\t10\tname\t1\n")
    with pytest.raises(ValueError, match="4 tab-separated"):
        pr.read_bedgraph(path)


def test_read_wig_sections(tmp_path) -> None:
    path = tmp_path / "s.wig.gz"
    path.write_bytes(
        gzip.compress(
            b"track type=wiggle_0\n"
            b"variableStep chrom=chr1\n101\t2\n111\t3\n"  # span 1
            b"fixedStep chrom=chr1 start=201 step=10 span=4\n1\n2\n"
            b"fixedStep chrom=chr2 start=1 step=5\n7\n"  # no span: the whole step
        )
    )
    gr = pr.read_wig(path)
    assert list(zip(gr["Chromosome"].astype(str), gr["Start"], gr["End"], gr["Value"], strict=True)) == [
        ("chr1", 100, 101, 2.0),
        ("chr1", 110, 111, 3.0),
        ("chr1", 200, 204, 1.0),
        ("chr1", 210, 214, 2.0),
        ("chr2", 0, 5, 7.0),
    ]


def test_read_wig_refuses_data_before_a_section(tmp_path) -> None:
    path = tmp_path / "bad.wig"
    path.write_text("1\t2\n")
    with pytest.raises(ValueError, match="before any fixedStep"):
        pr.read_wig(path)


TX = pr.PyRanges(
    {
        "Chromosome": ["chr1", "chr1"],
        "Start": [100, 500],
        "End": [200, 650],
        "Name": ["t1", "t2"],
        "Strand": ["+", "-"],
        "BlockCount": [2, 2],
        "BlockSizes": ["20,30,", "50,40,"],
        "BlockStarts": ["0,70,", "0,110,"],
    }
)


def test_explode_blocks_options() -> None:
    plain = TX.explode_blocks(use_strand=False, drop_block_columns=False)
    assert plain["BlockIndex"].tolist() == [0, 1, 0, 1]
    assert "BlockSizes" in plain.columns
    assert "BlockIndex" not in TX.explode_blocks(block_index_col=None).columns


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [("BlockCount", [3, 2], "BlockCount"), ("BlockStarts", ["0,", "0,110,"], "different numbers"), ("End", [210, 650], "last block")],
)
def test_explode_blocks_refuses_inconsistent_bed12(column, value, message) -> None:
    bad = TX.copy()
    bad[column] = value
    with pytest.raises(ValueError, match=message):
        bad.explode_blocks()
