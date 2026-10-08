import pandas as pd
import pytest

import pyranges1 as pr

BED12 = {
    "Chromosome": ["chr1"],
    "Start": [10],
    "End": [100],
    "Name": ["tx"],
    "Score": [0],
    "Strand": ["+"],
    "ThickStart": [20],
    "ThickEnd": [90],
    "ItemRGB": ["0"],
    "BlockCount": [2],
    "BlockSizes": ["10,20,"],
    "BlockStarts": ["0,70,"],
}
CANONICAL = "chr1\t10\t100\ttx\t0\t+\t20\t90\t0\t2\t10,20,\t0,70,\n"


def test_to_bed_writes_bed12_fields_in_their_positions() -> None:
    gr = pr.PyRanges(BED12)
    assert gr.to_bed() == CANONICAL
    reordered = gr[["Chromosome", "Start", "End", "Strand", "BlockCount", "Name", "Score", "BlockStarts", "ThickStart", "ThickEnd", "ItemRGB", "BlockSizes"]]
    assert reordered.to_bed() == CANONICAL


def test_to_bed_fills_defaultable_bed12_fields_and_keeps_extras_last() -> None:
    gr = pr.PyRanges({k: v for k, v in BED12.items() if k not in ("ThickStart", "ThickEnd", "ItemRGB")})
    gr["Extra"] = ["x"]
    assert gr.to_bed() == "chr1\t10\t100\ttx\t0\t+\t10\t100\t0\t2\t10,20,\t0,70,\tx\n"


def test_to_bed_refuses_a_block_field_it_cannot_default() -> None:
    gr = pr.PyRanges({k: v for k, v in BED12.items() if k != "BlockSizes"})
    with pytest.raises(ValueError, match="BlockSizes"):
        gr.to_bed()


def test_to_bed_keep_false_and_short_frames_are_unchanged() -> None:
    gr = pr.PyRanges(BED12)
    assert gr.to_bed(keep=False) == "chr1\t10\t100\ttx\t0\t+\n"
    assert pr.PyRanges({"Chromosome": ["chr1"], "Start": [1], "End": [2], "Other": [7]}).to_bed() == "chr1\t1\t2\t.\t.\t.\t7\n"


def test_gtf_attributes_print_nullable_integers_as_integers() -> None:
    gr = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [10, 20],
            "End": [15, 25],
            "Strand": ["+", "+"],
            "Feature": ["exon", "exon"],
            "gene_id": ["g", "g"],
            "exon_number": pd.array([1, None], dtype="Int64"),
        }
    )
    assert gr.to_gtf().splitlines() == [
        'chr1\t.\texon\t11\t15\t.\t+\t.\tgene_id "g"; exon_number "1";',
        'chr1\t.\texon\t21\t25\t.\t+\t.\tgene_id "g";',
    ]
    assert gr.to_gff3().splitlines()[0].endswith("gene_id=g;exon_number=1")
