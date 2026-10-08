"""liftover_ranges against UCSC liftOver; the expected values were produced with liftOver itself."""

import pandas as pd
import pytest

import pyranges1 as pr

# chr1:100-300 -> chrA (+, a 30 bp gap); chr1:500-600 -> chrB (-); chr1:700-800 -> chrC and chrD:700-800 overlapping.
CHAIN = (
    "chain 100 chr1 1000 + 100 300 chrA 2000 + 500 710 1\n50 30 40\n120\n\n"
    "chain 50 chr1 1000 + 500 600 chrB 900 - 0 100 2\n100\n\n"
    "chain 40 chr1 1000 + 700 800 chrC 500 + 0 100 3\n100\n\n"
    "chain 30 chr1 1000 + 750 850 chrD 500 + 0 100 4\n100\n\n"
)


@pytest.fixture
def chain_path(tmp_path):
    path = tmp_path / "toy.chain"
    path.write_text(CHAIN)
    return path


def test_read_chain_blocks(chain_path) -> None:
    blocks = pr.read_chain(chain_path)
    rows = list(zip(blocks["Start"], blocks["End"], blocks["QueryChromosome"], blocks["QueryStart"], blocks["QueryEnd"], blocks["QueryStrand"], strict=True))
    assert rows[:3] == [(100, 150, "chrA", 500, 550, "+"), (180, 300, "chrA", 590, 710, "+"), (500, 600, "chrB", 800, 900, "-")]


def test_liftover_matches_liftover(chain_path) -> None:
    gr = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": "chr1",
                "Start": [110, 140, 145, 520, 140, 900, 710, 760],
                "End": [140, 200, 290, 530, 160, 950, 730, 790],
                "Strand": ["+", "-", "+", "+", "+", "+", "+", "+"],
                "Name": list("abcdefgh"),
            },
            index=[10, 11, 12, 13, 14, 15, 16, 17],
        )
    )
    lifted, unmapped = gr.liftover_ranges(chain_path, return_unmapped=True, min_match=0.5)
    assert list(zip(lifted["Name"], lifted["Chromosome"], lifted["Start"], lifted["End"], lifted["Strand"], strict=True)) == [
        ("a", "chrA", 510, 540, "+"),
        ("b", "chrA", 540, 610, "-"),  # spans the gap: first to last mapped base
        ("c", "chrA", 545, 700, "+"),
        ("d", "chrB", 870, 880, "-"),  # minus chain flips the strand
        ("e", "chrA", 540, 550, "+"),  # exactly half its bases: min_match is inclusive
        ("g", "chrC", 10, 30, "+"),
    ]
    assert lifted.index.tolist() == [10, 11, 12, 13, 14, 16]
    assert dict(zip(unmapped["Name"], unmapped["LiftReason"], strict=True)) == {
        "f": "Deleted in new",
        "h": "Duplicated in new",
    }


def test_liftover_split_and_chain_as_pyranges(chain_path) -> None:
    gr = pr.PyRanges({"Chromosome": ["chr1"], "Start": [280], "End": [520]})
    lifted, unmapped = gr.liftover_ranges(pr.read_chain(chain_path), return_unmapped=True)
    assert len(lifted) == 0
    assert unmapped["LiftReason"].tolist() == ["Split in new"]
