import pandas as pd
import pytest

import pyranges1 as pr
from pyranges1.core.pyranges_helpers import ensure_pyranges

LOCI = {"Chromosome": ["chr1"], "Start": [1], "End": [2]}


def test_constructor_takes_data_as_a_keyword() -> None:
    assert isinstance(pr.PyRanges(data=LOCI), pr.PyRanges)
    assert isinstance(pr.PyRanges(data=pd.DataFrame(LOCI)), pr.PyRanges)
    assert pr.PyRanges(data=LOCI).equals(pr.PyRanges(LOCI))
    assert type(pr.PyRanges(data={"a": [1]})) is pd.DataFrame  # not genomic: a DataFrame, as positionally


def test_not_a_pyranges_names_the_missing_columns() -> None:
    with pytest.raises(TypeError, match=r"missing column\(s\) \['Chromosome', 'End'\]"):
        ensure_pyranges(pd.DataFrame({"chrom": ["chr1"], "Start": [1]}))


def test_from_string_reads_browser_positions_and_headerless_bed() -> None:
    gr = pr.from_string("chr1:1,000-2,000\nchrX:5-5")
    assert list(zip(gr["Chromosome"], gr["Start"], gr["End"], strict=True)) == [("chr1", 999, 2000), ("chrX", 4, 5)]
    assert "Strand" not in gr.columns
    bed = pr.from_string("chr1 10 20 a 0 +\nchr1 30 40 b 0 -")
    assert list(bed.columns) == ["Chromosome", "Start", "End", "Name", "Score", "Strand"]
    assert bed["Start"].tolist() == [10, 30]  # BED is already 0-based
    with_header = pr.from_string("Chromosome Start End\nchr1 10 20")
    assert with_header["Start"].tolist() == [10]
    with pytest.raises(ValueError, match="1-based"):
        pr.from_string("chr1:0-10")
