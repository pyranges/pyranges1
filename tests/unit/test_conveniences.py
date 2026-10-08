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
