import pandas as pd
import pytest

import pyranges1 as pr

pa = pytest.importorskip("pyarrow")


def test_from_arrow_gives_a_pyranges_for_genomic_columns() -> None:
    gr = pr.PyRanges.from_arrow(pa.table({"Chromosome": ["chr1", "chr2"], "Start": [10, 20], "End": [15, 30]}))
    assert isinstance(gr, pr.PyRanges)
    assert gr["Start"].tolist() == [10, 20]
    other = pr.PyRanges.from_arrow(pa.table({"a": [1]}))
    assert type(other) is pd.DataFrame
