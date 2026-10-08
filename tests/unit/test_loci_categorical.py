import numpy as np
import pandas as pd

import pyranges1 as pr


def test_loci_matches_categorical_chromosomes_as_the_rows_do() -> None:
    rng = np.random.default_rng(0)
    names = rng.choice(["chr1", "chr2", "chrX"], 500)
    plain = pr.PyRanges({"Chromosome": names, "Start": np.arange(500), "End": np.arange(500) + 3})
    categorical = plain.assign(Chromosome=pd.Categorical(names))
    for chrom in ("chr1", "chrX", "chrY"):
        assert categorical.loci[chrom].index.tolist() == plain.loci[chrom].index.tolist()
    numbered = pr.PyRanges({"Chromosome": pd.Categorical([1, 2, 1]), "Start": [0, 5, 9], "End": [3, 8, 12]})
    assert numbered.loci[1].index.tolist() == numbered.loci["1"].index.tolist() == [0, 2]
