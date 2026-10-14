"""Tests for the factorization helpers (factorize / factorize_binary).

These back the per-chromosome dispatch into ruranges. The fast path exploits the
categorical dtype the readers return, but must keep working for object/int dtypes
and must produce the same grouping (and group ordering) as the previous
``groupby(sort=True).ngroup()`` implementation.
"""

import numpy as np
import pandas as pd

from pyranges1.core.pyranges_helpers import factorize, factorize_binary


def _ngroup(df: pd.DataFrame, by: list[str]) -> np.ndarray:
    """The previous implementation, kept here as the reference oracle."""
    return df.groupby(by, observed=False).ngroup().to_numpy().astype(np.uint32)


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a and b induce the same grouping of rows (ignoring the id values).

    Canonicalised by first appearance, so two labelings that group the same rows
    together match even if the id *values* differ (e.g. int columns sort
    numerically while string columns sort lexicographically).
    """
    return bool((pd.factorize(a)[0] == pd.factorize(b)[0]).all())


CHROM = ["chr2", "chr1", "chr10", "chr1", "chr2", "chrX"]
STRAND = ["+", "-", "+", "+", "-", "-"]


def test_factorize_matches_ngroup_single_and_multi() -> None:
    df = pd.DataFrame({"Chromosome": pd.Categorical(CHROM), "Strand": pd.Categorical(STRAND)})
    for by in (["Chromosome"], ["Chromosome", "Strand"]):
        out = factorize(df, by)
        assert out.dtype == np.uint32
        # The genomic case (small cardinality) reproduces ngroup exactly.
        assert (out == _ngroup(df, by)).all()


def test_factorize_dtype_independent() -> None:
    """Same logical values group identically whether categorical, object, or int."""
    cat = pd.DataFrame({"Chromosome": pd.Categorical(CHROM)})
    obj = pd.DataFrame({"Chromosome": pd.Series(CHROM, dtype="object")})
    codes = {"chr1": 1, "chr2": 2, "chr10": 10, "chrX": 23}
    ints = pd.DataFrame({"Chromosome": [codes[c] for c in CHROM]})

    fc = factorize(cat, ["Chromosome"])
    fo = factorize(obj, ["Chromosome"])
    fi = factorize(ints, ["Chromosome"])
    assert _same_partition(fc, fo)
    assert _same_partition(fc, fi)
    # object and categorical (same sort order) give identical ids
    assert (fc == fo).all()


def test_factorize_empty_by() -> None:
    df = pd.DataFrame({"Chromosome": CHROM})
    out = factorize(df, [])
    assert out.tolist() == [0] * len(df)


def test_factorize_binary_cross_frame_consistent() -> None:
    df1 = pd.DataFrame({"Chromosome": pd.Categorical(["chr2", "chr1", "chrX"])})
    df2 = pd.DataFrame({"Chromosome": pd.Categorical(["chr1", "chrX", "chr3"])})
    f1, f2 = factorize_binary(df1, df2, ["Chromosome"])
    assert f1.dtype == np.uint32

    # The same Chromosome must get the same id in both frames.
    id_of = {}
    for chrom, code in zip(list(df1["Chromosome"]) + list(df2["Chromosome"]), list(f1) + list(f2)):
        id_of.setdefault(chrom, code)
        assert id_of[chrom] == code

    # Same grouping as the previous concat+ngroup implementation.
    joined = pd.concat([df1[["Chromosome"]], df2[["Chromosome"]]], ignore_index=True)
    ref = joined.groupby(["Chromosome"], observed=False).ngroup().to_numpy().astype(np.uint32)
    assert _same_partition(np.concatenate([f1, f2]), ref)


def test_factorize_binary_multikey() -> None:
    df1 = pd.DataFrame({"Chromosome": pd.Categorical(["chr1", "chr1"]), "Strand": pd.Categorical(["+", "-"])})
    df2 = pd.DataFrame({"Chromosome": pd.Categorical(["chr1", "chr2"]), "Strand": pd.Categorical(["+", "+"])})
    f1, f2 = factorize_binary(df1, df2, ["Chromosome", "Strand"])
    # (chr1,+) appears in both frames -> same id
    assert f1[0] == f2[0]
    # (chr1,+) and (chr1,-) differ
    assert f1[0] != f1[1]
