"""Statistics useful for genomics."""

import logging
from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations_with_replacement
from math import sqrt
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series

import pyranges1 as pr
from pyranges1.core.names import (
    CHROM_COL,
    END_COL,
    GENOME_LOC_COLS,
    STRAND_BEHAVIOR_IGNORE,
    STRAND_BEHAVIOR_SAME,
    VALID_STRAND_BEHAVIOR_TYPE,
)
from pyranges1.core.pyranges_helpers import (
    ensure_pyranges,
    strand_behavior_from_validated_use_strand,
    use_strand_from_validated_strand_behavior,
    validate_and_convert_strand_behavior,
)
from pyranges1.methods.statistics import _relative_distance

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pyranges1 import PyRanges

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


####################################################################################################
# methods and objects that are not exported
GenomeType = dict[str, int] | pd.DataFrame | None
LabelsType = list[str] | list[int]


def _chromsizes_as_int(chromsizes: "PyRanges | DataFrame | dict[Any, int]") -> int:
    if isinstance(chromsizes, dict):
        _chromsizes = sum(chromsizes.values())
    elif isinstance(chromsizes, pd.DataFrame | pr.PyRanges):
        _chromsizes = chromsizes.End.sum()
    else:
        msg = f"chromsizes must be dict, DataFrame or PyRanges, was {type(chromsizes)}"
        raise TypeError(msg)

    return int(_chromsizes)


def _mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    # https://stackoverflow.com/a/56875660/992687
    x = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return ((tp * tn) - (fp * fn)) / sqrt(x)


def _process_genome_data(grs: list[Any], genome: GenomeType, labels: LabelsType) -> tuple[pd.DataFrame, int, Iterable]:
    _genome = _find_chromosome_max_end_positions(grs) if genome is None else _ensure_genome_dataframe(genome)

    genome_length = _compute_genome_length(_genome)

    _labels = (
        _generate_labels(labels, grs) if labels is not None else combinations_with_replacement(np.arange(len(grs)), r=2)
    )

    return _genome, genome_length, _labels


def _ensure_genome_dataframe(genome: GenomeType) -> pd.DataFrame:
    if isinstance(genome, dict):
        return _create_genome_dataframe(genome)

    if isinstance(genome, pd.DataFrame):
        return genome

    msg = f"genome must be dict or DataFrame, was {type(genome)}"
    raise TypeError(msg)


def _find_chromosome_max_end_positions(grs: list["PyRanges"]) -> pd.DataFrame:
    """Find the largest end position in each chromosome.

    Examples
    --------
    >>> f1, f2 = pr.example_data.f1, pr.example_data.f2  # both only have chr1
    >>> int(f1["End"].max())
    9
    >>> int(f2["End"].max())
    7

    """
    genome: dict[str, int] = defaultdict(int)
    for gr in grs:
        for chrom, chrom_df in gr.groupby(CHROM_COL):
            genome[chrom] = max(chrom_df[END_COL].max(), genome[chrom])
    return _create_genome_dataframe(dict(genome))


def _create_genome_dataframe(genome: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame({"Chromosome": list(genome.keys()), "Start": 0, "End": list(genome.values())})


def _compute_genome_length(genome: pd.DataFrame) -> int:
    return int(genome[END_COL].sum())


def _generate_labels(labels: LabelsType, grs: list[Any]) -> Iterable:
    if len(labels) != len(grs):
        msg = "Labels length must match the length of grs"
        raise ValueError(msg)
    return combinations_with_replacement(labels, r=2)


def fdr(p_vals: Series) -> Series:
    """Adjust p-values with Benjamini-Hochberg.

    Parameters
    ----------
    p_vals : array-like
           P-values to adjust.

    Returns
    -------
    Pandas.DataFrame

        DataFrame where values are order of data.

    Examples
    --------
    >>> import pyranges1 as pr
    >>> d = {'Chromosome': ['chr3', 'chr6', 'chr13'], 'Start': [146419383, 39800100, 24537618], 'End': [146419483, 39800200, 24537718], 'Strand': ['-', '+', '-'], 'PValue': [0.0039591368855297175, 0.0037600512992788937, 0.0075061166500909205]}
    >>> gr = pr.PyRanges(d)
    >>> gr
      index  |    Chromosome        Start        End  Strand        PValue
      int64  |    str               int64      int64  str          float64
    -------  ---  ------------  ---------  ---------  --------  ----------
          0  |    chr3          146419383  146419483  -         0.00395914
          1  |    chr6           39800100   39800200  +         0.00376005
          2  |    chr13          24537618   24537718  -         0.00750612
    PyRanges with 3 rows, 5 columns, and 1 index columns.
    Contains 3 chromosomes and 2 strands.

    >>> gr["FDR"] = pr.stats.fdr(gr.PValue)
    >>> gr
      index  |    Chromosome        Start        End  Strand        PValue         FDR
      int64  |    str               int64      int64  str          float64     float64
    -------  ---  ------------  ---------  ---------  --------  ----------  ----------
          0  |    chr3          146419383  146419483  -         0.00395914  0.00593871
          1  |    chr6           39800100   39800200  +         0.00376005  0.0112802
          2  |    chr13          24537618   24537718  -         0.00750612  0.00750612
    PyRanges with 3 rows, 6 columns, and 1 index columns.
    Contains 3 chromosomes and 2 strands.

    """
    from scipy.stats import rankdata  # type: ignore[import]

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


def fisher_exact(tp: Series, fp: Series, fn: Series, tn: Series, pseudocount: int = 0) -> DataFrame:
    """Fisher's exact for contingency tables.

    Computes the hypotheses two-sided, less and greater at the same time.

    The odds-ratio is

    Parameters
    ----------
    tp : array-like of int
        Top left square of contingency table (true positives).

    fp : array-like of int
        Top right square of contingency table (false positives).

    fn : array-like of int
        Bottom left square of contingency table (false negatives).

    tn : array-like of int
        Bottom right square of contingency table (true negatives).

    pseudocount : float, default 0
        Values > 0 allow Odds Ratio to always be a finite number.

    Notes
    -----
    The odds-ratio is computed thusly:

    ``((tp + pseudocount) / (fp + pseudocount)) / ((fn + pseudocount) / (tn + pseudocount))``

    Returns
    -------
    pandas.DataFrame

        DataFrame with columns odds_ratio and P, PLeft and PRight.

    See Also
    --------
    pr.stats.fdr : correct for multiple testing

    Examples
    --------
    >>> d = {"TP": [12, 0], "FP": [5, 12], "TN": [29, 10], "FN": [2, 2]}
    >>> df = pd.DataFrame(d)
    >>> df
       TP  FP  TN  FN
    0  12   5  29   2
    1   0  12  10   2

    >>> pr.stats.fisher_exact(df.TP, df.FP, df.TN, df.FN) # doctest: +SKIP
       odds_ratio         P     PLeft    PRight
    0    0.165517  0.080269  0.044555  0.994525
    1    0.000000  0.000067  0.000034  1.000000

    """
    try:
        from fisher import pvalue_npy  # type: ignore[import]
    except ImportError:
        msg = (
            "fisher needs to be installed to use fisher exact. pip install fisher or conda install -c bioconda fisher."
        )
        raise ImportError(msg) from None

    _tp = np.array(tp, dtype=np.uint)
    _fp = np.array(fp, dtype=np.uint)
    _fn = np.array(fn, dtype=np.uint)
    _tn = np.array(tn, dtype=np.uint)

    left, right, twosided = pvalue_npy(_tp, _fp, _fn, _tn)

    odds_ratio = ((_tp + pseudocount) / (_fp + pseudocount)) / ((_fn + pseudocount) / (_tn + pseudocount))

    return pd.DataFrame({"odds_ratio": odds_ratio, "P": twosided, "PLeft": left, "PRight": right})


def mcc(
    grs: list["PyRanges"],
    *,
    labels: list[str],
    genome: "PyRanges | pd.DataFrame | dict[str, int] | None" = None,
    use_strand: bool = False,
) -> DataFrame:
    """Compute Matthew's correlation coefficient for PyRanges overlaps.

    Parameters
    ----------
    grs : list of PyRanges
        PyRanges to compare.

    genome : DataFrame or dict, default None
        Should contain chromosome sizes. By default, end position of the
        rightmost intervals are used as proxies for the chromosome size, but
        it is recommended to use a genome.

    labels : list of str, default None
        Names to give the PyRanges in the output.

    use_strand : bool, default False
        Whether to compute correlations per strand.

    Examples
    --------
    >>> grs = [pr.example_data.aorta, pr.example_data.aorta, pr.example_data.aorta2]
    >>> mcc = pr.stats.mcc(grs, labels="abc", genome={"chr1": 2100000})
    >>> mcc
       T  F   TP   FP       TN   FN      MCC
    0  a  a  728    0  2099272    0  1.00000
    1  a  b  728    0  2099272    0  1.00000
    3  a  c  457  485  2098787  271  0.55168
    2  b  a  728    0  2099272    0  1.00000
    5  b  b  728    0  2099272    0  1.00000
    6  b  c  457  485  2098787  271  0.55168
    4  c  a  457  271  2098787  485  0.55168
    7  c  b  457  271  2098787  485  0.55168
    8  c  c  942    0  2099058    0  1.00000

    To create a symmetric matrix (useful for heatmaps of correlations):

    >>> mcc.set_index(["T", "F"]).MCC.unstack().rename_axis(None, axis=0)
    F        a        b        c
    a  1.00000  1.00000  0.55168
    b  1.00000  1.00000  0.55168
    c  0.55168  0.55168  1.00000

    """
    _genome, genome_length, _labels = _process_genome_data(grs, labels=labels, genome=genome)

    if use_strand and not all(gr.strand_valid for gr in grs):
        msg = "use_strand=True but one or more PyRanges have invalid strand information."
        raise AssertionError(msg)

    # remove all non-loc columns before computation
    grs = [gr.merge_overlaps(use_strand=use_strand) for gr in grs]

    strand_behavior_same = all(
        strand_behavior_from_validated_use_strand(gr, use_strand) == STRAND_BEHAVIOR_SAME for gr in grs
    )
    strand_behavior: VALID_STRAND_BEHAVIOR_TYPE = (
        STRAND_BEHAVIOR_SAME if strand_behavior_same else STRAND_BEHAVIOR_IGNORE
    )

    rowdicts = []
    for (lt, lf), (t, f) in zip(_labels, combinations_with_replacement(grs, r=2), strict=True):
        if lt == lf:
            if not use_strand:
                tp = t.length
                fn = 0
                tn = genome_length - tp
                fp = 0
                rowdicts.append({"T": lt, "F": lf, "TP": tp, "FP": fp, "TN": tn, "FN": fn, "MCC": 1})
            else:
                for _strand in ["+", "-"]:
                    tp = t[use_strand].length
                    fn = 0
                    tn = genome_length - tp
                    fp = 0
                    rowdicts.append(
                        {
                            "T": lt,
                            "F": lf,
                            "Strand": _strand,
                            "TP": tp,
                            "FP": fp,
                            "TN": tn,
                            "FN": fn,
                            "MCC": 1,
                        },
                    )
            continue

        else:  # noqa: RET507
            j = t.join_overlaps(f, strand_behavior=strand_behavior)
            tp_gr = j.combine_interval_columns().merge_overlaps(use_strand=use_strand)
            if use_strand:
                for _strand in ["+", "-"]:
                    tp = tp_gr[_strand].length
                    fp = f[_strand].length - tp
                    fn = t[_strand].length - tp
                    tn = genome_length - (tp + fp + fn)
                    mcc = _mcc(tp, fp, tn, fn)
                    rowdicts.extend(
                        [
                            {
                                "T": lt,
                                "F": lf,
                                "Strand": _strand,
                                "TP": tp,
                                "FP": fp,
                                "TN": tn,
                                "FN": fn,
                                "MCC": mcc,
                            },
                            {
                                "T": lf,
                                "F": lt,
                                "Strand": _strand,
                                "TP": tp,
                                "FP": fn,
                                "TN": tn,
                                "FN": fp,
                                "MCC": mcc,
                            },
                        ],
                    )
            else:
                tp = tp_gr.length
                fp = f.length - tp
                fn = t.length - tp
                tn = genome_length - (tp + fp + fn)
                mcc = _mcc(tp, fp, tn, fn)

                rowdicts.extend(
                    [
                        {
                            "T": lt,
                            "F": lf,
                            "TP": tp,
                            "FP": fp,
                            "TN": tn,
                            "FN": fn,
                            "MCC": mcc,
                        },
                        {
                            "T": lf,
                            "F": lt,
                            "TP": tp,
                            "FP": fn,
                            "TN": tn,
                            "FN": fp,
                            "MCC": mcc,
                        },
                    ],
                )

    return pd.DataFrame.from_records(rowdicts).sort_values(["T", "F"])


def rowbased_spearman(x: ndarray, y: ndarray) -> ndarray:
    """Fast row-based Spearman's correlation.

    Parameters
    ----------
    x : matrix-like
        2D numerical matrix. Same size as y.

    y : matrix-like
        2D numerical matrix. Same size as x.

    Returns
    -------
    numpy.array

        Array with same length as input, where values are P-values.

    See Also
    --------
    pyranges.stats.rowbased_pearson : fast row-based Pearson's correlation.
    pyranges.stats.fdr : correct for multiple testing

    Examples
    --------
    >>> x = np.array([[7, 2, 9], [3, 6, 0], [0, 6, 3]])
    >>> y = np.array([[5, 3, 2], [9, 6, 0], [7, 3, 5]])

    Perform Spearman's correlation pairwise on each row in 10x10 matrixes:

    >>> pr.stats.rowbased_spearman(x, y)
    array([-0.5,  0.5, -1. ])

    """
    x = np.asarray(x)
    y = np.asarray(y)

    rx = rowbased_rankdata(x)
    ry = rowbased_rankdata(y)

    return rowbased_pearson(rx, ry)


def rowbased_pearson(x: ndarray | DataFrame, y: ndarray | DataFrame) -> ndarray:
    """Fast row-based Pearson's correlation.

    Parameters
    ----------
    x : matrix-like
        2D numerical matrix. Same size as y.

    y : matrix-like
        2D numerical matrix. Same size as x.

    Returns
    -------
    numpy.array

        Array with same length as input, where values are P-values.

    See Also
    --------
    pyranges.stats.rowbased_spearman : fast row-based Spearman's correlation.
    pyranges.stats.fdr : correct for multiple testing

    Examples
    --------
    >>> x = np.array([[7, 2, 9], [3, 6, 0], [0, 6, 3]])
    >>> y = np.array([[5, 3, 2], [9, 6, 0], [7, 3, 5]])

    Perform Pearson's correlation pairwise on each row in 10x10 matrixes:

    >>> pr.stats.rowbased_pearson(x, y)
    array([-0.09078413,  0.65465367, -1.        ])

    """
    # Thanks to https://github.com/dengemann

    def ss(a: "NDArray[np.float64]", axis: int) -> "NDArray[np.float64]":
        return np.sum(a * a, axis=axis)

    x = np.asarray(x)
    y = np.asarray(y)

    mx = x.mean(axis=-1)
    my = y.mean(axis=-1)

    xm, ym = x - mx[..., None], y - my[..., None]

    r_num = np.add.reduce(xm * ym, axis=-1)
    r_den = np.sqrt(ss(xm, axis=-1) * ss(ym, axis=-1))

    with np.errstate(divide="ignore", invalid="ignore"):
        return r_num / r_den


def rowbased_rankdata(data: ndarray) -> DataFrame:
    """Rank order of entries in each row.

    Same as SciPy rankdata with method=mean.

    Parameters
    ----------
    data : matrix-like
        The data to find order of.

    Returns
    -------
    Pandas.DataFrame

        DataFrame where values are order of data.

    Examples
    --------
    >>> x = np.random.randint(10, size=(3, 4))
    >>> x = np.array([[3, 7, 6, 0], [1, 3, 8, 9], [5, 9, 3, 5]])
    >>> pr.stats.rowbased_rankdata(x)
         0    1    2    3
    0  2.0  4.0  3.0  1.0
    1  1.0  2.0  3.0  4.0
    2  2.5  4.0  1.0  2.5

    """
    dc = np.asarray(data).copy()
    sorter = np.apply_along_axis(np.argsort, 1, data)

    inv = np.empty(data.shape, np.intp)

    ranks = np.tile(np.arange(data.shape[1]), (len(data), 1))

    np.put_along_axis(inv, sorter, ranks, axis=1)

    dc = np.take_along_axis(dc, sorter, 1)

    res = np.apply_along_axis(lambda r: r[1:] != r[:-1], 1, dc)

    obs = np.column_stack([np.ones(len(res), dtype=bool), res])

    dense = pd.DataFrame(np.take_along_axis(np.apply_along_axis(np.cumsum, 1, obs), inv, 1))

    len_r = obs.shape[1]

    nonzero = np.count_nonzero(obs, axis=1)

    _ranks = []
    for _nonzero, nzdf in pd.DataFrame(obs).groupby(pd.Series(nonzero), sort=False):
        nz = np.apply_along_axis(lambda r: np.nonzero(r)[0], 1, nzdf)

        _count = np.column_stack([nz, np.ones(len(nz)) * len_r])
        _dense = dense.reindex(nzdf.index).to_numpy()

        _result = 0.5 * (np.take_along_axis(_count, _dense, 1) + np.take_along_axis(_count, _dense - 1, 1) + 1)

        result = pd.DataFrame(_result, index=nzdf.index)
        _ranks.append(result)

    return pd.concat(_ranks).sort_index(kind="mergesort")


def simes(
    df: "pr.PyRanges",
    by: str | list[str],
    pcol: str,
    *,
    keep_position: bool = False,
) -> "pr.PyRanges | DataFrame":
    """Apply Simes method for giving dependent events a p-value.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to analyse with Simes.

    by : str or list of str
        Features equal in these columns will be merged with Simes.

    pcol : str
        Name of column with p-values.

    keep_position : bool, default False
        Keep columns "Chromosome", "Start", "End" and "Strand" if they exist.

    See Also
    --------
    pr.stats.fdr : correct for multiple testing

    Examples
    --------
    >>> s = '''Chromosome Start End Strand Gene PValue
    ... 1 10 20 + P53 0.0001
    ... 1 20 35 + P53 0.0002
    ... 1 30 40 + P53 0.0003
    ... 2 60 65 - FOX 0.05
    ... 2 70 75 - FOX 0.0000001
    ... 2 80 90 - FOX 0.0000021'''

    >>> gr = pr.from_string(s)
    >>> gr
      index  |      Chromosome    Start      End  Strand    Gene       PValue
      int64  |           int64    int64    int64  str       str       float64
    -------  ---  ------------  -------  -------  --------  ------  ---------
          0  |               1       10       20  +         P53       0.0001
          1  |               1       20       35  +         P53       0.0002
          2  |               1       30       40  +         P53       0.0003
          3  |               2       60       65  -         FOX       0.05
          4  |               2       70       75  -         FOX       1e-07
          5  |               2       80       90  -         FOX       2.1e-06
    PyRanges with 6 rows, 6 columns, and 1 index columns.
    Contains 2 chromosomes and 2 strands.

    >>> simes = pr.stats.simes(gr, "Gene", "PValue")
    >>> simes
      Gene         Simes
    0  FOX  3.000000e-07
    1  P53  3.000000e-04

    >>> pr.stats.simes(gr, "Gene", "PValue", keep_position=True)
      index  |      Chromosome    Start      End      Simes  Strand    Gene
      int64  |           int64    int64    int64    float64  str       str
    -------  ---  ------------  -------  -------  ---------  --------  ------
          0  |               2       60       90     1e-07   -         FOX
          1  |               1       10       40     0.0001  +         P53
    PyRanges with 2 rows, 6 columns, and 1 index columns.
    Contains 2 chromosomes and 2 strands.

    """
    if isinstance(by, str):
        by = [by]

    positions = []

    if keep_position:
        positions += GENOME_LOC_COLS
        if df.has_strand:
            positions += ["Strand"]

    sorter = [*by, pcol]

    sdf = df[positions + sorter].sort_values(sorter)
    g = sdf.groupby(positions + by)

    ranks = g.cumcount().to_numpy() + 1
    size = g.size().to_numpy()
    size = np.repeat(size, size)
    multiplied = sdf[pcol].to_numpy() * size

    simes = multiplied / ranks

    sdf.insert(sdf.shape[1], "Simes", simes)

    if keep_position:
        grpby_dict = {
            "Chromosome": "first",
            "Start": "min",
            "End": "max",
            "Simes": "min",
        }

        if sdf.has_strand:
            grpby_dict["Strand"] = "first"

        simes = sdf.groupby(by).agg(grpby_dict).reset_index()
        columns = list(simes.columns)
        columns.append(columns[0])
        del columns[0]
        _simes: PyRanges | DataFrame
        _simes = ensure_pyranges(simes[columns])
    else:
        _simes = sdf.groupby(by).Simes.min().reset_index()

    return _simes


####################################################################################################
# methods available at pr.stats and at pr.PyRanges.stats: (see usages of StatsManager for how to)


def forbes(
    p: "PyRanges",
    other: "PyRanges",
    chromsizes: "PyRanges | DataFrame | dict[Any, int]",
    strand_behavior: VALID_STRAND_BEHAVIOR_TYPE = "auto",
) -> float:
    """Compute Forbes coefficient.

    Ratio which represents observed versus expected co-occurence.

    Described in ``Forbes SA (1907): On the local distribution of certain Illinois fishes: an essay in statistical ecology.``

    Parameters
    ----------
    p : PyRanges
        Intervals to compare.

    other : PyRanges
        Intervals to compare with.

    chromsizes : int, dict, DataFrame or PyRanges
        Integer representing genome length or mapping from chromosomes
        to its length.

    strand_behavior : {"auto", "same", "opposite", "ignore"}, default "auto"
        Whether to consider overlaps of intervals on the same strand, the opposite or ignore strand
        information. The default, "auto", means use "same" if both PyRanges are stranded (see .strand_valid)
        otherwise ignore the strand information.

    Returns
    -------
    float

        Ratio of observed versus expected co-occurence.

    See Also
    --------
    pyranges.stats.jaccard : compute the jaccard coefficient

    Examples
    --------
    >>> gr, gr2 = pr.example_data.f1, pr.example_data.f2
    >>> float(pr.stats.forbes(gr, gr2, chromsizes={"chr1": 10}))
    0.8333333333333334

    """
    _chromsizes = _chromsizes_as_int(chromsizes)

    strand_behavior = validate_and_convert_strand_behavior(p, other, strand_behavior)
    use_strand = use_strand_from_validated_strand_behavior(p, other, strand_behavior)

    reference_length = p.merge_overlaps(use_strand=use_strand).length
    query_length = other.merge_overlaps(use_strand=use_strand).length

    intersection_sum = p.set_intersect_overlaps(other, strand_behavior=strand_behavior).lengths().sum()
    return _chromsizes * intersection_sum / (reference_length * query_length)


def jaccard(
    p: "PyRanges",
    other: "PyRanges",
    strand_behavior: VALID_STRAND_BEHAVIOR_TYPE = "auto",
) -> float:
    """Compute Jaccards coefficient.

    Ratio of the intersection and union of two sets.

    Parameters
    ----------
    p : PyRanges
        Intervals to compare.

    other : PyRanges
        Intervals to compare with.

    strand_behavior : {"auto", "same", "opposite", "ignore"}, default "auto"
        Whether to consider overlaps of intervals on the same strand, the opposite or ignore strand
        information. The default, "auto", means use "same" if both PyRanges are stranded (see .strand_valid)
        otherwise ignore the strand information.

    Returns
    -------
    float

        Ratio of the intersection and union of two sets.

    See Also
    --------
    pyranges.stats.forbes : compute the forbes coefficient

    Examples
    --------
    >>> gr, gr2 = pr.example_data.f1, pr.example_data.f2
    >>> chromsizes = pr.example_data.chromsizes
    >>> float(pr.stats.jaccard(gr, gr2))
    0.14285714285714285

    """
    strand_behavior = validate_and_convert_strand_behavior(p, other, strand_behavior)
    use_strand = use_strand_from_validated_strand_behavior(p, other, strand_behavior)

    intersection_sum = p.set_intersect_overlaps(other).lengths().sum()

    union_sum = 0
    for gr in [p, other]:
        union_sum += gr.merge_overlaps(use_strand=use_strand).lengths().sum()

    denominator = union_sum - intersection_sum
    if denominator == 0:
        return 1.0
    return intersection_sum / denominator


def relative_distance(p: "PyRanges", other: "PyRanges", **_) -> DataFrame:
    """Compute spatial correlation between two sets.

    Metric which describes relative distance between each interval in one
    set and two closest intervals in another.

    Parameters
    ----------
    p : PyRanges
        Intervals to compare.

    other : PyRanges
        Intervals to compare with.

    chromsizes : int, dict, DataFrame or PyRanges

        Integer representing genome length or mapping from chromosomes
        to its length.

    strandedness : {None, "same", "opposite", False}, default None, i.e. "auto"

        Whether to compute without regards to strand or on same or opposite.

    Returns
    -------
    pandas.DataFrame

        DataFrame containing the frequency of each relative distance.

    See Also
    --------
    pyranges.stats.jaccard : compute the jaccard coefficient
    pyranges.stats.forbes : compute the forbes coefficient

    Examples
    --------
    >>> gr1, gr2 = pr.example_data.chipseq, pr.example_data.chipseq_background
    >>> gr = pd.concat([gr1, gr1.head(4), gr2.tail(4)])
    >>> chromsizes = pr.example_data.chromsizes
    >>> pr.stats.relative_distance(gr, gr2)
        reldist  count  total  fraction
    0      0.00      4     18  0.222222
    1      0.03      1     18  0.055556
    2      0.04      1     18  0.055556
    3      0.10      1     18  0.055556
    4      0.12      1     18  0.055556
    5      0.13      1     18  0.055556
    6      0.19      1     18  0.055556
    7      0.23      1     18  0.055556
    8      0.24      1     18  0.055556
    9      0.38      1     18  0.055556
    10     0.41      2     18  0.111111
    11     0.42      1     18  0.055556
    12     0.43      2     18  0.111111

    """
    result = pd.Series(_relative_distance(p, other))

    not_nan = ~np.isnan(result)
    result.loc[not_nan] = np.floor(result[not_nan] * 100) / 100
    vc = result.value_counts(dropna=False).to_frame().reset_index()
    vc.columns = pd.Index(["reldist", "count"])
    vc.insert(vc.shape[1], "total", len(result))
    vc.insert(vc.shape[1], "fraction", vc["count"] / len(result))
    vc = vc.sort_values("reldist", ascending=True)
    return vc.reset_index(drop=True)
