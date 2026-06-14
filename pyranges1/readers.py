import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import gtfreader
import pandas as pd
from natsort import natsorted  # type: ignore[import]

from pyranges1.core.pyranges_helpers import ensure_pyranges

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyranges1.core.pyranges_main import PyRanges

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def from_string(s: str) -> "PyRanges":
    """Create a PyRanges from multiline string.

    Parameters
    ----------
    s : str
        String with data.

    Examples
    --------
    >>> import pyranges1 as pr
    >>> s = '''Chromosome      Start        End Strand
    ... chr1  246719402  246719502      +
    ... chr5   15400908   15401008      +
    ... chr9   68366534   68366634      +
    ... chr14   79220091   79220191      +
    ... chr14  103456471  103456571      -'''

    >>> pr.from_string(s)
      index  |    Chromosome        Start        End  Strand
      int64  |    str               int64      int64  str
    -------  ---  ------------  ---------  ---------  --------
          0  |    chr1          246719402  246719502  +
          1  |    chr5           15400908   15401008  +
          2  |    chr9           68366534   68366634  +
          3  |    chr14          79220091   79220191  +
          4  |    chr14         103456471  103456571  -
    PyRanges with 5 rows, 4 columns, and 1 index columns.
    Contains 4 chromosomes and 2 strands.

    """
    from io import StringIO

    df = pd.read_csv(StringIO(s), sep=r"\s+", index_col=None)

    return ensure_pyranges(df)


def read_bed(f: Path, /, nrows: int | None = None) -> "PyRanges":
    """Return bed file as PyRanges.

    This is a reader for files that follow the bed format. They can have from
    3-12 columns which will be named like so:

    Chromosome Start End Name Score Strand ThickStart ThickEnd ItemRGB
    BlockCount BlockSizes BlockStarts

    Parameters
    ----------
    f : str
        Path to bed file

    nrows : Optional int, default None
        Number of rows to return.

    Notes
    -----
    If you just want to create a PyRanges from a tab-delimited bed-like file,
    use `pr.PyRanges(pandas.read_table(f))` instead.

    Returns
    -------
    PyRanges


    Examples
    --------
    >>> import pyranges1 as pr
    >>> path = pr.example_data.files["aorta.bed"]
    >>> pr.read_bed(path, nrows=5)
      index  |    Chromosome      Start      End  Name        Score  Strand
      int64  |    category        int64    int64  str         int64  category
    -------  ---  ------------  -------  -------  --------  -------  ----------
          0  |    chr1             9916    10115  H3K27me3        5  -
          1  |    chr1             9939    10138  H3K27me3        7  +
          2  |    chr1             9951    10150  H3K27me3        8  -
          3  |    chr1             9953    10152  H3K27me3        5  +
          4  |    chr1             9978    10177  H3K27me3        7  -
    PyRanges with 5 rows, 6 columns, and 1 index columns.
    Contains 1 chromosomes and 2 strands.

    """
    columns = [
        "Chromosome",
        "Start",
        "End",
        "Name",
        "Score",
        "Strand",
        "ThickStart",
        "ThickEnd",
        "ItemRGB",
        "BlockCount",
        "BlockSizes",
        "BlockStarts",
    ]
    path = Path(f)
    if path.name.endswith(".gz"):
        import gzip

        first_start = gzip.open(path).readline().decode().split()[1]  # noqa: SIM115
    else:
        first_start = path.open().readline().split()[1]

    header = None

    try:
        int(first_start)
    except ValueError:
        header = 0

    ncols = pd.read_table(path, nrows=2).shape[1]

    df = pd.read_csv(
        path,
        dtype={"Chromosome": "category", "Strand": "category"},
        nrows=nrows,
        header=header,
        names=columns[:ncols] if header != 0 else None,
        sep="\t",
    )

    df.columns = pd.Index(columns[: df.shape[1]])

    return ensure_pyranges(df)


def read_bam(
    f: str | Path,
    /,
    mapq: int = 0,
    required_flag: int = 0,
    filter_flag: int = 1540,
    *,
    sparse: bool = True,
) -> "PyRanges":
    """Return bam file as PyRanges.

    Parameters
    ----------
    f : str
        Path to bam file

    sparse : bool, default True
        Whether to return only the columns Chromosome, Start, End, Strand, Flag.
        Set to False to return also columns
        QueryStart, QueryEnd, QuerySequence, Name, Cigar, Quality (more time consuming).

    mapq : int, default 0
        Minimum mapping quality score.

    required_flag : int, default 0
        Flags which must be present for the interval to be read.

    filter_flag : int, default 1540
        Ignore reads with these flags. Default 1540, which means that either
        the read is unmapped, the read failed vendor or platfrom quality
        checks, or the read is a PCR or optical duplicate.

    Returns
    -------
    PyRanges


    Notes
    -----
    This functionality requires the library `bamread`. It can be installed with
    `pip install bamread` or `conda install -c bioconda bamread`.

    Examples
    --------
    >>> import pyranges1 as pr
    >>> path = pr.example_data.files["smaller.bam"]
    >>> pr.read_bam(path)
    index    |    Chromosome    Start     End       Strand      Flag
    int64    |    category      int64     int64     category    uint16
    -------  ---  ------------  --------  --------  ----------  --------
    0        |    chr1          887771    887796    -           16
    1        |    chr1          994660    994685    -           16
    2        |    chr1          1041102   1041127   +           0
    3        |    chr1          1770383   1770408   -           16
    ...      |    ...           ...       ...       ...         ...
    96       |    chr1          18800901  18800926  +           0
    97       |    chr1          18800901  18800926  +           0
    98       |    chr1          18855123  18855148  -           16
    99       |    chr1          19373470  19373495  +           0
    PyRanges with 100 rows, 5 columns, and 1 index columns.
    Contains 1 chromosomes and 2 strands.

    """
    path = Path(f)
    try:
        import bamread  # type: ignore[import]
    except ImportError:
        LOGGER.exception(
            "bamread must be installed to read bam. Use `conda install -c bioconda bamread` or `pip install bamread` to install it.",
        )
        sys.exit(1)

    if bamread.__version__ in {
        "0.0.1",
        "0.0.2",
        "0.0.3",
        "0.0.4",
        "0.0.5",
        "0.0.6",
        "0.0.7",
        "0.0.8",
        "0.0.9",
    }:
        LOGGER.exception(
            "bamread not recent enough. Must be 0.0.10 or higher. Use `conda install -c bioconda 'bamread>=0.0.10'` or `pip install bamread>=0.0.10` to install it.",
        )
        sys.exit(1)

    if sparse:
        return ensure_pyranges(bamread.read_bam(path, mapq, required_flag, filter_flag))
    df = bamread.read_bam_full(path, mapq, required_flag, filter_flag)
    return ensure_pyranges(df)


def read_gtf(
    f: str | Path,
    /,
    *,
    nrows: int | None = None,
    duplicate_attr: bool = False,
    ignore_bad: bool = False,
) -> "PyRanges":
    r"""Read files in the Gene Transfer Format.

    Parameters
    ----------
    f : str
        Path to GTF file.

    nrows : int, default None
        Number of rows to read. Default None, i.e. all.

    duplicate_attr : bool, default False
        Whether to handle (potential) duplicate attributes or just keep last one.

    ignore_bad : bool, default False
        Whether to ignore bad lines or raise an error.


    Returns
    -------
    PyRanges

    Note
    ----
    The GTF format encodes both Start and End as 1-based included.
    PyRanges encodes intervals as 0-based, Start included and End excluded.

    See Also
    --------
    pyranges1.read_gff3 : read files in the General Feature Format

    Examples
    --------
    >>> import pyranges1 as pr
    >>> from tempfile import NamedTemporaryFile
    >>> contents = ['#!genome-build GRCh38.p10']
    >>> contents.append('1\thavana\tgene\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972"; gene_version "5"; gene_name "DDX11L1"; gene_source "havana"; gene_biotype "transcribed_unprocessed_pseudogene";')
    >>> contents.append('1\thavana\ttranscript\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972"; gene_version "5"; group_by "ENST00000456328"; transcript_version "2"; gene_name "DDX11L1"; gene_source "havana"; gene_biotype "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-202"; transcript_source "havana"; transcript_biotype "processed_transcript"; tag "basic"; transcript_support_level "1";')
    >>> f = NamedTemporaryFile("w")
    >>> _bytes_written = f.write("\n".join(contents))
    >>> f.flush()
    >>> pr.read_gtf(f.name)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
          index  |      Chromosome  Source      Feature       Start      End  Score    Strand      Frame       gene_id          ...
      int64  |        category  category    category      int64    int64  str      category    category    str              ...
    -------  ---  ------------  ----------  ----------  -------  -------  -------  ----------  ----------  ---------------  -----
          0  |               1  havana      gene          11868    14409  .        +           .           ENSG00000223972  ...
          1  |               1  havana      transcript    11868    14409  .        +           .           ENSG00000223972  ...
    PyRanges with 2 rows, 20 columns, and 1 index columns. (11 columns not shown: "gene_version", "gene_name", "gene_source", ...).
    Contains 1 chromosomes and 1 strands.

    """
    return read_gtf_full(
        Path(f),
        nrows=nrows,
        duplicate_attr=duplicate_attr,
        ignore_bad=ignore_bad,
    )


def read_gtf_full(
    f: str | Path,
    /,
    nrows: int | None = None,
    chunksize: int = int(1e5),  # for unit-testing purposes
    *,
    duplicate_attr: bool = False,
    ignore_bad: bool = False,
) -> "PyRanges":
    """Read files in the Gene Transfer Format into a PyRanges, including the annotation column.

    Parameters
    ----------
    f : str
        Path to GTF file.

    nrows : int, default None
        Number of rows to read. Default None, i.e. all.

    chunksize : int, default 100000
        Number of rows to read at a time. Default 100000.

    duplicate_attr : bool, default False
        Whether to handle (potential) duplicate attributes or just keep last one.

    ignore_bad : bool, default False
        Whether to ignore bad lines or raise an error.

    Returns
    -------
    PyRanges

    """
    path = Path(f)
    skiprows = gtfreader.find_first_data_line_index(path)
    df = gtfreader.read_gtf_full(
        path,
        nrows=nrows,
        skiprows=skiprows,
        chunksize=chunksize,
        duplicate_attr=duplicate_attr,
        ignore_bad=ignore_bad,
    )
    return ensure_pyranges(df)


def parse_kv_fields(line: str) -> list[tuple[str, str]]:
    """Parse GTF attribute column."""
    return gtfreader.parse_kv_fields(line)


def to_rows(anno: pd.Series, *, ignore_bad: bool = False) -> pd.DataFrame:
    """Parse GTF attribute column into a dataframe of attribute columns."""
    return gtfreader.to_rows(anno, ignore_bad=ignore_bad)


def to_rows_keep_duplicates(anno: pd.Series, *, ignore_bad: bool = False) -> pd.DataFrame:
    """If an entry is found multiple times in the attribute string, keep all of them.

    Examples
    --------
    >>> anno = pd.Series(['tag "DDX11L1"; tag "sonic"; unique "hi";'])
    >>> result = to_rows_keep_duplicates(anno)
    >>> result.to_dict(orient="records")
    [{'tag': 'DDX11L1,sonic', 'unique': 'hi'}]

    """
    return gtfreader.to_rows_keep_duplicates(anno, ignore_bad=ignore_bad)


def to_rows_gff3(anno: pd.Series) -> pd.DataFrame:
    """Parse GFF3 attribute column into a dataframe of attribute columns."""
    rowdicts = [to_keys_and_values(line) for line in list(anno)]

    return pd.DataFrame.from_records(rowdicts).set_index(anno.index)


def to_keys_and_values(line: str) -> dict[str, str]:
    """Parse GFF3 attribute column."""
    return dict(it.split("=") for it in line.rstrip("; ").split(";"))


def read_gff3(
    f: str | Path,
    nrows: int | None = None,
) -> "PyRanges":
    """Read files in the General Feature Format into a PyRanges.

    Parameters
    ----------
    f : str
        Path to GFF file.

    nrows : int, default None
        Number of rows to read. Default None, i.e. all.

    Returns
    -------
    PyRanges

    Notes
    -----
    The gff3 format encodes both Start and End as 1-based included.
    PyRanges (and also the DF returned by this function, if as_df=True), instead
    encodes intervals as 0-based, Start included and End excluded.

    See Also
    --------
    pyranges1.read_gtf : read files in the Gene Transfer Format

    """
    path = Path(f)

    dtypes: Mapping = {"Chromosome": "category", "Feature": "category", "Strand": "category"}

    names = ["Chromosome", "Source", "Feature", "Start", "End", "Score", "Strand", "Frame", "Attribute"]

    df_iter = pd.read_csv(
        path,
        comment="#",
        sep="\t",
        header=None,
        names=names,
        dtype=dtypes,
        chunksize=int(1e5),
        nrows=nrows,
    )

    dfs = []
    for df in df_iter:
        extra = to_rows_gff3(df.Attribute.astype(str))
        _df = df.drop("Attribute", axis=1)
        extra = extra.set_index(_df.index)
        ndf = pd.concat([_df, extra], axis=1, sort=False)
        dfs.append(ndf)

    df = pd.concat(dfs, sort=False)

    df.loc[:, "Start"] = df.Start - 1

    return ensure_pyranges(df)


def read_bigwig(f: str | Path) -> "PyRanges":
    """Read bigwig files into a PyRanges.

    Parameters
    ----------
    f : str
        Path to bw file.

    Returns
    -------
    PyRanges

    Note
    ----
    This function requires the library pyBigWig, it can be installed with pip install pyBigWig

    Examples
    --------
    >>> import pyranges1 as pr
    >>> path = pr.example_data.files["bigwig.bw"]
    >>> pr.read_bigwig(path)
      index  |      Chromosome    Start      End      Value
      int64  |             str    int64    int64    float64
    -------  ---  ------------  -------  -------  ---------
          0  |               1        0        1        0.1
          1  |               1        1        2        0.2
          2  |               1        2        3        0.3
          3  |               1      100      150        1.4
          4  |               1      150      151        1.5
          5  |              10      200      300        2
    PyRanges with 6 rows, 4 columns, and 1 index columns.
    Contains 2 chromosomes.


    """
    try:
        import pyBigWig  # type: ignore[import]
    except ModuleNotFoundError:
        LOGGER.exception(
            "pyBigWig must be installed to read bigwigs. Use `pip install pyBigWig` to install it.",
        )
        sys.exit(1)

    path = Path(f)
    bw = pyBigWig.open(str(path))

    size = int(1e5)
    chromosomes = bw.chroms()

    dfs = {}

    for chromosome in natsorted(chromosomes):
        outstarts = []
        outends = []
        outvalues = []

        length = chromosomes[chromosome]

        starts = list(range(0, length, size))
        ends = list(range(size, length + size, size))
        ends[-1] = length
        for start, end in zip(starts, ends, strict=True):
            intervals = bw.intervals(chromosome, start, end)
            if intervals is not None:
                for s, e, v in intervals:
                    outstarts.append(s)
                    outends.append(e)
                    outvalues.append(v)

        outstarts = pd.Series(outstarts)
        outends = pd.Series(outends)
        outvalues = pd.Series(outvalues)
        dfs[chromosome] = pd.DataFrame(
            {
                "Chromosome": chromosome,
                "Start": outstarts,
                "End": outends,
                "Value": outvalues,
            },
        )

    return ensure_pyranges(pd.concat(dfs).reset_index(drop=True))


# ENCODE narrowPeak is BED6 + 4 fixed extra columns (MACS2/MACS3 output).
_NARROWPEAK_COLUMNS = [
    "Chromosome",
    "Start",
    "End",
    "Name",
    "Score",
    "Strand",
    "SignalValue",
    "PValue",
    "QValue",
    "Peak",
]


def read_narrowPeak(f: "str | Path", /, nrows: int | None = None) -> "PyRanges":  # noqa: N802
    r"""Return ENCODE narrowPeak (BED6+4) file as PyRanges.

    Parameters
    ----------
    f : str or Path
        Path to narrowPeak file (may be gzip-compressed).

    nrows : int, default None
        Number of rows to read. Default None (all).

    Returns
    -------
    PyRanges

    Notes
    -----
    Columns: Chromosome, Start, End, Name, Score, Strand, SignalValue, PValue,
    QValue, Peak. This is the format produced by MACS2/MACS3 and used by ENCODE.
    ``#``-prefixed track/comment lines are skipped.

    See Also
    --------
    pyranges1.read_bed : read a plain BED file

    Examples
    --------
    >>> import pyranges1 as pr
    >>> from tempfile import NamedTemporaryFile
    >>> tmp = NamedTemporaryFile("w", suffix=".narrowPeak")
    >>> _ = tmp.write("chr1\t100\t200\tpeak1\t500\t+\t5.5\t3.2\t2.1\t50\n")
    >>> tmp.flush()
    >>> gr = pr.read_narrowPeak(tmp.name)
    >>> list(gr.columns)
    ['Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand', 'SignalValue', 'PValue', 'QValue', 'Peak']
    >>> gr["Start"].tolist(), gr["End"].tolist()
    ([100], [200])
    >>> gr["SignalValue"].tolist(), gr["Peak"].tolist()
    ([5.5], [50])

    """
    df = pd.read_csv(
        Path(f),
        sep="\t",
        header=None,
        names=_NARROWPEAK_COLUMNS,
        nrows=nrows,
        comment="#",
        dtype={"Chromosome": "category", "Strand": "category"},
    )
    return ensure_pyranges(df)


def read_parquet(
    f: "str | Path",
    /,
    columns: list[str] | None = None,
    **kwargs,
) -> "PyRanges":
    """Return a Parquet file as PyRanges.

    Parquet preserves column dtypes and loads much faster than CSV-based formats.
    Any Parquet file with at least ``Chromosome``, ``Start``, ``End`` columns can
    be read; use :meth:`PyRanges.to_parquet` to write one.

    Parameters
    ----------
    f : str or Path
        Path to the Parquet file.

    columns : list of str, optional
        Subset of columns to load. None (default) loads all columns. Must include
        at least Chromosome, Start, End.

    **kwargs
        Forwarded to :func:`pandas.read_parquet` (e.g. ``engine``, ``filters``).

    Returns
    -------
    PyRanges

    Notes
    -----
    This functionality requires a Parquet engine, e.g. ``pyarrow``
    (``pip install pyarrow``). ``Chromosome`` and ``Strand`` are restored as
    ``category`` dtype if they are not already categorical.

    See Also
    --------
    PyRanges.to_parquet : write a PyRanges to Parquet

    Examples
    --------
    >>> import pyranges1 as pr  # doctest: +SKIP
    >>> pr.read_parquet("intervals.parquet")  # doctest: +SKIP

    """
    try:
        df = pd.read_parquet(Path(f), columns=columns, **kwargs)
    except ImportError:
        LOGGER.exception(
            "A Parquet engine must be installed to read Parquet files. Use `pip install pyarrow` to install one.",
        )
        sys.exit(1)

    for col in ("Chromosome", "Strand"):
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")

    return ensure_pyranges(df)


def _flatten_vcf_info(value: object) -> object:
    """Collapse a multi-valued VCF INFO field (a tuple) to a comma-joined string."""
    if isinstance(value, (tuple, list)):
        return ",".join("" if v is None else str(v) for v in value)
    return value


def read_vcf(
    f: "str | Path",
    /,
    region: str | None = None,
    nrows: int | None = None,
    info_fields: list[str] | None = None,
) -> "PyRanges":
    """Return a VCF/BCF file as PyRanges.

    Each variant becomes one row spanning its REF allele: ``Start`` is the 0-based
    position (``POS - 1``) and ``End = Start + len(REF)``. Multi-allelic records
    keep their ALT alleles comma-joined in a single row.

    Parameters
    ----------
    f : str or Path
        Path to a VCF / VCF.gz / BCF file.

    region : str, optional
        Restrict to a region, e.g. ``"chr1:1000-2000"`` (requires a tabix/CSI index).

    nrows : int, optional
        Stop after this many records. Default None (all).

    info_fields : list of str, optional
        Which INFO fields to expand into columns. None (default) expands every
        INFO field present; pass an explicit list (possibly empty) to restrict.

    Returns
    -------
    PyRanges
        Columns: Chromosome, Start, End, ID, REF, ALT, QUAL, FILTER, plus one
        column per requested INFO field.

    Notes
    -----
    This functionality requires the library ``pysam`` (``pip install pysam`` or
    ``conda install -c bioconda pysam``).

    Examples
    --------
    >>> import pyranges1 as pr  # doctest: +SKIP
    >>> pr.read_vcf("variants.vcf.gz", region="chr1:1-100000")  # doctest: +SKIP

    """
    try:
        import pysam  # type: ignore[import]
    except ImportError:
        LOGGER.exception(
            "pysam must be installed to read VCF/BCF files. "
            "Use `conda install -c bioconda pysam` or `pip install pysam`.",
        )
        sys.exit(1)

    vf = pysam.VariantFile(str(Path(f)))
    iterator = vf.fetch(region=region) if region else vf

    chroms: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    ids: list[str | None] = []
    refs: list[str] = []
    alts: list[str | None] = []
    quals: list[float | None] = []
    filters: list[str] = []
    info_cols: dict[str, list] = {}

    for n, rec in enumerate(iterator):
        if nrows is not None and n >= nrows:
            break
        chroms.append(rec.chrom)
        starts.append(int(rec.start))
        ends.append(int(rec.stop))
        ids.append(rec.id)
        refs.append(rec.ref or "")
        alts.append(",".join(a for a in rec.alts if a is not None) if rec.alts else None)
        quals.append(rec.qual)
        filters.append(",".join(str(k) for k in rec.filter) if rec.filter else "PASS")

        keys = info_fields if info_fields is not None else list(rec.info.keys())
        for k in keys:
            info_cols.setdefault(k, [None] * n).append(
                _flatten_vcf_info(rec.info[k]) if k in rec.info else None,
            )
        for k, col in info_cols.items():
            if k not in keys:
                col.append(None)

    df = pd.DataFrame(
        {
            "Chromosome": pd.Categorical(chroms),
            "Start": starts,
            "End": ends,
            "ID": ids,
            "REF": refs,
            "ALT": alts,
            "QUAL": quals,
            "FILTER": pd.Categorical(filters),
        },
    )
    for k, v in info_cols.items():
        df[k] = v

    return ensure_pyranges(df)
