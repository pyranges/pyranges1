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
      int64  |        category    int64    int64    float64
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

    df = pd.concat(dfs).reset_index(drop=True)
    df["Chromosome"] = df["Chromosome"].astype("category")
    return ensure_pyranges(df)


# A PAF record has 12 mandatory tab fields; optional SAM-style KEY:TYPE:VALUE tags follow.
_PAF_MIN_FIELDS = 12
# SAM-style optional-tag TYPE codes we recognise.
_PAF_TAG_TYPES = frozenset({"i", "f", "A", "Z", "B", "H"})


def read_paf(  # noqa: C901, PLR0912, PLR0915
    f: "str | Path",
    /,
    nrows: int | None = None,
    anchor: str = "target",
) -> "PyRanges":
    r"""Return a PAF (minimap2 Pairwise mApping Format) file as PyRanges.

    PAF is a TSV with 12 mandatory fields per alignment (query name/length/start/end,
    strand, target name/length/start/end, matches, block length, mapping quality),
    optionally followed by SAM-style ``KEY:TYPE:VALUE`` tags. Each row pairs a *query*
    interval with a *target* interval; this reader promotes one side (the *anchor*) to
    the canonical ``Chromosome / Start / End`` columns and keeps the other in
    ``OtherChromosome / OtherStart / OtherEnd``.

    Parameters
    ----------
    f : str or Path
        Path to a ``.paf`` / ``.paf.gz`` file.

    nrows : int, optional
        Stop after this many records. Default None (all).

    anchor : ``"target"`` or ``"query"``, default ``"target"``
        Which alignment side becomes the genomic columns (``Chromosome / Start / End``).
        The default ``"target"`` puts the reference side there; ``"query"`` puts the
        read/contig side. The other side is kept in the ``Other*`` columns. PAF's single
        ``Strand`` (the query-vs-target orientation) is kept regardless.

    Returns
    -------
    PyRanges
        Columns ``Chromosome, Start, End, Strand, OtherChromosome, OtherStart, OtherEnd,
        QueryLength, TargetLength, Matches, BlockLength, MapQ`` plus one column per optional
        tag encountered (named by the tag key, typed from its TYPE code: ``i`` -> nullable
        Int64, ``f`` -> float, others -> string; a row lacking a tag gets a missing value).

    Notes
    -----
    PAF coordinates are 0-based half-open, matching PyRanges, so ``Start`` / ``End`` are
    used as-is. Lines with fewer than 12 fields are skipped.

    See Also
    --------
    pyranges1.read_bed : read a plain BED file

    Examples
    --------
    >>> import pyranges1 as pr
    >>> from tempfile import NamedTemporaryFile
    >>> rec = "q1\t100\t10\t90\t+\tchr1\t1000\t200\t280\t75\t80\t60\tNM:i:5\tdv:f:0.02"
    >>> tmp = NamedTemporaryFile("w", suffix=".paf")
    >>> _ = tmp.write(rec + "\n")
    >>> tmp.flush()
    >>> gr = pr.read_paf(tmp.name)
    >>> list(gr.columns)
    ['Chromosome', 'Start', 'End', 'Strand', 'OtherChromosome', 'OtherStart', 'OtherEnd', 'QueryLength', 'TargetLength', 'Matches', 'BlockLength', 'MapQ', 'NM', 'dv']
    >>> str(gr["Chromosome"][0]), int(gr["Start"][0]), int(gr["End"][0])
    ('chr1', 200, 280)
    >>> str(gr["OtherChromosome"][0]), int(gr["OtherStart"][0]), int(gr["OtherEnd"][0])
    ('q1', 10, 90)
    >>> int(gr["NM"][0]), float(gr["dv"][0])
    (5, 0.02)
    >>> str(pr.read_paf(tmp.name, anchor="query")["Chromosome"][0])
    'q1'

    """
    if anchor not in ("target", "query"):
        msg = f"anchor must be 'target' or 'query', got {anchor!r}"
        raise ValueError(msg)

    import gzip

    path = Path(f)
    opener = gzip.open if path.name.endswith(".gz") else open

    q_name: list[str] = []
    q_len: list[int] = []
    q_start: list[int] = []
    q_end: list[int] = []
    strand: list[str] = []
    t_name: list[str] = []
    t_len: list[int] = []
    t_start: list[int] = []
    t_end: list[int] = []
    matches: list[int] = []
    block_len: list[int] = []
    mapq: list[int] = []
    tag_values: dict[str, list] = {}
    tag_types: dict[str, set[str]] = {}

    n = 0
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < _PAF_MIN_FIELDS:
                continue
            q_name.append(parts[0])
            q_len.append(int(parts[1]))
            q_start.append(int(parts[2]))
            q_end.append(int(parts[3]))
            strand.append(parts[4])
            t_name.append(parts[5])
            t_len.append(int(parts[6]))
            t_start.append(int(parts[7]))
            t_end.append(int(parts[8]))
            matches.append(int(parts[9]))
            block_len.append(int(parts[10]))
            mapq.append(int(parts[11]))

            seen: dict[str, str] = {}
            for tag in parts[_PAF_MIN_FIELDS:]:
                key, _, rest = tag.partition(":")
                typ, _, val = rest.partition(":")
                if not val or typ not in _PAF_TAG_TYPES:
                    continue
                seen[key] = val
                tag_types.setdefault(key, set()).add(typ)
            for key, val in seen.items():
                tag_values.setdefault(key, [None] * n).append(val)
            for key, col in tag_values.items():
                if key not in seen:
                    col.append(None)

            n += 1
            if nrows is not None and n >= nrows:
                break

    if not q_name:
        empty: dict = {
            "Chromosome": pd.Series([], dtype="category"),
            "Start": pd.Series([], dtype="int64"),
            "End": pd.Series([], dtype="int64"),
        }
        return ensure_pyranges(pd.DataFrame(empty))

    if anchor == "target":
        chrom, start, end = t_name, t_start, t_end
        other_chrom, other_start, other_end = q_name, q_start, q_end
    else:
        chrom, start, end = q_name, q_start, q_end
        other_chrom, other_start, other_end = t_name, t_start, t_end

    df = pd.DataFrame(
        {
            "Chromosome": pd.Categorical(chrom),
            "Start": start,
            "End": end,
            "Strand": pd.Categorical(strand),
            "OtherChromosome": pd.Categorical(other_chrom),
            "OtherStart": other_start,
            "OtherEnd": other_end,
            "QueryLength": q_len,
            "TargetLength": t_len,
            "Matches": matches,
            "BlockLength": block_len,
            "MapQ": mapq,
        },
    )

    for key, vals in tag_values.items():
        types = tag_types[key]
        if types <= {"i"}:
            df[key] = pd.to_numeric(pd.Series(vals), errors="coerce").astype("Int64")
        elif types <= {"i", "f"}:
            df[key] = pd.to_numeric(pd.Series(vals), errors="coerce")
        else:
            df[key] = vals

    return ensure_pyranges(df)
