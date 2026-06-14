import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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


# Minimum number of tab-separated fields in a 4DN .pairs record
# (readID, chr1, pos1, chr2, pos2, strand1, strand2).
_PAIRS_MIN_FIELDS = 7

# The first three BigBed autoSql fields are always the coordinate columns
# (Chromosome, Start, End); any further fields are optional metadata columns.
_BED_COORD_NCOLS = 3

# autoSql field name -> PyRanges column name, for the optional columns embedded
# in a BigBed's SQL schema (the first three are always Chromosome/Start/End).
_BIGBED_AUTOSQL_TO_PYRANGES = {
    "name": "Name",
    "score": "Score",
    "strand": "Strand",
    "thickStart": "ThickStart",
    "thickEnd": "ThickEnd",
    "itemRgb": "ItemRGB",
    "blockCount": "BlockCount",
    "blockSizes": "BlockSizes",
    "blockStarts": "BlockStarts",
}


def _read_pysam_alignment(
    f: "str | Path",
    mode: Literal["r", "rc"],
    mapq: int,
    required_flag: int,
    filter_flag: int,
    reference_filename: "str | Path | None",
    *,
    sparse: bool,
) -> "PyRanges":
    """Read a SAM/CRAM file via pysam into a PyRanges.

    Shared backend for :func:`read_sam` (``mode="r"``) and :func:`read_cram`
    (``mode="rc"``). The output schema matches :func:`read_bam`: sparse reads
    return ``Chromosome, Start, End, Strand, Flag``; full reads additionally
    return ``QueryStart, QueryEnd, QuerySequence, Name, Cigar, Quality``.
    Unmapped reads are always skipped.
    """
    try:
        import pysam  # type: ignore[import]
    except ImportError:
        LOGGER.exception(
            "pysam must be installed to read SAM/CRAM files. "
            "Use `conda install -c bioconda pysam` or `pip install pysam`.",
        )
        sys.exit(1)

    open_kwargs: dict = {}
    if reference_filename is not None:
        open_kwargs["reference_filename"] = str(reference_filename)

    chromosomes: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    strands: list[str] = []
    flags: list[int] = []
    query_starts: list[int] = []
    query_ends: list[int] = []
    query_seqs: list[str | None] = []
    names: list[str | None] = []
    cigars: list[str | None] = []
    quals: list[str | None] = []

    with pysam.AlignmentFile(str(f), mode, **open_kwargs) as af:
        for read in af:
            if read.is_unmapped or read.mapping_quality < mapq:
                continue
            if required_flag and (read.flag & required_flag) != required_flag:
                continue
            if read.flag & filter_flag:
                continue
            end = read.reference_end
            if end is None:
                continue
            chromosomes.append(read.reference_name or "")
            starts.append(read.reference_start)
            ends.append(end)
            strands.append("-" if read.is_reverse else "+")
            flags.append(read.flag)
            if not sparse:
                query_starts.append(read.query_alignment_start)
                query_ends.append(read.query_alignment_end)
                query_seqs.append(read.query_sequence)
                names.append(read.query_name)
                cigars.append(read.cigarstring)
                quals.append(
                    "".join(chr(q + 33) for q in read.query_qualities) if read.query_qualities is not None else None,
                )

    data: dict = {
        "Chromosome": pd.Categorical(chromosomes),
        "Start": starts,
        "End": ends,
        "Strand": pd.Categorical(strands),
        "Flag": pd.Series(flags, dtype="uint16"),
    }
    if not sparse:
        data["QueryStart"] = query_starts
        data["QueryEnd"] = query_ends
        data["QuerySequence"] = query_seqs
        data["Name"] = names
        data["Cigar"] = cigars
        data["Quality"] = quals

    return ensure_pyranges(pd.DataFrame(data))


def read_sam(
    f: "str | Path",
    /,
    mapq: int = 0,
    required_flag: int = 0,
    filter_flag: int = 1540,
    *,
    sparse: bool = True,
) -> "PyRanges":
    """Return SAM file as PyRanges.

    Parameters
    ----------
    f : str or Path
        Path to SAM file.

    mapq : int, default 0
        Minimum mapping quality score. Reads below this are skipped.

    required_flag : int, default 0
        Flags which must all be present for the read to be kept (0 = no requirement).

    filter_flag : int, default 1540
        Ignore reads with any of these flags. Default 1540 = unmapped (4) +
        QC-fail (512) + PCR/optical duplicate (1024).

    sparse : bool, default True
        Whether to return only the columns Chromosome, Start, End, Strand, Flag.
        Set to False to additionally return QueryStart, QueryEnd, QuerySequence,
        Name, Cigar, Quality (more time consuming).

    Returns
    -------
    PyRanges

    Notes
    -----
    This functionality requires the library ``pysam``. It can be installed with
    ``pip install pysam`` or ``conda install -c bioconda pysam``. Unmapped reads
    are always skipped. The output mirrors :func:`read_bam`.

    See Also
    --------
    pyranges1.read_bam : read alignments from a BAM file
    pyranges1.read_cram : read alignments from a CRAM file

    Examples
    --------
    >>> import pyranges1 as pr  # doctest: +SKIP
    >>> pr.read_sam("reads.sam")  # doctest: +SKIP

    """
    return _read_pysam_alignment(f, "r", mapq, required_flag, filter_flag, None, sparse=sparse)


def read_cram(
    f: "str | Path",
    /,
    mapq: int = 0,
    required_flag: int = 0,
    filter_flag: int = 1540,
    *,
    sparse: bool = True,
    reference_filename: "str | Path | None" = None,
) -> "PyRanges":
    """Return CRAM file as PyRanges.

    Parameters
    ----------
    f : str or Path
        Path to CRAM file.

    mapq : int, default 0
        Minimum mapping quality score. Reads below this are skipped.

    required_flag : int, default 0
        Flags which must all be present for the read to be kept (0 = no requirement).

    filter_flag : int, default 1540
        Ignore reads with any of these flags. Default 1540 = unmapped (4) +
        QC-fail (512) + PCR/optical duplicate (1024).

    sparse : bool, default True
        Whether to return only the columns Chromosome, Start, End, Strand, Flag.
        Set to False to additionally return QueryStart, QueryEnd, QuerySequence,
        Name, Cigar, Quality (more time consuming).

    reference_filename : str or Path, optional
        Path to the reference FASTA used during CRAM encoding. Required to
        reconstruct sequence/quality (``sparse=False``) or when the CRAM's
        ``@SQ UR:`` header tag does not point to an accessible reference.
        Coordinate-only reads (``sparse=True``) can often be decoded without it.

    Returns
    -------
    PyRanges

    Notes
    -----
    This functionality requires the library ``pysam``. It can be installed with
    ``pip install pysam`` or ``conda install -c bioconda pysam``. Unmapped reads
    are always skipped. The output mirrors :func:`read_bam`.

    See Also
    --------
    pyranges1.read_bam : read alignments from a BAM file
    pyranges1.read_sam : read alignments from a SAM file

    Examples
    --------
    >>> import pyranges1 as pr  # doctest: +SKIP
    >>> pr.read_cram("reads.cram", reference_filename="genome.fa")  # doctest: +SKIP

    """
    return _read_pysam_alignment(f, "rc", mapq, required_flag, filter_flag, reference_filename, sparse=sparse)


def read_bigbed(f: "str | Path") -> "PyRanges":
    """Return BigBed file as PyRanges.

    Parameters
    ----------
    f : str or Path
        Path to BigBed (.bb) file.

    Returns
    -------
    PyRanges

    Notes
    -----
    This functionality requires the library ``pyBigWig`` (``pip install pyBigWig``).
    Column names beyond Chromosome/Start/End are taken from the autoSql schema
    embedded in the BigBed; the conventional BED field names map back to PyRanges
    columns (``name`` -> ``Name``, ``strand`` -> ``Strand`` ...).

    See Also
    --------
    pyranges1.read_bigwig : read signal values from a BigWig file

    Examples
    --------
    >>> import pyranges1 as pr  # doctest: +SKIP
    >>> pr.read_bigbed("annotations.bb")  # doctest: +SKIP

    """
    try:
        import pyBigWig  # type: ignore[import]
    except ModuleNotFoundError:
        LOGGER.exception(
            "pyBigWig must be installed to read BigBed files. Use `pip install pyBigWig` to install it.",
        )
        sys.exit(1)

    import re
    from io import StringIO

    bb = pyBigWig.open(str(Path(f)))
    sql = bb.SQL() or ""
    if isinstance(sql, bytes):
        sql = sql.decode("ascii")

    col_names = [
        _BIGBED_AUTOSQL_TO_PYRANGES.get(m.group(1), m.group(1))
        for line in sql.splitlines()
        if (m := re.match(r"\s*[A-Za-z_]\w*(?:\[[^\]]*\])?\s+`?(\w+)`?\s*;", line))
    ]

    rows: list[tuple] = []
    for chrom, chrom_len in bb.chroms().items():
        entries = bb.entries(chrom, 0, chrom_len)
        if entries:
            rows.extend((chrom, beg, end, rest) for beg, end, rest in entries)

    if not rows:
        empty: dict = {
            "Chromosome": pd.Series([], dtype="category"),
            "Start": pd.Series([], dtype="int64"),
            "End": pd.Series([], dtype="int64"),
        }
        return ensure_pyranges(pd.DataFrame(empty))

    coords = pd.DataFrame(
        {
            "Chromosome": pd.Categorical([r[0] for r in rows]),
            "Start": [r[1] for r in rows],
            "End": [r[2] for r in rows],
        },
    )

    extra_cols = col_names[_BED_COORD_NCOLS:] if len(col_names) > _BED_COORD_NCOLS else []
    if extra_cols:
        meta_text = "\n".join(r[3] for r in rows if r[3] is not None)
        meta = pd.read_csv(StringIO(meta_text), sep="\t", names=extra_cols, header=None)
        df = pd.concat([coords.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    else:
        df = coords

    if "Strand" in df.columns:
        df["Strand"] = df["Strand"].astype("category")

    return ensure_pyranges(df)


def read_pairs(
    f: "str | Path",
    /,
    nrows: int | None = None,
    anchor: str = "1",
) -> "PyRanges":
    r"""Return a 4DN Hi-C ``.pairs`` / ``.pairs.gz`` file as PyRanges.

    The 4DN pairs format is a TSV with a ``#``-prefixed header block followed by
    records ``readID, chr1, pos1, chr2, pos2, strand1, strand2``. Each Hi-C
    contact is a pair of point coordinates; this reader keeps both mates on a
    single row, promoting one mate (the *anchor*) to the canonical
    ``Chromosome / Start / End / Strand`` columns and demoting the other to
    ``OtherChromosome / OtherStart / OtherEnd / OtherStrand``. Point positions
    become single-base intervals (``End = Start + 1``).

    Parameters
    ----------
    f : str or Path
        Path to a ``.pairs`` or ``.pairs.gz`` file.

    nrows : int, optional
        Stop after reading this many records. Default None (all).

    anchor : ``"1"`` or ``"2"``, default ``"1"``
        Which mate of the pair becomes the genomic columns
        (``Chromosome / Start / End / Strand``). The other mate is kept in the
        ``Other*`` columns. Downstream PyRanges operations act on the anchor mate.

    Returns
    -------
    PyRanges
        Columns ``Chromosome, Start, End, Strand, OtherChromosome, OtherStart,
        OtherEnd, OtherStrand, ReadID``.

    Notes
    -----
    The ``pos`` fields are 1-based per the 4DN spec; this reader subtracts 1 so
    coordinates are stored 0-based (Start included, End excluded), consistent
    with the rest of PyRanges.

    Examples
    --------
    >>> import pyranges1 as pr
    >>> from tempfile import NamedTemporaryFile
    >>> contents = '''## pairs format v1.0
    ... #columns: readID chr1 pos1 chr2 pos2 strand1 strand2
    ... r1\tchr1\t100\tchr2\t200\t+\t-
    ... r2\tchr1\t150\tchr1\t900\t-\t+'''
    >>> tmp = NamedTemporaryFile("w", suffix=".pairs")
    >>> _ = tmp.write(contents)
    >>> tmp.flush()
    >>> gr = pr.read_pairs(tmp.name)
    >>> list(gr.columns)
    ['Chromosome', 'Start', 'End', 'Strand', 'OtherChromosome', 'OtherStart', 'OtherEnd', 'OtherStrand', 'ReadID']
    >>> gr["Start"].tolist()
    [99, 149]
    >>> gr["OtherStart"].tolist()
    [199, 899]
    >>> pr.read_pairs(tmp.name, anchor="2")["Start"].tolist()
    [199, 899]

    """
    if anchor not in ("1", "2"):
        msg = f"anchor must be '1' or '2', got {anchor!r}"
        raise ValueError(msg)

    path = Path(f)
    read_ids: list[str] = []
    chr1: list[str] = []
    pos1: list[int] = []
    chr2: list[str] = []
    pos2: list[int] = []
    strand1: list[str] = []
    strand2: list[str] = []

    import gzip

    opener = gzip.open if path.name.endswith(".gz") else open

    n = 0
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < _PAIRS_MIN_FIELDS:
                continue
            read_ids.append(parts[0])
            chr1.append(parts[1])
            pos1.append(int(parts[2]) - 1)
            chr2.append(parts[3])
            pos2.append(int(parts[4]) - 1)
            strand1.append(parts[5])
            strand2.append(parts[6])
            n += 1
            if nrows is not None and n >= nrows:
                break

    if anchor == "1":
        a_chrom, a_pos, a_strand = chr1, pos1, strand1
        o_chrom, o_pos, o_strand = chr2, pos2, strand2
    else:
        a_chrom, a_pos, a_strand = chr2, pos2, strand2
        o_chrom, o_pos, o_strand = chr1, pos1, strand1

    df = pd.DataFrame(
        {
            "Chromosome": pd.Categorical(a_chrom),
            "Start": a_pos,
            "End": [p + 1 for p in a_pos],
            "Strand": pd.Categorical(a_strand),
            "OtherChromosome": pd.Categorical(o_chrom),
            "OtherStart": o_pos,
            "OtherEnd": [p + 1 for p in o_pos],
            "OtherStrand": pd.Categorical(o_strand),
            "ReadID": read_ids,
        },
    )

    return ensure_pyranges(df)
