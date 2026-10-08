import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import gtfreader
import numpy as np
import pandas as pd
from natsort import natsorted  # type: ignore[import]

from pyranges1.core.options import option_manager
from pyranges1.core.pyranges_helpers import ensure_pyranges

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

    import pyarrow as pa

    from pyranges1.core.pyranges_main import PyRanges

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

_HASH = ord("#")


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

    If `pyarrow` is installed, the file is parsed on every core rather than one,
    which is 10-18x faster on large files; the result is identical either way.
    Install it with `pip install pyranges1[fast-io]` (also included in
    `[add-ons]` and `[all]`).

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
    names = columns[:ncols] if header != 0 else None

    # `nrows` stays on the pandas path: pyarrow.csv has no row limit, so reading
    # the whole file to throw most of it away would be slower, not faster.
    df = None if nrows is not None else _read_bed_pyarrow(path, names=names, header=header)
    if df is None:
        df = pd.read_csv(
            path,
            dtype={"Chromosome": "category", "Strand": "category"},
            nrows=nrows,
            header=header,
            names=names,
            sep="\t",
        )

    df.columns = pd.Index(columns[: df.shape[1]])

    # Whichever reader ran, the dtype contract is the same. The pyarrow path
    # usually gets the dictionary for free by asking for it up front; this makes
    # the guarantee hold either way.
    _as_sorted_categorical(df, ("Chromosome", "Strand"))

    return ensure_pyranges(df)


def _pyarrow_csv() -> "tuple | None":
    """Return `(pyarrow, pyarrow.csv)`, or None when pyarrow is not installed.

    Every fast path in this module goes through here, so there is exactly one
    place to disable, and the tests that run each reader both ways patch it.

    `pr.options.set_option("use_pyarrow", False)` also lands here. It is an
    option rather than a reader argument because the readers mirror
    polaranges', which parses through Polars and has no pyarrow to switch; a
    `use_pyarrow=` parameter would be a pyranges1-only argument on a shared
    signature, while options are free to differ between the two.
    """
    requested = option_manager.get_option("use_pyarrow")
    if requested is False:
        return None
    try:
        import pyarrow as pa
        from pyarrow import csv as pacsv
    except ImportError:
        if requested:
            msg = (
                'pr.options.set_option("use_pyarrow", True) was set but pyarrow is '
                "not installed. Install it with `pip install pyranges1[fast-io]`, or "
                "set the option back to None to fall back to the pandas parser."
            )
            raise ImportError(msg) from None
        return None
    return pa, pacsv


def _as_sorted_categorical(df: pd.DataFrame, columns: "tuple[str, ...]") -> pd.DataFrame:
    """Make `columns` categorical with pandas' lexicographically sorted order.

    Category *order* is not cosmetic: pandas groups and sorts a categorical by
    it, so `sort_values("Chromosome")` follows it. `astype` sorts, but pyarrow's
    dictionary is in order of first appearance, so without this the row order of
    a sort or a groupby would depend on whether pyarrow happened to be
    installed.
    """
    for column in columns:
        if column not in df.columns:
            continue
        values = df[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            categories = values.cat.categories
            if not categories.is_monotonic_increasing:
                df[column] = values.cat.reorder_categories(categories.sort_values())
        else:
            df[column] = values.astype("category")
    return df


def _skip_comment_rows(row) -> str:
    """Skip a whole-line comment; refuse any other ragged row.

    pandas drops `#` lines because of `comment="#"`; to pyarrow they are rows
    with the wrong number of fields. A ragged row that is not a comment -- the
    sequence lines of a `##FASTA` section, say -- is a file this path does not
    model, so it errors out and the caller falls back to pandas.
    """
    return "skip" if row.text.lstrip().startswith("#") else "error"


def _contains_hash(pa: "ModuleType", column: "pa.ChunkedArray") -> bool:
    """Report whether the byte `#` occurs anywhere in a text column.

    `comment="#"` truncates a line at a `#` in *any* position, not only at the
    start, so a file with one inside a field is read differently by the two
    parsers. Rather than model that, detect it and let pandas have the file.

    Only the raw character buffer is needed, so this runs at memory speed --
    about 0.03 s over a 764 MB column, against 0.7 s for
    `pyarrow.compute.match_substring`.
    """
    for chunk in column.chunks:
        # A dictionary column keeps its text in the (tiny) dictionary.
        values = chunk.dictionary if pa.types.is_dictionary(chunk.type) else chunk
        if not (pa.types.is_string(values.type) or pa.types.is_large_string(values.type)):
            continue
        data = values.buffers()[2]
        if data is None:
            continue
        if (np.frombuffer(data, dtype=np.uint8) == _HASH).any():
            return True
    return False


def _widen_empty_columns(pa: "ModuleType", table: "pa.Table") -> "pa.Table":
    """Give a wholly-empty column the float64 pandas would have inferred.

    An empty field is null on both sides, but a column that is *nothing but*
    empty fields has no type to infer: pandas calls it float64 full of NaN,
    pyarrow calls it `null`, and null converts to an object column of None.
    """
    fields = [field.with_type(pa.float64()) if pa.types.is_null(field.type) else field for field in table.schema]
    if all(field.type == original.type for field, original in zip(fields, table.schema, strict=True)):
        return table
    return table.cast(pa.schema(fields))


def _arrow_read_csv(
    path: Path,
    *,
    column_names: list[str] | None,
    dictionary_columns: "tuple[str, ...]" = (),
    integer_columns: "tuple[str, ...]" = (),
    skip_comment_lines: bool = False,
) -> "pa.Table | None":
    """Parse a tab-separated file with `pyarrow.csv`, or return None for pandas.

    pandas' C parser is single-threaded, and on a large file it is most of the
    cost of reading. `pyarrow.csv` parses on every core and hands back an Arrow
    table; asking for the categorical columns as dictionaries up front means
    they arrive without a second pass. Measured on a 12-core machine:

        rows       file        pandas   pyarrow
        10^7       hg38          1.27      0.12
        10^8       hg38         12.93      1.19
        10^8       proteome     51.49      3.16

    Returns None whenever pandas should read the file instead. pandas is the
    reference implementation, so anything this path does not model exactly is
    handed back rather than approximated.
    """
    modules = _pyarrow_csv()
    if modules is None:
        return None
    pa, pacsv = modules

    dictionary: object = pa.dictionary(pa.int32(), pa.string())
    column_types: dict[str, object] = dict.fromkeys(dictionary_columns, dictionary)
    # Pinned rather than inferred: inference turns a coordinate too large for
    # int64 into a float, quietly rounding it, where pandas keeps the integer.
    # Pinned, pyarrow refuses the file and pandas gets it.
    column_types.update({name: pa.int64() for name in integer_columns})

    parse_options = pacsv.ParseOptions(
        delimiter="\t",
        **({"invalid_row_handler": _skip_comment_rows} if skip_comment_lines else {}),
    )
    read_options = (
        pacsv.ReadOptions(column_names=column_names, use_threads=True)
        if column_names is not None
        # The file names its own columns; let pyarrow read them.
        else pacsv.ReadOptions(use_threads=True)
    )

    try:
        table = pacsv.read_csv(
            path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=pacsv.ConvertOptions(
                column_types=column_types,
                # pandas applies its NA list to text columns as well, so an
                # empty field reads back as NaN rather than "". pyarrow only
                # does that when asked, and its default null list is the same
                # set of spellings.
                strings_can_be_null=True,
            ),
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, UnicodeDecodeError, OSError):
        # Ragged rows, an encoding pandas tolerates, an unreadable compression:
        # all of these are pandas' problem to solve, not reasons to fail.
        return None

    if skip_comment_lines and any(_contains_hash(pa, table.column(name)) for name in table.column_names):
        return None

    # A file with no data lines is not worth modelling: pandas' empty frame
    # carries the dtypes of the chunk it never filled, which the Arrow schema
    # does not reproduce.
    if table.num_rows == 0:
        return None

    return _widen_empty_columns(pa, table)


def _arrow_to_pandas(table: "pa.Table") -> pd.DataFrame:
    """Convert to pandas using pandas' own spelling of a missing value.

    pyarrow puts `None` in an object column where pandas' reader puts `NaN`.
    Under pandas 3 both are NA and the difference does not arise; under pandas 2
    they are distinguishable, and pandas already warns that a future version
    will stop treating them as equal.

    `null_count` is Arrow metadata, so a column without nulls costs nothing.
    """
    df = table.to_pandas()
    for name in df.columns:
        if df[name].dtype == object and table.column(name).null_count:
            df[name] = df[name].fillna(np.nan)
    return df


def _read_bed_pyarrow(
    path: Path,
    *,
    names: list[str] | None,
    header: int | None,
) -> "pd.DataFrame | None":
    """Read a BED file with `pyarrow.csv`, or return None to use pandas."""
    named = names if header != 0 and names is not None else []
    table = _arrow_read_csv(
        path,
        column_names=names if header != 0 else None,
        dictionary_columns=tuple(c for c in ("Chromosome", "Strand") if c in named),
        integer_columns=tuple(c for c in ("Start", "End") if c in named),
    )
    return None if table is None else _arrow_to_pandas(table)


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
        Repeated attributes are joined with commas. GENCODE gives most rows several
        "tag" attributes, so pass True to keep e.g. Ensembl_canonical and MANE_Select
        (see pyranges1.genes.select_transcripts).

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
        # gtfreader parses the fixed GTF columns through its own pyarrow path,
        # which `_pyarrow_csv` above cannot reach, so the option is forwarded
        # rather than applied here.
        use_pyarrow=cast("bool | None", option_manager.get_option("use_pyarrow")),
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


def _compiled_gff3_parser() -> "Callable | None":
    """Return gtfreader's compiled GFF3 attribute parser, or None without it.

    Imported from the extension module rather than the package, because that is
    the import that actually fails when the extension was not built or gtfreader
    predates the parser -- the package re-exports a stub that raises on call.
    Tests patch this to force the Python parser.
    """
    try:
        from gtfreader._parser import parse_gff3_chunk_columns
    except ImportError:
        return None
    return parse_gff3_chunk_columns


def to_rows_gff3(anno: pd.Series) -> pd.DataFrame:
    """Parse GFF3 attribute column into a dataframe of attribute columns.

    With the tabular parse handed to pyarrow, this is about three quarters of a
    GFF3 read. The Python path below builds a dict per row and lets pandas
    reconcile them, which costs as much again as the parsing does; the compiled
    parser fills one list per column and skips `from_records` entirely. 2.37x
    at 10^6 rows, and the same frame either way.
    """
    # A row with no attribute at all -- an empty ninth field, or the sequence
    # lines of a `##FASTA` section, which arrive as ragged rows padded with NaN
    # -- has nothing to expand. That is not a parse error.
    normalized = anno.where(anno.notna(), "")

    parse = _compiled_gff3_parser()
    if parse is not None:
        return pd.DataFrame(parse(normalized.to_numpy(copy=False)), index=anno.index)

    rowdicts = [to_keys_and_values(line) for line in normalized]

    return pd.DataFrame.from_records(rowdicts).set_index(anno.index)


def to_keys_and_values(line: str) -> dict[str, str]:
    """Parse GFF3 attribute column."""
    # Split once: a value may contain `=`, and GFF3 in the wild does not always
    # percent-encode it. Segments without one are not tag=value pairs and are
    # skipped, which is also what makes an empty attribute an empty dict.
    return dict(it.split("=", 1) for it in line.rstrip("; ").split(";") if "=" in it)


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

    If `pyarrow` is installed, the nine fixed columns are parsed on every core
    rather than one; the result is identical either way. Install it with
    `pip install pyranges1[fast-io]`. Expanding the attribute column is the bulk
    of the work and is unaffected, so the gain is around 1.3x, not the 10-18x
    `read_bed` sees.

    See Also
    --------
    pyranges1.read_gtf : read files in the Gene Transfer Format

    """
    path = Path(f)

    dtypes: Mapping = {"Chromosome": "category", "Feature": "category", "Strand": "category"}

    names = ["Chromosome", "Source", "Feature", "Start", "End", "Score", "Strand", "Frame", "Attribute"]
    chunksize = int(1e5)

    # `nrows` stays on the pandas path: pyarrow.csv has no row limit, so reading
    # the whole file to throw most of it away would be slower, not faster.
    table = (
        None
        if nrows is not None
        else _arrow_read_csv(
            path,
            column_names=names,
            dictionary_columns=tuple(dtypes),
            integer_columns=("Start", "End"),
            skip_comment_lines=True,
        )
    )

    if table is not None:
        dfs = _gff3_frames_from_arrow(table, chunksize)
    else:
        df_iter = pd.read_csv(
            path,
            comment="#",
            sep="\t",
            header=None,
            names=names,
            dtype=dtypes,
            chunksize=chunksize,
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

    # One dtype whatever the file. Without this it is an accident of chunking:
    # pd.concat demotes a categorical to object when the chunks carry different
    # categories, so on a coordinate-sorted GFF3 Chromosome came back object
    # while Feature came back category, and the answer depended on a chunk size
    # that is a tuning knob rather than a contract.
    _as_sorted_categorical(df, tuple(dtypes))

    return ensure_pyranges(df)


def _gff3_frames_from_arrow(table: "pa.Table", chunksize: int) -> list[pd.DataFrame]:
    """Expand attributes a chunk at a time, as the pandas path does.

    Slicing the Arrow table rather than converting it whole keeps the raw
    attribute strings of one chunk alive at a time.
    """
    dfs = []
    for start in range(0, table.num_rows, chunksize):
        df = _arrow_to_pandas(table.slice(start, chunksize))
        # pandas numbers its chunks continuously across the file; a slice
        # converted on its own would restart at zero and the concat would end up
        # with a repeated index.
        df.index = pd.RangeIndex(start, start + len(df))
        extra = to_rows_gff3(df.Attribute.astype(str))
        _df = df.drop("Attribute", axis=1)
        extra = extra.set_index(_df.index)
        dfs.append(pd.concat([_df, extra], axis=1, sort=False))
    return dfs


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
    # Chromosome as categorical, consistent with the other readers (read_bed / read_gtf /
    # read_pairs). Without this it follows pandas' string default — `object` on pandas 2.x,
    # the new `str` dtype on pandas 3.0 — which breaks the doctype-sensitive doctest and
    # diverges from the rest of the library.
    df["Chromosome"] = df["Chromosome"].astype("category")
    return ensure_pyranges(df)


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
