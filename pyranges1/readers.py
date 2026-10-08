import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
