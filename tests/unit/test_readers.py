"""Every reader, both with and without pyarrow, must return the same frame.

pandas is the reference implementation: whatever it returns today is the
behaviour users have. The parity tests compare strictly -- dtypes and category
*order* included, because pandas groups and sorts a categorical by its category
order, so an order-of-appearance dictionary silently reorders `groupby` output
and `sort_values`.

`read_gtf` is included even though the parse lives in gtfreader: pyranges1 is
where the two libraries meet, and a gtfreader release that broke parity should
fail here too.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import pyranges1 as pr
from gtfreader import readers as gtf_readers
from pyranges1 import readers
from pyranges1.readers import to_keys_and_values, to_rows_gff3

DATA = Path(pr.__file__).parent / "data"

BED_CORPUS = {
    "bed3": "chr1\t1\t10\nchr2\t20\t30\n",
    "bed6": "chr1\t1\t10\tn1\t5\t+\nchr2\t20\t30\tn2\t7\t-\n",
    "bed9": "chr1\t1\t10\tn1\t5\t+\t1\t10\t\nchr2\t20\t30\tn2\t7\t-\t20\t30\t\n",
    "bed12": (
        "chr1\t1\t10\tn1\t5\t+\t1\t10\t0,0,0\t1\t9,\t0,\n"
        "chr2\t20\t30\tn2\t7\t-\t20\t30\t0,0,0\t1\t10,\t0,\n"
    ),
    "unsorted_chromosomes": "chr9\t1\t10\nchr1\t2\t20\nchr22\t3\t30\nchr2\t4\t40\n",
    "with_header": "Chromosome\tStart\tEnd\nchr1\t1\t10\nchr2\t20\t30\n",
    "single_row": "chr1\t1\t10\n",
    "no_trailing_newline": "chr1\t1\t10\nchr2\t20\t30",
    "empty_name_column": "chr1\t1\t10\t\nchr2\t20\t30\t\n",
    "crlf": "chr1\t1\t10\r\nchr2\t20\t30\r\n",
    "hash_in_a_field": "chr1\t1\t10\tname#1\nchr2\t20\t30\tname2\n",
    "many_rows": "".join(f"chr{(i % 5) + 1}\t{i}\t{i + 10}\n" for i in range(300)),
}

GFF3_LINE = "chr1\tsource\tgene\t11869\t14409\t.\t+\t.\tID=g1;Name=DDX11L1\n"

GFF3_CORPUS = {
    "bare": GFF3_LINE,
    "leading_directives": "##gff-version 3\n##sequence-region chr1 1 100\n" + GFF3_LINE,
    "separators_in_body": (
        "##gff-version 3\n"
        + GFF3_LINE
        + "###\n"
        + "chr2\tsource\texon\t20\t30\t.\t-\t.\tID=e1;Parent=g1\n"
        + "##sequence-region chr3 1 100\n"
        + "chr3\tsource\tCDS\t40\t50\t.\t+\t0\tID=c1\n"
    ),
    "unsorted_chromosomes": (
        "chr9\tsrc\tgene\t1\t10\t.\t+\t.\tID=a\n"
        "chr1\tsrc\tgene\t2\t20\t.\t-\t.\tID=b\n"
        "chr22\tsrc\tgene\t3\t30\t.\t+\t.\tID=c\n"
        "chr2\tsrc\tgene\t4\t40\t.\t-\t.\tID=d\n"
    ),
    "many_rows": "".join(
        f"chr{(i % 7) + 1}\tsrc\tgene\t{i + 1}\t{i + 100}\t.\t+\t.\tID=g{i};Name=n{i}\n" for i in range(400)
    ),
    "empty_score_column": "chr1\tsrc\tgene\t1\t10\t\t+\t.\tID=a\n" * 3,
    "numeric_score": "chr1\tsrc\tgene\t1\t10\t42\t+\t.\tID=a\n" * 3,
    "utf8_attribute": "chr1\tsrc\tgene\t1\t10\t.\t+\t.\tID=éèü\n",
    "crlf": "chr1\tsrc\tgene\t1\t10\t.\t+\t.\tID=a\r\n" * 3,
}


@pytest.fixture(params=["pandas", "pyarrow"])
def mode(request, monkeypatch) -> str:
    """Run the test body once per reader.

    Both gates are patched: pyranges1 owns read_bed and read_gff3, gtfreader
    owns the GTF parse, and "without pyarrow" has to mean both.
    """
    if request.param == "pandas":
        disable_pyarrow(monkeypatch)
    elif readers._pyarrow_csv() is None:
        pytest.skip("pyarrow is not installed")
    return request.param


@pytest.fixture
def requires_pyarrow():
    if readers._pyarrow_csv() is None:
        pytest.skip("pyarrow is not installed")


@pytest.fixture(params=["python", "compiled"])
def attribute_parser(request, monkeypatch) -> str:
    """Run the test body once per GFF3 attribute parser.

    The compiled one lives in gtfreader and the Python one in pyranges1; they
    have to produce the same frame, column order included.
    """
    if request.param == "python":
        monkeypatch.setattr(readers, "_compiled_gff3_parser", lambda: None)
    elif readers._compiled_gff3_parser() is None:
        pytest.skip("gtfreader's compiled extension is not available")
    return request.param


@pytest.fixture
def requires_compiled_parser():
    if readers._compiled_gff3_parser() is None:
        pytest.skip("gtfreader's compiled extension is not available")


def disable_pyarrow(monkeypatch) -> None:
    monkeypatch.setattr(readers, "_pyarrow_csv", lambda: None)
    monkeypatch.setattr(gtf_readers, "_pyarrow_csv", lambda: None)


def write(tmp_path: Path, contents: str, name: str, *, compress: bool = False) -> Path:
    path = tmp_path / (name + ".gz" if compress else name)
    if compress:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(contents)
    else:
        path.write_text(contents, encoding="utf-8")
    return path


def read_both(reader, path: Path, monkeypatch, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pandas frame, pyarrow frame) for the same file."""
    disable_pyarrow(monkeypatch)
    try:
        from_pandas = reader(path, **kwargs)
    finally:
        monkeypatch.undo()
    return from_pandas, reader(path, **kwargs)


# --------------------------------------------------------------------------
# read_bed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BED_CORPUS))
@pytest.mark.parametrize("compress", [False, True], ids=["plain", "gzip"])
def test_read_bed_readers_agree(tmp_path, monkeypatch, requires_pyarrow, name, compress):
    path = write(tmp_path, BED_CORPUS[name], "test.bed", compress=compress)
    from_pandas, from_arrow = read_both(pr.read_bed, path, monkeypatch)
    assert_frame_equal(from_pandas, from_arrow)


@pytest.mark.parametrize("name", sorted(p.name for p in DATA.glob("*.bed")) + ["ucsc_human.bed.gz"])
def test_read_bed_readers_agree_on_bundled_files(monkeypatch, requires_pyarrow, name):
    from_pandas, from_arrow = read_both(pr.read_bed, DATA / name, monkeypatch)
    assert_frame_equal(from_pandas, from_arrow)


def test_read_bed_categories_are_sorted(tmp_path, mode):
    """The regression test for the bug this contract exists to prevent.

    pyarrow's dictionary is in order of first appearance and pandas' is sorted,
    so before this the row order of `sort_values("Chromosome")` depended on
    whether pyarrow happened to be installed.
    """
    path = write(tmp_path, BED_CORPUS["unsorted_chromosomes"], "test.bed")
    categories = list(pr.read_bed(path)["Chromosome"].cat.categories)
    assert categories == ["chr1", "chr2", "chr22", "chr9"]


def test_read_bed_sort_order_does_not_depend_on_pyarrow(tmp_path, monkeypatch, requires_pyarrow):
    path = write(tmp_path, BED_CORPUS["unsorted_chromosomes"], "test.bed")
    from_pandas, from_arrow = read_both(pr.read_bed, path, monkeypatch)
    assert_frame_equal(from_pandas.sort_values("Chromosome"), from_arrow.sort_values("Chromosome"))


@pytest.mark.parametrize("nrows", [1, 2, 5])
def test_read_bed_nrows_agrees(tmp_path, monkeypatch, requires_pyarrow, nrows):
    path = write(tmp_path, BED_CORPUS["many_rows"], "test.bed")
    from_pandas, from_arrow = read_both(pr.read_bed, path, monkeypatch, nrows=nrows)
    assert len(from_arrow) == nrows
    assert_frame_equal(from_pandas, from_arrow)


def test_read_bed_fast_path_is_really_taken(tmp_path, requires_pyarrow):
    """Without this the pyarrow half of every test above could be a no-op."""
    path = write(tmp_path, BED_CORPUS["bed6"], "test.bed")
    names = ["Chromosome", "Start", "End", "Name", "Score", "Strand"]
    assert readers._read_bed_pyarrow(path, names=names, header=None) is not None


@pytest.mark.parametrize("column", ["ItemRGB"])
def test_read_bed_nulls_use_pandas_spelling(monkeypatch, requires_pyarrow, column):
    """pyarrow puts None in an object column where pandas' reader puts NaN.

    Indistinguishable under pandas 3, where both are NA, and distinguishable
    under pandas 2 -- which also warns that a future version will stop treating
    them as equal. `ucsc_human.bed.gz` has 390 of them in its empty ninth field.
    """
    from_pandas, from_arrow = read_both(pr.read_bed, DATA / "ucsc_human.bed.gz", monkeypatch)
    assert from_pandas[column].isna().sum() > 0
    assert {type(v) for v in from_pandas[column][from_pandas[column].isna()]} == {
        type(v) for v in from_arrow[column][from_arrow[column].isna()]
    }


def test_read_bed_dtype_contract(tmp_path, mode):
    path = write(tmp_path, BED_CORPUS["bed6"], "test.bed")
    frame = pr.read_bed(path)
    assert isinstance(frame["Chromosome"].dtype, pd.CategoricalDtype)
    assert isinstance(frame["Strand"].dtype, pd.CategoricalDtype)


# --------------------------------------------------------------------------
# read_gff3
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GFF3_CORPUS))
@pytest.mark.parametrize("compress", [False, True], ids=["plain", "gzip"])
def test_read_gff3_readers_agree(tmp_path, monkeypatch, requires_pyarrow, name, compress):
    path = write(tmp_path, GFF3_CORPUS[name], "test.gff3", compress=compress)
    from_pandas, from_arrow = read_both(pr.read_gff3, path, monkeypatch)
    assert_frame_equal(from_pandas, from_arrow)


def test_read_gff3_readers_agree_on_the_bundled_file(monkeypatch, requires_pyarrow):
    from_pandas, from_arrow = read_both(pr.read_gff3, DATA / "ncbi.gff.gz", monkeypatch)
    assert_frame_equal(from_pandas, from_arrow)


def test_read_gff3_fast_path_is_really_taken(tmp_path, requires_pyarrow):
    path = write(tmp_path, GFF3_CORPUS["separators_in_body"], "test.gff3")
    names = ["Chromosome", "Source", "Feature", "Start", "End", "Score", "Strand", "Frame", "Attribute"]
    table = readers._arrow_read_csv(
        path,
        column_names=names,
        dictionary_columns=("Chromosome", "Feature", "Strand"),
        integer_columns=("Start", "End"),
        skip_comment_lines=True,
    )
    assert table is not None


@pytest.mark.parametrize(
    ("name", "contents"),
    [
        ("hash inside an attribute", "chr1\tsrc\tgene\t1\t10\t.\t+\t.\tID=a;note=grade #2\n"),
        ("fasta section", GFF3_LINE + "##FASTA\n>chr1\nACGTACGT\n"),
        ("directive with nine fields", "#chr1\tx\ty\t1\t2\t.\t+\t.\tz\n" + GFF3_LINE),
    ],
)
def test_read_gff3_fast_path_declines(tmp_path, requires_pyarrow, name, contents):
    path = write(tmp_path, contents, "test.gff3")
    names = ["Chromosome", "Source", "Feature", "Start", "End", "Score", "Strand", "Frame", "Attribute"]
    table = readers._arrow_read_csv(
        path,
        column_names=names,
        dictionary_columns=("Chromosome", "Feature", "Strand"),
        integer_columns=("Start", "End"),
        skip_comment_lines=True,
    )
    assert table is None, name


@pytest.mark.parametrize(
    ("name", "contents"),
    [
        ("hash inside an attribute", "chr1\tsrc\tgene\t1\t10\t.\t+\t.\tID=a;note=grade #2\n"),
        ("fasta section", GFF3_LINE + "##FASTA\n>chr1\nACGTACGT\n"),
    ],
)
def test_read_gff3_declined_files_still_read(tmp_path, monkeypatch, requires_pyarrow, name, contents):
    """Declining means pandas reads it, not that the read fails."""
    path = write(tmp_path, contents, "test.gff3")
    from_pandas, from_arrow = read_both(pr.read_gff3, path, monkeypatch)
    assert_frame_equal(from_pandas, from_arrow)


def test_read_gff3_reads_a_file_with_a_fasta_section(tmp_path, mode):
    """A `##FASTA` section used to crash the reader.

    Its sequence lines are ragged rows, so pandas pads them with NaN, and
    `to_keys_and_values` called `rstrip` on the NaN. A row with no attributes
    has nothing to expand; that is not a parse error.
    """
    path = write(tmp_path, GFF3_LINE + "##FASTA\n>chr1\nACGTACGT\n", "test.gff3")
    frame = pr.read_gff3(path)
    assert frame.iloc[0]["ID"] == "g1"
    assert pd.isna(frame.iloc[-1]["ID"])


def test_read_gff3_reads_an_empty_attribute_field(tmp_path, mode):
    path = write(tmp_path, "chr1\tsrc\tgene\t1\t10\t.\t+\t.\t\n" + GFF3_LINE, "test.gff3")
    frame = pr.read_gff3(path)
    assert pd.isna(frame.iloc[0]["ID"])
    assert frame.iloc[1]["ID"] == "g1"


def test_read_gff3_keeps_equals_signs_inside_a_value(tmp_path, mode):
    """`dict(it.split("="))` raised on these; GFF3 does not always encode them."""
    path = write(tmp_path, "chr1\tsrc\tgene\t1\t10\t.\t+\t.\tID=a;Note=x=y=z\n", "test.gff3")
    assert pr.read_gff3(path).iloc[0]["Note"] == "x=y=z"


def test_read_gff3_directives_are_dropped_not_parsed(tmp_path, mode):
    path = write(tmp_path, GFF3_CORPUS["separators_in_body"], "test.gff3")
    frame = pr.read_gff3(path)
    assert list(frame["Chromosome"]) == ["chr1", "chr2", "chr3"]


def test_read_gff3_dtype_contract(tmp_path, mode, attribute_parser):
    path = write(tmp_path, GFF3_CORPUS["many_rows"], "test.gff3")
    frame = pr.read_gff3(path)
    for column in ("Chromosome", "Feature", "Strand"):
        assert isinstance(frame[column].dtype, pd.CategoricalDtype), column


def test_read_gff3_index_is_continuous(tmp_path, mode):
    path = write(tmp_path, GFF3_CORPUS["many_rows"], "test.gff3")
    frame = pr.read_gff3(path)
    assert list(frame.index) == list(range(len(frame)))


@pytest.mark.parametrize("chunksize", [1, 7, 399, 100_000])
def test_read_gff3_arrow_chunking_is_invisible(tmp_path, requires_pyarrow, chunksize):
    """`read_gff3` hard-codes a 100k chunk, so end-to-end tests never split.

    A file has to exceed that for the chunk seam to appear at all, which is
    every real GFF3 and no affordable test. Driving the helper directly is what
    actually covers it: without a continuous index the concat below would carry
    a repeated one.
    """
    path = write(tmp_path, GFF3_CORPUS["many_rows"], "test.gff3")
    names = ["Chromosome", "Source", "Feature", "Start", "End", "Score", "Strand", "Frame", "Attribute"]
    table = readers._arrow_read_csv(
        path,
        column_names=names,
        dictionary_columns=("Chromosome", "Feature", "Strand"),
        integer_columns=("Start", "End"),
        skip_comment_lines=True,
    )
    frame = pd.concat(readers._gff3_frames_from_arrow(table, chunksize), sort=False)
    assert list(frame.index) == list(range(table.num_rows))
    assert not frame.index.has_duplicates


def test_read_gff3_start_is_zero_based(tmp_path, mode):
    path = write(tmp_path, GFF3_LINE, "test.gff3")
    assert pr.read_gff3(path).iloc[0]["Start"] == 11868


@pytest.mark.parametrize("nrows", [1, 2])
def test_read_gff3_nrows_agrees(tmp_path, monkeypatch, requires_pyarrow, nrows):
    path = write(tmp_path, GFF3_CORPUS["many_rows"], "test.gff3")
    from_pandas, from_arrow = read_both(pr.read_gff3, path, monkeypatch, nrows=nrows)
    assert len(from_arrow) == nrows
    assert_frame_equal(from_pandas, from_arrow)


# --------------------------------------------------------------------------
# The two GFF3 attribute parsers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GFF3_CORPUS))
def test_gff3_attribute_parsers_agree(tmp_path, monkeypatch, requires_compiled_parser, name):
    path = write(tmp_path, GFF3_CORPUS[name], "test.gff3")
    monkeypatch.setattr(readers, "_compiled_gff3_parser", lambda: None)
    from_python = pr.read_gff3(path)
    monkeypatch.undo()
    assert_frame_equal(from_python, pr.read_gff3(path))


@pytest.mark.parametrize(
    "attribute",
    [
        "ID=a;Name=b",
        "ID=a;Name=b;",
        "ID=a; Name=b",
        "ID=a;;Name=b",
        "novalue;ID=a",
        "ID=a=b=c",
        "",
        ";;;",
        "ID=a;ID=b",
        "ID=;Name=b",
        "=v;ID=a",
        "ID=éèü",
    ],
)
def test_gff3_attribute_parsers_agree_on_edge_cases(requires_compiled_parser, attribute):
    """The compiled parser has to reproduce `to_keys_and_values` exactly.

    Including where it is odd: `ID=a; Name=b` really does yield a key of
    `" Name"`, because nothing trims whitespace around a key.
    """
    series = pd.Series([attribute, "ID=z"])
    from_compiled = to_rows_gff3(series)
    rowdicts = [to_keys_and_values(line) for line in series]
    from_python = pd.DataFrame.from_records(rowdicts).set_index(series.index)
    assert_frame_equal(from_compiled, from_python)


def test_gff3_compiled_parser_is_really_used(requires_compiled_parser):
    assert readers._compiled_gff3_parser() is not None


def test_gff3_reader_works_without_the_compiled_parser(tmp_path, monkeypatch):
    monkeypatch.setattr(readers, "_compiled_gff3_parser", lambda: None)
    path = write(tmp_path, GFF3_CORPUS["many_rows"], "test.gff3")
    assert len(pr.read_gff3(path)) == 400


# --------------------------------------------------------------------------
# read_gtf, which pyranges1 gets from gtfreader
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["ensembl.gtf", "ensembl_human.gtf.gz", "gencode_human.gtf.gz"])
def test_read_gtf_readers_agree_on_bundled_files(monkeypatch, requires_pyarrow, name):
    from_pandas, from_arrow = read_both(pr.read_gtf, DATA / name, monkeypatch)
    assert_frame_equal(from_pandas, from_arrow)


def test_read_gtf_dtype_contract(mode):
    frame = pr.read_gtf(DATA / "gencode_human.gtf.gz")
    for column in ("Chromosome", "Source", "Feature", "Strand", "Frame"):
        assert isinstance(frame[column].dtype, pd.CategoricalDtype), column


# --------------------------------------------------------------------------
# The fallback is a supported state, not an error path
# --------------------------------------------------------------------------


@pytest.fixture
def no_pyarrow_import(monkeypatch):
    """Make `import pyarrow` fail, as it does when pyarrow is not installed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            msg = "No module named 'pyarrow'"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_gate_reports_missing_pyarrow(no_pyarrow_import):
    assert readers._pyarrow_csv() is None


@pytest.mark.parametrize(
    ("reader", "name", "contents"),
    [
        (pr.read_bed, "test.bed", BED_CORPUS["bed6"]),
        (pr.read_gff3, "test.gff3", GFF3_CORPUS["separators_in_body"]),
    ],
)
def test_readers_work_without_pyarrow(tmp_path, no_pyarrow_import, reader, name, contents):
    path = write(tmp_path, contents, name)
    assert len(reader(path)) > 0
