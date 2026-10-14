import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from natsort import natsorted  # type: ignore[import]
from pandas.core.frame import DataFrame

from pyranges1.core.names import BIGWIG_SCORE_COL, CHROM_COL, END_COL, PANDAS_COMPRESSION_TYPE, START_COL
from pyranges1.core.pyranges_helpers import ensure_pyranges
from pyranges1.core.pyranges_main import PyRanges

if TYPE_CHECKING:
    from runlength import RleBatch  # type: ignore[import-not-found]

GTF_COLUMNS_TO_PYRANGES = {
    "seqname": "Chromosome",
    "source": "Source",
    "feature": "Feature",
    "start": "Start",
    "end": "End",
    "score": "Score",
    "strand": "Strand",
    "frame": "Frame",
}

GFF3_COLUMNS_TO_PYRANGES = GTF_COLUMNS_TO_PYRANGES.copy()
GFF3_COLUMNS_TO_PYRANGES["phase"] = GFF3_COLUMNS_TO_PYRANGES.pop("frame")

_ordered_gtf_columns = [
    "seqname",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attribute",
]
_ordered_gff3_columns = [
    "seqname",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "phase",
    "attribute",
]


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _fill_missing(df: DataFrame, all_columns: list[str]) -> DataFrame:
    columns = list(df.columns)

    if set(columns).intersection(set(all_columns)) == set(all_columns):
        return df[all_columns]
    missing = set(all_columns) - set(columns)
    missing_idx = {all_columns.index(m): m for m in missing}
    not_missing = set(columns).intersection(set(all_columns))
    not_missing_ordered = sorted(not_missing, key=all_columns.index)
    outdf = df[not_missing_ordered]

    for idx, _missing in sorted(missing_idx.items()):
        outdf.insert(idx, _missing, ".")

    return outdf


def _bed(df: DataFrame, *, keep: bool) -> DataFrame:
    bed_columns = ["Chromosome", "Start", "End", "Name", "Score", "Strand"]

    outdf = _fill_missing(df, bed_columns)
    if not keep:
        return outdf

    # BED columns are positional, so the BED12 fields go in their places whatever the
    # frame's column order, up to the last one present; the rest follow as extras.
    bed12 = _BED_EXTRA_ORDER[3:]
    present = [c for c in bed12 if c in df.columns]
    standard = bed12[: bed12.index(present[-1]) + 1] if present else []
    defaults = {"ThickStart": df[START_COL], "ThickEnd": df[END_COL], "ItemRGB": "0"}
    filled = {}
    for column in standard:
        if column in df.columns:
            filled[column] = df[column]
        elif column in defaults:
            filled[column] = defaults[column]
        else:
            msg = f"BED columns are positional: {present[-1]} needs {column} before it."
            raise ValueError(msg)

    noncanonical = [c for c in df.columns if c not in bed_columns and c not in standard]
    return pd.concat([outdf, pd.DataFrame(filled, index=df.index), df[noncanonical]], axis=1)


def _resolve_compression(compression: PANDAS_COMPRESSION_TYPE) -> PANDAS_COMPRESSION_TYPE:
    """Resolve the ``compression`` argument of the interval writers.

    pandas reads ``None`` as *no compression, even for a .gz path*. The interval
    writers infer from the path suffix instead, so ``to_bed("x.bed.gz")`` writes
    gzip. Every other value is passed to pandas untouched.
    """
    return "infer" if compression is None else compression


def _to_gff_like(
    gr: PyRanges,
    out_format: Literal["gtf", "gff3"],
    path: Path | None = None,
    compression: PANDAS_COMPRESSION_TYPE = "infer",
    map_cols: dict | None = None,
) -> str | None:
    df = _pyranges_to_gtf_like(
        gr,
        out_format=out_format,
        map_cols=map_cols,
    )
    return df.to_csv(
        path,
        index=False,
        header=False,
        compression=_resolve_compression(compression),
        mode="w",
        sep="\t",
        quoting=csv.QUOTE_NONE,
    )


def _to_csv(
    self: PyRanges,
    path: Path | str | None = None,
    sep: str = ",",
    compression: PANDAS_COMPRESSION_TYPE = None,
    *,
    header: bool = True,
) -> str | None:
    gr = self

    if path:
        mode = "w+"
        for _, outdf in natsorted(gr.dfs.items()):
            outdf.to_csv(
                path,
                index=False,
                compression=compression,
                header=header,
                mode=mode,
                sep=sep,
                quoting=csv.QUOTE_NONE,
            )
            mode = "a"
            header = False
        return None
    return "".join(
        [
            outdf.to_csv(index=False, header=header, sep=sep, quoting=csv.QUOTE_NONE)
            for _, outdf in sorted(gr.dfs.items())
        ],
    )


def _to_bed(
    self: PyRanges,
    path: str | None = None,
    compression: PANDAS_COMPRESSION_TYPE = "infer",
    *,
    keep: bool = True,
    tabix: bool = False,
) -> str | None:
    df = _bed(self, keep=keep)

    if tabix:
        _to_bed_tabix(df, path, compression)
        return None

    return df.to_csv(
        path,
        index=False,
        header=False,
        compression=_resolve_compression(compression),
        mode="w+",
        sep="\t",
        quoting=csv.QUOTE_NONE,
    )


def _to_bed_tabix(df: DataFrame, path: str | Path | None, compression: PANDAS_COMPRESSION_TYPE) -> None:
    """Write `df` sorted and BGZF-compressed to `path`, and index it with tabix beside it."""
    if path is None or not str(path).endswith((".gz", ".bgz")) or compression not in ("infer", "gzip", None):
        msg = "tabix=True writes a bgzip-compressed file: give a path ending in .gz or .bgz, and no other compression."
        raise ValueError(msg)
    try:
        import pysam  # type: ignore[import]
    except ImportError:
        msg = "pysam must be installed to write a tabix-indexed BED file. Use `pip install pysam`."
        raise ImportError(msg) from None

    # tabix needs each chromosome contiguous and sorted by start.
    text = df.sort_values([CHROM_COL, START_COL, END_COL], kind="stable").to_csv(
        index=False, header=False, sep="\t", quoting=csv.QUOTE_NONE
    )
    with pysam.BGZFile(str(path), "wb", index=None) as fh:
        fh.write(text.encode())
    pysam.tabix_index(str(path), preset="bed", force=True)


def _nonzero_runs(rles: "RleBatch") -> pd.DataFrame:
    """Return the runs of every track as Chromosome, Start, End and Score, without zero-valued runs."""
    runs = rles.to_pandas().rename(columns={"start": START_COL, "end": END_COL, "value": BIGWIG_SCORE_COL})
    runs = runs.astype({CHROM_COL: "category", BIGWIG_SCORE_COL: np.float64})
    return runs[runs[BIGWIG_SCORE_COL] != 0]


def _log2(values: np.ndarray) -> np.ndarray:
    # Callbacks run on worker threads, where numpy's error state is the default.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(values)


def _to_bigwig(
    self: PyRanges,
    path: None,
    chromosome_sizes: PyRanges | pd.DataFrame | dict,
    value_col: str | None = None,
    *,
    divide: bool = False,
    rpm: bool = True,
    precomputed: bool = False,
    return_data: bool = False,
) -> PyRanges | None:
    try:
        import pyBigWig  # type: ignore[import]
    except ModuleNotFoundError:
        msg = "pybigwig must be installed to create bigwigs. Use `conda install -c bioconda pybigwig` or `pip install pybigwig` to install it."
        raise ImportError(msg) from None

    if precomputed:
        if value_col is None:
            msg = "precomputed=True requires value_col."
            raise ValueError(msg)
        if divide:
            msg = "divide=True is incompatible with precomputed=True."
            raise ValueError(msg)
        df = self.get_with_loc_columns(value_col).rename(columns={value_col: BIGWIG_SCORE_COL})
    elif not divide:
        df = _nonzero_runs(self.to_rle(rpm=rpm, strand=False, value_col=value_col))
    else:
        # rpm scales numerator and denominator alike, so it cancels in the ratio;
        # scaling first would only add rounding noise that splits equal ratios.
        numerator = self.to_rle(strand=False, value_col=value_col)
        denominator = self.to_rle(strand=False)
        ratio = (numerator / denominator).map(_log2, backend="numpy")
        df = _nonzero_runs(ratio)
    gr = ensure_pyranges(df)
    gr = gr.remove_strand()
    gr = gr.sort_ranges()
    gr = gr.get_with_loc_columns(BIGWIG_SCORE_COL)
    unique_chromosomes = gr.chromosomes

    if return_data:
        return gr

    if not isinstance(chromosome_sizes, dict):
        size_df = chromosome_sizes
        chromosome_sizes = dict(zip(size_df[CHROM_COL], size_df[END_COL], strict=True))

    missing = [c for c in unique_chromosomes if c not in chromosome_sizes]
    if missing:
        msg = f"chromosome_sizes has no size for {missing}."
        raise ValueError(msg)
    header = [(c, int(chromosome_sizes[c])) for c in unique_chromosomes]

    bw = pyBigWig.open(path, "w")
    bw.addHeader(header)

    chromosomes = gr[CHROM_COL].tolist()
    starts = gr[START_COL].tolist()
    ends = gr[END_COL].tolist()
    values = gr[BIGWIG_SCORE_COL].tolist()

    bw.addEntries(chromosomes, starts, ends=ends, values=values)

    return None


def _pyranges_to_gtf_like(
    df: pd.DataFrame,
    out_format: Literal["gtf", "gff3"],
    map_cols: dict | None = None,
) -> pd.DataFrame:
    if out_format == "gtf":
        all_columns = _ordered_gtf_columns[:-1]
        # first: gff column to pyranges column
        rename_columns = GTF_COLUMNS_TO_PYRANGES.copy()
    elif out_format == "gff3":
        all_columns = _ordered_gff3_columns[:-1]
        # first: gff column to pyranges column
        rename_columns = GFF3_COLUMNS_TO_PYRANGES.copy()
    else:
        msg = f"Invalid output format: {out_format}. Must be one of 'gtf' or 'gff3'."
        raise ValueError(msg)

    map_cols = map_cols or {}
    valid_keys = set(rename_columns) | {"attribute"}
    invalid_map_cols = set(map_cols) - valid_keys
    if invalid_map_cols:
        msg = f"Invalid column mapping: {invalid_map_cols}. Must be one of {set(rename_columns) | {'attribute'}}."
        raise ValueError(msg)

    rename_columns.update(map_cols)
    # from here on, rename_columns is: pyranges column to gff column
    rename_columns = {v: k for k, v in rename_columns.items()}

    df = df.rename(columns=rename_columns)
    df.loc[:, "start"] = df.start + 1

    columns = list(df.columns)

    # filling missing columns with "."
    outdf = _fill_missing(df, all_columns)

    if "attribute" not in map_cols:
        _rest = set(df.columns) - set(all_columns)
        rest = sorted(_rest, key=columns.index)
        outdf.insert(outdf.shape[1], column="attribute", value=_attribute_column(df[rest], out_format))
    else:
        outdf.insert(outdf.shape[1], column="attribute", value=df["attribute"].copy())

    return outdf


def _attribute_column(rest_df: pd.DataFrame, out_format: Literal["gtf", "gff3"]) -> "pd.Series[str]":
    """Build the GTF/GFF3 attribute column from the remaining columns, leaving out missing values.

    Whole columns at a time. Each value is written as its column's dtype prints it, so
    a nullable integer is "1", not "1.0".
    """
    attribute = pd.Series("", index=rest_df.index, dtype=object)
    for name in rest_df.columns:
        column = rest_df[name]
        text = column.astype(str)
        entry = (f'{name} "' + text + '"; ') if out_format == "gtf" else (f"{name}=" + text + ";")
        attribute = attribute + entry.where(column.notna(), "")
    return attribute.str.replace(" $" if out_format == "gtf" else ";$", "", regex=True)


# ── narrowPeak / pairs / bigBed writers (companions to the readers) ─────────────

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


# What the ENCODE formats write for a value that is not available.
_PEAK_DEFAULTS = {"Name": ".", "Score": 0, "Strand": ".", "SignalValue": -1, "PValue": -1, "QValue": -1, "Peak": -1}


def _to_peaks(
    self: PyRanges,
    columns: list[str],
    path: str | None = None,
    compression: PANDAS_COMPRESSION_TYPE = None,
) -> str | None:
    """Write an ENCODE peak file with these columns, filling missing ones with the formats' defaults."""
    defaults: dict[str, object] = dict(_PEAK_DEFAULTS)
    if "BlockCount" in columns:  # gappedPeak: one block spanning each peak, unless there are blocks
        block_columns = ["BlockCount", "BlockSizes", "BlockStarts"]
        present = [c for c in block_columns if c in self.columns]
        if present and len(present) < len(block_columns):
            # Filling the others with one whole-peak block would contradict the given ones.
            msg = f"gappedPeak blocks need all of {block_columns}, or none; got only {present}."
            raise ValueError(msg)
        defaults |= {
            "ThickStart": self[START_COL],
            "ThickEnd": self[END_COL],
            "ItemRGB": "0",
            "BlockCount": 1,
            "BlockSizes": (self[END_COL] - self[START_COL]).astype(str) + ",",
            "BlockStarts": "0,",
        }
    filled = {c: self[c] if c in self.columns else defaults[c] for c in columns}
    return pd.DataFrame(filled, index=self.index).to_csv(
        path,
        index=False,
        header=False,
        compression=_resolve_compression(compression),
        mode="w+",
        sep="\t",
        quoting=csv.QUOTE_NONE,
    )


def _to_narrowpeak(
    self: PyRanges,
    path: str | None = None,
    compression: PANDAS_COMPRESSION_TYPE = None,
) -> str | None:
    return _to_peaks(self, _NARROWPEAK_COLUMNS, path, compression)


def _to_broadpeak(
    self: PyRanges,
    path: str | None = None,
    compression: PANDAS_COMPRESSION_TYPE = None,
) -> str | None:
    return _to_peaks(self, _NARROWPEAK_COLUMNS[:-1], path, compression)


def _to_gappedpeak(
    self: PyRanges,
    path: str | None = None,
    compression: PANDAS_COMPRESSION_TYPE = None,
) -> str | None:
    from pyranges1.readers import _GAPPEDPEAK_COLUMNS

    return _to_peaks(self, _GAPPEDPEAK_COLUMNS, path, compression)


_PAIRS_REQUIRED = ["Chromosome", "Start", "Strand", "OtherChromosome", "OtherStart", "OtherStrand"]
_PAIRS_HEADER = "## pairs format v1.0\n#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n"


def _to_pairs(self: PyRanges, path: str | None = None) -> str | None:
    missing = [c for c in _PAIRS_REQUIRED if c not in self.columns]
    if missing:
        msg = (
            f"to_pairs requires the paired columns {missing}; write a Hi-C/paired PyRanges "
            "(e.g. the output of read_pairs, which carries Other* columns)."
        )
        raise ValueError(msg)

    read_ids = list(self["ReadID"]) if "ReadID" in self.columns else ["."] * len(self)
    records = pd.DataFrame(
        {
            "readID": read_ids,
            "chr1": self["Chromosome"].astype(str).to_numpy(),
            "pos1": self["Start"].astype("int64").to_numpy() + 1,
            "chr2": self["OtherChromosome"].astype(str).to_numpy(),
            "pos2": self["OtherStart"].astype("int64").to_numpy() + 1,
            "strand1": self["Strand"].astype(str).to_numpy(),
            "strand2": self["OtherStrand"].astype(str).to_numpy(),
        },
    )
    text = _PAIRS_HEADER + records.to_csv(index=False, header=False, sep="\t", quoting=csv.QUOTE_NONE)

    if path is None:
        return text
    out = Path(path)
    if out.name.endswith(".gz"):
        import gzip

        with gzip.open(out, "wt") as fh:
            fh.write(text)
    else:
        out.write_text(text)
    return None


# PyRanges BED column -> conventional autoSql field name (inverse of read_bigbed's map),
# so a to_bigbed -> read_bigbed round-trip restores the column names.
_PYRANGES_TO_AUTOSQL_NAME = {
    "Name": "name",
    "Score": "score",
    "Strand": "strand",
    "ThickStart": "thickStart",
    "ThickEnd": "thickEnd",
    "ItemRGB": "itemRgb",
    "BlockCount": "blockCount",
    "BlockSizes": "blockSizes",
    "BlockStarts": "chromStarts",
}
_BED_EXTRA_ORDER = list(_PYRANGES_TO_AUTOSQL_NAME)


def _bigbed_autosql_type(dtype, name: str) -> str:
    if name == "Strand":
        return "char[1]"
    if pd.api.types.is_integer_dtype(dtype):
        return "uint" if pd.api.types.is_unsigned_integer_dtype(dtype) else "int"
    if pd.api.types.is_float_dtype(dtype):
        return "double"
    return "lstring"


def _bigbed_field_name(name: str) -> str:
    import re

    if name in _PYRANGES_TO_AUTOSQL_NAME:
        return _PYRANGES_TO_AUTOSQL_NAME[name]
    if re.fullmatch(r"[A-Za-z_]\w*", name):
        return name
    return re.sub(r"\W", "_", name) or "field"


def _build_bigbed_autosql(rest_cols: list[str], dtypes: list) -> str:
    lines = [
        "table pyranges",
        '"Generated by PyRanges.to_bigbed"',
        "(",
        'string chrom;      "Reference sequence chromosome or scaffold"',
        'uint   chromStart; "Start position in chromosome"',
        'uint   chromEnd;   "End position in chromosome"',
    ]
    for col, dt in zip(rest_cols, dtypes, strict=True):
        lines.append(f'{_bigbed_autosql_type(dt, col)} {_bigbed_field_name(col)}; "{col}"')
    lines.append(")")
    return "\n".join(lines) + "\n"


def _to_bigbed(
    self: PyRanges,
    path: str,
    chromosome_sizes: "PyRanges | pd.DataFrame | dict | None" = None,
    autosql: str | None = None,
) -> None:
    try:
        import pybigtools  # type: ignore[import]
    except ImportError:
        msg = "pybigtools must be installed to write BigBed files. Use `pip install pybigtools` to install it."
        raise ImportError(msg) from None

    present_bed = [c for c in _BED_EXTRA_ORDER if c in self.columns]
    extras = [c for c in self.columns if c not in (CHROM_COL, START_COL, END_COL) and c not in present_bed]
    rest_cols = present_bed + extras

    df = self.copy()
    df[CHROM_COL] = df[CHROM_COL].astype(str)
    df = df.sort_values([CHROM_COL, START_COL, END_COL]).reset_index(drop=True)

    if chromosome_sizes is None:
        sizes = {str(k): int(v) for k, v in df.groupby(CHROM_COL)[END_COL].max().items()}
    elif isinstance(chromosome_sizes, dict):
        sizes = {str(k): int(v) for k, v in chromosome_sizes.items()}
    else:
        sizes = dict(zip(chromosome_sizes[CHROM_COL].astype(str), chromosome_sizes[END_COL].astype(int), strict=True))

    chroms = df[CHROM_COL].tolist()
    starts = df[START_COL].astype("int64").tolist()
    ends = df[END_COL].astype("int64").tolist()

    if rest_cols:
        rest = df[rest_cols].astype(str).agg("\t".join, axis=1).tolist()
        vals = zip(chroms, starts, ends, rest, strict=True)
        sql = autosql or _build_bigbed_autosql(rest_cols, [self[c].dtype for c in rest_cols])
    else:
        # pybigtools takes (chrom, start, end, rest) even when there is no rest.
        vals = zip(chroms, starts, ends, [""] * len(chroms), strict=True)
        sql = autosql

    ordered_sizes = {c: sizes[c] for c in sorted(sizes)}
    writer = pybigtools.open(str(path), "w")  # type: ignore[attr-defined]
    writer.write(ordered_sizes, iter(vals), autosql=sql)
