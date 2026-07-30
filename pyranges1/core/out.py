import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import pandas as pd
from natsort import natsorted  # type: ignore[import]
from pandas.core.frame import DataFrame

from pyranges1.core.names import BIGWIG_SCORE_COL, CHROM_COL, END_COL, PANDAS_COMPRESSION_TYPE, START_COL
from pyranges1.core.pyranges_helpers import ensure_pyranges
from pyranges1.core.pyranges_main import PyRanges

if TYPE_CHECKING:
    from pyrle import RleDict  # type: ignore[import]

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

    noncanonical = list(set(df.columns) - set(bed_columns))
    noncanonical = [c for c in df.columns if c in noncanonical]

    if keep:
        return pd.concat([outdf, df[noncanonical]], axis=1)
    return outdf


def _to_gff_like(
    gr: PyRanges,
    out_format: Literal["gtf", "gff3"],
    path: Path | None = None,
    compression: PANDAS_COMPRESSION_TYPE = None,
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
        compression=compression,
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
    compression: PANDAS_COMPRESSION_TYPE = None,
    *,
    keep: bool = True,
) -> str | None:
    df = _bed(self, keep=keep)

    return df.to_csv(
        path,
        index=False,
        header=False,
        compression=compression,
        mode="w+",
        sep="\t",
        quoting=csv.QUOTE_NONE,
    )


def _merged_runs(rles: "RleDict") -> "RleDict":
    """Merge consecutive runs holding the same value, in every track.

    This is what ``pyrle``'s ``defragment`` is for, but that implementation
    zeroes the value whenever the merged result collapses to a single run --
    ``Rle([5], [3.0]).defragment()`` yields a value of ``0.0``. Since
    ``to_ranges`` then discards zero-valued runs as uncovered, relying on it
    silently deletes any track that reduces to one run.

    ``NaN`` is treated as equal to ``NaN`` so that adjacent undefined runs
    merge, matching ``defragment``'s behaviour on inputs where it is correct.
    """
    from pyrle import Rle, RleDict

    merged = {}
    for chromosome, rle in rles.items():
        runs = np.asarray(rle.runs)
        values = np.asarray(rle.values, dtype=float)
        if values.size <= 1:
            merged[chromosome] = Rle(runs.copy(), values.copy())
            continue
        previous, current = values[:-1], values[1:]
        same = (current == previous) | (np.isnan(current) & np.isnan(previous))
        starts = np.concatenate(([0], np.flatnonzero(~same) + 1))
        merged_values = values[starts]
        # `defragment` normalises negative zero; keep that.
        merged_values[merged_values == 0] = 0.0
        merged[chromosome] = Rle(np.add.reduceat(runs, starts), merged_values)
    return RleDict(merged)


def _to_bigwig(
    self: PyRanges,
    path: None,
    chromosome_sizes: PyRanges | pd.DataFrame | dict,
    value_col: str | None = None,
    *,
    divide: bool = False,
    rpm: bool = True,
    return_data: bool = False,
) -> PyRanges | None:
    try:
        import pyBigWig  # type: ignore[import]
    except ModuleNotFoundError:
        LOGGER.exception(
            "pybigwig must be installed to create bigwigs. Use `conda install -c bioconda pybigwig` or `pip install pybigwig` to install it.",
        )
        import sys

        sys.exit(1)

    if not divide:
        rles = self.to_rle(rpm=rpm, strand=False, value_col=value_col)
        df = rles.to_ranges()
    else:
        numerator = self.to_rle(rpm=rpm, strand=False, value_col=value_col)
        denominator = self.to_rle(rpm=rpm, strand=False)
        ratio = numerator / denominator
        for rle in ratio.values():
            rle.values = np.log2(rle.values)
        df = _merged_runs(ratio).to_ranges()
    gr = ensure_pyranges(df)
    unique_chromosomes = gr.chromosomes

    gr = gr.remove_strand()
    gr = gr.sort_ranges()
    gr = gr.get_with_loc_columns(BIGWIG_SCORE_COL)

    if return_data:
        return gr

    if not isinstance(chromosome_sizes, dict):
        size_df = chromosome_sizes
        chromosome_sizes = dict(zip(size_df[CHROM_COL], size_df[END_COL], strict=True))

    header = [(c, int(chromosome_sizes[c])) for c in unique_chromosomes]

    bw = pyBigWig.open(path, "w")
    bw.addHeader(header)

    chromosomes = df[CHROM_COL].tolist()
    starts = df[START_COL].tolist()
    ends = df[END_COL].tolist()
    values = df.Score.tolist()

    bw.addEntries(chromosomes, starts, ends=ends, values=values)

    return None


class AttributeFormatter(Protocol):
    def __call__(self, colname: str, col: "pd.Series[str]") -> "pd.Series[str]":
        """Stub to properly annotate forced named args (..., *, ...)."""
        ...


def _pyranges_to_gtf_like(
    df: pd.DataFrame,
    out_format: Literal["gtf", "gff3"],
    map_cols: dict | None = None,
) -> pd.DataFrame:
    attribute_formatter: AttributeFormatter

    if out_format == "gtf":
        all_columns = _ordered_gtf_columns[:-1]
        # first: gff column to pyranges column
        rename_columns = GTF_COLUMNS_TO_PYRANGES.copy()
        attribute_formatter = gtf_formatter
    elif out_format == "gff3":
        all_columns = _ordered_gff3_columns[:-1]
        # first: gff column to pyranges column
        rename_columns = GFF3_COLUMNS_TO_PYRANGES.copy()
        attribute_formatter = gff3_formatter
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
        rest_df = df[rest].copy()
        # putting all remaining columns into the attribute column
        for colname in rest_df.columns:
            col = pd.Series(rest_df[colname])
            isnull = col.isna()
            new_val = attribute_formatter(colname, col)  # type: ignore[call-arg]

            # not working to convert cat to str:  rest_df.loc[:, colname] = rest_df[colname].astype(str)
            # so doing this instead:
            rest_df[colname] = rest_df[colname].astype(str)
            rest_df.loc[~isnull, colname] = new_val
            rest_df.loc[isnull, colname] = ""

        attribute = merge_attributes(rest_df, out_format)
        outdf.insert(outdf.shape[1], column="attribute", value=attribute)
    else:
        outdf.insert(outdf.shape[1], column="attribute", value=df["attribute"].copy())

    return outdf


def gtf_formatter(colname: str, col: "pd.Series[str]") -> "pd.Series[str]":
    """Format a column as a GTF attribute column."""
    attribute_template_string = f'{colname} "{{col}}"; '
    return col.apply(lambda x: attribute_template_string.format(col=x))


def gff3_formatter(colname: str, col: "pd.Series[str]") -> "pd.Series[str]":
    """Format a column as a GFF3 attribute column."""
    attribute_template_string = f"{colname}={{col}};"
    return col.apply(lambda x: attribute_template_string.format(col=x))


def merge_attributes(attributes: pd.DataFrame, out_format: Literal["gtf", "gff3"]) -> "pd.Series[str]":
    """Merge attributes into a single column.

    The final column in gtf/gff is not separated by tabs, but by other separators.
    """
    regex_to_remove = " $" if out_format == "gtf" else ";$"
    return attributes.apply(lambda r: "".join([v for v in r if v]), axis=1).str.replace(regex_to_remove, "", regex=True)
