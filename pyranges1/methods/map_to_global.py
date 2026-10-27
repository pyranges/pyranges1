from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyranges1._ruranges import require_ruranges
from pyranges1.core.names import (
    CHROM_COL,
    END_COL,
    START_COL,
    STRAND_COL,
)
from pyranges1.core.pyranges_helpers import ensure_pyranges, factorize_binary

if TYPE_CHECKING:
    from pyranges1 import PyRanges

ruranges = require_ruranges()

out_local_start = "Start_local"
out_local_end = "End_local"
out_local_strand = "Strand_local"


def _map_to_global_pandas(
    local_gr: "PyRanges",
    global_gr: "PyRanges",
    global_on: str,
    *,
    local_on: str = "Chromosome",
    keep_id: bool = False,
    keep_loc: bool = False,
) -> "PyRanges":
    local_has_strand, global_has_strand = local_gr.has_strand, global_gr.has_strand
    original_order, original_columns = local_gr.index.copy(), local_gr.columns.copy()

    local_gr = local_gr.copy()
    local_gr["_temp_index_"] = local_gr.index.copy()

    global_gr = global_gr.get_with_loc_columns(global_on)

    m = local_gr.merge(
        global_gr,
        left_on=local_on,
        right_on=global_on,
        suffixes=("_local", ""),
    )

    z = m.slice_ranges(
        group_by="_temp_index_",
        start=m[out_local_start].to_numpy(),
        end=m[out_local_end].to_numpy(),
        use_strand=global_has_strand,
    )

    cols_to_drop = (
        [CHROM_COL + "_local"]
        # With local_on == global_on the merge keeps one key column: the local frame's own.
        + ([] if keep_id or local_on == global_on else [global_on])
        + ([] if keep_loc else [out_local_start, out_local_end])
    )

    if local_has_strand and global_has_strand:
        z[STRAND_COL] = np.where(z[out_local_strand] == z[STRAND_COL], "+", "-")
        if not keep_loc:
            cols_to_drop.append(out_local_strand)

    z = z.set_index("_temp_index_", drop=True)
    z.index.name = local_gr.index.name
    if (z := z.drop(columns=cols_to_drop)) is None:
        raise RuntimeError
    z = z.loc[original_order.intersection(z.index, sort=False)]

    bottom_cols: list[str] = []
    if keep_id:
        bottom_cols.append(global_on)
    elif (global_on + "_local") in z.columns:
        z = z.rename(columns={global_on + "_local": global_on})
    if keep_loc:
        bottom_cols += [out_local_start, out_local_end]
        if local_has_strand and global_has_strand:
            bottom_cols.append(out_local_strand)
    if global_has_strand and not local_has_strand:
        bottom_cols.append(STRAND_COL)

    cols_output = [c for c in original_columns if c not in bottom_cols] + bottom_cols
    return ensure_pyranges(z[cols_output])


def _transcript_offsets(
    local_gr: "PyRanges", q_tx_code: np.ndarray, ex_tx_code: np.ndarray, ex_local_end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpret negative query offsets relative to their transcript length."""
    q_start = local_gr[START_COL].to_numpy(copy=False)
    q_end = local_gr[END_COL].to_numpy(copy=False)

    # slice_ranges historically interprets negative offsets from the transcript
    # end. Normalize those offsets before the core clips to exon coverage.
    if (q_start < 0).any() or (q_end < 0).any():
        n_groups = max(int(q_tx_code.max(initial=0)), int(ex_tx_code.max(initial=0))) + 1
        transcript_lengths = np.zeros(n_groups, dtype=np.int64)
        np.maximum.at(transcript_lengths, ex_tx_code, ex_local_end)
        lengths = transcript_lengths[q_tx_code]
        q_start = np.where(q_start < 0, lengths + q_start, q_start)
        q_end = np.where(q_end < 0, lengths + q_end, q_end)

    return q_start, q_end


def _map_to_global_ruranges(
    local_gr: "PyRanges",
    global_gr: "PyRanges",
    global_on: str,
    *,
    local_on: str = "Chromosome",
    keep_id: bool = False,
    keep_loc: bool = False,
    preserve_input_order: bool = True,
) -> "PyRanges":
    """Project queries in Rust and rebuild their original index and metadata."""
    local_has_strand = local_gr.has_strand
    global_has_strand = global_gr.has_strand
    reference = global_gr.get_with_loc_columns(global_on)

    # The core models one spliced coordinate system per feature. Preserve the
    # existing pandas behavior for ids reused on different chromosomes/strands.
    locus_columns = [CHROM_COL] + ([STRAND_COL] if global_has_strand else [])
    locus_columns = [column for column in locus_columns if column != global_on]
    if locus_columns and (
        reference.groupby(global_on, observed=True, dropna=False)[locus_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=None)
    ):
        return _map_to_global_pandas(
            local_gr,
            global_gr,
            global_on,
            local_on=local_on,
            keep_id=keep_id,
            keep_loc=keep_loc,
        )

    # Keep temporary cumulative-coordinate columns off the calling frame.
    local_start_column, local_end_column = "__map_start", "__map_end"
    while local_start_column in reference or local_start_column == global_on:
        local_start_column += "_"
    while local_end_column in reference or local_end_column == global_on:
        local_end_column += "_"
    ex_df = reference.group_cumsum(
        group_by=global_on,
        use_strand="auto",
        cumsum_start_column=local_start_column,
        cumsum_end_column=local_end_column,
        keep_order=True,
    )
    query_ids = pd.DataFrame({"feature": local_gr[local_on].array})
    exon_ids = pd.DataFrame({"feature": ex_df[global_on].array})
    q_tx_code, ex_tx_code = factorize_binary(query_ids, exon_ids, by="feature")
    ex_local_start = ex_df[local_start_column].to_numpy(copy=False)
    ex_local_end = ex_df[local_end_column].to_numpy(copy=False)
    q_start, q_end = _transcript_offsets(local_gr, q_tx_code, ex_tx_code, ex_local_end)

    keep_idx, out_start, out_end, out_strand, exon_idx = ruranges.numpy.map_to_global(
        groups=q_tx_code,
        starts=q_start,
        ends=q_end,
        strand=(local_gr[STRAND_COL] == "+").to_numpy(np.bool_, copy=False)
        if local_has_strand
        else np.ones(len(local_gr), dtype=bool),
        groups2=ex_tx_code,
        starts2=ex_local_start,
        ends2=ex_local_end,
        chr_code2=np.zeros(len(ex_df), dtype=np.uint32),
        genome_start2=ex_df[START_COL].to_numpy(copy=False),
        genome_end2=ex_df[END_COL].to_numpy(copy=False),
        strand2=(ex_df[STRAND_COL] == "+").to_numpy(np.bool_, copy=False)
        if global_has_strand
        else np.ones(len(ex_df), dtype=bool),
        sort_output=preserve_input_order,
        return_exon_indices=True,
    )
    # An empty merge reproduces pandas' key/category dtype rules without
    # materializing the query-by-exon table used by the previous implementation.
    schema = local_gr.head(0).merge(reference.head(0), left_on=local_on, right_on=global_on, suffixes=("_local", ""))
    mapped_df = local_gr.take(keep_idx).copy()
    if keep_id:
        mapped_df[global_on] = pd.array(ex_df[global_on].iloc[exon_idx], dtype=schema[global_on].dtype)
    elif local_on == global_on and local_on != CHROM_COL:
        mapped_df[local_on] = mapped_df[local_on].astype(schema[local_on].dtype)
    if keep_loc:
        mapped_df[out_local_start] = mapped_df[START_COL].array.copy()
        mapped_df[out_local_end] = mapped_df[END_COL].array.copy()
        if local_has_strand and global_has_strand:
            mapped_df[out_local_strand] = mapped_df[STRAND_COL].array.copy()
    mapped_df[CHROM_COL] = pd.array(ex_df[CHROM_COL].iloc[exon_idx], dtype=schema[CHROM_COL].dtype)
    mapped_df[START_COL] = pd.array(out_start, dtype=global_gr[START_COL].dtype)
    mapped_df[END_COL] = pd.array(out_end, dtype=global_gr[END_COL].dtype)
    if local_has_strand and global_has_strand:
        mapped_df[STRAND_COL] = np.where(out_strand, "+", "-")
    elif global_has_strand:
        mapped_df[STRAND_COL] = ex_df[STRAND_COL].iloc[exon_idx].array
    return ensure_pyranges(mapped_df)
