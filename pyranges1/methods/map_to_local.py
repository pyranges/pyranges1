from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyranges1.core.names import (
    CHROM_COL,
    END_COL,
    START_COL,
    STRAND_COL,
)
from pyranges1.core.pyranges_helpers import arg_to_list, ensure_pyranges

if TYPE_CHECKING:
    from pyranges1 import PyRanges

cumsum_start = "_refcumstart"
cumsum_end = "_refcumend"
suffix = "__ref"
start_b = START_COL + suffix
end_b = END_COL + suffix
strand_b = STRAND_COL + suffix
row_id = "__row"

out_global_chromosome = "Chromosome_global"
out_global_start = "Start_global"
out_global_end = "End_global"
out_global_strand = "Strand_global"


def _map_to_local(
    gr,
    ref,
    ref_on,
    match_by,
    keep_chrom,
    keep_loc,
    *,
    split_at_junctions: bool = True,
) -> "PyRanges":
    # record ordered columns of gr
    output_cols = gr.columns.tolist()
    gr_has_strand = gr.has_strand
    ref_has_strand = ref.has_strand
    match_by = arg_to_list(match_by)
    ref_cols_to_keep = [*match_by, ref_on]

    if ref_on in match_by:
        # dealing with case in which user used ref_on as match_by
        fixed_idcol = ref_on
        output_cols = [c for c in output_cols if c != ref_on]
        ref_cols_to_keep.pop(-1)  # remove idcol from match_by
    else:
        fixed_idcol = "__idcol"

    ref = (
        ref.get_with_loc_columns(ref_cols_to_keep)
        .group_cumsum(
            group_by=ref_on,
            use_strand="auto",
            cumsum_start_column=cumsum_start,
            cumsum_end_column=cumsum_end,
            keep_order=False,
        )
        .rename(columns={ref_on: fixed_idcol})
    )

    gr = gr.assign(**{row_id: np.arange(len(gr))})
    gr = gr.join_overlaps(ref, strand_behavior="ignore", match_by=match_by, suffix=suffix)

    # removing gr regions that do not overlap with ref
    gr = gr.combine_interval_columns("intersect", drop_old_columns=False, start2=start_b, end2=end_b)

    columns_to_drop = [cumsum_start, cumsum_end]
    if not keep_chrom:
        columns_to_drop = [*columns_to_drop, CHROM_COL]
    if not keep_loc:
        columns_to_drop = [*columns_to_drop, start_b, end_b]

    # dealing with - strand intervals on ref.
    if ref_has_strand:
        # depending on whether gr had strand, STRAND_COL + suffix may be missing in join_overlaps output
        strand_col_of_ref = strand_b if gr_has_strand else STRAND_COL

        ref_strand_is_neg = gr[strand_col_of_ref] == "-"

        # computing start and end for neg intervals at the same time, to allow replacement in one go
        gr.loc[ref_strand_is_neg, START_COL], gr.loc[ref_strand_is_neg, END_COL] = (
            (
                gr.loc[ref_strand_is_neg, end_b]
                - gr.loc[ref_strand_is_neg, END_COL]
                + gr.loc[ref_strand_is_neg, cumsum_start]
            ),
            (
                gr.loc[ref_strand_is_neg, end_b]
                - gr.loc[ref_strand_is_neg, START_COL]
                + gr.loc[ref_strand_is_neg, cumsum_start]
            ),
        )

        if gr_has_strand:  # and ref_has_strand: # implicit
            # transform Strand: for rows with Strand == Strand_b, it becomes '+', otherwise '-'
            gr[STRAND_COL] = np.where(gr[STRAND_COL] == gr[strand_b], "+", "-")
            if not keep_loc:
                columns_to_drop = [*columns_to_drop, strand_b]
    else:
        ref_strand_is_neg = pd.Series(data=False, index=gr.index, dtype=np.bool_)

    # dealing with + strand intervals in ref
    gr.loc[~ref_strand_is_neg, START_COL] = (
        gr.loc[~ref_strand_is_neg, START_COL]
        - gr.loc[~ref_strand_is_neg, start_b]
        + gr.loc[~ref_strand_is_neg, cumsum_start]
    )
    gr.loc[~ref_strand_is_neg, END_COL] = (
        gr.loc[~ref_strand_is_neg, END_COL]
        - gr.loc[~ref_strand_is_neg, start_b]
        + gr.loc[~ref_strand_is_neg, cumsum_start]
    )

    # drop useless columns

    if keep_chrom:
        gr = gr.rename(columns={CHROM_COL: out_global_chromosome})
        output_cols = [*output_cols, out_global_chromosome]

    gr = gr.drop(columns=columns_to_drop).rename(columns={fixed_idcol: CHROM_COL})

    if keep_loc:
        gr = gr.rename(columns={start_b: out_global_start, end_b: out_global_end})
        output_cols = [*output_cols, out_global_start, out_global_end]
        if gr_has_strand and ref_has_strand:
            gr = gr.rename(columns={strand_b: out_global_strand})
            output_cols = [*output_cols, out_global_strand]

    gr = gr if split_at_junctions else _join_exon_pieces(gr)

    # reordering columns to match original gr
    return ensure_pyranges(gr.reindex(columns=output_cols))


def _join_exon_pieces(gr: "PyRanges") -> "PyRanges":
    """One row per (interval, transcript): its exon pieces are adjacent in transcript coordinates."""
    keys = [row_id, CHROM_COL] + ([STRAND_COL] if STRAND_COL in gr.columns else [])
    grouped = gr.groupby(keys, sort=False, dropna=False)
    lefts = [c for c in (START_COL, out_global_start) if c in gr.columns]
    rights = [c for c in (END_COL, out_global_end) if c in gr.columns]
    gr = gr.copy()
    for c in lefts:
        gr[c] = grouped[c].transform("min")
    for c in rights:
        gr[c] = grouped[c].transform("max")
    return ensure_pyranges(gr.loc[~gr.duplicated(keys).to_numpy()])
