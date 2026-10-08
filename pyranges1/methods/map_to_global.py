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

cumsum_start = "_local_start"
cumsum_end = "_local_end"
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

    z = m.slice_ranges(  # type: ignore[reportCallIssue]
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


# buggy:
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
    """Lift intervals from local_gr to genomic coordinates using global_gr exon annotations.

    Lift intervals in *local_gr* (coordinates relative to a transcript) onto absolute genomic coordinates using the exon annotation in *global_gr*.
    """
    local_has_strand = local_gr.has_strand
    global_has_strand = global_gr.has_strand

    global_cum = global_gr.group_cumsum(
        group_by=global_on,
        use_strand="auto",
        cumsum_start_column=cumsum_start,
        cumsum_end_column=cumsum_end,
        keep_order=True,  # False,
    )
    ex_df = global_cum.copy()

    ex_df["__global_start__"] = ex_df[START_COL]
    ex_df["__global_end__"] = ex_df[END_COL]
    ex_df[START_COL] = ex_df[cumsum_start]
    ex_df[END_COL] = ex_df[cumsum_end]

    tmp_tx_id = "__txid__"
    local_df = local_gr.copy()

    ex_df[tmp_tx_id] = ex_df[global_on]
    local_df[tmp_tx_id] = local_df[local_on]

    q_tx_code, ex_tx_code = factorize_binary(local_df, ex_df, by=tmp_tx_id)

    ex_chr_code, chr_table = pd.factorize(ex_df[CHROM_COL], sort=False)
    ex_chr_code = ex_chr_code.astype(ex_tx_code.dtype, copy=False)

    ex_local_start = ex_df[START_COL].to_numpy(copy=False)
    ex_local_end = ex_df[END_COL].to_numpy(copy=False)
    ex_genome_start = ex_df["__global_start__"].to_numpy(copy=False)
    ex_genome_end = ex_df["__global_end__"].to_numpy(copy=False)
    ex_fwd = (
        (ex_df[STRAND_COL] == "+").to_numpy(np.bool_, copy=False)
        if global_has_strand
        else np.ones(len(ex_df), dtype=bool)
    )

    q_start = local_df[START_COL].to_numpy(copy=False)
    q_end = local_df[END_COL].to_numpy(copy=False)
    q_fwd = (
        (local_df[STRAND_COL] == "+").to_numpy(np.bool_, copy=False)
        if local_has_strand
        else np.ones(len(local_df), dtype=bool)
    )

    keep_idx, out_start, out_end, out_strand_bool = ruranges.numpy.map_to_global(
        groups=ex_tx_code,
        starts=ex_local_start,
        ends=ex_local_end,
        strand=ex_fwd,
        groups2=q_tx_code,
        starts2=q_start,
        ends2=q_end,
        chr_code2=ex_chr_code,
        genome_start2=ex_genome_start,
        genome_end2=ex_genome_end,
        strand2=q_fwd,
        sort_output=preserve_input_order,
    )

    if len(keep_idx) == 0:
        return ensure_pyranges(local_df.head(0))

    max_code = int(ex_tx_code.max()) + 1
    tx2chr_code = np.empty(max_code, dtype=ex_chr_code.dtype)
    tx2chr_code[ex_tx_code] = ex_chr_code
    out_chr_code = tx2chr_code[q_tx_code[keep_idx]]
    out_chr_str = chr_table[out_chr_code]

    mapped_df = local_gr.take(keep_idx).copy()  # type: ignore[arg-type]

    if keep_id:
        mapped_df[global_on] = mapped_df[CHROM_COL].copy()
    if keep_loc:
        mapped_df[out_local_start] = mapped_df[START_COL].copy()
        mapped_df[out_local_end] = mapped_df[END_COL].copy()
        if local_has_strand and global_has_strand:
            mapped_df[out_local_strand] = mapped_df[STRAND_COL].copy()

    mapped_df[CHROM_COL] = out_chr_str
    mapped_df[START_COL] = out_start
    mapped_df[END_COL] = out_end

    if local_has_strand or global_has_strand:
        mapped_df[STRAND_COL] = np.where(out_strand_bool, "+", "-")

    if tmp_tx_id in mapped_df.columns:
        mapped_df = mapped_df.drop(columns=[tmp_tx_id])
        if mapped_df is None:
            raise ValueError

    return ensure_pyranges(mapped_df)
