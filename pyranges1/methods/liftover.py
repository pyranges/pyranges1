import gzip
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyranges1.core.names import CHROM_COL, END_COL, START_COL, STRAND_COL
from pyranges1.core.pyranges_helpers import ensure_pyranges

if TYPE_CHECKING:
    from pyranges1 import PyRanges

CHAIN_COLUMNS = [
    CHROM_COL,
    START_COL,
    END_COL,
    "QueryChromosome",
    "QueryStart",
    "QueryEnd",
    "QueryStrand",
    "ChainId",
    "ChainScore",
]


def _read_chain(f: "str | Path") -> "PyRanges":
    """Read a UCSC chain file into one row per gapless block.

    Chromosome/Start/End is the block on the assembly lifted from; QueryChromosome,
    QueryStart and QueryEnd the same bases on the assembly lifted to, in forward-strand
    coordinates even on a "-" chain.
    """
    path = Path(f)
    with path.open("rb") as fh:
        compressed = fh.read(2) == b"\x1f\x8b"
    rows: list[tuple] = []
    # (source name, destination name, size, strand, id, score) of the chain being read,
    # and the positions its next block starts at on either side.
    header: tuple[str, str, int, str, int, int] | None = None
    t_pos = q_pos = 0
    with gzip.open(path, "rt") if compressed else path.open() as fh:
        for line in fh:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "chain":
                header = (fields[2], fields[7], int(fields[8]), fields[9], int(fields[12]), int(fields[1]))
                t_pos, q_pos = int(fields[5]), int(fields[10])
                continue
            if header is None:
                msg = f"{path}: an alignment block before any chain line: {line.strip()!r}."
                raise ValueError(msg)
            t_name, q_name, q_size, q_strand, chain_id, score = header
            size = int(fields[0])
            q_from, q_to = (q_pos, q_pos + size) if q_strand == "+" else (q_size - q_pos - size, q_size - q_pos)
            rows.append((t_name, t_pos, t_pos + size, q_name, q_from, q_to, q_strand, chain_id, score))
            if len(fields) == 3:  # noqa: PLR2004 -- size, gap on the source, gap on the destination
                t_pos += size + int(fields[1])
                q_pos += size + int(fields[2])
    return ensure_pyranges(pd.DataFrame(rows, columns=CHAIN_COLUMNS))


def _liftover(self: "PyRanges", chain: "PyRanges", *, min_match: float) -> "tuple[PyRanges, PyRanges]":
    """Lift intervals through chain blocks, as UCSC liftOver does without -multiple.

    An interval is lifted when exactly one chain covers at least `min_match` of its
    bases; it maps to the span between the first and last of those bases. Returns the
    lifted rows and the rest, with a LiftReason as liftOver reports it.
    """
    rows = self.reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    pairs = rows[[CHROM_COL, START_COL, END_COL, "__row__"]].join_overlaps(
        chain, strand_behavior="ignore", suffix="_chain"
    )
    start = np.maximum(pairs[START_COL].to_numpy(np.int64), pairs[START_COL + "_chain"].to_numpy(np.int64))
    end = np.minimum(pairs[END_COL].to_numpy(np.int64), pairs[END_COL + "_chain"].to_numpy(np.int64))
    offset_start = start - pairs[START_COL + "_chain"].to_numpy(np.int64)
    offset_end = end - pairs[START_COL + "_chain"].to_numpy(np.int64)
    q_start, q_end = pairs["QueryStart"].to_numpy(np.int64), pairs["QueryEnd"].to_numpy(np.int64)
    minus = (pairs["QueryStrand"] == "-").to_numpy()
    pieces = pd.DataFrame(
        {
            "row": pairs["__row__"].to_numpy(),
            "chain": pairs["ChainId"].to_numpy(),
            "bases": end - start,
            "to_start": np.where(minus, q_end - offset_end, q_start + offset_start),
            "to_end": np.where(minus, q_end - offset_start, q_start + offset_end),
            "to_chrom": pairs["QueryChromosome"].to_numpy(),
            "to_minus": minus,
        }
    )
    per_chain = pieces.groupby(["row", "chain"], sort=False).agg(
        bases=("bases", "sum"),
        to_start=("to_start", "min"),
        to_end=("to_end", "max"),
        to_chrom=("to_chrom", "first"),
        to_minus=("to_minus", "first"),
    )
    lengths = (rows[END_COL] - rows[START_COL]).to_numpy(np.int64)
    per_chain["enough"] = per_chain["bases"].to_numpy() >= min_match * lengths[per_chain.index.get_level_values("row")]
    passing = per_chain[per_chain["enough"]].reset_index()
    n_passing = passing.groupby("row").size().reindex(range(len(rows)), fill_value=0).to_numpy()
    n_chains = per_chain.groupby(level="row").size().reindex(range(len(rows)), fill_value=0).to_numpy()

    lifted_rows = np.flatnonzero(n_passing == 1)
    chosen = passing.set_index("row").loc[lifted_rows]
    lifted = rows.iloc[lifted_rows].copy()
    lifted[CHROM_COL] = chosen["to_chrom"].to_numpy()
    lifted[START_COL] = chosen["to_start"].to_numpy()
    lifted[END_COL] = chosen["to_end"].to_numpy()
    if STRAND_COL in lifted.columns:
        strand = lifted[STRAND_COL].astype(object).to_numpy()
        flipped = np.where(strand == "+", "-", np.where(strand == "-", "+", strand))
        lifted[STRAND_COL] = np.where(chosen["to_minus"].to_numpy(), flipped, strand)

    unlifted = rows.iloc[np.flatnonzero(n_passing != 1)].copy()
    # liftOver's reasons: no chain at all; one chain covering too little of it; several
    # chains, none covering enough; several covering enough.
    reason = np.select(
        [n_passing > 1, n_chains > 1, n_chains == 1],
        ["Duplicated in new", "Split in new", "Partially deleted in new"],
        default="Deleted in new",
    )
    unlifted["LiftReason"] = reason[n_passing != 1]
    original_index = self.index.to_numpy()
    lifted.index = original_index[lifted_rows]
    unlifted.index = original_index[np.flatnonzero(n_passing != 1)]
    return (
        ensure_pyranges(lifted.loc[:, lifted.columns != "__row__"]),
        ensure_pyranges(unlifted.loc[:, unlifted.columns != "__row__"]),
    )
