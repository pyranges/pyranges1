"""Helpers for gene annotations (GTF/GFF3 as read_gtf and read_gff3 return them)."""

import re
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pyranges1.core.names import END_COL, START_COL, STRAND_COL
from pyranges1.core.pyranges_helpers import ensure_pyranges

if TYPE_CHECKING:
    from pyranges1 import PyRanges

# Tags ranked first by select_transcripts, per mode.
_PREFERRED_TAGS = {"canonical": ["Ensembl_canonical"], "mane": ["MANE_Select", "Ensembl_canonical"], "longest": []}


def has_tag(gr: "PyRanges", tag: str, tag_col: str = "tag") -> pd.Series:
    """Whether each row carries a tag, in a column of comma-joined tags.

    read_gtf(duplicate_attr=True) joins a row's repeated attributes with commas, e.g.
    "basic,Ensembl_canonical"; with the default duplicate_attr=False only the last tag of
    each row is kept. Whole tags are matched, so "start_NF" does not match "mRNA_start_NF".

    Parameters
    ----------
    gr : PyRanges
        Annotation rows.

    tag : str
        The tag, e.g. "Ensembl_canonical" or "MANE_Select".

    tag_col : str, default "tag"
        Column of comma-joined tags.

    Returns
    -------
    pandas.Series of bool, aligned with gr

    Examples
    --------
    >>> import pyranges1 as pr
    >>> gr = pr.PyRanges({"Chromosome": "chr1", "Start": [0, 10, 20], "End": [5, 15, 25],
    ...                   "tag": ["basic,Ensembl_canonical", "mRNA_start_NF", None]})
    >>> pr.genes.has_tag(gr, "Ensembl_canonical").tolist()
    [True, False, False]
    >>> pr.genes.has_tag(gr, "start_NF").tolist()
    [False, False, False]

    """
    pattern = rf"(?:^|,){re.escape(tag)}(?:,|$)"
    return gr[tag_col].astype("string").str.contains(pattern, regex=True).fillna(value=False).astype(bool)


def select_transcripts(
    gr: "PyRanges",
    how: Literal["canonical", "mane", "longest"] = "canonical",
    *,
    gene_col: str = "gene_id",
    transcript_col: str = "transcript_id",
    tag_col: str = "tag",
) -> "PyRanges":
    """Keep one transcript per gene.

    Transcripts are ranked within their gene by tag, then by span (first to last base of
    any of their rows), longest first, then by id. Rows of the chosen transcripts are
    kept, along with rows that belong to no transcript (gene lines); row order and index
    are unchanged.

    Read the annotation with read_gtf(..., duplicate_attr=True): by default only the last
    tag of each row is kept, and most canonical and MANE tags are lost with it.

    Parameters
    ----------
    gr : PyRanges
        Annotation rows: transcripts, exons, CDS, ... in any mix.

    how : {"canonical", "mane", "longest"}, default "canonical"
        "canonical": the transcript tagged Ensembl_canonical. "mane": MANE_Select, else
        Ensembl_canonical. "longest": the longest span. Genes without the tag fall back to
        the next rule, with a warning saying how many.

    gene_col, transcript_col, tag_col : str
        Columns holding the gene id, the transcript id and the comma-joined tags.

    Returns
    -------
    PyRanges

    See Also
    --------
    pyranges1.genes.has_tag : whether rows carry a tag

    Examples
    --------
    >>> import pyranges1 as pr
    >>> gr = pr.PyRanges({"Chromosome": "chr1", "Start": [100, 100, 100, 500], "End": [900, 400, 300, 700],
    ...                   "Strand": "+", "gene_id": ["g1", "g1", "g1", "g2"],
    ...                   "transcript_id": ["t1", "t2", "t3", "t4"],
    ...                   "tag": ["basic", "basic,Ensembl_canonical,MANE_Select", "Ensembl_canonical", "basic"]})
    >>> pr.genes.select_transcripts(gr, "longest")["transcript_id"].tolist()
    ['t1', 't4']
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     pr.genes.select_transcripts(gr)["transcript_id"].tolist()
    ['t2', 't4']
    >>> print(w[0].message)
    Ensembl_canonical tags 1 of 2 genes; the rest keep their longest transcript.

    """
    if how not in _PREFERRED_TAGS:
        msg = f"how must be 'canonical', 'mane' or 'longest', got {how!r}."
        raise ValueError(msg)
    tags = _PREFERRED_TAGS[how]
    if tags and tag_col not in gr.columns:
        msg = f"No {tag_col!r} column to find {tags[0]} in; use how='longest' or pass tag_col."
        raise ValueError(msg)

    in_transcript = gr[transcript_col].notna().to_numpy()
    # Only the four ranking columns are subset: copying every column of a large annotation dominates otherwise.
    columns = {"gene": gr[gene_col], "transcript": gr[transcript_col], "start": gr[START_COL], "end": gr[END_COL]}
    columns |= {f"__{tag}": has_tag(gr, tag, tag_col) for tag in tags}
    ranked = (
        pd.DataFrame({name: column.to_numpy()[in_transcript] for name, column in columns.items()})
        .groupby("transcript", sort=False)
        .agg(
            gene=("gene", "first"),
            start=("start", "min"),
            end=("end", "max"),
            **{f"__{tag}": (f"__{tag}", "any") for tag in tags},
        )
        .reset_index()
    )
    ranked["length"] = ranked["end"] - ranked["start"]
    order = [*(f"__{tag}" for tag in tags), "length", "transcript"]
    ranked = ranked.sort_values(order, ascending=[False] * (len(order) - 1) + [True], kind="stable")
    chosen = ranked.drop_duplicates("gene")

    if tags:
        covered, genes = int(chosen[f"__{tags[0]}"].sum()), len(chosen)
        if covered < genes:
            rest = "Ensembl_canonical, then their longest transcript" if len(tags) > 1 else "their longest transcript"
            msg = f"{tags[0]} tags {covered} of {genes} genes; the rest keep {rest}."
            if not gr[tag_col].astype("string").str.contains(",", regex=False).any():
                msg += " No row has more than one tag: was the GTF read with read_gtf(..., duplicate_attr=True)?"
            warnings.warn(msg, stacklevel=2)

    keep = ~in_transcript | gr[transcript_col].isin(chosen["transcript"]).to_numpy()
    return ensure_pyranges(gr.loc[keep])


def label_utrs(gr: "PyRanges", *, transcript_col: str = "transcript_id", feature_col: str = "Feature") -> "PyRanges":
    """Label unsplit UTR rows as 5' or 3', from the transcript's coding extent.

    GENCODE GTFs have only "UTR" rows; Ensembl GTFs say five_prime_utr and three_prime_utr.
    A UTR row wholly before the first CDS base of its transcript (in transcription order)
    becomes "five_prime_utr", wholly after the last "three_prime_utr". UTR rows of
    transcripts without CDS rows, or overlapping the CDS, keep "UTR". Other rows are
    unchanged.

    Parameters
    ----------
    gr : PyRanges
        Annotation rows with Strand, feature and transcript columns, including the CDS rows.

    transcript_col : str, default "transcript_id"
        Column holding the transcript id.

    feature_col : str, default "Feature"
        Column holding the feature type.

    Returns
    -------
    PyRanges

    Examples
    --------
    >>> import pyranges1 as pr
    >>> gr = pr.PyRanges({"Chromosome": "chr1", "Start": [100, 150, 400, 100, 150, 400],
    ...                   "End": [150, 400, 500, 150, 400, 500], "Strand": ["+"] * 3 + ["-"] * 3,
    ...                   "Feature": ["UTR", "CDS", "UTR"] * 2, "transcript_id": ["t1"] * 3 + ["t2"] * 3})
    >>> pr.genes.label_utrs(gr)[["Strand", "Feature", "transcript_id"]]
      Strand          Feature transcript_id
    0      +   five_prime_utr            t1
    1      +              CDS            t1
    2      +  three_prime_utr            t1
    3      -  three_prime_utr            t2
    4      -              CDS            t2
    5      -   five_prime_utr            t2

    """
    if STRAND_COL not in gr.columns:
        msg = "label_utrs needs a Strand column: which side is 5' depends on it."
        raise ValueError(msg)
    feature = gr[feature_col].astype(str).str.lower().to_numpy()
    is_utr = feature == "utr"
    labels = gr[feature_col].astype(object).to_numpy(copy=True)
    if is_utr.any():
        cds = gr.loc[feature == "cds"]
        first = cds.groupby(transcript_col)[START_COL].min()
        last = cds.groupby(transcript_col)[END_COL].max()
        utr = gr.loc[is_utr]
        before = (utr[END_COL] <= utr[transcript_col].map(first)).to_numpy()
        after = (utr[START_COL] >= utr[transcript_col].map(last)).to_numpy()
        minus = (utr[STRAND_COL] == "-").to_numpy()
        side = np.select(
            [(before & ~minus) | (after & minus), (after & ~minus) | (before & minus)],
            ["five_prime_utr", "three_prime_utr"],
            default="UTR",
        )
        labels[np.flatnonzero(is_utr)] = side
    out = gr.copy()
    out[feature_col] = pd.Categorical(labels) if isinstance(gr[feature_col].dtype, pd.CategoricalDtype) else labels
    return ensure_pyranges(out)
