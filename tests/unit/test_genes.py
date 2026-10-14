"""pyranges1.genes: select_transcripts, has_tag, label_utrs."""

import warnings

import pandas as pd
import pytest

import pyranges1 as pr


def annotation() -> pr.PyRanges:
    # g1: t1 longest, t2 MANE_Select and (on its exon row) Ensembl_canonical, t3 Ensembl_canonical.
    # g2: t4 and t5 equally long, no tags: t4 by id. A gene row belongs to no transcript.
    return pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": "chr1",
                "Start": [100, 100, 100, 300, 100, 500, 500, 600],
                "End": [900, 900, 400, 400, 300, 700, 700, 650],
                "Strand": "+",
                "Feature": [
                    "gene",
                    "transcript",
                    "transcript",
                    "exon",
                    "transcript",
                    "transcript",
                    "transcript",
                    "exon",
                ],
                "gene_id": ["g1", "g1", "g1", "g1", "g1", "g2", "g2", "g2"],
                "transcript_id": [None, "t1", "t2", "t2", "t3", "t5", "t4", "t4"],
                "tag": [
                    None,
                    "basic",
                    "basic,MANE_Select",
                    "basic,Ensembl_canonical",
                    "Ensembl_canonical,basic",
                    None,
                    None,
                    None,
                ],
            },
            index=[7, 7, 3, 3, 9, 1, 2, 2],
        )
    )


@pytest.mark.parametrize(
    ("how", "expected"), [("longest", {"t1", "t4"}), ("canonical", {"t2", "t4"}), ("mane", {"t2", "t4"})]
)
def test_select_transcripts(how: str, expected: set[str]) -> None:
    gr = annotation()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = pr.genes.select_transcripts(gr, how)  # type: ignore[arg-type]
    assert set(got["transcript_id"].dropna()) == expected
    kept = gr["transcript_id"].isna() | gr["transcript_id"].isin(expected)
    pd.testing.assert_frame_equal(got, gr[kept.to_numpy()])  # every row of a chosen transcript, order and index kept


def test_select_transcripts_tag_spread_over_rows() -> None:
    # t3 carries Ensembl_canonical on its transcript row and t2 only on its exon row: both count.
    gr = annotation()
    gr = pr.PyRanges(gr.assign(tag=gr["tag"].str.replace("MANE_Select", "x")))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = pr.genes.select_transcripts(gr, "canonical")
    assert set(got["transcript_id"].dropna()) == {"t2", "t4"}  # t2 and t3 both canonical: longer t2 wins
    assert [str(x.message) for x in w] == [
        "Ensembl_canonical tags 1 of 2 genes; the rest keep their longest transcript."
    ]


def test_select_transcripts_hints_duplicate_attr() -> None:
    gr = annotation()
    gr = pr.PyRanges(gr.assign(tag=gr["tag"].str.split(",").str[-1]))  # what duplicate_attr=False leaves
    with pytest.warns(UserWarning, match="duplicate_attr=True"):
        pr.genes.select_transcripts(gr, "canonical")


def test_select_transcripts_errors() -> None:
    gr = annotation()
    with pytest.raises(ValueError, match="how must be"):
        pr.genes.select_transcripts(gr, "first")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="No 'tag' column"):
        pr.genes.select_transcripts(pr.PyRanges(gr.drop(columns="tag")), "canonical")


def test_has_tag_whole_tags() -> None:
    gr = pr.PyRanges(
        {"Chromosome": "chr1", "Start": [0, 1, 2, 3], "End": [1, 2, 3, 4], "tag": ["a.b,c", "c", "xc,cx", None]}
    )
    assert pr.genes.has_tag(gr, "c").tolist() == [True, True, False, False]
    assert pr.genes.has_tag(gr, "a.b").tolist() == [True, False, False, False]
    assert pr.genes.has_tag(gr, "a").tolist() == [False, False, False, False]


def test_label_utrs() -> None:
    gr = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": "chr1",
                "Start": [0, 10, 50, 60, 0, 10, 50, 15, 20, 0],
                "End": [10, 50, 60, 70, 10, 50, 60, 25, 30, 30],
                "Strand": ["+", "+", "+", "+", "-", "-", "-", "+", "+", "-"],
                "Feature": ["UTR", "CDS", "UTR", "five_prime_UTR", "utr", "CDS", "UTR", "UTR", "CDS", "UTR"],
                "transcript_id": ["a", "a", "a", "a", "b", "b", "b", "c", "c", "d"],
            },
            index=[5, 5, 4, 4, 3, 3, 2, 2, 1, 1],
        ).astype({"Feature": "category"})
    )
    got = pr.genes.label_utrs(gr)
    assert got["Feature"].tolist() == [
        "five_prime_utr",
        "CDS",
        "three_prime_utr",
        "five_prime_UTR",  # +; an explicit label is left alone
        "three_prime_utr",
        "CDS",
        "five_prime_utr",  # -
        "UTR",
        "CDS",  # overlaps the CDS: not guessed
        "UTR",  # no CDS
    ]
    assert isinstance(got["Feature"].dtype, pd.CategoricalDtype)
    pd.testing.assert_frame_equal(got.drop(columns="Feature"), gr.drop(columns="Feature"))
