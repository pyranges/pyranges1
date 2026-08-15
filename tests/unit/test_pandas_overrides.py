"""Test that overriden pandas methods work and return PyRanges objects."""

import pickle

import pandas as pd
import pandas.core.groupby
import pytest

import pyranges1 as pr
from pyranges1.core.pyranges_groupby import PyRangesDataFrameGroupBy


def test_getitem() -> None:
    gr = pr.PyRanges({"Chromosome": ["chr1"], "Start": [0], "End": [40]})
    res = gr[gr.Chromosome == "chr1"]
    assert isinstance(res, pr.PyRanges)


def test_loc_get() -> None:
    gr = pr.PyRanges({"Chromosome": ["chr1"], "Start": [0], "End": [40]})
    res = gr.loc[gr.Chromosome == "chr1", ["Chromosome", "Start", "End"]]
    assert isinstance(res, pr.PyRanges)

    res = gr.loc[gr.Chromosome == "chr1", ["Chromosome", "Start"]]
    assert isinstance(res, pd.DataFrame)

    res = gr.loc[gr.Chromosome != "chr1"]
    assert isinstance(res, pr.PyRanges)


@pytest.fixture
def gr():
    return pr.PyRanges(
        {
            "Chromosome": ["chr1"] * 2,
            "Start": [0, 10],
            "End": [40, 20],
            "Gene": ["DonkeyKong", "Mario"],
            "Val": [50, 30],
        },
    )


def test_loc_set() -> None:
    gr = pr.PyRanges({"Chromosome": ["chr1"], "Start": [0], "End": [40]})

    gr.loc[gr.Chromosome == "chr1", "Start"] = 1
    assert isinstance(gr, pr.PyRanges)
    assert gr.Start[0] == 1


def test_groupby_agg(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.agg("first"), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.agg("first"), pr.PyRanges)


def test_groupby_aggregate(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.aggregate("first"), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.aggregate("first"), pr.PyRanges)


def test_groupby_apply(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.apply(lambda df: df.head()), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.apply(lambda df: df.head()), pr.PyRanges)


def test_groupby_bfill(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.bfill(), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.bfill(), pr.PyRanges)


def test_groupby_ffill(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.ffill(), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.ffill(), pr.PyRanges)


def test_groupby_filter(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.filter(lambda df: True), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.filter(lambda df: True), pr.PyRanges)


def test_groupby_first(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.first(), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.first(), pr.PyRanges)


def test_groupby_get_group(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.get_group("chr1"), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.get_group("Mario"), pr.PyRanges)


def test_groupby_head(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.head(), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.head(), pr.PyRanges)


def test_groupby_last(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.last(), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.last(), pr.PyRanges)


def test_groupby_min(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.min(), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.min(), pr.PyRanges)


def test_groupby_pipe(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.pipe(lambda df: df).agg("first"), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.pipe(lambda df: df).agg("first"), pr.PyRanges)


def test_groupby_prod(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.prod("first"), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.prod(), pr.PyRanges)


def test_groupby_sample(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.sample(), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.sample(), pr.PyRanges)


def test_groupby_sum(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.sum(), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.sum(), pr.PyRanges)


def test_groupby_tail(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.tail(1), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.tail(1), pr.PyRanges)


def test_groupby_transform(gr) -> None:
    g = gr.groupby("Chromosome")
    assert isinstance(g.transform("first"), pd.DataFrame)

    g = gr.groupby("Gene")
    assert isinstance(g.transform("first"), pr.PyRanges)


def test_groupby_getitem_frame(gr) -> None:
    g = gr.groupby("Chromosome")
    result = g[["Start", "Gene"]]
    assert isinstance(result, PyRangesDataFrameGroupBy)

    # The Chromosome is now the index, so it is not a PyRanges object
    assert isinstance(result.agg("first"), pd.DataFrame)


def test_groupby_getitem_frame_as_index_false(gr) -> None:
    g = gr.groupby("Chromosome", as_index=False)
    result = g[["Start", "Gene"]]
    assert type(result) is PyRangesDataFrameGroupBy, type(result)

    agg = result.agg("first")

    assert agg.columns.tolist() == ["Chromosome", "Start", "Gene"]
    assert type(agg) is pd.DataFrame, type(agg)


def test_groupby_getitem_series(gr) -> None:
    g = gr.groupby("Chromosome")
    result = g["Start"]
    assert isinstance(result, PyRangesDataFrameGroupBy)

    assert isinstance(result.agg("first"), pd.Series)


def test_groupby_getitem_series_as_index_false(gr) -> None:
    g = gr.groupby("Chromosome", as_index=False)
    result = g.Start
    assert isinstance(result, pandas.core.groupby.SeriesGroupBy), type(result)

    agg = result.agg("first")
    assert isinstance(agg, pd.DataFrame), agg


def test_groupby_getattr_series(gr) -> None:
    g = gr.groupby("Chromosome")
    result = g["Start"]
    assert isinstance(result, PyRangesDataFrameGroupBy)

    assert isinstance(result.agg("first"), pd.Series)


def test_groupby_getattr_series_as_index_false(gr) -> None:
    g = gr.groupby("Chromosome", as_index=False)
    result = g.Start
    assert isinstance(result, pd.core.groupby.SeriesGroupBy), type(result)

    res = result.agg("first")
    # DataFrame because as_index=False
    assert isinstance(res, pd.DataFrame)


def test_drop(gr) -> None:
    assert type(gr.drop("Val", axis=1)) is pr.PyRanges

    # dropping a column a PyRanges requires degrades to a DataFrame
    assert type(gr.drop("Chromosome", axis=1)) is pd.DataFrame


def test_drop_and_return(gr) -> None:
    assert type(gr.drop_and_return("Val", axis=1)) is pr.PyRanges
    assert type(gr.drop_and_return("Chromosome", axis=1)) is pd.DataFrame


def test_reindex(gr) -> None:
    assert type(gr.reindex(columns=["Chromosome", "Start", "End"])) is pr.PyRanges
    assert type(gr.reindex(columns=["Start", "End"])) is pd.DataFrame


def test_range_frame_never_degrades() -> None:
    # a RangeFrame requires no columns, so these always stay a RangeFrame
    rf = pr.RangeFrame({"Start": [0, 10], "End": [40, 20], "Val": [50, 30]})
    assert type(rf.drop("Val", axis=1)) is pr.RangeFrame
    assert type(rf.drop_and_return("Val", axis=1)) is pr.RangeFrame
    assert type(rf.reindex(columns=["Start"])) is pr.RangeFrame


def test_rebuilt_frames_are_pyranges(gr) -> None:
    # pandas builds these from a block manager, without going through PyRanges(...)
    assert type(gr.head(1)) is pr.PyRanges
    assert type(gr[gr.Start >= 0]) is pr.PyRanges
    assert type(gr.copy()) is pr.PyRanges
    assert type(gr.reset_index(drop=True)) is pr.PyRanges


def test_loci_survives_a_rebuild(gr) -> None:
    # the loci accessor is built on demand, because pandas rebuilds frames without __init__
    assert len(gr.head(1).loci["chr1"]) == 1
    assert len(gr.copy().loci["chr1"]) == 2


def test_loci_survives_a_pickle_roundtrip(gr) -> None:
    unpickled = pickle.loads(pickle.dumps(gr))  # noqa: S301
    assert type(unpickled) is pr.PyRanges
    assert len(unpickled.loci["chr1"]) == 2


def test_data_can_be_passed_by_keyword() -> None:
    df = pd.DataFrame({"Chromosome": ["chr1"], "Start": [0], "End": [40]})
    assert type(pr.PyRanges(data=df)) is pr.PyRanges
