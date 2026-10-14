import pandas as pd
import pytest

import pyranges1 as pr


def test_write_bed(chip_10, tmp_path) -> None:
    outfile = tmp_path / "deleteme.bed"
    chip_10.to_bed(outfile)


def test_write_bed_no_path(chip_10) -> None:
    result = chip_10.to_bed()
    assert isinstance(result, str)


def test_write_gtf(chip_10, tmp_path) -> None:
    outfile = tmp_path / "deleteme.gtf"
    chip_10.to_gtf(outfile)


def test_write_gff3(chip_10, tmp_path) -> None:
    outfile = tmp_path / "deleteme.gff3"
    chip_10.to_gff3(outfile)


def test_write_gtf_no_path(chip_10) -> None:
    result = chip_10.to_gtf()
    assert isinstance(result, str)


def test_write_bigwig(chip_10, tmp_path, chromsizes) -> None:
    try:
        outfile = tmp_path / "deleteme.bigwig"
        outpath = str(outfile)
        chip_10.to_bigwig(outpath, chromosome_sizes=chromsizes)
    except SystemExit:
        pass


def test_to_bigwig_divide_keeps_single_run_tracks() -> None:
    """A divided track that reduces to one run must survive.

    ``pyrle``'s ``defragment`` zeroes the value whenever the merged result is a
    single run, and ``to_ranges`` then discards zero-valued runs as uncovered,
    so such a track disappeared entirely (gh-160).
    """
    import numpy as np

    # One interval, value 2 over coverage 1: log2(2 / 1) == 1.0.
    single = pr.PyRanges(
        pd.DataFrame({"Chromosome": ["chr1"], "Start": [0], "End": [4], "Value": [2]}),
    )
    result = single.to_bigwig(
        None, chromosome_sizes={"chr1": 10}, value_col="Value", divide=True, rpm=False, return_data=True
    )
    assert len(result) == 1
    assert result["Score"].tolist() == [1.0]

    # Two chromosomes, one of which reduces to a single run: neither may be lost.
    pair = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": ["chr2", "chr1"],
                "Start": [0, 1],
                "End": [1, 2],
                "Value": [2, 1],
            },
        ),
    )
    both = pair.to_bigwig(
        None,
        chromosome_sizes={"chr1": 22, "chr2": 21},
        value_col="Value",
        divide=True,
        rpm=False,
        return_data=True,
    )
    assert sorted(both["Chromosome"].astype(str).tolist()) == ["chr1", "chr2"]
    scores = dict(zip(both["Chromosome"].astype(str), both["Score"], strict=True))
    assert scores["chr2"] == 1.0
    # chr1 is uncovered where its value track is defined, so the ratio is undefined.
    assert np.isnan(scores["chr1"])


def test_to_bigwig_divide_merges_adjacent_equal_runs() -> None:
    """Consecutive runs with the same ratio still collapse into one interval."""
    gr = pr.PyRanges(
        pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr1"],
                "Start": [0, 4],
                "End": [4, 8],
                "Value": [2, 2],
            },
        ),
    )
    result = gr.to_bigwig(
        None, chromosome_sizes={"chr1": 20}, value_col="Value", divide=True, rpm=False, return_data=True
    )
    assert len(result) == 1
    assert result["Start"].tolist() == [0]
    assert result["End"].tolist() == [8]
    assert result["Score"].tolist() == [1.0]


def test_to_bigwig_precomputed_does_not_materialize_float_rounding_gaps(tmp_path) -> None:
    """A change in scale must not turn an uncovered range into data."""
    import pyBigWig

    expected_coordinates = [(0, 1), (1, 2), (3, 4)]

    for index, scale in enumerate((0.5152221054097652, 0.5152221054097644)):
        precomputed_coverage = pr.PyRanges(
            pd.DataFrame(
                {
                    "Chromosome": ["chr1"] * 3,
                    "Start": [0, 1, 3],
                    "End": [1, 2, 4],
                    "Score": [3 * scale, scale, scale],
                },
            ),
        )

        path = tmp_path / f"precomputed_{index}.bw"
        precomputed_coverage.to_bigwig(
            str(path),
            chromosome_sizes={"chr1": 10},
            value_col="Score",
            precomputed=True,
        )

        bw = pyBigWig.open(str(path))
        intervals = bw.intervals("chr1")
        bw.close()

        assert [(start, end) for start, end, _ in intervals] == expected_coordinates
        assert [value for _, _, value in intervals] == pytest.approx([3 * scale, scale, scale])


def test_to_bigwig_weighted_coverage_does_not_materialize_float_rounding_gaps() -> None:
    expected_coordinates = [(0, 1), (1, 2), (3, 4)]

    for scale in (0.5152221054097652, 0.5152221054097644):
        weighted_ranges = pr.PyRanges(
            {
                "Chromosome": ["chr1"] * 3,
                "Start": [0, 1, 3],
                "End": [1, 2, 4],
                "Score": [3 * scale, scale, scale],
            },
        )
        result = weighted_ranges.to_bigwig(
            None,
            chromosome_sizes={"chr1": 10},
            value_col="Score",
            rpm=False,
            return_data=True,
        )

        assert list(zip(result["Start"], result["End"], strict=True)) == expected_coordinates

    tiny = pr.PyRanges({"Chromosome": ["chr1"], "Start": [0], "End": [1], "Score": [1e-20]})
    tiny_result = tiny.to_bigwig(
        None,
        chromosome_sizes={"chr1": 1},
        value_col="Score",
        rpm=False,
        return_data=True,
    )
    assert tiny_result["Score"].tolist() == [1e-20]


def test_to_bigwig_precomputed_validates_arguments() -> None:
    gr = pr.PyRanges({"Chromosome": ["chr1"], "Start": [0], "End": [1], "Score": [1.0]})

    with pytest.raises(ValueError, match="requires value_col"):
        gr.to_bigwig(None, chromosome_sizes={"chr1": 1}, precomputed=True, return_data=True)

    with pytest.raises(ValueError, match="incompatible"):
        gr.to_bigwig(
            None,
            chromosome_sizes={"chr1": 1},
            value_col="Score",
            divide=True,
            precomputed=True,
            return_data=True,
        )
