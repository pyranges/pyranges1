import pandas as pd
import pytest

import pyranges1 as pr

PEAKS = pr.PyRanges(
    {
        "Chromosome": ["chr1", "chr2"],
        "Start": [100, 50],
        "End": [200, 90],
        "Name": ["p1", "p2"],
        "Score": [500, 20],
        "Strand": ["+", "-"],
        "SignalValue": [5.5, 1.0],
        "PValue": [3.2, 0.5],
        "QValue": [2.1, 0.1],
        "Peak": [50, 10],
    }
)


@pytest.mark.parametrize(
    ("write", "read", "columns"),
    [
        ("to_narrowPeak", pr.read_narrowPeak, 10),
        ("to_broadPeak", pr.read_broadPeak, 9),
        ("to_gappedPeak", pr.read_gappedPeak, 15),
    ],
)
@pytest.mark.parametrize("suffix", ["", ".gz"])
def test_peak_files_round_trip(tmp_path, write, read, columns, suffix) -> None:
    path = tmp_path / f"peaks{suffix}"
    getattr(PEAKS, write)(str(path))
    if suffix:
        assert path.read_bytes()[:2] == b"\x1f\x8b"
    back = read(path)
    assert back.shape[1] == columns
    for column in ("Chromosome", "Start", "End", "Name", "Score", "SignalValue", "QValue"):
        assert back[column].astype(str).tolist() == PEAKS[column].astype(str).tolist()


def test_missing_values_are_written_as_the_formats_define_them(tmp_path) -> None:
    bare = pr.PyRanges({"Chromosome": ["chr1"], "Start": [100], "End": [200]})
    assert bare.to_narrowPeak() == "chr1\t100\t200\t.\t0\t.\t-1\t-1\t-1\t-1\n"
    path = tmp_path / "bare.narrowPeak"
    bare.to_narrowPeak(str(path))
    back = pr.read_narrowPeak(path)
    assert all(pd.api.types.is_numeric_dtype(back[c]) for c in ("Score", "SignalValue", "PValue", "QValue", "Peak"))
    assert bare.to_gappedPeak().split("\t")[6:12] == ["100", "200", "0", "1", "100,", "0,"]


def test_peak_readers_skip_track_lines_and_refuse_other_formats(tmp_path) -> None:
    path = tmp_path / "t.narrowPeak"
    path.write_text('track type=narrowPeak name="x"\nbrowser position chr1:1-1000\nchr1\t100\t200\tp\t5\t.\t1.5\t2\t3\t50\n')
    assert pr.read_narrowPeak(path)["Name"].tolist() == ["p"]
    with pytest.raises(ValueError, match="broadPeak file has 9"):
        pr.read_broadPeak(path)
