"""map_to_local(split_at_junctions=False) against a per-base oracle; map_to_global with the id named alike on both sides."""

import numpy as np
import pandas as pd
import pytest

import pyranges1 as pr


def random_transcripts(rng: np.random.Generator, n: int) -> pd.DataFrame:
    rows = []
    for t in range(n):
        pos = int(rng.integers(0, 5000))
        strand = "+" if rng.random() < 0.5 else "-"
        for _ in range(int(rng.integers(1, 5))):
            start = pos + int(rng.integers(1, 60))
            pos = start + int(rng.integers(1, 80))
            rows.append(("chr1", start, pos, strand, f"tx{t}"))
    exons = pd.DataFrame(rows, columns=["Chromosome", "Start", "End", "Strand", "transcript_id"])
    return exons.sample(frac=1, random_state=0).reset_index(drop=True)  # exons in no particular order


def oracle(features: pd.DataFrame, exons: pd.DataFrame) -> set[tuple]:
    """(feature index, transcript, local start, local end, strand) from base-by-base mapping."""
    out = set()
    for tx, ex in exons.groupby("transcript_id"):
        minus = ex["Strand"].iloc[0] == "-"
        bases = np.concatenate([np.arange(s, e) for s, e in zip(ex["Start"], ex["End"], strict=True)])
        bases = np.sort(bases)[::-1] if minus else np.sort(bases)
        local = {b: i for i, b in enumerate(bases)}
        for idx, f in features.iterrows():
            hit = [local[b] for b in range(f["Start"], f["End"]) if b in local]
            if hit:
                strand = "+" if (f["Strand"] == "-") == minus else "-"
                out.add((idx, tx, min(hit), max(hit) + 1, strand))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_map_to_local_joined_matches_per_base(seed: int) -> None:
    rng = np.random.default_rng(seed)
    exons = random_transcripts(rng, 8)
    starts = rng.integers(0, 6000, 60)
    features = pd.DataFrame(
        {
            "Chromosome": "chr1",
            "Start": starts,
            "End": starts + rng.integers(1, 400, 60),
            "Strand": rng.choice(["+", "-"], 60),
        },
        index=np.arange(100, 160),
    )
    got = pr.PyRanges(features).map_to_local(pr.PyRanges(exons), "transcript_id", split_at_junctions=False)
    rows = set(zip(got.index, got["Chromosome"], got["Start"], got["End"], got["Strand"], strict=True))
    assert len(rows) == len(got)  # one row per (feature, transcript)
    assert rows == oracle(features, exons)


def test_map_to_local_joined_keep_loc_spans_exons() -> None:
    exons = pr.PyRanges(
        {"Chromosome": "chr1", "Start": [5000, 5100, 5200], "End": [5050, 5150, 5250], "Strand": "-", "transcript_id": "tx3"}
    )
    feature = pr.PyRanges({"Chromosome": ["chr1"], "Start": [5040], "End": [5110], "Strand": ["-"]})
    got = feature.map_to_local(exons, "transcript_id", keep_loc=True, split_at_junctions=False)
    assert got[["Chromosome", "Start", "End", "Start_global", "End_global"]].values.tolist() == [["tx3", 90, 110, 5000, 5150]]


@pytest.mark.parametrize("keep_id", [False, True])
@pytest.mark.parametrize("keep_loc", [False, True])
def test_map_to_global_on_same_id_column(keep_id: bool, keep_loc: bool) -> None:
    exons = pr.PyRanges(
        {"Chromosome": "chr1", "Start": [100, 300], "End": [200, 400], "Strand": "+", "transcript_id": "tx1"}
    )
    local = pr.PyRanges({"Chromosome": ["x"], "Start": [90], "End": [120], "transcript_id": ["tx1"]})
    got = local.map_to_global(exons, "transcript_id", local_on="transcript_id", keep_id=keep_id, keep_loc=keep_loc)
    assert got[["Chromosome", "Start", "End"]].values.tolist() == [["chr1", 190, 200], ["chr1", 300, 320]]
    assert got["transcript_id"].tolist() == ["tx1", "tx1"]
