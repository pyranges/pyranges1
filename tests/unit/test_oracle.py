"""Interval verbs against brute-force implementations of their definitions.

Intervals are half-open, [Start, End). Two intervals a and b on the same chromosome
pair under slack k when

    a.Start < b.End + k  and  b.Start < a.End + k,
    i.e. max(a.Start, b.Start) - min(a.End, b.End) < k,

so slack=0 is strict overlap, slack=1 also pairs bookended intervals, and slack=k
bridges gaps of up to k - 1 bases. Each test builds random small frames with
bookended and nested intervals and compares a verb with the same definition written
out base by base or pair by pair.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

import pyranges1 as pr

SEEDS = range(25)
SLACKS = range(4)


def random_frame(seed: int, n: int = 14) -> pr.PyRanges:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, 60, n)
    ends = starts + rng.integers(1, 13, n)
    bookend = rng.random(n) < 0.3  # put some intervals right after the previous one
    starts[1:][bookend[1:]] = ends[:-1][bookend[1:]]
    ends = np.maximum(ends, starts + 1)
    return pr.PyRanges(
        pd.DataFrame({"Chromosome": rng.choice(["chr1", "chr2"], n), "Start": starts, "End": ends}),
    )


def rows(gr: pd.DataFrame) -> list[tuple[str, int, int]]:
    return list(zip(gr["Chromosome"], gr["Start"].astype(int), gr["End"].astype(int), strict=True))


def pairs(a: tuple[str, int, int], b: tuple[str, int, int], slack: int = 0) -> bool:
    return a[0] == b[0] and a[1] < b[2] + slack and b[1] < a[2] + slack


def bases(intervals: list[tuple[str, int, int]]) -> set[tuple[str, int]]:
    return {(c, x) for c, s, e in intervals for x in range(s, e)}


def runs(base_set: set[tuple[str, int]]) -> set[tuple[str, int, int]]:
    """Maximal runs of consecutive bases."""
    out = set()
    for chrom in {c for c, _ in base_set}:
        xs = sorted(x for c, x in base_set if c == chrom)
        start = prev = xs[0]
        for x in xs[1:]:
            if x != prev + 1:
                out.add((chrom, start, prev + 1))
                start = x
            prev = x
        out.add((chrom, start, prev + 1))
    return out


def components(intervals: list[tuple[str, int, int]], slack: int) -> list[set[int]]:
    """Connected components of the 'pairs under slack' graph, by union-find."""
    parent = list(range(len(intervals)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(len(intervals)), 2):
        if pairs(intervals[i], intervals[j], slack):
            parent[find(i)] = find(j)
    groups: dict[int, set[int]] = {}
    for i in range(len(intervals)):
        groups.setdefault(find(i), set()).add(i)
    return list(groups.values())


def hull(intervals: list[tuple[str, int, int]], members: set[int]) -> tuple[str, int, int]:
    return (
        intervals[next(iter(members))][0],
        min(intervals[i][1] for i in members),
        max(intervals[i][2] for i in members),
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("slack", SLACKS)
def test_overlap_count_join(seed: int, slack: int) -> None:
    a, b = random_frame(seed), random_frame(seed + 1000)
    ra, rb = rows(a), rows(b)
    hits = {(i, j) for i, x in enumerate(ra) for j, y in enumerate(rb) if pairs(x, y, slack)}

    assert sorted(a.overlap(b, slack=slack).index) == sorted({i for i, _ in hits})
    assert a.count_overlaps(b, slack=slack)["Count"].tolist() == [sum(h == k for h, _ in hits) for k in range(len(ra))]
    joined = a.assign(i=range(len(ra))).join_overlaps(b.assign(j=range(len(rb))), slack=slack)
    assert sorted(zip(joined["i"], joined["j"], strict=True)) == sorted(hits)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("slack", SLACKS)
def test_merge_and_cluster(seed: int, slack: int) -> None:
    """merge(A) = the hulls of the components; cluster(A) = the components."""
    a = random_frame(seed)
    ra = rows(a)
    comps = components(ra, slack)

    merged = a.merge_overlaps(slack=slack)
    assert sorted(rows(merged)) == sorted(hull(ra, c) for c in comps)

    clustered = a.cluster_overlaps(slack=slack)
    got = {}
    for position, cluster in enumerate(clustered.loc[a.index, "Cluster"]):
        got.setdefault(cluster, set()).add(position)
    assert sorted(map(sorted, got.values())) == sorted(map(sorted, comps))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("between", [False, True])
def test_split(seed: int, between: bool) -> None:
    """split(A) = the pieces between consecutive distinct ends that some interval covers
    (with between=True, also the uncovered pieces between the first and last end)."""
    a = random_frame(seed)
    ra = rows(a)
    expected = set()
    for chrom in {c for c, _, _ in ra}:
        ends = sorted({x for c, s, e in ra if c == chrom for x in (s, e)})
        for lo, hi in itertools.pairwise(ends):
            if between or any(c == chrom and s <= lo and hi <= e for c, s, e in ra):
                expected.add((chrom, lo, hi))
    got = rows(a.split_overlaps(between=between))
    assert len(got) == len(set(got))
    assert set(got) == expected


@pytest.mark.parametrize("seed", SEEDS)
def test_subtract_intersect_set_ops(seed: int) -> None:
    a, b = random_frame(seed), random_frame(seed + 1000)
    ra, rb = rows(a), rows(b)
    b_bases = bases(rb)

    # subtract(A, B): each interval of A loses the bases B covers; what is left of it stays its own row(s).
    subtracted = a.assign(i=range(len(ra))).subtract_overlaps(b)
    for i, x in enumerate(ra):
        mine = subtracted[subtracted["i"] == i]
        assert bases(rows(mine)) == bases([x]) - b_bases
        assert len(bases(rows(mine))) == (mine["End"] - mine["Start"]).sum()  # its pieces do not overlap

    # intersect(A, B): one interval per overlapping pair, [max of starts, min of ends).
    expected = sorted((x[0], max(x[1], y[1]), min(x[2], y[2])) for x in ra for y in rb if pairs(x, y))
    assert sorted(rows(a.intersect_overlaps(b))) == expected

    # Set operations work on the merged inputs (slack 0, so bookended intervals stay apart):
    # union(A, B) = merge(A + B); intersect(A, B) = pairwise intersections of merge(A) and merge(B).
    def merged(intervals: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
        return [hull(intervals, c) for c in components(intervals, 0)]

    union = rows(a.set_union_overlaps(b))
    assert sorted(union) == sorted(merged(ra + rb))
    assert bases(union) == bases(ra) | b_bases
    ma, mb = merged(ra), merged(rb)
    inter = rows(a.set_intersect_overlaps(b))
    assert sorted(inter) == sorted((x[0], max(x[1], y[1]), min(x[2], y[2])) for x in ma for y in mb if pairs(x, y))
    assert bases(inter) == bases(ra) & b_bases


@pytest.mark.parametrize("seed", SEEDS)
def test_complement(seed: int) -> None:
    """complement(A) = the uncovered bases between A's first and last base, per chromosome; with
    chromsizes and include_first_interval, all uncovered bases of each chromosome that has intervals."""
    a = random_frame(seed)
    ra = rows(a)
    covered = bases(ra)
    present = {c for c, _, _ in ra}
    sizes = {c: size for c, size in {"chr1": 90, "chr2": 80}.items() if c in present}
    inner, whole = set(), set()
    for chrom in present:
        lo = min(s for c, s, _ in ra if c == chrom)
        hi = max(e for c, _, e in ra if c == chrom)
        inner |= {(chrom, x) for x in range(lo, hi)} - covered
        whole |= {(chrom, x) for x in range(sizes[chrom])} - covered
    assert set(rows(a.complement_ranges())) == (runs(inner) if inner else set())
    got = a.complement_ranges(chromsizes=sizes, include_first_interval=True)
    assert set(rows(got)) == runs(whole)


@pytest.mark.parametrize("seed", SEEDS)
def test_nearest(seed: int) -> None:
    """Distance is 0 for overlapping intervals, otherwise max(a.Start, b.Start) - min(a.End, b.End) + 1
    (bookended intervals at 1, as bedtools closest -d); ties="all" reports every b at the smallest one."""
    a, b = random_frame(seed), random_frame(seed + 1000)
    ra, rb = rows(a), rows(b)

    def distance(x: tuple[str, int, int], y: tuple[str, int, int]) -> int:
        return 0 if pairs(x, y) else max(x[1], y[1]) - min(x[2], y[2]) + 1

    expected = set()
    for i, x in enumerate(ra):
        candidates = [(distance(x, y), j) for j, y in enumerate(rb) if y[0] == x[0]]
        if candidates:
            best = min(d for d, _ in candidates)
            expected |= {(i, j, best) for d, j in candidates if d == best}
    got = a.assign(i=range(len(ra))).nearest_ranges(b.assign(j=range(len(rb))))
    assert set(zip(got["i"], got["j_b"], got["Distance"], strict=True)) == expected
    assert len(got) == len(expected)
