# pyranges

## Introduction

Pyranges is a Python library with a Rust backend for efficient and intuitive manipulation of genomics data,
particularly genomic intervals (like genes, genomic features, or reads).
The library is optimized for fast querying and manipulation of genomic annotations.
It enables intuitive and highly efficient pipelines for genomic analysis.

*"Finally ... This was what Python badly needed for years."* - Heng Li

## Version 1
This is version 1.x of pyranges. It is a complete rewrite of the original pyranges library, 
soon to replace the "default" original one (version 0). If you are a v0 user, check the migration guide 
in the documentation.

## Documentation

The pyranges documentation, including installation instructions, API, tutorial, and how-to-pages, is 
available at https://pyranges1.readthedocs.io/

## Recent Changelog

```
# 1.5.2 (27.10.26)
- Project transcript queries through Rust, retaining original query/exon order, indices, strand composition and optional metadata. Negative offsets retain their transcript-end interpretation. References whose ids span multiple chromosomes or strands retain the pandas path. Requires ruranges >= 0.2.9.
- Fix the `stats.fisher_exact` docstring example, which passed TN and FN in each other's slots and showed the odds ratio and p-values for the wrong table. Found and fixed by Mohit Arvind Khakharia (@Mohit-Ak) in pyranges/pyranges0#411

# 1.5.1 (25.10.26)
- Fix the readers under pandas 3.1, which infers the categories of a `"category"` column in `read_csv`: a chromosome named `1` came back as the integer `1` (and `01` merged with it). BED, GFF3, peak and signal files are now read as text and made categorical after; GTF needs gtfreader >= 0.3.1
- Fix `ValueError`/`TypeError: The given array is not contiguous` from most range operations on frames whose columns are strided views (pandas >= 3.1 can return them; coordinates past int32 always reached the kernels uncopied); requires ruranges >= 0.2.8
- Doctests and the unit tests pass on pandas 3.1

# 1.5.0 (14.10.26)
Most of this release is by Piotr Balwierz (@balwierz), and the parallel readers by Marco Mariotti (@marco-mariotti).

Breaking and behaviour changes
- **breaking, gh-204**: pyrle is no longer used: `to_rle` returns a `runlength` `RleBatch` instead of a pyrle `RleDict`, and `to_bigwig` computes its coverage with runlength. `runlength` replaces `pyrle` in the `add-ons` extra
- **breaking, gh-185**: `stats.fdr` is now the Benjamini-Hochberg procedure. It applied p·m/rank without the running minimum, so q-values were not monotone in p and could be too large; it now matches `scipy.stats.false_discovery_control` and R's `p.adjust(method="BH")`. Missing p-values stay missing and are not counted. By @balwierz
- **breaking, gh-184**: `to_bigwig` without `chromosome_sizes` takes each chromosome's size from the data instead of the bundled hg19 sizes, and a chromosome missing from `chromosome_sizes` raises a `ValueError` naming it. By @balwierz
- **breaking, gh-196**: `to_narrowPeak` writes missing columns with the format's "not available" values (Score 0, -1 for SignalValue, PValue, QValue and Peak) instead of `.`, so they read back as numbers; the peak readers refuse a file with the wrong number of fields instead of shifting its columns. By @balwierz
- **breaking, gh-183**: a missing optional dependency raises `ImportError` instead of logging and calling `sys.exit`. By @balwierz
- **gh-191**: `complement_ranges(chromsizes=..., include_first_interval=True)` includes chromosomes without intervals, as `bedtools complement` does; a chromosome missing from `chromsizes` raises a `ValueError` naming it. By @balwierz
- **gh-206**: a missing value in a `match_by` or `group_by` column is a group of its own, as with `groupby(dropna=False)`: it matches only other missing values on the same chromosome
- `read_bigwig` returns `Chromosome` as `category`, like the other readers (gh-181). By @balwierz
- `read_bedgraph`, like `read_bed`, skips `track`/`browser`/`#` lines only at the top of the file (gh-197)

Performance
- `read_bed`, `read_gtf` and `read_gff3` parse with pyarrow on every core when it is installed (the new `fast-io` extra, included in `add-ons`): `read_bed` is up to ~15x faster, `read_gtf` ~3x and `read_gff3` ~5x, with identical output. `pr.set_num_threads` sets the number of threads, and an option chooses the parser. By @marco-mariotti
- grouping (`factorize`) codes each column with `pd.factorize` instead of a groupby: 3-6x faster on every grouped operation (gh-181). By @balwierz
- `to_gtf` and `to_gff3` build the attribute column a whole column at a time: ~12x faster, and nullable integers print as `1`, not `1.0` (gh-194). By @balwierz
- `gr.loci[...]` on a categorical `Chromosome` is ~17x faster (gh-195). By @balwierz

New readers and writers
- `read_sam`, `read_cram`, `read_vcf`, `read_bigbed`, `read_parquet`, `read_pairs`, `read_narrowPeak` and `read_paf`; `to_parquet`, `to_narrowPeak`, `to_pairs` and `to_bigbed` (gh-181). By @balwierz
- `region=` on `read_bed` (tabix), `read_bigwig`, `read_bigbed`, `read_bam`, `read_sam` and `read_cram`: only the records overlapping a samtools-style region are read, through the file's index (gh-189). By @balwierz
- the text readers and `read_parquet` read from URLs: http(s) and ftp with the standard library, s3://, gs:// and others through fsspec with `storage_options` (the new `remote` extra) (gh-190). By @balwierz
- `read_broadPeak`/`to_broadPeak` and `read_gappedPeak`/`to_gappedPeak`; `to_narrowPeak` to a `.gz` path is gzipped, and the peak readers skip track lines (gh-196). By @balwierz
- `read_bedgraph`, `to_bedgraph` and `read_wig` (gh-197). By @balwierz
- `read_chain` reads UCSC chain files (gh-199). By @balwierz
- `read_gff` reads GFF3 or GTF by content; `read_bed` skips track/browser lines and reads bgzip by content; `read_gff3` stops at `##FASTA` (gh-187). By @balwierz
- `to_bed(tabix=True)` writes a sorted, bgzip-compressed, tabix-indexed BED (gh-194). By @balwierz
- `from_string` reads genome browser positions (`chr1:1,000-2,000`) and headerless BED; `PyRanges.from_arrow` builds a PyRanges from Arrow, polars or DuckDB results; `pyranger` offers every reader (gh-195). By @balwierz

New methods
- `liftover_ranges`: UCSC liftOver on PyRanges, identical to `liftOver` including its unmapped reasons (gh-199). By @balwierz
- `coverage_ranges`: depth of coverage as runs of constant depth, as `bedtools genomecov -bg` (gh-198). By @balwierz
- `bigwig_stats`: bigWig signal summarised over each interval, or as a regions x bins matrix (gh-188). By @balwierz
- `resize_ranges`, `promoters` and `shuffle_ranges` (gh-192). By @balwierz
- `nearest_ranges(signed="self"|"other")`: strand-oriented distance, as `bedtools closest -D a` / `-D b` (gh-193). By @balwierz
- `count_overlaps(calculate_coverage=True)`: the fraction of each interval covered (gh-186). By @balwierz
- `explode_blocks`: one row per BED12 block, as `bedtools bed12tobed6` (gh-197). By @balwierz
- `map_to_local(split_at_junctions=False)`: one interval per (interval, transcript) across exon junctions (gh-200). By @balwierz
- `pr.genes`: `select_transcripts` (canonical, MANE or longest), `has_tag` and `label_utrs` for GTF/GFF3 annotations (gh-201). By @balwierz

Fixes
- **fix, gh-206**: a missing value in a `match_by` column could pair intervals on different chromosomes; `window_ranges(group_by=...)` windowed rows with another row's strand when sorting reordered them
- **fix, gh-194**: `to_bed` writes BED12 fields in their positions; with `ThickEnd` but no `ThickStart`, ThickEnd was written as column 7. By @balwierz
- **fix, gh-184**: `read_bigwig` no longer duplicates intervals that cross its internal read windows; BED3 frames write to bigBed. By @balwierz
- **fix, gh-185**: `reverse_complement` complements the IUPAC ambiguity codes. By @balwierz
- **fix, gh-200**: `map_to_global` with the id named alike on both sides raised `KeyError`. By @balwierz
- **fix, gh-187**: `read_gff` passes `duplicate_attr` and `ignore_bad` on to `read_gtf`. By @balwierz
- the `PyRanges` constructor's error lists the columns present as well as the missing ones (gh-195)
- docs: exact definitions of every overlap operation, checked by brute-force oracle tests; `merge_overlaps`' `slack` bridges gaps of up to `slack - 1` (gh-202). By @balwierz

Dependencies
- `gtfreader>=0.3.0`; `runlength` replaces `pyrle`; new extras `fast-io` (pyarrow) and `remote` (fsspec), both in `add-ons`

# 1.4.5 (21.09.26)
- **fix**: explicitly convert the chromosome-size total used by `stats.forbes` to a Python `int`, matching the helper's return type and fixing CI type checking with current pandas stubs

# 1.4.4 (08.09.26)
- `to_bigwig` takes a keyword-only `precomputed` option. With `precomputed=True`, it writes each non-overlapping input range with its `value_col` value directly, bypassing coverage calculation and run merging; `rpm` is ignored and `divide=True` is rejected (gh-176)
- **fix**: weighted coverage no longer leaves floating-point residue in genuinely uncovered regions. An exact active-interval count identifies gaps without erasing legitimate tiny values in covered regions
- **fix, gh-169**: `nearest_ranges(direction="upstream"/"downstream", preserve_input_order=True)` now preserves self's original order across strands. Directional queries split by strand and then concatenated their results, silently grouping rows by strand and making the option inert; `preserve_input_order=False` remains grouped
- **breaking, gh-171**: direct `PyRanges(...)` construction now either returns a `PyRanges` or raises `ValueError` listing missing required columns; it no longer silently returns a plain `DataFrame`. Pandas' internal reconstruction still degrades to a `DataFrame` when an operation removes a required column
- **performance, gh-174**: `RangeFrame` and `PyRanges` are rebuilt directly from pandas' block manager and methods return frames pandas has already constructed, removing redundant frame construction from operations such as `head`, `reset_index`, `reindex`, `drop` and direct `PyRanges(DataFrame)` construction
- **fix, gh-174**: pandas operations now preserve `RangeFrame` or `PyRanges` while their required columns remain and return a plain `DataFrame` otherwise. This restores `RangeFrame` type preservation for `head`, slicing, masking and copying under pandas 3, while preventing invalid `RangeFrame` objects without `Start` or `End`
- **fix, gh-174**: the `loci` accessor is initialized lazily so it works after pandas rebuilds and pickle round-trips; constructing `PyRanges` from an array with an explicit `columns` argument now validates the intended columns
- ignore files generated when doctests run from the repository root (gh-171)

# 1.4.3 (10.08.26)
- `nearest_ranges` takes `ties`, on both `PyRanges` and `RangeFrame`. The default, `"all"`, is unchanged: every interval of `other` at the winning distance is reported. `ties="first"` reports one of them, so each interval of self appears at most once. Which one is not specified — only that the same input gives the same answer every time; nothing is sorted to decide it, because the sort would cost more than the option saves
- this matters most where it is least visible. Overlapping intervals are all at distance 0, so with the default `exclude_overlaps=False` "every tied interval" means "every interval of other that self overlaps": one interval of self covered by 500 intervals of other produces 500 rows, all of them equally the nearest. On 100 million hg38-like intervals against themselves that is 1.1 billion output rows where one per query would be 7.7 million — 144x — which is the difference between fitting in memory and not
- the tied rows are never built, so `ties="first"` saves the memory and not only the time. Materialising them and keeping one per query afterwards would have cost the same as `"all"` and more time; instead the overlap sweep stops at the first hit per interval of self, each directional sweep emits one row per distinct distance, and the merge keeps one row per distance bucket. That work is in `ruranges_core::nearest`, so the floor on `ruranges` rises to 0.1.8
- `k` still counts distinct distances, not rows. `ties="first"` with `k=2` reports two rows, one per distance, rather than the combination being rejected
- `ties` names the same thing here as elsewhere in the field: bedtools spells it `closest -t first`, GenomicRanges `select="arbitrary"`, BEDOPS `--closest`. bioframe, polars-bio and pyranges 0.x report one interval per query and offer no way to ask for the other behaviour
- an unknown `ties` raises `ValueError` naming the valid options, as `direction` and `multiple` already do, rather than reaching the kernel: a Rust panic is not an `Exception`, so `except Exception` cannot catch it and the interpreter is aborted

# 1.4.2 (31.07.26)
- **fix, severe**: `PyRanges.nearest_ranges(direction="upstream"/"downstream")` ignored strand on minus-strand intervals, returning the neighbour on the wrong side and so inverting the biological meaning of every directional nearest query on the reverse strand — nearest promoter, nearest TSS. For a minus-strand feature, upstream is the *higher* coordinate, as `PyRanges.upstream()` has always had it. `self` was split into its forward and reverse halves and both halves were then queried with the same coordinate direction. The strand-to-coordinate-direction mapping is now a single four-entry table rather than four hand-picked direction strings at four call sites, which is what let the bug through. Reported and fixed by Mike German (@steps-re) in gh-162
- **fix**: `RangeFrame.nearest_ranges` did not validate `direction`. Any value the coordinate space does not know — including the strand-aware `"upstream"`/`"downstream"`, which belong to `PyRanges` — reached `ruranges` and raised a `PanicException` reading `Invalid direction string`. That is not an `Exception`, so `except Exception` could not catch it and the interpreter was aborted. It now raises `ValueError` naming the valid options, as `multiple` has since 1.4.0
- **fix**: `PyRanges.nearest_ranges` with a directional `direction` on a frame without valid strand failed with `UndefinedVariableError: name 'Strand' is not defined`, leaked from a pandas query. It now raises `ValueError` explaining that upstream and downstream are strand-aware. `direction="any"` is unaffected
- the two direction vocabularies are named for what they are: `VALID_NEAREST_TYPE` is now `VALID_GENOMIC_DIRECTION_TYPE` (`any`/`upstream`/`downstream`, strand-aware) and `VALID_DIRECTION_TYPE` is now `VALID_COORDINATE_DIRECTION_TYPE` (`any`/`forward`/`backward`, where forward is always the higher coordinate). Both are internal to `pyranges1.core.names`. `VALID_NEAREST_OPTIONS` listed `upstream` twice and never `downstream`; it was unused, and the corrected list now backs the validation above
- **fix, gh-166**: `strand_behavior="opposite"` dropped every non-location column of `other`. To flip the strand, `prepare_by_binary` took `other.loc[:, [*RANGE_COLS, *by]].copy()` — a copy that was also a projection — so `join_overlaps` and `nearest_ranges` lost the columns they exist to report. `left.join_overlaps(right, strand_behavior="opposite")` returned no `ID_b`/`Score_b` at all, while the same call with `"same"` returned both, making the loss depend on the strand option rather than on the data. Downstream code indexing those columns raised `KeyError` for one strand setting and worked for the others. The copy is now of the whole frame; the projection was never needed
- **fix**: `complement_ranges` emitted intervals of non-positive length at the externally bounded edges — an interval no validator in the library accepts. With `chromsizes` and a last interval ending exactly on the chromosome size it emitted a terminal gap of zero length (`Start == End == size`); that gap is now dropped, as is any other gap the kernel returns with `End <= Start`
- **breaking**: `complement_ranges(include_first_interval=True)` now raises `ValueError` when any interval starts below zero. The first complement interval runs from coordinate 0 to the first interval, so a negative start produced a gap of *negative* length: one row `chr1:[-1, 0)+` came back as `Start=0, End=-1`. Negative coordinates remain legal everywhere else — `extend_ranges`, `upstream` and `five_end` all produce them by design — this is checked only where it would produce an invalid result
- **fix**: `join_overlaps` with `report_overlap_column` reported an overlap for rows that matched nothing. `DataFrame.min`/`.max` skip nulls, so an unmatched row of a `left`, `right` or `outer` join got the length of the one interval present — `Overlap = 1` on a row whose every `_b` column is null. Unmatched rows now report null
- **breaking**: `split_overlaps` no longer drops `Strand` when `use_strand=False`. It dropped the column merely because it was not a grouping key, so `gr.split_overlaps(use_strand=False)` lost strand information the input carried and no other method discards this way. Every output interval descends from an input interval and now keeps its metadata. `between=True` is unchanged: gap intervals descend from no input row, so they keep only the location columns, and `Strand` among them only when it was a grouping key
- **breaking**: `group_cumsum` accumulated across chromosomes. With no `group_by` it ran a single cumulative sum over the whole frame, so two rows of length 10 on `chr1` and `chr2` came back as `[0,10)` and `[10,20)` — one coordinate space spanning two chromosomes. `Chromosome`, and `Strand` when strand-aware, now always partition the cumulative space. The documented behaviour ("when *None* all intervals on the same chromosome are cumulated together") was already this; only the code disagreed. Naming a column in `group_by` no longer drops the chromosome partition either. The caller's own keys are placed *before* the implicit ones, because the key order decides how groups are numbered and therefore the order rows come back in when `keep_order=False`; only the key set decides which intervals share a cumulative space. `map_to_local`, `map_to_global` and `tile_ranges(add_window_id=True)` all call `group_cumsum` internally with a transcript-level key, and their output row order is unchanged by this fix
- **breaking**: `to_bed`, `to_gtf` and `to_gff3` always infer compression from the file extension. The default was `None`, which pandas reads as *no compression even for a .gz path*, while all three docstrings said "infer" — `to_bed("x.bed.gz")` silently wrote plain text under a `.gz` name. `None` is now treated as `"infer"`, so there is no longer a way to write uncompressed output to a `.gz` path; pass a different extension

# 1.4.1 (31.07.26)
- **fix, severe**: `RangeFrame.sort_ranges` passed the kernel's arguments in the wrong order — group ids landed in `starts`, `Start` in `ends` and `End` in `groups` — so the primary sort key was `End`, not `Start`. On a 100,000-row frame with realistic coordinates, 97,964 rows came back in the wrong position. `PyRanges.sort_ranges` was unaffected. The `groups` argument of `ruranges.numpy.sort_intervals` is keyword-only from `ruranges>=0.1.7`, so the transposition can no longer happen silently
- **fix, severe**: `RangeFrame.sort_ranges()` with no `by` ran a full Python-level pass over every row to build an array that is provably all zeros — 97% of the total cost of an unkeyed sort (5.77 s of 5.95 s at 5,000,000 rows). The replacement ranker is never invoked for zero columns
- **breaking**: `RangeFrame.sort_ranges` no longer takes `natsort`, and orders string keys lexically. Natural ordering exists for chromosome names, and a `RangeFrame` has no `Chromosome` column; `PyRanges.sort_ranges` keeps `natsort` unchanged. `rf.sort_ranges(by="transcript_id")` therefore now puts `t10` before `t9`
- **breaking**: `RangeFrame.sort_ranges` no longer takes `sort_rows_reverse_order`. It was declared and never called, and it was the one parameter in the family requiring a per-row Python sequence. Callers wanting 5'→3' ordering use `PyRanges.sort_ranges`
- both `sort_ranges` methods take `by` keys that *relocate*: the full key list is `Chromosome, Strand, *by, Start, End`, and any column named in `by` is taken out of its implicit position and used where you put it. `by=["Strand", "Chromosome"]` sorts by strand first; `by=["Start", "End", "score"]` sorts by a key *after* the coordinates. Both halves of gh-94. The rule applies per column, so `by=["score", "Chromosome"]` gives `Strand, score, Chromosome, Start, End` — name every key you care about
- both `sort_ranges` methods take `sort_descending`, naming keys to reverse. Any key qualifies, the implicit `Chromosome`, `Strand`, `Start` and `End` included. A name that is not a sort key raises `ValueError` rather than being ignored. On the coordinate keys it composes with `use_strand` by XOR, so a reversed row and a reversed key cancel out
- **deprecated**: `RangeFrame.sort_by_position` now warns and will be removed next release. `sort_ranges()` with no arguments sorts by the same columns, and on `PyRanges` `sort_by_position` is a trap: it sorts globally by position and ignores `Chromosome`
- sorting is faster and the ordering rule is now shared with `polaranges`. Key columns are coded one at a time rather than as key tuples — the cost is the number of distinct values per column, not the number of distinct combinations, which approaches the row count — and the natural ordering of a column's distinct string values happens in Rust, in `ruranges_core::ranks`, which `polaranges` also calls. Doing it in Python cost 24.9 s for 10 million distinct values against 0.7 s in Rust and was the largest line item in every large sort. `natsort` is no longer a runtime dependency of the sort path
- document that `use_strand=False` does not remove `Strand` from the sort keys: it only stops negative-strand rows being ordered 3'→5'. To sort without grouping by strand, drop or rename the column

# 1.4.0 (30.07.26)
- **breaking**: `RangeFrame.overlap` now defaults to `multiple=False`, matching `PyRanges.overlap`. It previously defaulted to `"all"`, so `rf.overlap(other)` and `gr.overlap(other)` — the same bare call on a class and its subclass — meant different things: one reported every overlapping pair, the other filtered. A bare `overlap()` is now a filter everywhere. Pass `multiple=True` for the old generic behaviour
- `RangeFrame.overlap` also takes `multiple` as a bool now, like `PyRanges.overlap`. `overlap` returns rows of self and nothing from other, so the only thing this option can change is how many times a row appears: `"first"` and `"last"` would select the same rows and differ in output order alone. A string now raises `TypeError` instead of being silently truthy, and the option is keyword-only on both
- `PyRanges.overlap` used to derive the string the kernel wants with a bare truthiness test, so `multiple="first"` and `multiple="last"` silently meant `"all"` — the opposite of what was asked, with no error. On a 200k x 500k overlap that is a 47% row-count difference
- `intersect_overlaps`, `join_overlaps` and `set_intersect_overlaps` keep the `{"all", "first", "last"}` vocabulary: they return information taken from other, so `"first"` and `"last"` genuinely select different output there
- reject an unknown `multiple` with a `ValueError` naming the valid options; it used to reach `ruranges` and abort the interpreter with a Rust panic reading `invalid overlap_type string: "Invalid direction string"`, which `except Exception` cannot catch
- correct the `multiple` docstrings that described `overlap` and `join_overlaps` as reporting "subintervals". Only `intersect_overlaps`, `set_intersect_overlaps` and `subtract_overlaps` return subintervals; `overlap` returns whole intervals of self and `join_overlaps` returns them alongside other's columns
- remove `"contained"` from `VALID_OVERLAP_TYPE`. It was declared but `ruranges>=0.1.5` rejects the overlap type, so it only ever panicked, and it duplicated `contained_intervals_only=True`, which is now the single way to express containment. `multiple="contained"` raises a `ValueError` naming that argument
- annotate `tile_ranges`'s `use_strand` as `VALID_USE_STRAND_TYPE`, like every other method taking it. It was annotated `bool` while already accepting and handling `"auto"` at runtime, so type checkers rejected a valid call

```

## Install

Pyranges1 requires python ≥3.12. Minimal installation: 

```bash
pip install pyranges1
```

This installs and requires `ruranges>=0.1.3` automatically.

Installation including all optional dependencies:

```bash
pip install pyranges1[all]
```

For faster file reading, install `pyarrow`. `read_bed` then parses on every
core instead of one, which is 10-18x faster on large files and returns exactly
the same frame:

```bash
pip install pyranges1[fast-io]   # or [add-ons], or [all]
```

Reading then runs on every core by default. To pin how many, or to go back to
the single-threaded parser:

```python
import pyranges1 as pr

pr.set_num_threads(4)                          # kernels and readers alike
pr.options.set_option("use_pyarrow", False)    # the pandas parser, same frame
```

`set_num_threads` reaches the interval kernels too, but those read their count
once, when the first operation runs — call it before that or they keep the
count they started with.

Details at https://pyranges1.readthedocs.io/en/latest/installation.html


## Features

  - fast
  - memory-efficient
  - featureful
  - pythonic/pandastic

## Paper/Cite

For v1:

Stovner EB, Ticó M, Muñoz del Campo E, Pallarès-Albanell J, Chawla K, Sætrom P, Mariotti M (2025) Pyranges v1: a Python framework for ultrafast sequence interval operations.
*bioRxiv* 2025.12.11.693639; doi: https://doi.org/10.64898/2025.12.11.693639


For v0:

Stovner EB, Sætrom P (2020) PyRanges: efficient comparison of genomic intervals in Python. 
*Bioinformatics 36(3):918-919*  http://dx.doi.org/10.1093/bioinformatics/btz615

## Supporting pyranges

  - most importantly, cite pyranges if you use it. It is the main metric funding sources care about.
  - use pyranges in Stack Overflow/biostars questions and answers
  - star the repo (possibly important for github visibility and as a proxy for project popularity)

## Asking for help

If you encounter bugs, or the documentation is not enough a cannot accomplish a specific task of interest, 
or if you'd like new features implemented, open an Issue at github: https://github.com/pyranges/pyranges/issues

## Contributing to pyranges

Pyranges accepts code contributions in form of pull request. 
For details, visit [https://pyranges1.readthedocs.io/developer_guide.html](https://pyranges1.readthedocs.io/en/latest/developer_guide.html)

## Cheatsheet
![cheatsheet](https://raw.githubusercontent.com/pyranges/pyrangeyes/for_pyranges1_1/images/pyranges_cheatsheet.png)
(The cheatsheet above was created with pyrangeyes, a companion graphical library:  https://pyrangeyes.readthedocs.io/)
