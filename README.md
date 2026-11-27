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

### 1.5.3 (27.11.26)

- Fix Simes p-values with `keep_position=True`. Thanks to Chen-Yang Su (@chenyangsu). [#218](https://github.com/pyranges/pyranges1/pull/218)
- Honor grouping columns in overlap counts and retain them in the output of `split_overlaps(between=True)`. Thanks to Ricardo Pinho (@rpinho). [#216](https://github.com/pyranges/pyranges1/pull/216)
- Use Rust for transcript-to-genome projection in `map_to_global`. Requires `ruranges>=0.2.9`. [#211](https://github.com/pyranges/pyranges1/pull/211)
- Correct the `stats.fisher_exact` example. Thanks to Mohit Arvind Khakharia (@Mohit-Ak). [#212](https://github.com/pyranges/pyranges1/pull/212)

### 1.5.2 (27.10.26)

Superseded by 1.5.3 before publication.

### 1.5.1 (25.10.26)

- Preserve chromosome labels such as `01` in readers under pandas 3.1. GTF reading requires `gtfreader>=0.3.1`.
- Fix range operations on strided arrays from pandas 3.1. Requires `ruranges>=0.2.8`. [#210](https://github.com/pyranges/pyranges1/pull/210)

[Full changelog](https://github.com/pyranges/pyranges1/blob/master/CHANGELOG.txt)

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
