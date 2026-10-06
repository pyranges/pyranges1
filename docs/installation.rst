
Installation
~~~~~~~~~~~~

Pyranges1 requires python ≥ 3.12. If necessary, create a new conda environment::

    conda create -yn pr python
    conda activate pr


The preferred way to install pyranges1 is via pip::

    pip install pyranges1

The command above will install a **minimal version** of pyranges1.
Pyranges has several optional dependencies, required for certain functionalities.

To **install all optional dependencies**, use::

    pip install pyranges1[all]

Here you can see the optional dependencies grouped by functionality::

    # user add-ons: to fetch sequences, read BAM files, faster file reading ...
    pip install pyranges1[add-ons]

    # faster file reading only: installs pyarrow, which makes read_bed
    # 10-18x faster on large files. Included in add-ons and in all.
    pip install pyranges1[fast-io]

    # command line: to use the pyranger command-line tool
    pip install pyranges1[cli]

    # development: for testing, linting, type checking, generating documentation
    pip install pyranges1[dev]

    # documentation: for building the documentation
    pip install pyranges1[docs]

To inspect the list of dependencies, check the pyproject.toml file in the repository.
