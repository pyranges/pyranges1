Columns operations
~~~~~~~~~~~~~~~~~~

.. contents::
   :local:
   :depth: 2



Fetching or writing a column
----------------------------
Most column operations are analogous to pandas.
A single PyRanges column (which are pandas Series) can be extracted through the dot notation, when reading it:

  >>> import pyranges1 as pr
  >>> gr = pr.example_data.chipseq
  >>> gr
  index    |    Chromosome    Start      End        Name    Score    Strand
  int64    |    category      int64      int64      str     int64    category
  -------  ---  ------------  ---------  ---------  ------  -------  ----------
  0        |    chr8          28510032   28510057   U0      0        -
  1        |    chr7          107153363  107153388  U0      0        -
  2        |    chr5          135821802  135821827  U0      0        -
  3        |    chr14         19418999   19419024   U0      0        -
  ...      |    ...           ...        ...        ...     ...      ...
  16       |    chr9          120803448  120803473  U0      0        +
  17       |    chr6          89296757   89296782   U0      0        -
  18       |    chr1          194245558  194245583  U0      0        +
  19       |    chr8          57916061   57916086   U0      0        +
  PyRanges with 20 rows, 6 columns, and 1 index columns.
  Contains 15 chromosomes and 2 strands.

  >>> gr.Chromosome.head()  # doctest: +NORMALIZE_WHITESPACE
  0     chr8
  1     chr7
  2     chr5
  3    chr14
  4    chr12
  Name: Chromosome, dtype: category
  Categories (15, str): ['chr1', 'chr10', 'chr11', 'chr12', ..., 'chr6', 'chr7', 'chr8', 'chr9']


  >>> ( (gr.End + gr.Start)/2 ).head()
  0     28510044.5
  1    107153375.5
  2    135821814.5
  3     19419011.5
  4    106679773.5
  dtype: float64


The ``gr[column_name]`` syntax also extracts a column from a PyRanges object:

  >>> gr['Chromosome'].head()  # doctest: +NORMALIZE_WHITESPACE
  0     chr8
  1     chr7
  2     chr5
  3    chr14
  4    chr12
  Name: Chromosome, dtype: category
  Categories (15, str): ['chr1', 'chr10', 'chr11', 'chr12', ..., 'chr6', 'chr7', 'chr8', 'chr9']


The ``gr[column_name]`` syntax is the only one accepted for assignment (i.e. create or edit a column):

  >>> gr['newchr'] = gr['Chromosome'].str.replace('chr', '')
  >>> gr
  index    |    Chromosome    Start      End        Name    Score    Strand      newchr
  int64    |    category      int64      int64      str     int64    category    str
  -------  ---  ------------  ---------  ---------  ------  -------  ----------  --------
  0        |    chr8          28510032   28510057   U0      0        -           8
  1        |    chr7          107153363  107153388  U0      0        -           7
  2        |    chr5          135821802  135821827  U0      0        -           5
  3        |    chr14         19418999   19419024   U0      0        -           14
  ...      |    ...           ...        ...        ...     ...      ...         ...
  16       |    chr9          120803448  120803473  U0      0        +           9
  17       |    chr6          89296757   89296782   U0      0        -           6
  18       |    chr1          194245558  194245583  U0      0        +           1
  19       |    chr8          57916061   57916086   U0      0        +           8
  PyRanges with 20 rows, 7 columns, and 1 index columns.
  Contains 15 chromosomes and 2 strands.

Extracting multiple columns
---------------------------

As in pandas, you can extract a dataframe with a subset of columns by indexing it with list of column names:

  >>> gr[ ['Chromosome', 'Start'] ].head()
    Chromosome      Start
  0       chr8   28510032
  1       chr7  107153363
  2       chr5  135821802
  3      chr14   19418999
  4      chr12  106679761

When the resulting dataframe has all required genomic location columns (Chromosome, Start, End), then
a PyRanges is returned:

  >>> gr[ ['Chromosome', 'Start', 'End', 'Name'] ].head()
    index  |    Chromosome        Start        End  Name
    int64  |    category          int64      int64  str
  -------  ---  ------------  ---------  ---------  ------
        0  |    chr8           28510032   28510057  U0
        1  |    chr7          107153363  107153388  U0
        2  |    chr5          135821802  135821827  U0
        3  |    chr14          19418999   19419024  U0
        4  |    chr12         106679761  106679786  U0
  PyRanges with 5 rows, 4 columns, and 1 index columns.
  Contains 5 chromosomes.

The method :func:`get_with_loc_columns <pyranges1.PyRanges.get_with_loc_columns>` is a shortcut to extract
any column together with the genomic location columns:

  >>> gr.get_with_loc_columns('Name').head()
    index  |    Chromosome        Start        End  Strand      Name
    int64  |    category          int64      int64  category    str
  -------  ---  ------------  ---------  ---------  ----------  ------
        0  |    chr8           28510032   28510057  -           U0
        1  |    chr7          107153363  107153388  -           U0
        2  |    chr5          135821802  135821827  -           U0
        3  |    chr14          19418999   19419024  -           U0
        4  |    chr12         106679761  106679786  -           U0
  PyRanges with 5 rows, 5 columns, and 1 index columns.
  Contains 5 chromosomes and 1 strands.

  >>> gr.get_with_loc_columns(['Name', 'Score']).head()
    index  |    Chromosome        Start        End  Strand      Name      Score
    int64  |    category          int64      int64  category    str       int64
  -------  ---  ------------  ---------  ---------  ----------  ------  -------
        0  |    chr8           28510032   28510057  -           U0            0
        1  |    chr7          107153363  107153388  -           U0            0
        2  |    chr5          135821802  135821827  -           U0            0
        3  |    chr14          19418999   19419024  -           U0            0
        4  |    chr12         106679761  106679786  -           U0            0
  PyRanges with 5 rows, 6 columns, and 1 index columns.
  Contains 5 chromosomes and 1 strands.


Dropping columns
----------------

Alternatively, you can specify which columns to remove with the pandas dataframe ``drop`` method.
Again, a PyRanges object is returned only if genomic location columns are maintained:

  >>> gr.drop('Name', axis=1)
  index    |    Chromosome    Start      End        Score    Strand      newchr
  int64    |    category      int64      int64      int64    category    str
  -------  ---  ------------  ---------  ---------  -------  ----------  --------
  0        |    chr8          28510032   28510057   0        -           8
  1        |    chr7          107153363  107153388  0        -           7
  2        |    chr5          135821802  135821827  0        -           5
  3        |    chr14         19418999   19419024   0        -           14
  ...      |    ...           ...        ...        ...      ...         ...
  16       |    chr9          120803448  120803473  0        +           9
  17       |    chr6          89296757   89296782   0        -           6
  18       |    chr1          194245558  194245583  0        +           1
  19       |    chr8          57916061   57916086   0        +           8
  PyRanges with 20 rows, 6 columns, and 1 index columns.
  Contains 15 chromosomes and 2 strands.

  >>> gr.drop(['Name', 'Chromosome', 'newchr'], axis=1).head()
         Start        End  Score Strand
  0   28510032   28510057      0      -
  1  107153363  107153388      0      -
  2  135821802  135821827      0      -
  3   19418999   19419024      0      -
  4  106679761  106679786      0      -

The PyRanges method :func:`remove_strand <pyranges1.PyRanges.remove_strand>` is a shortcut to remove the Strand column:

  >>> gr.remove_strand().head()
    index  |    Chromosome        Start        End  Name      Score    newchr
    int64  |    category          int64      int64  str       int64       str
  -------  ---  ------------  ---------  ---------  ------  -------  --------
        0  |    chr8           28510032   28510057  U0            0         8
        1  |    chr7          107153363  107153388  U0            0         7
        2  |    chr5          135821802  135821827  U0            0         5
        3  |    chr14          19418999   19419024  U0            0        14
        4  |    chr12         106679761  106679786  U0            0        12
  PyRanges with 5 rows, 6 columns, and 1 index columns.
  Contains 5 chromosomes.
