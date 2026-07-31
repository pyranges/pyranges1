Rows operations
~~~~~~~~~~~~~~~

.. contents::
   :local:
   :depth: 2


Indexing with iloc, loc
-----------------------

PyRanges inherits all the indexing and slicing capabilities of pandas, e.g. boolean Series indexing,
``iloc``, ``loc``, ``at``, ``iat``.
Note that these methods return a view, not a copy, with the caveats that it implies.
See the pandas documentation for details.
Briefly, to avoid ambiguity it is best to explicitly call ``copy`` if you want an object to not be linked
to the original object from which it was extracted. For example:

  >>> import pyranges1 as pr
  >>> gr = pr.example_data.aorta
  >>> gr
  index    |    Chromosome    Start    End      Name      Score    Strand
  int64    |    category      int64    int64    str       int64    category
  -------  ---  ------------  -------  -------  --------  -------  ----------
  0        |    chr1          9916     10115    H3K27me3  5        -
  1        |    chr1          9939     10138    H3K27me3  7        +
  2        |    chr1          9951     10150    H3K27me3  8        -
  3        |    chr1          9953     10152    H3K27me3  5        +
  ...      |    ...           ...      ...      ...       ...      ...
  7        |    chr1          10127    10326    H3K27me3  1        -
  8        |    chr1          10241    10440    H3K27me3  6        -
  9        |    chr1          10246    10445    H3K27me3  4        +
  10       |    chr1          110246   110445   H3K27me3  1        +
  PyRanges with 11 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.

  >>> sgr = gr.iloc[0:3].copy()
  >>> sgr
    index  |    Chromosome      Start      End  Name        Score  Strand
    int64  |    category        int64    int64  str         int64  category
  -------  ---  ------------  -------  -------  --------  -------  ----------
        0  |    chr1             9916    10115  H3K27me3        5  -
        1  |    chr1             9939    10138  H3K27me3        7  +
        2  |    chr1             9951    10150  H3K27me3        8  -
  PyRanges with 3 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.

  >>> sgr['Score'] = 100  # does not modify gr
  >>> gr.head(3)
    index  |    Chromosome      Start      End  Name        Score  Strand
    int64  |    category        int64    int64  str         int64  category
  -------  ---  ------------  -------  -------  --------  -------  ----------
        0  |    chr1             9916    10115  H3K27me3        5  -
        1  |    chr1             9939    10138  H3K27me3        7  +
        2  |    chr1             9951    10150  H3K27me3        8  -
  PyRanges with 3 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.

On the other hand, to modify even a few lines of a PyRanges object,
use ``loc`` (label-based indexing) or ``iloc`` (positional indexing) on the whole object, not on a view:

  >>> gr.loc[0:2, 'Score'] = 100  # modifies gr
  >>> gr.head(5)
    index  |    Chromosome      Start      End  Name        Score  Strand
    int64  |    category        int64    int64  str         int64  category
  -------  ---  ------------  -------  -------  --------  -------  ----------
        0  |    chr1             9916    10115  H3K27me3      100  -
        1  |    chr1             9939    10138  H3K27me3      100  +
        2  |    chr1             9951    10150  H3K27me3      100  -
        3  |    chr1             9953    10152  H3K27me3        5  +
        4  |    chr1             9978    10177  H3K27me3        7  -
  PyRanges with 5 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.

Using boolean indexers
-----------------------

In PyRanges, boolean indexers work analogously as in pandas, as already seen in the tutorial:

  >>> sel = (gr['Score'] == 100)
  >>> gr[sel]
    index  |    Chromosome      Start      End  Name        Score  Strand
    int64  |    category        int64    int64  str         int64  category
  -------  ---  ------------  -------  -------  --------  -------  ----------
        0  |    chr1             9916    10115  H3K27me3      100  -
        1  |    chr1             9939    10138  H3K27me3      100  +
        2  |    chr1             9951    10150  H3K27me3      100  -
  PyRanges with 3 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.

  >>> gr[gr.Score==5]
    index  |    Chromosome      Start      End  Name        Score  Strand
    int64  |    category        int64    int64  str         int64  category
  -------  ---  ------------  -------  -------  --------  -------  ----------
        3  |    chr1             9953    10152  H3K27me3        5  +
        5  |    chr1            10001    10200  H3K27me3        5  -
  PyRanges with 2 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.

Combined with ``loc``, boolean indexers can be used to add or update column values:

  >>> gr.loc[gr['Score'] < 6, 'Score'] = -10  # modifies gr
  >>> gr.head(5)
    index  |    Chromosome      Start      End  Name        Score  Strand
    int64  |    category        int64    int64  str         int64  category
  -------  ---  ------------  -------  -------  --------  -------  ----------
        0  |    chr1             9916    10115  H3K27me3      100  -
        1  |    chr1             9939    10138  H3K27me3      100  +
        2  |    chr1             9951    10150  H3K27me3      100  -
        3  |    chr1             9953    10152  H3K27me3      -10  +
        4  |    chr1             9978    10177  H3K27me3        7  -
  PyRanges with 5 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.

In pandas, these logical operators can be employed with boolean Series:

* "&" =  element-wise AND operator
* "|" = element-wise OR operator
* "~" = NOT operator, inverts the values of the Series on its right

When using logical operators, make sure to parenthesize properly.

Let's get the + intervals with Score<8 starting before 10,000 or ending after 100,000:

  >>> gr[ (gr.Score<8) & (gr.Strand=='+') &
  ...     ((gr.Start<10000) | (gr.End>100000)) ]
    index  |    Chromosome      Start      End  Name        Score  Strand
    int64  |    category        int64    int64  str         int64  category
  -------  ---  ------------  -------  -------  --------  -------  ----------
        3  |    chr1             9953    10152  H3K27me3      -10  +
       10  |    chr1           110246   110445  H3K27me3      -10  +
  PyRanges with 2 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 1 strands.

Let's invert the selection:
  >>> gr[~(
  ...      (gr.Score<8) & (gr.Strand=='+') &
  ...      ((gr.Start<10000) | (gr.End>100000)) )]
  index    |    Chromosome    Start    End      Name      Score    Strand
  int64    |    category      int64    int64    str       int64    category
  -------  ---  ------------  -------  -------  --------  -------  ----------
  0        |    chr1          9916     10115    H3K27me3  100      -
  1        |    chr1          9939     10138    H3K27me3  100      +
  2        |    chr1          9951     10150    H3K27me3  100      -
  4        |    chr1          9978     10177    H3K27me3  7        -
  ...      |    ...           ...      ...      ...       ...      ...
  6        |    chr1          10024    10223    H3K27me3  -10      +
  7        |    chr1          10127    10326    H3K27me3  -10      -
  8        |    chr1          10241    10440    H3K27me3  6        -
  9        |    chr1          10246    10445    H3K27me3  -10      +
  PyRanges with 9 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.

Using PyRanges .loci
---------------------

PyRanges provides the method :func:`loci <pyranges1.PyRanges.loci>`
to select rows by genomic region:

  >>> gr2 = pr.example_data.aorta2.sort_ranges()
  >>> gr2
  index    |    Chromosome    Start    End      Name    Score    Strand
  int64    |    category      int64    int64    str     int64    category
  -------  ---  ------------  -------  -------  ------  -------  ----------
  1        |    chr1          10073    10272    Input   1        +
  5        |    chr1          10280    10479    Input   1        +
  6        |    chr1          16056    16255    Input   1        +
  7        |    chr1          16064    16263    Input   1        +
  ...      |    ...           ...      ...      ...     ...      ...
  4        |    chr1          10149    10348    Input   1        -
  3        |    chr1          10082    10281    Input   1        -
  2        |    chr1          10079    10278    Input   1        -
  0        |    chr1          9988     10187    Input   1        -
  PyRanges with 10 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.

Various syntaxes are accepted, see its API. For example:

  >>> gr2.loci['-'] # get all rows with strand '-'
    index  |    Chromosome      Start      End  Name      Score  Strand
    int64  |    category        int64    int64  str       int64  category
  -------  ---  ------------  -------  -------  ------  -------  ----------
        9  |    chr1            19958    20157  Input         1  -
        4  |    chr1            10149    10348  Input         1  -
        3  |    chr1            10082    10281  Input         1  -
        2  |    chr1            10079    10278  Input         1  -
        0  |    chr1             9988    10187  Input         1  -
  PyRanges with 5 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 1 strands.

  >>> gr2.loci['chr1', '+'] # get all rows with chromosome 'chr1' and strand '+'
    index  |    Chromosome      Start      End  Name      Score  Strand
    int64  |    category        int64    int64  str       int64  category
  -------  ---  ------------  -------  -------  ------  -------  ----------
        1  |    chr1            10073    10272  Input         1  +
        5  |    chr1            10280    10479  Input         1  +
        6  |    chr1            16056    16255  Input         1  +
        7  |    chr1            16064    16263  Input         1  +
        8  |    chr1            16109    16308  Input         1  +
  PyRanges with 5 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 1 strands.

  >>> gr2.loci['chr1', 10000:11000] # get all rows on 'chr1' and overlapping 10000:11000
    index  |    Chromosome      Start      End  Name      Score  Strand
    int64  |    category        int64    int64  str       int64  category
  -------  ---  ------------  -------  -------  ------  -------  ----------
        1  |    chr1            10073    10272  Input         1  +
        5  |    chr1            10280    10479  Input         1  +
        4  |    chr1            10149    10348  Input         1  -
        3  |    chr1            10082    10281  Input         1  -
        2  |    chr1            10079    10278  Input         1  -
        0  |    chr1             9988    10187  Input         1  -
  PyRanges with 6 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.

  >>> gr2.loci['chr1', '+', 10000:11000] # get all rows on 'chr1', strand '+', and overlapping 10000:11000
    index  |    Chromosome      Start      End  Name      Score  Strand
    int64  |    category        int64    int64  str       int64  category
  -------  ---  ------------  -------  -------  ------  -------  ----------
        1  |    chr1            10073    10272  Input         1  +
        5  |    chr1            10280    10479  Input         1  +
  PyRanges with 2 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 1 strands.

  Loci also support assignment. In this case, you must provide a DataFrame with the same shape as the selection:

  >>> gr2.loci['chr1', '+', 10000:11000] = gr2.loci['chr1', '+', 10000:11000].copy().assign(Score=100)
  >>> gr2.loci['chr1', '+', 10000:11000]  # see below that the Score was altered
    index  |    Chromosome      Start      End  Name      Score  Strand
    int64  |    category        int64    int64  str       int64  category
  -------  ---  ------------  -------  -------  ------  -------  ----------
        1  |    chr1            10073    10272  Input       100  +
        5  |    chr1            10280    10479  Input       100  +
  PyRanges with 2 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 1 strands.

  For more flexible assignment, you can use ``loc`` and provide use the ``index`` attribute of ``loci`` output:

  >>> sindex=gr2.loci['chr1', '+', 16000:17000].index
  >>> gr2.loc[sindex, "Score"]=150
  >>> gr2
  index    |    Chromosome    Start    End      Name    Score    Strand
  int64    |    category      int64    int64    str     int64    category
  -------  ---  ------------  -------  -------  ------  -------  ----------
  1        |    chr1          10073    10272    Input   100      +
  5        |    chr1          10280    10479    Input   100      +
  6        |    chr1          16056    16255    Input   150      +
  7        |    chr1          16064    16263    Input   150      +
  ...      |    ...           ...      ...      ...     ...      ...
  4        |    chr1          10149    10348    Input   1        -
  3        |    chr1          10082    10281    Input   1        -
  2        |    chr1          10079    10278    Input   1        -
  0        |    chr1          9988     10187    Input   1        -
  PyRanges with 10 rows, 6 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.



Sorting PyRanges
----------------

PyRanges objects can be sorted (i.e. altering the order of rows) by calling the pandas dataframe method ``sort_values``,
or the PyRanges method :func:`sort_ranges <pyranges1.PyRanges.sort_ranges>`.

  >>> import random; random.seed(1)
  >>> c = pr.example_data.chipseq.remove_nonloc_columns()
  >>> c['peak'] = [random.randint(0, 1000) for _ in range(len(c))] # add a column with random values
  >>> c
  index    |    Chromosome    Start      End        Strand      peak
  int64    |    category      int64      int64      category    int64
  -------  ---  ------------  ---------  ---------  ----------  -------
  0        |    chr8          28510032   28510057   -           137
  1        |    chr7          107153363  107153388  -           582
  2        |    chr5          135821802  135821827  -           867
  3        |    chr14         19418999   19419024   -           821
  ...      |    ...           ...        ...        ...         ...
  16       |    chr9          120803448  120803473  +           96
  17       |    chr6          89296757   89296782   -           499
  18       |    chr1          194245558  194245583  +           29
  19       |    chr8          57916061   57916086   +           914
  PyRanges with 20 rows, 5 columns, and 1 index columns.
  Contains 15 chromosomes and 2 strands.


Pandas ``sort_values`` sorts the whole dataframe by the specified columns. See its API for details.
For example, let's sort by column ``peak``:

  >>> c.sort_values(by='peak', ascending=False)
  index    |    Chromosome    Start      End        Strand      peak
  int64    |    category      int64      int64      category    int64
  -------  ---  ------------  ---------  ---------  ----------  -------
  19       |    chr8          57916061   57916086   +           914
  2        |    chr5          135821802  135821827  -           867
  3        |    chr14         19418999   19419024   -           821
  14       |    chr2          152562484  152562509  -           807
  ...      |    ...           ...        ...        ...         ...
  7        |    chr19         19571102   19571127   +           120
  16       |    chr9          120803448  120803473  +           96
  5        |    chr21         40099618   40099643   +           64
  18       |    chr1          194245558  194245583  +           29
  PyRanges with 20 rows, 5 columns, and 1 index columns.
  Contains 15 chromosomes and 2 strands.


PyRanges :func:`sort_ranges <pyranges1.PyRanges.sort_ranges>` is designed for genomic ranges.
By default, it sorts by Chromosome, Strand, then interval coordinates. If Strands are valid (
see :func:`strand_valid <pyranges1.PyRanges.strand_valid>`), then intervals on the reverse strand are
sorted in reverse order:

  >>> c.sort_ranges()
  index    |    Chromosome    Start      End        Strand      peak
  int64    |    category      int64      int64      category    int64
  -------  ---  ------------  ---------  ---------  ----------  -------
  12       |    chr1          38457520   38457545   +           667
  18       |    chr1          194245558  194245583  +           29
  13       |    chr1          80668132   80668157   -           388
  14       |    chr2          152562484  152562509  -           807
  ...      |    ...           ...        ...        ...         ...
  4        |    chr12         106679761  106679786  -           782
  3        |    chr14         19418999   19419024   -           821
  7        |    chr19         19571102   19571127   +           120
  5        |    chr21         40099618   40099643   +           64
  PyRanges with 20 rows, 5 columns, and 1 index columns.
  Contains 15 chromosomes and 2 strands.


Above, ``chr8`` appears before ``chr10`` because of 'natural sorting'. We can force alphabetical sorting instead:

  >>> c.sort_ranges(natsort=False)
  index    |    Chromosome    Start      End        Strand      peak
  int64    |    category      int64      int64      category    int64
  -------  ---  ------------  ---------  ---------  ----------  -------
  12       |    chr1          38457520   38457545   +           667
  18       |    chr1          194245558  194245583  +           29
  13       |    chr1          80668132   80668157   -           388
  9        |    chr10         35419784   35419809   -           779
  ...      |    ...           ...        ...        ...         ...
  19       |    chr8          57916061   57916086   +           914
  0        |    chr8          28510032   28510057   -           137
  6        |    chr8          22714402   22714427   -           261
  16       |    chr9          120803448  120803473  +           96
  PyRanges with 20 rows, 5 columns, and 1 index columns.
  Contains 15 chromosomes and 2 strands.
    
To sort by a different column, use the first argument (``by``). This is used after Chromosome and Strand, but before
coordinates:

  >>> c.sort_ranges('peak')
  index    |    Chromosome    Start      End        Strand      peak
  int64    |    category      int64      int64      category    int64
  -------  ---  ------------  ---------  ---------  ----------  -------
  18       |    chr1          194245558  194245583  +           29
  12       |    chr1          38457520   38457545   +           667
  13       |    chr1          80668132   80668157   -           388
  14       |    chr2          152562484  152562509  -           807
  ...      |    ...           ...        ...        ...         ...
  4        |    chr12         106679761  106679786  -           782
  3        |    chr14         19418999   19419024   -           821
  7        |    chr19         19571102   19571127   +           120
  5        |    chr21         40099618   40099643   +           64
  PyRanges with 20 rows, 5 columns, and 1 index columns.
  Contains 15 chromosomes and 2 strands.

The full list of sort keys is ``Chromosome, Strand, *by, Start, End``, and **any column you
name in** ``by`` **is taken out of its implicit position and used where you put it**. That is
how you reprioritise the location columns — here ``peak`` comes first, so the frame is
ordered by peak height across the whole genome:

  >>> c.sort_ranges(['peak', 'Chromosome', 'Strand'])
  index    |    Chromosome    Start      End        Strand      peak
  int64    |    category      int64      int64      category    int64
  -------  ---  ------------  ---------  ---------  ----------  -------
  18       |    chr1          194245558  194245583  +           29
  5        |    chr21         40099618   40099643   +           64
  16       |    chr9          120803448  120803473  +           96
  7        |    chr19         19571102   19571127   +           120
  ...      |    ...           ...        ...        ...         ...
  14       |    chr2          152562484  152562509  -           807
  3        |    chr14         19418999   19419024   -           821
  2        |    chr5          135821802  135821827  -           867
  19       |    chr8          57916061   57916086   +           914
  PyRanges with 20 rows, 5 columns, and 1 index columns.
  Contains 15 chromosomes and 2 strands.

The same rule lets you sort by a key *after* the coordinates: name ``Start`` and ``End`` in
``by`` and put the extra key behind them, as in ``by=['Start', 'End', 'peak']``.

The rule applies per column, so it is worth naming every key you care about. Writing
``by=['peak', 'Chromosome']`` on a stranded frame gives ``Strand, peak, Chromosome, Start,
End``: only ``Chromosome`` was pulled out of the implicit head, and ``Strand`` stayed where
it was.

Any key can be reversed with ``sort_descending``, including the implicit ``Chromosome``,
``Strand``, ``Start`` and ``End``. Naming something that is not a sort key raises
``ValueError`` rather than being ignored:

  >>> c.sort_ranges(sort_descending='Chromosome').head(4)
    index  |    Chromosome        Start        End  Strand         peak
    int64  |    category          int64      int64  category      int64
  -------  ---  ------------  ---------  ---------  ----------  -------
        5  |    chr21          40099618   40099643  +                64
        7  |    chr19          19571102   19571127  +               120
        3  |    chr14          19418999   19419024  -               821
        4  |    chr12         106679761  106679786  -               782
  PyRanges with 4 rows, 5 columns, and 1 index columns.
  Contains 4 chromosomes and 2 strands.

Note that ``use_strand=False`` does not remove ``Strand`` from the sort keys — it only stops
intervals on the negative strand being ordered 5'→3'. To sort without grouping by strand,
drop or rename the column first.
