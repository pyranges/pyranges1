Command-line interface: pyranger
================================

Pyranger lets you access core functionalities directly from your shell:
build a pyranges1-based pipeline in one command, without writing any
Python code.

Installation
------------

Pyranger is based on `Google's Python Fire <https://github.com/google/python-fire>`__.
which is an optional dependency. To install it, use::

   pip install pyranges1[cli]


Quick Start
-----------

Run pyranger with no arguments to print a usage summary
along these lines::

   Read sequence interval data into PyRanges and apply a chain of methods

   Usage:
     pyranger reader <args> , [var=reader <args>]… , method <args> , …

     • The command line defines a pipeline of actions separated by " , " and starting with readers
     • The first reader loads the main PyRanges object
     • Every reader beyond the first *must* be named (e.g. b=read_bed b.bed)
     • Methods are invoked on the main object; others can be provided as arguments, e.g. intersect_overlaps b
     • The result replaces the main object in the pipeline

Readers
-------

Every pyranger command line must begin with one of these possible readers:

- :func:`read_bed <pyranges1.read_bed>` <path> [<options>]
- :func:`read_gtf <pyranges1.read_gtf>` <path> [<options>]
- :func:`read_gff <pyranges1.read_gff>` <path> [<options>]
- :func:`read_gff3 <pyranges1.read_gff3>` <path> [<options>]
- :func:`read_bam <pyranges1.read_bam>` <path> [<options>]
- :func:`read_bigwig <pyranges1.read_bigwig>` <path> [<options>]
- :func:`read_bigbed <pyranges1.read_bigbed>` <path> [<options>]
- :func:`read_sam <pyranges1.read_sam>` <path> [<options>]
- :func:`read_cram <pyranges1.read_cram>` <path> [<options>]
- :func:`read_vcf <pyranges1.read_vcf>` <path> [<options>]
- :func:`read_pairs <pyranges1.read_pairs>` <path> [<options>]
- :func:`read_paf <pyranges1.read_paf>` <path> [<options>]
- :func:`read_narrowPeak <pyranges1.read_narrowPeak>` <path> [<options>]
- :func:`read_parquet <pyranges1.read_parquet>` <path> [<options>]
- read_csv  <path> [<options>]   # wrapped from Pandas


The first reader (unnamed) becomes the “main” PyRanges object.
Without further operations, pyranger will print its content to stdout::

   pyranger read_bed sample1.bed

    index  |    Chromosome      Start      End  Name      Score     Strand        ThickStart
    int64  |    category        int64    int64  str    str    category           int64
  -------  ---  ------------  -------  -------  --------  --------  ----------  ------------
        0  |    chr1                1        5  .         .         +                      1
        1  |    chr1                6        8  .         .         -                      2
  PyRanges with 2 rows, 7 columns, and 1 index columns.
  Contains 1 chromosomes and 2 strands.


*(From here onwards, the output of commands is omitted.)*

Some operations (e.g. intersect_overlaps) require two objects.
You can chain multiple readers and methods together, separated by a comma surrounded by spaces
(:literal:` , `). After the first one, all subsequent readers
must be named into variables that can be used later.
For example, to read two BED files and intersect them, use::

   pyranger read_bed sample1.bed , b=read_bed sample2.bed , intersect_overlaps b

Methods
-------

After reader(s), pyranger pipelines consists of methods applied to the main PyRanges object.
As before, the separator is a comma surrounded by spaces (:literal:` , `)::

   pyranger read_bed a.bed , downstream 10

   pyranger read_bed a.bed , head

Methods available include all those implemented by :class:`PyRanges <pyranges1.PyRanges>` objects,
as well as those from Pandas DataFrames, which are inherited.

Arguments
---------

Arguments can be passed to readers/methods either positionally or by name::

   pyranger read_bed sample1.bed , downstream 10 --gap=5

   pyranger read_bed sample1.bed , b=read_bed sample2.bed , intersect_overlaps b --multiple first

Typically, the last method in a pipeline will be the one that outputs the result
(e.g., :func:`to_bed <pyranges1.PyRanges.to_bed>`, :func:`to_csv <pyranges1.PyRanges.to_csv>`,
:func:`to_gtf <pyranges1.PyRanges.to_gtf>`, :func:`to_gff3 <pyranges1.PyRanges.to_csv>`)::

   pyranger read_bed sample1.bed , downstream 10 --gap=5 , to_bed output.bed
   pyranger read_bed sample1.bed , b=read_bed sample2.bed , intersect_overlaps b , to_csv output.tsv --sep $'\t'

Getting help
------------

To view arguments for a specific reader/method, append ``--help`` immediately after its name.
For methods, include the full context of your pipeline::

   pyranger read_bed --help

   pyranger read_bed a.bed , downstream --help


Examples
--------

Below are some common workflows. Everything after a separator (:literal:` , `) is
either a named reader or a method invocation, in sequence:

1. **Load a single BED file**::

     pyranger read_bed sample1.bed

   Loads `sample1.bed` into a `PyRanges` object and prints its content.

2. **Load + inspect first 5 lines**::

     pyranger read_bed sample1.bed , head 5

   - ``read_bed sample1.bed`` becomes the main object  
   - ``head 5`` takes the first five rows of that `PyRanges`.

3. **Intersect two BED files**::

     pyranger read_bed a.bed , other=read_bed b.bed , intersect_overlaps other

   - ``read_bed a.bed`` → main object  
   - ``other=read_bed b.bed`` → variable ``other``  
   - ``intersect_overlaps other`` → runs ``.intersect_overlaps(other)``

4. **Chain three readers and two methods**::

     pyranger read_bed a.bed , b=read_bed b.bed , c=read_bed c.bed , join_overlaps b , intersect_overlaps c

   - Load `a.bed` as main  
   - Load `b.bed` into ``b`` and `c.bed` into ``c``  
   - Run ``.join_overlaps(b)`` on the main object, then ``.intersect_overlaps(c)`` on the result


Final notes
-----------

To discover the functionalities available in pyranger,
we recommend reading the rest of the pyranges1 documentation, especially the
:doc:`tutorial <./tutorial>` and :doc:`how-to pages <./how_to_pages>`.
While these are written in Python, the same concepts and methods are
accessible through pyranger.

Note that some cases may not be fully supported in pyranger.
If you struggle to express a specific operation,
consider building a custom Python script using pyranges1 instead.

