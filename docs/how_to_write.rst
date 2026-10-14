Writing to disk
~~~~~~~~~~~~~~~

.. contents::
   :local:
   :depth: 2


The PyRanges can be written to several formats, namely csv, gtf, gff3 and bigwig.
If no path-argument is given, the string representation of the data is returned. (It may potentially be very large.)
If a path is given, it is taken as the path to the file to be written; in this case, the return value is the object
itself, to allow inserting write methods into method call chains.


Writing genomic formats
-----------------------

Pyranges supports the most popular for genomic annotations, such as bed, gtf and gff3.
You can readily write them using the correspondent methods (see
:func:`to_bed <pyranges1.PyRanges.to_bed>`,
:func:`to_gtf <pyranges1.PyRanges.to_gtf>`,
:func:`to_gff3 <pyranges1.PyRanges.to_gff3>`).

  >>> import pyranges1 as pr
  >>> gr = pr.example_data.chipseq
  >>> gr.to_gtf("chipseq.gtf")
  >>> #file chipseq.gtf has been created


Methods to_gff3 and to_gtf have a default mapping of PyRanges columns to GFF/GTF fields.
All extra ("metadata") columns are put in the last field:

  >>> gr['Label']='something'
  >>> print(gr.head().to_gtf()) # doctest: +NORMALIZE_WHITESPACE
  chr8	.	.	28510033	28510057	0	-	.	Name "U0"; Label "something";
  chr7	.	.	107153364	107153388	0	-	.	Name "U0"; Label "something";
  chr5	.	.	135821803	135821827	0	-	.	Name "U0"; Label "something";
  chr14	.	.	19419000	19419024	0	-	.	Name "U0"; Label "something";
  chr12	.	.	106679762	106679786	0	-	.	Name "U0"; Label "something";

Such mapping, as well as which attribute(s) are included as last field, can be altered. See the API for details.

The `bigwig <http://genome.ucsc.edu/goldenPath/help/bigWig.html>`_ format differs substantially from
the formats above. Bigwig is a binary format, and it is typically used for large continuous quantitative
data along a genome sequence.

The pyranges1 library can also create bigwigs, but it needs the libraries pybigwig and runlength which are not installed by default.

Use this to install it::

	pip install pybigwig runlength

The bigwig writer needs to know the chromosome sizes, e.g. provided as a dictionary {chromosome_name: size}.
You can also derive chromosome sizes from a fasta file using pyfaidx (see above to install it).

.. doctest::

  >>> import pyfaidx
  >>> chromsizes=pyfaidx.Fasta('your_genome.fa') # doctest: +SKIP


Once you obtained chromosome sizes, you are ready to write your PyRanges object to a bigwig file:

  >>> gr.to_bigwig("chipseq.bw", chromsizes) # doctest: +SKIP
  >>> # file chipseq.bw has been created

Bigwig is typically used to represent a coverage of some type.
To compute it from an arbitrary value column, use the value_col argument. See the API for additional options.
If you want to write one bigwig for each strand, you need to do it manually.


  >>> gr.loci["+"].to_bigwig("chipseq_plus.bw", chromsizes) # doctest: +SKIP
  >>> gr.loci["-"].to_bigwig("chipseq_minus.bw", chromsizes) # doctest: +SKIP


Writing tabular formats
-----------------------

The csv format is the most flexible format, as it allows for any column to be included, and any separator to be used.
The method ``to_csv`` is directly inherited by pandas, so search for its API for details.


The ``to_csv`` method takes the arguments header and sep:

  >>> print(gr.drop(['Label'], axis=1).head().to_csv(sep="\t", header=False, index=False)) # doctest: +NORMALIZE_WHITESPACE
  chr8	28510032	28510057	U0	0	-
  chr7	107153363	107153388	U0	0	-
  chr5	135821802	135821827	U0	0	-
  chr14	19418999	19419024	U0	0	-
  chr12	106679761	106679786	U0	0	-
  <BLANKLINE>

Remember that ``to_csv`` will not alter coordinates, so the output
will have the same pythonic convention as PyRanges. Adjust accordingly if needed.
