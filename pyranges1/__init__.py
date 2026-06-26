# must import here every module/object X that will be available as pyranges1.X
# For each 'extension' e.g. named foobar, you must have:
# a ext/foobar.py file in pyranges1/ which contains the actual code for methods etc
# a foobar.py file in pyranges1/ which just imports the minimal objects to be exposed

from pyranges1 import orfs, seqs, stats  # noqa: F401
from pyranges1.core.assistant import assistant  # noqa: F401
from pyranges1.core.example_data import example_data  # noqa: F401
from pyranges1.core.multioverlap import count_overlaps  # noqa: F401
from pyranges1.core.options import option_manager as options  # noqa: F401
from pyranges1.core.pyranges_main import PyRanges  # noqa: F401
from pyranges1.core.random import random  # noqa: F401
from pyranges1.core.version import __version__  # noqa: F401
from pyranges1.methods.concat import concat  # noqa: F401
from pyranges1.methods.tile_genome import tile_genome  # noqa: F401
from pyranges1.range_frame.range_frame import RangeFrame  # noqa: F401
from pyranges1.readers import (  # noqa: F401
    from_string,
    read_bam,
    read_bed,
    read_bigbed,
    read_bigwig,
    read_cram,
    read_gff3,
    read_gtf,
    read_pairs,
    read_sam,
)

read_gff = read_gtf
