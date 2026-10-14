import logging
import sys
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from pyranges1.core.names import (
    CHROM_COL,
    END_COL,
    FORWARD_STRAND,
    FRAME_COL,
    REVERSE_STRAND,
    START_COL,
    STRAND_COL,
    TEMP_CUMSUM_COL,
    TEMP_INDEX_COL,
    TEMP_LENGTH_COL,
)
from pyranges1.core.pyranges_helpers import arg_to_list, ensure_pyranges

if TYPE_CHECKING:
    import pyranges1 as pr

## Joan Pallares started this function
#  Marco Mariotti later optimized it, made it independent from pyranges

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

DIRECTION_OPTIONS = Literal["up", "down"]

STARTS_NUCLEOTIDE_SEQ = ["ATG"]
STOPS_NUCLEOTIDE_SEQ = ["TAG", "TGA", "TAA"]

N_PRINTED_NON_MULTIPLE_3 = 10


def calculate_frame(p: "pr.PyRanges", group_by: str | list[str], frame_col: str = "Frame") -> "pr.PyRanges":
    """Calculate the frame of genomic intervals, assuming all are coding sequences (CDS), and add it as column.

    A stranded
    After this, the input Pyranges will contain an added "Frame" column, which determines the nucleotide of the CDS
    that is the first base of a codon.Resulting values are in range between 0 and 2 included.
    0 indicates that the first nucleotide of that interval is the first base of a codon,
    1 indicates the second base and 2 indicates the third base.
    While the 5'-most interval of each transcript has always 0 frame, the following ones may have any of these values.

    Parameters
    ----------
    p : PyRanges
        Input CDS intervals.

    group_by : str or list of str
        Column(s) to group by the intervals: coding exons belonging to the same transcript have the same values in this/these column(s).

    frame_col : str, default 'Frame'
        Name of the column to store the frame values.

    Returns
    -------
    PyRanges

    Examples
    --------
    >>> import pyranges1 as pr
    >>> p = pr.PyRanges({"Chromosome": [1,1,1,2,2],
    ...                   "Strand": ["+","+","+","-","-"],
    ...                   "Start": [1,31,52,101,201],
    ...                   "End": [10,45,90,130,218],
    ...                   "transcript_id": ["t1","t1","t1","t2","t2"]})
    >>> p
      index  |      Chromosome  Strand      Start      End  transcript_id
      int64  |           int64  str         int64    int64  str
    -------  ---  ------------  --------  -------  -------  ---------------
          0  |               1  +               1       10  t1
          1  |               1  +              31       45  t1
          2  |               1  +              52       90  t1
          3  |               2  -             101      130  t2
          4  |               2  -             201      218  t2
    PyRanges with 5 rows, 5 columns, and 1 index columns.
    Contains 2 chromosomes and 2 strands.

    >>> pr.orfs.calculate_frame(p, group_by=['transcript_id'])
      index  |      Chromosome  Strand      Start      End  transcript_id      Frame
      int64  |           int64  str         int64    int64  str                int64
    -------  ---  ------------  --------  -------  -------  ---------------  -------
          0  |               1  +               1       10  t1                     0
          1  |               1  +              31       45  t1                     0
          2  |               1  +              52       90  t1                     2
          3  |               2  -             101      130  t2                     2
          4  |               2  -             201      218  t2                     0
    PyRanges with 5 rows, 6 columns, and 1 index columns.
    Contains 2 chromosomes and 2 strands.

    """
    if not p.strand_valid:
        msg = "Strand must be valid to run calculate_frame."
        raise AssertionError(msg)

    gr = p.copy()

    # Column to save the initial index
    gr[TEMP_INDEX_COL] = np.arange(len(p))

    # Filtering for desired columns
    sorted_p = gr.get_with_loc_columns([TEMP_INDEX_COL, *arg_to_list(group_by)])

    # Sorting by 5' (Intervals on + are sorted by ascending order and - are sorted by descending order)
    sorted_p = sorted_p.sort_ranges()

    # Creating a column saving the length for the intervals
    sorted_p[TEMP_LENGTH_COL] = sorted_p.lengths()

    # Creating a column saving the cumulative length for the intervals
    sorted_p[TEMP_CUMSUM_COL] = sorted_p.groupby(group_by)[TEMP_LENGTH_COL].cumsum()

    # Creating a frame column
    sorted_p[FRAME_COL] = (sorted_p[TEMP_CUMSUM_COL] - sorted_p[TEMP_LENGTH_COL]) % 3

    # Appending the Frame of sorted_p by the index of p
    sorted_p = ensure_pyranges(sorted_p.sort_values(by=TEMP_INDEX_COL))

    gr[frame_col] = sorted_p[FRAME_COL]

    # TEMP_INDEX_COL isn't a required column, so this drop can never fall back to a DataFrame.
    return cast("pr.PyRanges", gr.drop_and_return(TEMP_INDEX_COL, axis=1))


def extend_orfs(  # noqa: C901,PLR0912,PLR0915
    p: "pr.PyRanges",
    fasta_path: str,
    group_by: str | list[str] | None = None,
    *,
    direction: Iterable[DIRECTION_OPTIONS] | DIRECTION_OPTIONS | None = None,
    starts: list[str] = STARTS_NUCLEOTIDE_SEQ,
    stops: list[str] = STOPS_NUCLEOTIDE_SEQ,
    keep_off_bounds: bool = False,
    record_extensions: bool = False,
    chunk_size: int = 900,
    verbose: bool = False,
) -> "pr.PyRanges":
    r"""Extend PyRanges intervals to form complete open reading frames.

    The input intervals are extended their next Stop codon downstream, and to their leftmost Start codon upstream
    before encountering a Stop.

    Parameters
    ----------
    p : PyRanges
        Input CDS intervals.

    fasta_path : location of the Fasta file from which the sequences
        for the extensions will be retrieved.

    group_by : str or list of str or None
        Name(s) of column(s) to group intervals into transcripts

    starts : list containing the nucleotide pattern to look for upstream.
        if not provided, ORFs are delimited by stops
        Default ['ATG']

    stops : list containing the nucleotide pattern to look for downstream.
        Default ['TAG', 'TGA', 'TAA']

    direction : whether the extension should be upstream ('up'), downstream
        ('down') or both. Default (None) means: ['up', 'down']

    keep_off_bounds : if True, those intervals that reached out of bounds during extension
        without finding any stop are returned in their largest (3-nt multiple) extension.
        In this case, these intervals will not begin with a start or end with a stop

    record_extensions : if True, add columns extension_up and extension_down
        with the extensions amounts. Default: False

    chunk_size : the amount of nucleotides to be extended on each iteration. Does not affect output, only speed/memory.
        Default 900.

    verbose : if True, print information about the progress of the extension.
        Default: False

    Note
    ----

    This function requires the library pyfaidx, it can be installed with
    ``conda install -c bioconda pyfaidx`` or ``pip install pyfaidx``.

    Sorting the PyRanges is likely to improve the speed.
    Intervals on the negative strand will be reverse complemented.

    Examples
    --------
    >>> import pyranges1 as pr
    >>> p = pr.PyRanges({"Chromosome": ['seq1'], "Start":[20], "End":[29], "Strand" : ["+"]})
    >>> p
      index  |    Chromosome      Start      End  Strand
      int64  |    str             int64    int64  str
    -------  ---  ------------  -------  -------  --------
          0  |    seq1               20       29  +
    PyRanges with 1 rows, 4 columns, and 1 index columns.
    Contains 1 chromosomes and 1 strands.

    >>> #            *       ^       ^      ... ... ...          *       #  ... = p interval
    >>> seq1 = " AA TAA TGT ATG GTA ATG GGC GCC GGG ATT CCA CAG TAA GTG C".replace(' ', '')
    >>> tmp_handle = open("temp.fasta", "w+")
    >>> _ = tmp_handle.write(">seq1\n")
    >>> _ = tmp_handle.write(seq1+'\n')
    >>> tmp_handle.close()

    >>> p.get_sequence("temp.fasta")
    0    GCCGGGATT
    Name: Sequence, dtype: object

    >>> ep = pr.orfs.extend_orfs(p, fasta_path="temp.fasta")
    >>> ep
      index  |    Chromosome      Start      End  Strand
      int64  |    str             int64    int64  str
    -------  ---  ------------  -------  -------  --------
          0  |    seq1                8       38  +
    PyRanges with 1 rows, 4 columns, and 1 index columns.
    Contains 1 chromosomes and 1 strands.

    >>> ep.get_sequence("temp.fasta")
    0    ATGGTAATGGGCGCCGGGATTCCACAGTAA
    Name: Sequence, dtype: object

    >>> pr.orfs.extend_orfs(p, fasta_path="temp.fasta", record_extensions=True)
      index  |    Chromosome      Start      End  Strand      extension_up    extension_down
      int64  |    str             int64    int64  str                int64             int64
    -------  ---  ------------  -------  -------  --------  --------------  ----------------
          0  |    seq1                8       38  +                     12                 9
    PyRanges with 1 rows, 6 columns, and 1 index columns.
    Contains 1 chromosomes and 1 strands.

    Extending only in one direction:

    >>> pr.orfs.extend_orfs(p, fasta_path="temp.fasta", direction='up')
      index  |    Chromosome      Start      End  Strand
      int64  |    str             int64    int64  str
    -------  ---  ------------  -------  -------  --------
          0  |    seq1                8       29  +
    PyRanges with 1 rows, 4 columns, and 1 index columns.
    Contains 1 chromosomes and 1 strands.

    With starts=[], any codon can be used as a start (i.e. ORFs defined as stop-delimited sequences):

    >>> ep=pr.orfs.extend_orfs(p, fasta_path="temp.fasta", starts=[])
    >>> ep
      index  |    Chromosome      Start      End  Strand
      int64  |    str             int64    int64  str
    -------  ---  ------------  -------  -------  --------
          0  |    seq1                5       38  +
    PyRanges with 1 rows, 4 columns, and 1 index columns.
    Contains 1 chromosomes and 1 strands.

    >>> ep.get_sequence("temp.fasta")
    0    TGTATGGTAATGGGCGCCGGGATTCCACAGTAA
    Name: Sequence, dtype: object

    Example with multi-exon input intervals
    Intervals on the negative strand are extended accordingly

    >>> #                   :    *           ^      ... .        ..      *      # ... = p interval
    >>> # reverse complement: C TAG CGT TTG ATG TTG GGC CAG GTG TTT CAG TAG CCC GG
    >>> seq2 = " CC GGG CTA CTG AAA CAC CTG GCC CAA CAT CAA ACG CTA G".replace(' ', '')
    >>> tmp_handle = open("temp1.fasta", "w+")
    >>> _ = tmp_handle.write(">seq2\n")
    >>> _ = tmp_handle.write(seq2+'\n')
    >>> tmp_handle.close()


    >>> np = pr.PyRanges({"Chromosome": ['seq2']*2, "Start":[19, 11], "End":[23, 13],
    ...                   "Strand" : ["-"]*2, "ID":["a", "a"]})
    >>> np
      index  |    Chromosome      Start      End  Strand    ID
      int64  |    str             int64    int64  str       str
    -------  ---  ------------  -------  -------  --------  -----
          0  |    seq2               19       23  -         a
          1  |    seq2               11       13  -         a
    PyRanges with 2 rows, 5 columns, and 1 index columns.
    Contains 1 chromosomes and 1 strands.

    >>> np.get_sequence("temp1.fasta")
    0    GGCC
    1      TT
    Name: Sequence, dtype: object

    >>> ep = pr.orfs.extend_orfs(np, fasta_path="temp1.fasta", group_by='ID')
    >>> ep.get_sequence("temp1.fasta")
    0    ATGTTGGGCC
    1      TTCAGTAG
    Name: Sequence, dtype: object


    A sequence with no in-frame stops after the input interval before the end:

    >>> #             *       ^       ^      ... ... ...                   #  ... = p interval
    >>> seq1b = " AA TAA TGT ATG GTA ATG GGC GCC GGG ATT CCA CAG AAA GTG C".replace(' ', '')
    >>> tmp_handle = open("temp2.fasta", "w+")
    >>> _ = tmp_handle.write(">seq1\n")
    >>> _ = tmp_handle.write(seq1b+'\n')
    >>> tmp_handle.close()

    >>> pr.orfs.extend_orfs(p, fasta_path="temp2.fasta", record_extensions=True)
      index  |    Chromosome      Start      End  Strand      extension_up    extension_down
      int64  |    str             int64    int64  str                int64             int64
    -------  ---  ------------  -------  -------  --------  --------------  ----------------
          0  |    seq1                8       29  +                     12                 0
    PyRanges with 1 rows, 6 columns, and 1 index columns.
    Contains 1 chromosomes and 1 strands.

    Showcasing keep_off_bounds:

    >>> ep=pr.orfs.extend_orfs(p, fasta_path="temp2.fasta", record_extensions=True, keep_off_bounds=True)
    >>> ep
      index  |    Chromosome      Start      End  Strand      extension_up    extension_down
      int64  |    str             int64    int64  str                int64             int64
    -------  ---  ------------  -------  -------  --------  --------------  ----------------
          0  |    seq1                8       41  +                     12                12
    PyRanges with 1 rows, 6 columns, and 1 index columns.
    Contains 1 chromosomes and 1 strands.
    >>> ep.get_sequence("temp2.fasta")
    0    ATGGTAATGGGCGCCGGGATTCCACAGAAAGTG
    Name: Sequence, dtype: object

    A sequence with no in-frame stops BEFORE the input interval:

    >>> #                     ^       ^      ... ... ...          *        #  ... = p interval
    >>> seq1c = " AA TAC TGT ATG GTA ATG GGC GCC GGG ATT CCA CAG TAA GTG C".replace(' ', '')
    >>> tmp_handle = open("temp3.fasta", "w+")
    >>> _ = tmp_handle.write(">seq1\n")
    >>> _ = tmp_handle.write(seq1c+'\n')
    >>> tmp_handle.close()

    >>> pr.orfs.extend_orfs(p, fasta_path="temp3.fasta", record_extensions=True)
      index  |    Chromosome      Start      End  Strand      extension_up    extension_down
      int64  |    str             int64    int64  str                int64             int64
    -------  ---  ------------  -------  -------  --------  --------------  ----------------
          0  |    seq1                8       38  +                     12                 9
    PyRanges with 1 rows, 6 columns, and 1 index columns.
    Contains 1 chromosomes and 1 strands.

    >>> ep = pr.orfs.extend_orfs(p, fasta_path="temp3.fasta", record_extensions=True, keep_off_bounds=True)
    >>> ep
      index  |    Chromosome      Start      End  Strand      extension_up    extension_down
      int64  |    str             int64    int64  str                int64             int64
    -------  ---  ------------  -------  -------  --------  --------------  ----------------
          0  |    seq1                2       38  +                     18                 9
    PyRanges with 1 rows, 6 columns, and 1 index columns.
    Contains 1 chromosomes and 1 strands.

    >>> ep.get_sequence("temp3.fasta")
    0    TACTGTATGGTAATGGGCGCCGGGATTCCACAGTAA
    Name: Sequence, dtype: object

    """
    try:
        import pyfaidx  # type: ignore[import]
    except ImportError:
        LOGGER.exception(
            "To use extend_orfs, pyfaidx must be installed. Use `conda install -c bioconda pyfaidx` or `pip install pyfaidx` to install pyfaidx.",
        )
        sys.exit(1)

    def pverbose(msg: Any) -> None:
        if verbose:
            LOGGER.info(msg)

    # Sanity Checks
    if (chunk_size % 3) != 0:
        msg = "Chunk size must be a multiple of three."
        raise AssertionError(msg)

    if not all(len(pattern) == len(STARTS_NUCLEOTIDE_SEQ[0]) for pattern in starts + stops):
        msg = "Ensure that all patterns have a length of 3 nt."
        raise AssertionError(msg)

    if not stops:
        msg = "At least one stop codon must be provided."
        raise AssertionError(msg)

    direction = direction if direction is not None else ["up", "down"]

    p = p.copy()

    if not p.strand_valid:
        msg = "Intervals must be have valid strands to call extend_orfs"
        raise AssertionError(msg)

    if group_by is None:
        cds_id = "Custom_ID"
        p[cds_id] = np.arange(len(p))  # Generates a New Column
    else:
        cds_id = group_by

    if isinstance(cds_id, str):  # ensure it is a list
        cds_id = [cds_id]

    ##################
    # Show a warning if some transcript lengths are not divisible by 3:
    p["__length"] = p[END_COL] - p[START_COL]
    module_3 = p.groupby(by=cds_id).__length.sum().mod(3)  # noqa: SLF001
    ndv = module_3[module_3 != 0].index
    if len(ndv) > 0:
        warnings.warn("\nWARNING! Some input lenghts are not divisible by 3:", stacklevel=2)
        warnings.warn(
            " ".join(ndv[:N_PRINTED_NON_MULTIPLE_3]) + (" and others " if len(ndv) > N_PRINTED_NON_MULTIPLE_3 else ""),
            stacklevel=2,
        )

    # "__length" isn't a required column, so this drop can never fall back to a DataFrame.
    p = cast("pr.PyRanges", p.drop_and_return(["__length"], axis=1))
    ##################

    # Load Sequence Data from a Fasta file
    fs = pyfaidx.Fasta(fasta_path)  # pyfaidx_fasta object, fasta sequences

    # get a minimal interval per group, with min start and max end
    minp = (
        (p[[STRAND_COL, START_COL, END_COL, CHROM_COL, *cds_id]])
        .groupby(cds_id, as_index=False)
        .agg(
            {
                STRAND_COL: "first",
                START_COL: "min",
                END_COL: "max",
                CHROM_COL: "first",
            },
        )
    )
    ncds = len(minp)
    nexons = len(p)

    # add chromosome sizes as column
    minp = minp.merge(
        pd.DataFrame(
            {
                CHROM_COL: [chrom for chrom, _ in fs.items()],
                "__chromsize": [len(fs[chrom]) for chrom, _ in fs.items()],
            },
        ),
        on=CHROM_COL,
    ).set_index(cds_id)

    if len(minp) != ncds:
        msg = "ERROR some sequences were not found in the pyfaidx object"
        raise AssertionError(msg)

    ######################  Extend Sequence Upstream ###########################

    if "up" in direction or direction == "up":
        pup = minp.copy()
        pup["__out_of_bounds"] = False
        pup["__up_stop_dist"] = -1  # -1 means not found
        pup["__up_start_dist"] = -1  # -1 means not found
        selector = pd.Series(
            data=True,
            index=pup.index,
        )  # initialized as all True; keeps track of intervals with work to do
        ic = 0  # iteration counter

        while (selector).any():
            pverbose(
                f"Upstream | iteration {ic},  left to do: {selector.sum()}",
            )
            pverbose(pup[selector])
            # selector= ... everything which is not done nor out of bounds
            _extend_df(pup, "5", chunk_size, selector)
            _subseq_df(pup, "5", chunk_size, selector)
            _correct_bounds(pup, adjust="5")
            _get_seqs(pup, fs, selector)  # seqs length is only selector

            ## looking for a stop if didn't find it already
            z = pup.loc[selector, "__seq"].apply(lambda x: _find_rightmost_stop(x, stops))

            found_stop_selector = selector.copy()  # made to index pup
            found_stop_selector[found_stop_selector] = z != -1
            pup.loc[found_stop_selector, "__up_stop_dist"] = z[z != -1] + ic * chunk_size

            no_stop_off_bounds = selector.copy()
            no_stop_off_bounds[no_stop_off_bounds] = z == -1

            if keep_off_bounds:
                # focus on intervals that reached out of bounds in this iteration and did not find a stop:
                #  record their max extension as up_stop_dist and __up_start_dist so they are considered whether starts was provided or not
                # btw this means these intervals can be identified by looking at rows with __up_start_dist == __up_stop_dist

                pup.loc[no_stop_off_bounds, "__up_stop_dist"] = (
                    pup.loc[no_stop_off_bounds, END_COL] - pup.loc[no_stop_off_bounds, START_COL]
                ) + ic * chunk_size
                pup.loc[no_stop_off_bounds, "__up_start_dist"] = pup.loc[no_stop_off_bounds, "__up_stop_dist"]

            ## looking for a start
            if starts:
                w = pup.loc[selector, ["__seq", "__up_stop_dist"]].apply(
                    lambda x: _find_leftmost_start(x, starts, chunk_size),
                    axis=1,
                )
                ## if start is not found now, but was found in a previous iteration, the previous will be kept (not overwriting a -1)
                found_start_selector = selector.copy()  # made to index pup
                found_start_selector[found_start_selector] = w != -1
                if not keep_off_bounds:
                    pup.loc[found_start_selector, "__up_start_dist"] = w[w != -1] + ic * chunk_size
                else:
                    found_start_and_stop_selector = found_start_selector & ~no_stop_off_bounds
                    pup.loc[found_start_and_stop_selector, "__up_start_dist"] = w[w != -1] + ic * chunk_size
            else:
                # if no starts are provided, the start is whatever comes after the stop
                pup.loc[found_stop_selector, "__up_start_dist"] = pup.loc[found_stop_selector, "__up_stop_dist"] - 3

            pup.loc[found_stop_selector, "__seq"] = ""  ## frees memory
            pverbose(" ---> After iteration:")
            pverbose(pup[selector])

            selector = (pup.__up_stop_dist == -1) & ~(pup.__out_of_bounds)  # noqa: SLF001
            ic += 1

        ext_up = pup["__up_start_dist"].rename("extension_up")

        ext_up[ext_up == -1] = 0
        del pup

    else:  # No extension upstream (i.e. the extension is 0)
        ext_up = pd.Series(0, index=minp.index)

    ######################  Extend Sequence Downstream #########################
    if "down" in direction or direction == "down":
        pdo = minp.copy()
        pdo["__out_of_bounds"] = False
        pdo["__down_stop_dist"] = -1  # -1 means not found
        selector = pd.Series(
            data=True,
            index=pdo.index,
        )  # initialized as all True; keeps track of intervals with work to do
        ic = 0  # iteration counter

        while (selector).any():
            pverbose(
                f"Downstream | iteration {ic},  left to do: {selector.sum()}",
            )
            pverbose(pdo[selector])
            # selector= ... everything which is not done nor out of bounds
            _extend_df(pdo, "3", chunk_size, selector)
            _subseq_df(pdo, "3", chunk_size, selector)
            _correct_bounds(pdo, adjust="3")
            _get_seqs(pdo, fs, selector)  # seqs length is only selector

            ## looking for a stop if didn't find it already
            z = pdo.loc[selector, "__seq"].apply(lambda x: _find_leftmost_stop(x, stops))
            found_stop_selector = selector.copy()  # made to index pdo
            found_stop_selector[found_stop_selector] = z != -1
            pdo.loc[found_stop_selector, "__down_stop_dist"] = z[z != -1] + ic * chunk_size

            if keep_off_bounds:
                # focus on intervals that reached out of bounds in this iteration and did not find a stop:
                #  record their max extension as down_stop_dist

                no_stop_off_bounds = selector.copy()
                no_stop_off_bounds[no_stop_off_bounds] = z == -1

                pdo.loc[no_stop_off_bounds, "__down_stop_dist"] = (
                    pdo.loc[no_stop_off_bounds, END_COL] - pdo.loc[no_stop_off_bounds, START_COL]
                ) + ic * chunk_size

            pdo.loc[found_stop_selector, "__seq"] = ""  ## frees memory
            pverbose(" ---> After iteration:")
            pverbose(pdo[selector])
            selector = (pdo.__down_stop_dist == -1) & ~(pdo.__out_of_bounds)  # noqa: SLF001
            ic += 1

        ext_down = pdo["__down_stop_dist"].rename("extension_down")
        ext_down[ext_down == -1] = 0
        del pdo

    else:  # No extension downstream (i.e. the extension is 0)
        ext_down = pd.Series(data=0, index=minp.index)

    ### Extensions have been determined. Now let's apply them to the original DF
    p["__order"] = np.arange(len(p))
    zp = ensure_pyranges(
        p.merge(
            pd.DataFrame({"__extension_up": ext_up, "__extension_down": ext_down}),
            left_on=cds_id,
            right_index=True,
        ),
    )
    if len(zp) != nexons:
        msg = "ERROR malformed gene structures"
        raise AssertionError(msg)

    p = zp

    p = p.sort_ranges()
    _extend_groups(p, cds_id)
    p = ensure_pyranges(p.sort_values("__order"))  # restore order

    # getting ready to return
    if record_extensions:
        p = ensure_pyranges(
            p.rename(
                columns={
                    "__extension_up": "extension_up",
                    "__extension_down": "extension_down",
                },
            ),
        )
        to_drop = ["__order"]
    else:
        to_drop = ["__extension_up", "__extension_down", "__order"]

    if group_by is None:
        to_drop.extend(cds_id)

    return ensure_pyranges(p.drop_and_return(to_drop, axis=1))


# MM # efficients methods to work inplace. All methods with df as input have one row per CDS group
def _extend_df(df, direction, extension, selector) -> None:
    """Inplace."""
    if direction == "5":
        df.loc[selector & (df.Strand == FORWARD_STRAND), [START_COL]] -= extension
        df.loc[selector & (df.Strand == "-"), [END_COL]] += extension
    elif direction == "3":
        df.loc[selector & (df.Strand == REVERSE_STRAND), [START_COL]] -= extension
        df.loc[selector & (df.Strand == FORWARD_STRAND), [END_COL]] += extension


def _extend_groups(dfg, cds_id) -> None:
    """In place, dfg may contain multiple exons."""
    first_exon_indices = dfg.groupby(cds_id).apply(lambda x: x.index[0]).reset_index(drop=True)
    last_exon_indices = dfg.groupby(cds_id).apply(lambda x: x.index[-1]).reset_index(drop=True)

    # boolean Series to select first exons
    first_exon_selector = dfg.index.isin(set(first_exon_indices))
    last_exon_selector = dfg.index.isin(set(last_exon_indices))

    # boolean Series to select positive and negative strand
    sp = dfg.Strand == FORWARD_STRAND
    sm = dfg.Strand == REVERSE_STRAND

    # Find the intersection to get, e.g. first exon on positive strand
    fe_sp = first_exon_selector & sp
    fe_sm = first_exon_selector & sm
    le_sp = last_exon_selector & sp
    le_sm = last_exon_selector & sm

    dfg.loc[fe_sp, START_COL] -= dfg.loc[fe_sp, "__extension_up"]
    dfg.loc[le_sp, END_COL] += dfg.loc[le_sp, "__extension_down"]
    dfg.loc[fe_sm, END_COL] += dfg.loc[fe_sm, "__extension_up"]
    dfg.loc[le_sm, START_COL] -= dfg.loc[le_sm, "__extension_down"]


def _get_seqs(df, fs, selector) -> None:
    """Creates/updates column __seq."""
    seqs = []
    d = df[selector]
    for start, end, strand, chrom in zip(d.Start, d.End, d.Strand, d.Chromosome, strict=False):
        _fasta = fs[chrom]
        if strand == REVERSE_STRAND:
            seqs.append((-_fasta[start:end]).seq.upper())  # reverse complement
        elif strand == FORWARD_STRAND:
            seqs.append(_fasta[start:end].seq.upper())
    df.loc[selector, "__seq"] = seqs


def _correct_bounds(df, adjust=None) -> None:
    """Similar to pyranges clip_ranges, inplace, but see below.

    but also: adjust the 5' or 3' end of intervals so they are multiples of 3
    and set boolean attribute __out_of_bounds.
    """
    out_end = df[END_COL] > df["__chromsize"]
    df.loc[out_end, END_COL] = df.loc[out_end, [END_COL, "__chromsize"]].min(axis=1)
    out_start = df[START_COL] < 0
    df.loc[out_start, START_COL] = 0

    sp = df.Strand == FORWARD_STRAND
    sm = df.Strand == REVERSE_STRAND

    if adjust == "5":
        df.loc[out_start & sp, START_COL] += (df.loc[out_start & sp, END_COL] - df.loc[out_start & sp, START_COL]).mod(
            3,
        )
        df.loc[out_end & sm, END_COL] -= (df.loc[out_end & sm, END_COL] - df.loc[out_end & sm, START_COL]).mod(3)

    elif adjust == "3":
        df.loc[out_end & sp, END_COL] -= (df.loc[out_end & sp, END_COL] - df.loc[out_end & sp, START_COL]).mod(3)
        df.loc[out_start & sm, START_COL] += (df.loc[out_start & sm, END_COL] - df.loc[out_start & sm, START_COL]).mod(
            3,
        )

    df.loc[(out_end | out_start), "__out_of_bounds"] = True


def _subseq_df(df, from_end, amount, selector) -> None:
    """Similar to pyranges subseq but inplace."""
    sp = selector & (df.Strand == FORWARD_STRAND)
    sm = selector & (df.Strand == REVERSE_STRAND)
    if from_end == "5":
        df.loc[sp, END_COL] = df.loc[sp, START_COL] + amount
        df.loc[sm, START_COL] = df.loc[sm, END_COL] - amount
    elif from_end == "3":
        df.loc[sp, START_COL] = df.loc[sp, END_COL] - amount
        df.loc[sm, END_COL] = df.loc[sm, START_COL] + amount


def _find_rightmost_stop(seq, motifs) -> int:
    """Return the distance from end of seq of the 1st codon position of the rightmost stop; e.g. 3 is for last codon."""
    for i in range(len(seq) - 3, -1, -3):
        if seq[i : i + 3] in motifs:
            return len(seq) - i
    return -1


def _find_leftmost_start(row, motifs, chunk_size) -> int:
    """Return distance from end of seq of 1st codon position of leftmost start AFTER the pos in column __up_stop_dist."""
    seq = row["__seq"]
    stop_pos = row["__up_stop_dist"]
    start_search = (3 + len(seq) - (stop_pos % chunk_size)) if stop_pos != -1 else 0
    # we search starting from the position next to the stop
    for i in range(start_search, len(seq), 3):
        if seq[i : i + 3] in motifs:
            return len(seq) - i
    return -1


def _find_leftmost_stop(seq, motifs) -> int:
    """Return the position (distance from the start of the string) of the 3rd codon position of the leftmost stop.

    e.g. 3 is for the first codon.
    """
    for i in range(0, len(seq), 3):
        if seq[i : i + 3] in motifs:
            return i + 3
    return -1
