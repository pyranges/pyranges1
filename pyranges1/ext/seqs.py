import itertools
from collections.abc import Iterable

import pandas as pd

import pyranges1 as pr  # noqa: F401

# thanks to Devon Ryan at https://bioinformatics.stackexchange.com/questions/3583/what-is-the-fastest-way-to-get-the-reverse-complement-of-a-dna-sequence-in-pytho
# IUPAC ambiguity codes complement too: R (A/G) <-> Y (C/T), K (G/T) <-> M (A/C),
# B (not A) <-> V (not T), D (not C) <-> H (not G); S, W and N are their own.
complement = str.maketrans("ACTGRYKMBVDHactgrykmbvdh", "TGACYRMKVBHDtgacyrmkvbhd")
rnacomplement = str.maketrans("ACUGRYKMBVDHacugrykmbvdh", "UGACYRMKVBHDugacyrmkvbhd")

# build alternative genetic code translation tables based on NCBI codes
GENETIC_CODES = {}  # will store dict of dicts: {genetic_code_id: {codon:aminoacid} }
GENETIC_CODE_AAS = {
    "1": "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "2": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG",
    "3": "FFLLSSSSYY**CCWWTTTTPPPPHHQQRRRRIIMMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "4": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "5": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSSSSVVVVAAAADDEEGGGG",
    "6": "FFLLSSSSYYQQCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "9": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    "10": "FFLLSSSSYY**CCCWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "11": "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "12": "FFLLSSSSYY**CC*WLLLSPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "13": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSSGGVVVVAAAADDEEGGGG",
    "14": "FFLLSSSSYYY*CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    "16": "FFLLSSSSYY*LCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "21": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    "22": "FFLLSS*SYY*LCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "23": "FF*LSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "24": "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSSKVVVVAAAADDEEGGGG",
    "25": "FFLLSSSSYY**CCGWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "26": "FFLLSSSSYY**CC*WLLLAPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "27": "FFLLSSSSYYQQCCWWLLLAPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "28": "FFLLSSSSYYQQCCWWLLLAPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "29": "FFLLSSSSYYYYCC*WLLLAPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "30": "FFLLSSSSYYEECC*WLLLAPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "31": "FFLLSSSSYYEECCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    "33": "FFLLSSSSYYY*CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSSKVVVVAAAADDEEGGGG",
}

for gc_code in GENETIC_CODE_AAS:
    GENETIC_CODES[gc_code] = {"---": "-"}
    for codon_index, combo in enumerate(
        itertools.product("TCAG", repeat=3),
    ):  # iterates over ('T', 'T, 'T),  ('T', 'T', 'C'), ... etc
        codon = "".join(combo)
        GENETIC_CODES[gc_code][codon] = GENETIC_CODE_AAS[gc_code][codon_index]
    GENETIC_CODES[gc_code + "+U"] = GENETIC_CODES[gc_code].copy()
    GENETIC_CODES[gc_code + "+U"]["TGA"] = "U"


def reverse_complement(
    seq: str | pd.Series | Iterable[str],
    *,
    is_rna: bool = False,
    check: bool = False,
) -> str | pd.Series:
    """Reverse complement a DNA sequence.

    The input sequence is in DNA alphabet, upper or lowercase (ATGCatgc). The case is maintained in output.
    Other characters are left unchanged.

    Parameters
    ----------
    seq : str | Iterable[str]
        nucleotide sequence in DNA format (characters: ATGCatgc), or an iterable of such sequences, e.g. a Series

    is_rna : bool
        use this to provide the input seq in RNA format instead (characters: AUGCaugc)

    check : bool
        check if the input string contains only characters present in the translation table, raising an error if not

    Returns
    -------
    revcompseq : str
        Reverse complement nucleotide sequence in DNA format (or RNA if is_rna was set to True).
        If an iterable was provided, a Series of reverse complement sequences is returned.

    Examples
    --------
    >>> pr.seqs.reverse_complement("ATGAAATTTGGGTGA")
    'TCACCCAAATTTCAT'

    >>> pr.seqs.reverse_complement("AUGAAAUUUGGGUGA", is_rna=True)
    'UCACCCAAAUUUCAU'

    >>> pr.seqs.reverse_complement("aaaATCcccGGG")
    'CCCgggGATttt'

    >>> pr.seqs.reverse_complement("ATCWWWCCCTTT")
    'AAAGGGWWWGAT'

    >>> pr.seqs.reverse_complement("AUGAAATGGGTGA", check=True)
    Traceback (most recent call last):
    ...
    ValueError: One or more characters in the input string are not present in the translation table.

    >>> some_seqs=["ATGAAATTTGGGTGA", "AAAGAAATGGGTGACCCCC"]
    >>> pr.seqs.reverse_complement(some_seqs)
    0        TCACCCAAATTTCAT
    1    GGGGGTCACCCATTTCTTT
    dtype: object

    If a Series is provided, the output Series preserve its index:

    >>> some_seqs = pd.Series(["ATGAAATTTGGGTGA", "AAAGAAATGGGTGACCCCC"], index=["s1", "s2"])
    >>> pr.seqs.reverse_complement(some_seqs)
    s1        TCACCCAAATTTCAT
    s2    GGGGGTCACCCATTTCTTT
    dtype: object

    """

    def _rev_comp(seq: str, transtable: dict, *, check: bool) -> str:
        if check:
            accepted_chars = {chr(charint) for charint in transtable}
            if not set(seq).issubset(accepted_chars):
                msg = "One or more characters in the input string are not present in the translation table."
                raise ValueError(msg)
        return seq.translate(transtable)[::-1]

    def _vectorized_rev_comp(seqs: Iterable[str], transtable: dict, *, check: bool) -> pd.Series:
        seqs = pd.Series(list(seqs)) if not isinstance(seqs, pd.Series) else seqs
        if check:
            accepted_chars = {chr(charint) for charint in transtable}
            wrong_seq_sel = seqs.str.contains(f"[^{accepted_chars}]")
            if wrong_seq_sel.any():
                example = str(seqs[wrong_seq_sel].head(1)).split("\n")[0]
                msg = (
                    f"One or more characters in {wrong_seq_sel.sum()} input string(s) are not present in the "
                    f"translation table, such as: {example}"
                )
                raise ValueError(msg)

        return seqs.str.translate(transtable).str[::-1].astype(object)

    transtable = complement if not is_rna else rnacomplement
    return (
        _rev_comp(seq, transtable, check=check)
        if isinstance(seq, str)
        else _vectorized_rev_comp(seq, transtable, check=check)
    )


GENETIC_CODE_OPTION = str | int | dict


def translate(  # noqa: C901
    seq: str | pd.Series | Iterable[str],
    *,
    genetic_code: GENETIC_CODE_OPTION = "1",
    unknown: str = "X",
    sanitize: bool = True,
    cache: int | bool = False,
) -> str | pd.Series:
    """Translate a coding sequence into protein.

    The input sequence is expected to be a DNA sequence in uppercase format (characters: ATGC).
    Incomplete codons at the end of the sequence, as well as non-canonical codons, result in the unknown character "X".

    Parameters
    ----------
    seq : str | Iterable[str]
        nucleotide sequence in uppercase DNA format (characters: ATGC), or an iterable of such sequences, e.g. a Series

    genetic_code : int | str | dict
        int or string-converted NCBI index for genetic code (https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi)
        or dictionary with keys for each codon, values are amino acids (remember to
        include the translation for gaps ``'---':'-'``)
        or string-converted NCBI index with a '+U' suffix to have UGA as selenocysteine (U character)

    unknown : str | None
        codons that are not found in the genetic code table will be translated as this character
        if None, finding an unknown codon will raise an exception instead

    sanitize : bool
        Whether the input is converted to DNA uppercase (from lowercase and/or RNA) to ensure translation works.
        Set to False if you are sure sequences are already DNA uppercase, to speed up execution

    cache : bool | int
        speeds up translation by caching the result of translation of multicodon strings.
        The 1st time, the function is slow since precomputing all results; then, it is ~3 times faster than non-caching translate.
        With cache=True, all 3-codon translations are cached (memory 10Mb, precompute ~ 230ms).
        Provide an int to define how many codons to cache; this is approx the speedup that will be obtained.
        Note: memory and precomputing grow exponentially with the N of codons cached;
        You may use pyranges.seqs.clear_kmer_memory() to free memory.

    Returns
    -------
    pep : str
        Protein sequence resulting from translation, with gaps as '-' and unknown characters as 'X'.
        If an iterable was provided, a Series of protein sequences is returned.

    Examples
    --------
    >>> pr.seqs.translate("ATGAAATTTGGGTGA")
    'MKFG*'

    >>> pr.seqs.translate("ATGTTGCTGAA")
    'MLLX'

    Translate with the vertebrate mithochondrial genetic code:

    >>> pr.seqs.translate("ATGAAATTTGGGTGA", genetic_code=2)
    'MKFGW'

    Translate with a custom genetic code (all codons starting with A are translated as A, the rest as Q):

    >>> gc={codon:'A' if codon.startswith('A') else 'Q' for codon in map(''.join, itertools.product("TCAG", repeat=3))}
    >>> pr.seqs.translate("ATGAAATTTGGGTGA", genetic_code=gc)
    'AAQQQ'

    >>> pr.seqs.translate("AUGAAATTtGGGTGA", sanitize=False)
    'XKXG*'

    >>> pr.seqs.translate("AUGAAATTtGGGTGA", sanitize=False, unknown=None)
    Traceback (most recent call last):
    ...
    ValueError: translate ERROR cannot find codon AUG (pos 0-3) in genetic_code!

    Output for list of sequences is a Series:

    >>> pr.seqs.translate(['ACTGCATAA', 'ATGGGGTACTAG'])
    0     TA*
    1    MGY*
    dtype: object

    >>> pr.seqs.translate(['AAUUUtACTGCACTACGACTAGCTAC', 'ACACTGACTGACTATCTGATCGAC'], sanitize=True)
    0    NFTALRLAX
    1     TLTDYLID
    dtype: object

    When the input is a Series, the output is a Series with the same index:

    >>> x = pd.Series(['ACTGCATAA', 'ATGGGGTACTAG'], index=['s1', 's2'])
    >>> pr.seqs.translate(x)
    s1     TA*
    s2    MGY*
    dtype: object

    Caching makes sense when translating many sequences.
    For large enough data, translate with ``cache=True`` is 3x faster than non-caching translate:

    >>> import random
    >>> random.seed(42)
    >>> many_seqs = [ "".join(random.choices("ATGC", k=1000)) for _ in range(100000) ]
    >>> translated_seqs = pr.seqs.translate(many_seqs, cache=True)
    >>> translated_seqs[0][:50]
    'DRHKKEHVLTRLRGINVIGDRANYLS*CYESYRQGDS*PGRDTLPYHV*D'

    """

    def _translate_noncached(seq: str, *, sanitize: bool = False) -> str:
        seq = seq.upper().replace("U", "T") if sanitize else seq
        output = []
        for pos in range(0, len(seq), 3):
            codon = seq[pos : pos + 3]
            if codon in codon_table:
                output.append(codon_table[codon])
            elif unknown is not None:
                output.append(unknown)
            else:
                msg = f"translate ERROR cannot find codon {codon} (pos {pos}-{pos + 3}) in genetic_code!"
                raise ValueError(msg)
        return "".join(output)

    def _translate_cached(
        seq: str,
        *,
        sanitize: bool = False,
    ) -> str:
        seq = seq.upper().replace("U", "T") if sanitize else seq
        output = []
        for pos in range(0, len(seq), step):
            multicodon = seq[pos : pos + step]
            output.append(
                kmer_codon_table[multicodon]
                if multicodon in kmer_codon_table
                # for final bits of sequences which are shorter than k codons, and also for sequences containing Ns:
                else _translate_noncached(multicodon, sanitize=False),
            )
        return "".join(output)

    def _vectorized_translate(
        seqs: Iterable[str],
        *,
        sanitize: bool,
    ) -> pd.Series:
        seqs = pd.Series(list(seqs)) if not isinstance(seqs, pd.Series) else seqs
        seqs = seqs.str.upper().str.replace("U", "T") if sanitize else seqs

        return (
            seqs.apply(_translate_noncached)
            if not cache  # above: no cache
            else seqs.apply(_translate_cached)
        ).astype(object)

    if isinstance(genetic_code, dict):
        codon_table = genetic_code
    elif str(genetic_code) in GENETIC_CODES:
        codon_table = GENETIC_CODES[str(genetic_code)]
    else:
        msg = f"translate ERROR genetic_code input not recognized: {genetic_code}"
        raise ValueError(msg)

    if cache:
        # precomputing all possible translations for k codons, if not computed already (see _get_kmer_codon_table)
        cache = 3 if cache is True else cache
        kmer_codon_table = _get_kmer_codon_table(genetic_code, cache)
        step = cache * 3

    return (
        _translate_noncached(seq, sanitize=sanitize)
        if (isinstance(seq, str) and not cache)
        else (
            _translate_cached(seq, sanitize=sanitize)
            if isinstance(seq, str) and cache
            else _vectorized_translate(seq, sanitize=sanitize)
        )
    )


kmer_codon_tables = {}


def clear_kmer_memory() -> None:
    """Clear memory used by the translation cache."""
    kmer_codon_tables.clear()


def _get_kmer_codon_table(genetic_code, k) -> dict[str, str]:
    if (genetic_code, k) not in kmer_codon_tables:
        all_codons = list(GENETIC_CODES["1"].keys())
        codon_table_multik = {
            multicodon: translate(multicodon, genetic_code=genetic_code)
            for multicodon in map("".join, itertools.product(*(all_codons for i in range(k))))
        }
        kmer_codon_tables[(genetic_code, k)] = codon_table_multik

    return kmer_codon_tables[(genetic_code, k)]
