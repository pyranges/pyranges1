"""Module of example data.

See Also
--------
pyranges.random : generate random PyRanges

Examples
--------
>>> pr.example_data
Available example data:
-----------------------
example_data.chipseq            : Example ChIP-seq data.
example_data.chipseq_background : Example ChIP-seq data.
example_data.chromsizes         : Example chromsizes data (hg19).
example_data.ensembl_gtf        : Example gtf file from Ensembl.
example_data.interpro_hits      : Example of InterPro protein hits.
example_data.rfam_hits          : Example of RNA motifs (Rfam) as 1-based dataframe.
example_data.f1                 : Example bed file.
example_data.f2                 : Example bed file.
example_data.aorta              : Example ChIP-seq data.
example_data.aorta2             : Example ChIP-seq data.
example_data.ncbi_gff           : Example NCBI GFF data.
example_data.ncbi_fasta         : Example NCBI fasta.
example_data.files              : Return a dict of basenames to file paths of available data.

>>> pr.example_data.f1
  index  |    Chromosome      Start      End  Name         Score  Strand
  int64  |    category        int64    int64  str          int64  category
-------  ---  ------------  -------  -------  ---------  -------  ----------
      0  |    chr1                3        6  interval1        0  +
      1  |    chr1                5        7  interval2        0  -
      2  |    chr1                8        9  interval3        0  +
PyRanges with 3 rows, 6 columns, and 1 index columns.
Contains 1 chromosomes and 2 strands.


"""

import logging
import sys
import tempfile
import typing
from importlib.resources import files
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

import pyranges1 as pr
from pyranges1.core.pyranges_helpers import ensure_pyranges

if typing.TYPE_CHECKING:
    import pyfaidx

    from pyranges1.core.pyranges_main import PyRanges

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# Define a custom class property decorator
class ClassProperty:
    def __init__(self, fget: Any) -> None:
        self.fget = fget
        # Propagate the docstring from the getter function
        self.__doc__ = fget.__doc__

    def __get__(self, instance: Any | None, owner: type[Any]) -> Any:
        return self.fget(owner)


class ExampleData:
    _files: ClassVar[dict[str, Path]] = {}

    def __repr__(self) -> str:
        methods_info = "Available example data:\n-----------------------\n"
        items = [
            (name, prop)
            for name, prop in self.__class__.__dict__.items()
            if isinstance(prop, property) and not name.startswith("_")
        ]
        max_name_len = max(len(name) for name, _ in (*items, ("files", "")))
        for name, prop in items:
            doc_line = prop.fget.__doc__.split("\n")[0] if prop.fget.__doc__ else ""
            methods_info += f"example_data.{name:<{max_name_len}} : {doc_line}\n"
        for name in ["files"]:
            doc_line = self.__class__.__dict__[name].__doc__.strip().split("\n")[0]
            methods_info += f"example_data.{name:<{max_name_len}} : {doc_line}\n"
        return methods_info.strip()

    @ClassProperty
    def files(self) -> dict[str, Path]:
        """Return a dict of basenames to file paths of available data.

        Examples
        --------
        >>> bam = ExampleData.files["smaller.bam"]
        >>> bam.exists()
        True

        """
        if ExampleData._files:
            return ExampleData._files

        paths = []
        for f in files("pyranges1").joinpath("data").iterdir():
            if not isinstance(f, Path):
                msg = f"Expected Path, got {type(f)}"
                raise TypeError(msg)
            if "__" not in f.name:
                paths.append(f)
        ExampleData._files = {f.name: Path(f) for f in paths}
        return ExampleData._files

    @staticmethod
    def _read_bed_from_string(contents: str) -> "PyRanges":
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as f:
            f.write(contents)
            f.flush()
            return pr.read_bed(Path(f.name))

    @staticmethod
    def _read_gtf_from_string(contents: str) -> "PyRanges":
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as f:
            f.write(contents)
            f.flush()
            return pr.read_gtf(Path(f.name))

    @staticmethod
    def _read_tsv_from_string(contents: str) -> "pd.DataFrame":
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as f:
            f.write(contents)
            f.flush()
            return pd.read_csv(Path(f.name), sep="\t")

    @property
    def chipseq(self) -> "pr.PyRanges":
        """Example ChIP-seq data.

        From the SICER software.
        """
        contents = """chr8	28510032	28510057	U0	0	-
chr7	107153363	107153388	U0	0	-
chr5	135821802	135821827	U0	0	-
chr14	19418999	19419024	U0	0	-
chr12	106679761	106679786	U0	0	-
chr21	40099618	40099643	U0	0	+
chr8	22714402	22714427	U0	0	-
chr19	19571102	19571127	U0	0	+
chr3	140986358	140986383	U0	0	-
chr10	35419784	35419809	U0	0	-
chr4	98488749	98488774	U0	0	+
chr11	22225193	22225218	U0	0	+
chr1	38457520	38457545	U0	0	+
chr1	80668132	80668157	U0	0	-
chr2	152562484	152562509	U0	0	-
chr4	153155301	153155326	U0	0	+
chr9	120803448	120803473	U0	0	+
chr6	89296757	89296782	U0	0	-
chr1	194245558	194245583	U0	0	+
chr8	57916061	57916086	U0	0	+"""
        return self._read_bed_from_string(contents)

    @property
    def chipseq_background(self) -> "pr.PyRanges":
        """Example ChIP-seq data.

        From the SICER software.
        """
        contents = """chr7	20246668	20246693	U0	0	+
chr1	39036822	39036847	U0	0	+
chr19	47109000	47109025	U0	0	-
chr10	90059861	90059886	U0	0	-
chr3	55648137	55648162	U0	0	+
chr7	91135110	91135135	U0	0	+
chr13	100938475	100938500	U0	0	+
chr3	115816130	115816155	U0	0	+
chr19	43528773	43528798	U0	0	+
chr10	73781101	73781126	U0	0	+"""
        return self._read_bed_from_string(contents)

    @property
    def chromsizes(self) -> "pr.PyRanges":
        """Example chromsizes data (hg19)."""
        contents = """chr1	0	249250621
chr2	0	243199373
chr3	0	198022430
chr4	0	191154276
chr5	0	180915260
chr6	0	171115067
chr7	0	159138663
chrX	0	155270560
chr8	0	146364022
chr9	0	141213431
chr10	0	135534747
chr11	0	135006516
chr12	0	133851895
chr13	0	115169878
chr14	0	107349540
chr15	0	102531392
chr16	0	90354753
chr17	0	81195210
chr18	0	78077248
chr20	0	63025520
chrY	0	59373566
chr19	0	59128983
chr22	0	51304566
chr21	0	48129895
chrM	0	16571"""
        return self._read_bed_from_string(contents)

    @property
    def ensembl_gtf(self) -> "pr.PyRanges":
        """Example gtf file from Ensembl."""
        contents = """#!genome-build GRCh38.p10
#!genome-version GRCh38
#!genome-date 2013-12
#!genome-build-accession NCBI:GCA_000001405.25
#!genebuild-last-updated 2017-06
1	havana	gene	11869	14409	.	+	.	gene_id "ENSG00000223972"; gene_version "5"; gene_name "DDX11L1"; gene_source "havana"; gene_biotype "transcribed_unprocessed_pseudogene";
1	havana	transcript	11869	14409	.	+	.	gene_id "ENSG00000223972"; gene_version "5"; transcript_id "ENST00000456328"; transcript_version "2"; gene_name "DDX11L1"; gene_source "havana"; gene_biotype "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-202"; transcript_source "havana"; transcript_biotype "processed_transcript"; tag "basic"; transcript_support_level "1";
1	havana	exon	11869	12227	.	+	.	gene_id "ENSG00000223972"; gene_version "5"; transcript_id "ENST00000456328"; transcript_version "2"; exon_number "1"; gene_name "DDX11L1"; gene_source "havana"; gene_biotype "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-202"; transcript_source "havana"; transcript_biotype "processed_transcript"; exon_id "ENSE00002234944"; exon_version "1"; tag "basic"; transcript_support_level "1";
1	havana	exon	12613	12721	.	+	.	gene_id "ENSG00000223972"; gene_version "5"; transcript_id "ENST00000456328"; transcript_version "2"; exon_number "2"; gene_name "DDX11L1"; gene_source "havana"; gene_biotype "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-202"; transcript_source "havana"; transcript_biotype "processed_transcript"; exon_id "ENSE00003582793"; exon_version "1"; tag "basic"; transcript_support_level "1";
1	havana	exon	13221	14409	.	+	.	gene_id "ENSG00000223972"; gene_version "5"; transcript_id "ENST00000456328"; transcript_version "2"; exon_number "3"; gene_name "DDX11L1"; gene_source "havana"; gene_biotype "transcribed_unprocessed_pseudogene"; transcript_name "DDX11L1-202"; transcript_source "havana"; transcript_biotype "processed_transcript"; exon_id "ENSE00002312635"; exon_version "1"; tag "basic"; transcript_support_level "1";
1	havana	exon	112700	112804	.	-	.	gene_id "ENSG00000238009"; gene_version "6"; transcript_id "ENST00000471248"; transcript_version "1"; exon_number "2"; gene_name "AL627309.1"; gene_source "ensembl_havana"; gene_biotype "lincRNA"; transcript_name "AL627309.1-203"; transcript_source "havana"; transcript_biotype "lincRNA"; exon_id "ENSE00001957285"; exon_version "1"; transcript_support_level "5";
1	havana	exon	110953	111357	.	-	.	gene_id "ENSG00000238009"; gene_version "6"; transcript_id "ENST00000471248"; transcript_version "1"; exon_number "3"; gene_name "AL627309.1"; gene_source "ensembl_havana"; gene_biotype "lincRNA"; transcript_name "AL627309.1-203"; transcript_source "havana"; transcript_biotype "lincRNA"; exon_id "ENSE00001879696"; exon_version "1"; transcript_support_level "5";
1	ensembl	transcript	120725	133723	.	-	.	gene_id "ENSG00000238009"; gene_version "6"; transcript_id "ENST00000610542"; transcript_version "1"; gene_name "AL627309.1"; gene_source "ensembl_havana"; gene_biotype "lincRNA"; transcript_name "AL627309.1-205"; transcript_source "ensembl"; transcript_biotype "lincRNA"; tag "basic"; transcript_support_level "5";
1	ensembl	exon	133374	133723	.	-	.	gene_id "ENSG00000238009"; gene_version "6"; transcript_id "ENST00000610542"; transcript_version "1"; exon_number "1"; gene_name "AL627309.1"; gene_source "ensembl_havana"; gene_biotype "lincRNA"; transcript_name "AL627309.1-205"; transcript_source "ensembl"; transcript_biotype "lincRNA"; exon_id "ENSE00003748456"; exon_version "1"; tag "basic"; transcript_support_level "5";
1	ensembl	exon	129055	129223	.	-	.	gene_id "ENSG00000238009"; gene_version "6"; transcript_id "ENST00000610542"; transcript_version "1"; exon_number "2"; gene_name "AL627309.1"; gene_source "ensembl_havana"; gene_biotype "lincRNA"; transcript_name "AL627309.1-205"; transcript_source "ensembl"; transcript_biotype "lincRNA"; exon_id "ENSE00003734824"; exon_version "1"; tag "basic"; transcript_support_level "5";
1	ensembl	exon	120874	120932	.	-	.	gene_id "ENSG00000238009"; gene_version "6"; transcript_id "ENST00000610542"; transcript_version "1"; exon_number "3"; gene_name "AL627309.1"; gene_source "ensembl_havana"; gene_biotype "lincRNA"; transcript_name "AL627309.1-205"; transcript_source "ensembl"; transcript_biotype "lincRNA"; exon_id "ENSE00003740919"; exon_version "1"; tag "basic"; transcript_support_level "5";"""
        return self._read_gtf_from_string(contents)

    @property
    def interpro_hits(self) -> "pr.PyRanges":
        """Example of InterPro protein hits."""
        contents = """Chromosome	predictor	feature	Start	End
CAD5126873.1	SignalP_EUK	SignalP-noTM	1	18
CAD5126492.1	SignalP_EUK	SignalP-noTM	1	16
CAD5126498.1	SignalP_EUK	SignalP-noTM	1	21
CAD5126499.1	SignalP_EUK	SignalP-noTM	1	21"""
        d = pr.PyRanges(self._read_tsv_from_string(contents))
        d["Start"] -= 1
        return ensure_pyranges(d)

    @property
    def rfam_hits(self) -> "pd.DataFrame":
        """Example of RNA motifs (Rfam) as 1-based dataframe."""
        contents = """target_name	accession	query_name	accession.1	mdl	mdl_from	mdl_to	seq_from	seq_to	strand	trunc	pass	gc	bias	score	E-value	inc	description_of_target
rna-DGYR_LOCUS12552	-	GAIT	RF00179	cm	1	71	267	311	+	no	1	0.31	0.0	19.2	0.0067	!	-
rna-DGYR_LOCUS12552-2	-	GAIT	RF00179	cm	1	71	288	332	+	no	1	0.31	0.0	19.2	0.0067	!	-
rna-DGYR_LOCUS13738	-	REN-SRE	RF00180	cm	1	37	1641	1678	+	no	1	0.37	0.0	16.6	0.0015	!	-
rna-DGYR_LOCUS14091	-	IFN_gamma	RF00259	cm	1	169	137	43	-	no	1	0.27	0.7	19.6	0.004	!	-
rna-DGYR_LOCUS14091	-	snoZ30	RF00288	cm	1	97	547	616	+	no	1	0.33	0.0	18.5	0.005	!	-
rna-DGYR_LOCUS13734	-	snoR21	RF00352	cm	1	82	3276	3189	-	no	1	0.35	0.0	30.4	0.00027	!	-
rna-DGYR_LOCUS13734	-	HIV_GSL3	RF00376	cm	1	84	3356	3278	-	no	1	0.34	0.0	21.2	0.0022	!	-
rna-DGYR_LOCUS14091	-	SNORA53	RF00563	cm	1	248	833	748	-	no	1	0.29	0.1	15.4	0.0026	!	-
rna-DGYR_LOCUS13737	-	mir-320	RF00736	cm	1	76	1264	1310	+	no	1	0.28	0.1	15.6	0.0095	!	-
rna-DGYR_LOCUS13739	-	sR43	RF01128	hmm	7	38	432	468	+	-	6	0.43	0.0	10.7	0.0074	!	-
rna-DGYR_LOCUS13734	-	SNORD107	RF01164	cm	1	74	981	1052	+	no	1	0.25	0.6	21.2	0.0019	!	-
rna-DGYR_LOCUS14092	-	snR77	RF01181	hmm	17	61	572	617	+	-	6	0.33	0.7	12.9	0.0016	!	-
rna-DGYR_LOCUS14092	-	snR50	RF01190	cm	1	89	716	792	+	no	1	0.30	0.1	21.0	0.0085	!	-
rna-DGYR_LOCUS13737	-	snoR31	RF01288	cm	1	93	244	172	-	no	1	0.23	1.0	21.6	0.0045	!	-
rna-DGYR_LOCUS13738	-	CRISPR-DR42	RF01351	hmm	7	27	312	332	+	-	6	0.24	0.3	13.8	0.0017	!	-
rna-DGYR_LOCUS13734	-	rli62	RF01486	cm	1	135	276	401	+	no	1	0.36	0.0	20.1	0.0024	!	-
rna-DGYR_LOCUS13738	-	6S-Flavo	RF01685	cm	1	108	2414	2503	+	no	1	0.28	0.4	25.2	0.002	!	-
rna-DGYR_LOCUS13734	-	alpha_tmRNA	RF01849	cm	40	242	2457	2581	+	no	1	0.33	0.2	13.5	0.00042	!	-
rna-DGYR_LOCUS13734	-	STnc420	RF02054	cm	1	58	94	48	-	no	1	0.19	0.6	18.2	0.0076	!	-
rna-DGYR_LOCUS14095	-	MESTIT1_1	RF02148	hmm	60	124	174	112	-	-	6	0.25	1.1	11.7	0.0022	!	-
rna-DGYR_LOCUS14095-2	-	MESTIT1_1	RF02148	hmm	60	124	174	112	-	-	6	0.25	1.1	11.7	0.0022	!	-
rna-DGYR_LOCUS13738	-	PVT1_5	RF02168	hmm	17	117	739	635	-	-	6	0.26	0.1	13.6	0.0008	!	-
rna-DGYR_LOCUS13736	-	TtnuHACA18	RF02325	cm	1	130	376	263	-	no	1	0.36	0.0	28.3	0.00024	!	-
rna-DGYR_LOCUS13740	-	TtnuHACA18	RF02325	cm	1	130	538	425	-	no	1	0.36	0.0	28.3	0.00024	!	-
rna-DGYR_LOCUS13741	-	TtnuHACA18	RF02325	cm	1	130	538	425	-	no	1	0.36	0.0	28.3	0.00024	!	-
rna-DGYR_LOCUS12552	-	psRNA6	RF02350	cm	242	333	884	798	-	5'	2	0.21	8.4	11.5	0.0063	!	-
rna-DGYR_LOCUS12552-2	-	psRNA6	RF02350	cm	242	333	905	819	-	5'	2	0.21	8.4	11.5	0.0063	!	-
rna-DGYR_LOCUS13736	-	ToxI	RF02519	cm	1	34	393	436	+	no	1	0.32	0.0	19.4	0.005	!	-
rna-DGYR_LOCUS14092	-	OppA_thermometer	RF02777	cm	1	65	850	913	+	3'	3	0.25	0.2	12.7	0.0042	!	-
rna-DGYR_LOCUS13737	-	RT-8	RF03029	cm	1	92	163	77	-	no	1	0.33	0.0	21.4	0.0026	!	-
rna-DGYR_LOCUS13734	-	mir-5697	RF03269	cm	1	59	976	1026	+	no	1	0.18	3.6	18.7	0.005	!	-
rna-DGYR_LOCUS13737	-	mir-3047	RF03339	cm	1	61	550	600	+	no	1	0.33	0.0	18.2	0.0074	!	-
rna-DGYR_LOCUS13737	-	mir-3156	RF03360	cm	1	77	544	605	+	no	1	0.32	0.0	26.3	4.9e-05	!	-
rna-DGYR_LOCUS13737	-	MIR1523	RF04113	cm	1	92	530	612	+	no	1	0.30	0.1	18.8	0.0053	!	-
rna-DGYR_LOCUS13737	-	MIR8001	RF04255	cm	1	67	550	600	+	no	1	0.33	0.0	18.7	0.0068	!	-"""
        return self._read_tsv_from_string(contents)

    @property
    def f1(self) -> "pr.PyRanges":
        """Example bed file."""
        contents = """chr1	3	6	interval1	0	+
chr1	5	7	interval2	0	-
chr1	8	9	interval3	0	+"""
        return self._read_bed_from_string(contents)

    @property
    def f2(self) -> "pr.PyRanges":
        """Example bed file."""
        contents = """chr1	1	2	a	0	+
chr1	6	7	b	0	-"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as f:
            f.write(contents)
            f.flush()
            return pr.read_bed(Path(f.name))

    @property
    def aorta(self) -> "pr.PyRanges":
        """Example ChIP-seq data.

        From the epigenomics roadmap.
        """
        return pr.read_bed(ExampleData.files["aorta.bed"])

    @property
    def aorta2(self) -> "pr.PyRanges":
        """Example ChIP-seq data.

        From the epigenomics roadmap.
        """
        return pr.read_bed(ExampleData.files["aorta2.bed"])

    @property
    def ncbi_gff(self) -> "pr.PyRanges":
        """Example NCBI GFF data.

        Subset of the NCBI annotation of D.gyrociliatus assembly GCA_904063045.1.
        """
        return pr.read_gff3(ExampleData.files["ncbi.gff.gz"])

    @property
    def ncbi_fasta(self) -> "pyfaidx.Fasta":
        """Example NCBI fasta.

        Subset of the NCBI D.gyrociliatus assembly GCA_904063045.1.

        A pyfaidx.Fasta object is returned. If not installed, an exception is raised.
        To retrieve the location of the fasta file, use pyranges.example_data.files['ncbi.fasta']
        """
        try:
            import pyfaidx
        except ImportError:
            LOGGER.exception(
                "To use this method, pyfaidx must be installed. To get just the fasta file path, use pyranges.example_data.files['ncbi.fasta']. Use `conda install -c bioconda pyfaidx` or `pip install pyfaidx` to install pyfaidx.",
            )
            sys.exit(1)

        # note: example data include ncbi.fasta.fai, the pyfaidx index
        return pyfaidx.Fasta(ExampleData.files["ncbi.fasta"])


example_data = ExampleData()
