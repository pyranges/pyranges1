import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

import pandas as pd
from tabulate import tabulate

if TYPE_CHECKING:
    from pyranges1 import PyRanges


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _summary(
    self: "PyRanges",
    *,
    return_df: bool = False,
) -> pd.DataFrame | None:
    lengths = {}
    total_lengths = {}
    lengths["pyrange"] = self.lengths()
    total_lengths["pyrange"] = [self.length]

    if self.strand_valid:
        c = self.merge_overlaps(use_strand=True)
        lengths["coverage_forward"] = c.loci["+"].lengths()
        lengths["coverage_reverse"] = c.loci["-"].lengths()
        total_lengths["coverage_forward"] = [c.loci["+"].length]
        total_lengths["coverage_reverse"] = [c.loci["-"].length]
    else:
        c = self

    c = c.merge_overlaps(use_strand=False)
    lengths["coverage_unstranded"] = c.lengths()
    total_lengths["coverage_unstranded"] = [c.length]

    summaries = OrderedDict()

    # stats for lengths
    for summary, s in lengths.items():
        summaries[summary] = s.describe()

    summary_lengths = pd.concat(summaries.values(), axis=1)
    summary_lengths.columns = pd.Index(summaries)

    df = pd.DataFrame.from_dict(total_lengths)
    df.index = pd.Index(["sum"])
    summary_lengths = pd.concat([summary_lengths, df])

    if return_df:
        return summary_lengths
    str_repr = tabulate(summary_lengths.itertuples(), headers=list(summary_lengths.columns))
    print(str_repr)  # noqa: T201
    return None
