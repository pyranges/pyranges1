from typing import TYPE_CHECKING

import numpy as np

from pyranges1.core.names import END_COL, START_COL, TEMP_ID_COL, VALID_BY_TYPES, VALID_STRAND_BEHAVIOR_TYPE
from pyranges1.core.pyranges_helpers import use_strand_from_validated_strand_behavior

if TYPE_CHECKING:
    from pyranges1 import PyRanges


def _fraction_covered(
    self: "PyRanges",
    other: "PyRanges",
    strand_behavior: VALID_STRAND_BEHAVIOR_TYPE,
    match_by: VALID_BY_TYPES,
) -> np.ndarray:
    """Fraction of each row of self covered by the union of other, 0 for rows of length 0."""
    use_strand = use_strand_from_validated_strand_behavior(self, other, strand_behavior)
    # Merged first, so a base covered by several rows of other is counted once.
    merged = other.merge_overlaps(use_strand=use_strand, match_by=match_by)
    rows = self.copy()
    rows[TEMP_ID_COL] = np.arange(len(self))
    pieces = rows.intersect_overlaps(merged, strand_behavior=strand_behavior, match_by=match_by)
    covered = np.zeros(len(self))
    np.add.at(covered, pieces[TEMP_ID_COL].to_numpy(), (pieces[END_COL] - pieces[START_COL]).to_numpy())
    lengths = (self[END_COL] - self[START_COL]).to_numpy()
    return np.divide(covered, lengths, out=np.zeros(len(self)), where=lengths > 0)
