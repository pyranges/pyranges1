import gzip
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np
import pandas as pd

from pyranges1.core.names import CHROM_COL, END_COL, START_COL
from pyranges1.core.pyranges_helpers import ensure_pyranges

if TYPE_CHECKING:
    from pyranges1 import PyRanges

VALUE_COL = "Value"
_HEADER_LINES = ("track", "browser", "#")


def _open_text(path: Path) -> TextIO:
    """Open a text file, gzip-compressed or not, telling which by its content."""
    with path.open("rb") as fh:
        compressed = fh.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if compressed else path.open()


def _read_bedgraph(f: "str | Path", nrows: int | None) -> "PyRanges":
    path = Path(f)
    with _open_text(path) as fh:
        lines = [line for line in fh if line.strip() and not line.startswith(_HEADER_LINES)]
    if lines and len(lines[0].split("\t")) != 4:  # noqa: PLR2004
        msg = f"A bedGraph file has 4 tab-separated fields; {path} has {len(lines[0].split(chr(9)))}."
        raise ValueError(msg)
    df = pd.read_csv(
        StringIO("".join(lines[:nrows] if nrows is not None else lines)),
        sep="\t",
        header=None,
        names=[CHROM_COL, START_COL, END_COL, VALUE_COL],
        dtype={CHROM_COL: "category", VALUE_COL: "float64"},
    )
    return ensure_pyranges(df)


def _wig_section(header: str) -> dict[str, str]:
    """Parse the key=value settings of a fixedStep or variableStep line."""
    return dict(field.split("=", 1) for field in header.split()[1:])


def _read_wig(f: "str | Path") -> "PyRanges":
    """Read the fixedStep and variableStep sections of a WIG file, positions 1-based as WIG has them."""
    path = Path(f)
    frames = []
    section: tuple[str, dict[str, str]] | None = None
    body: list[str] = []

    def flush() -> None:
        if section is None or not body:
            return
        kind, settings = section
        # Without a span, a fixedStep value covers its whole step, as UCSC's wigToBigWig reads it.
        span = int(settings.get("span", settings.get("step", 1) if kind == "fixedStep" else 1))
        values = pd.read_csv(StringIO("".join(body)), sep=r"\s+", header=None, dtype="float64")
        if kind == "variableStep":
            starts = values[0].to_numpy(np.int64) - 1
            scores = values[1].to_numpy()
        else:
            step = int(settings.get("step", 1))
            starts = int(settings["start"]) - 1 + step * np.arange(len(values), dtype=np.int64)
            scores = values[0].to_numpy()
        frames.append(
            pd.DataFrame({CHROM_COL: settings["chrom"], START_COL: starts, END_COL: starts + span, VALUE_COL: scores})
        )

    with _open_text(path) as fh:
        for line in fh:
            if line.startswith(("fixedStep", "variableStep")):
                flush()
                section, body = (line.split()[0], _wig_section(line)), []
            elif line.strip() and not line.startswith(_HEADER_LINES):
                if section is None:
                    msg = f"{path}: a data line before any fixedStep or variableStep line: {line.strip()!r}."
                    raise ValueError(msg)
                body.append(line)
        flush()

    columns = [CHROM_COL, START_COL, END_COL, VALUE_COL]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    df[CHROM_COL] = df[CHROM_COL].astype("category")
    return ensure_pyranges(df.astype({START_COL: "int64", END_COL: "int64", VALUE_COL: "float64"}))


def _to_bedgraph(self: "PyRanges", path: "str | None", value_col: str) -> "str | None":
    if value_col not in self.columns:
        msg = f"to_bedgraph writes the column {value_col!r}, which this PyRanges does not have."
        raise ValueError(msg)
    out = self[[CHROM_COL, START_COL, END_COL, value_col]]
    return out.to_csv(path, sep="\t", header=False, index=False, compression="infer")
