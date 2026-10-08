import io
import sys
from contextlib import redirect_stdout
from functools import wraps

import pandas as pd

import pyranges1 as pr
from pyranges1.core.pyranges_helpers import ensure_pyranges

try:
    import fire  # type: ignore[reportMissingImports]
except ImportError:
    fire = None


@wraps(pd.read_csv)
def read_csv(path: str, **kwargs) -> pr.PyRanges:
    """Read a CSV with pandas, then convert the resulting DataFrame into a PyRanges."""
    df = pd.read_csv(path, **kwargs)
    return ensure_pyranges(pr.PyRanges(df))


# 1) Available readers (no from_string)
READERS = {
    "read_bed": pr.read_bed,
    "read_gtf": pr.read_gtf,
    "read_gff": pr.read_gff,
    "read_gff3": pr.read_gff3,
    "read_bam": pr.read_bam,
    "read_bigwig": pr.read_bigwig,
    "read_bigbed": pr.read_bigbed,
    "read_sam": pr.read_sam,
    "read_cram": pr.read_cram,
    "read_vcf": pr.read_vcf,
    "read_pairs": pr.read_pairs,
    "read_paf": pr.read_paf,
    "read_narrowPeak": pr.read_narrowPeak,
    "read_parquet": pr.read_parquet,
    "read_csv": read_csv,
}


def show_usage() -> None:
    """Display help/usage for the pyranges1 CLI."""
    prog = "pyranger"
    sys.stdout.write(
        f"""
pyranger: read sequence interval data into pyranges1 and apply a chain of methods

Usage:
  {prog} reader <args> , [var=reader <args>]… , method <args> , …

  • The command line defines a pipeline of actions separated by " , " and starting with readers
  • The first reader loads the main PyRanges object
  • Every reader beyond the first *must* be named (e.g. b=read_bed b.bed)
  • Methods are invoked on the main object; others can be provided as arguments, e.g. intersect_overlaps b
  • The result replaces the main object in the main

Available readers:
  {", ".join(READERS)}

PyRanges methods:
   see https://pyranges1.readthedocs.io/en/latest/pyranges_objects.html

Examples:
  1. Load only:
       {prog} read_bed sample1.bed

  2. Load + inspect first 5 lines:
       {prog} read_bed sample1.bed , head 5

  3. intersect_overlaps two files:
       {prog} read_bed a.bed , other=read_bed b.bed , intersect_overlaps other

  4. Chain involving three files:
       {prog} read_bed a.bed , b=read_bed b.bed , c=read_bed c.bed , join_overlaps b , intersect_overlaps c

Tip:
  Append `--help` immediately after any reader or method to see its documentation

""".strip()
        + "\n",
    )


def cast_literal(x: str) -> str | int | float | bool:
    """Try int, float, bool, else return the original string."""
    lx = x.lower()
    if lx in ("true", "false"):
        return lx == "true"
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        pass
    return x


def main() -> None:  # noqa: C901,PLR0912,PLR0915
    """Entry point for the pyranges1 CLI."""
    if fire is None:
        sys.stderr.write(
            "Error: the CLI entry-point requires the Python package 'fire'.\n"
            "Please install it with:\n  pip install pyranges1[cli]\n"
        )
        sys.exit(1)

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        show_usage()
        sys.exit(0)

    # 2) Split on literal commas
    segments: list[list[str]] = []
    buf: list[str] = []
    for tok in args:
        if tok == ",":
            if not buf:
                sys.exit("Error: empty segment before comma")
            segments.append(buf)
            buf = []
        else:
            buf.append(tok)
    if buf:
        segments.append(buf)

    # 3) Process the first reader: must be unnamed
    seg0 = segments[0]
    head0 = seg0[0]
    if "=" in head0 or head0 not in READERS:
        sys.exit(f"Error: the first segment must be an unnamed reader ({','.join(READERS)})")
    reader_fn = READERS[head0]
    reader_args = seg0[1:]
    # help for first reader?
    if any(a in ("-h", "--help") for a in reader_args):
        fire.Fire(reader_fn, name=head0, command=["--help"])
        sys.exit(0)
    # invoke reader (suppress Fire's print)
    with redirect_stdout(io.StringIO()):
        pr_obj = fire.Fire(reader_fn, command=reader_args)
    registry = {"pr": pr_obj}

    # 4) Process any additional *reader* segments (they must be named)
    n_readers = 1
    for seg in segments[1:]:
        head = seg[0]
        if "=" in head:
            var, cmd = head.split("=", 1)
            if cmd not in READERS:
                break  # not a reader segment
            reader_fn = READERS[cmd]
            reader_args = seg[1:]
            if any(a in ("-h", "--help") for a in reader_args):
                fire.Fire(reader_fn, name=cmd, command=["--help"])
                sys.exit(0)
            with redirect_stdout(io.StringIO()):
                obj = fire.Fire(reader_fn, command=reader_args)
            registry[var] = obj
            n_readers += 1
        else:
            break

    # 5) The rest are methods on registry['pr']
    primary = registry["pr"]
    for seg in segments[n_readers:]:
        head = seg[0]
        method_args = seg[1:]

        # method help?
        if any(a in ("-h", "--help") for a in method_args):
            fn = getattr(primary, head, None)
            if fn is None:
                sys.exit(f"Error: unknown method '{head}'")
            fire.Fire(fn, name=head, command=["--help"])
            sys.exit(0)

        fn = getattr(primary, head, None)
        if fn is None:
            sys.exit(f"Error: unknown method '{head}' on PyRanges")

        # parse flags & positionals, substituting any var names
        pos: list[object] = []
        flags: dict[str, object] = {}
        i = 0
        while i < len(method_args):
            tok = method_args[i]
            if tok.startswith("--"):
                if "=" in tok:
                    k, v = tok[2:].split("=", 1)
                    flags[k] = cast_literal(v)
                    i += 1
                else:
                    k = tok[2:]
                    if i + 1 < len(method_args) and not method_args[i + 1].startswith("--"):
                        flags[k] = cast_literal(method_args[i + 1])
                        i += 2
                    else:
                        flags[k] = True
                        i += 1
            else:
                if tok in registry:
                    pos.append(registry[tok])
                else:
                    pos.append(cast_literal(tok))
                i += 1

        # call the method
        primary = fn(*pos, **flags)
        registry["pr"] = primary  # update pr to the latest

    # 6) Print the final result
    if primary is not None:
        sys.stdout.write(str(primary) + "\n")


if __name__ == "__main__":
    main()
