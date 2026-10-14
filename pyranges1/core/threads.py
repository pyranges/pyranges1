"""One place to set every thread pool PyRanges work runs on.

Three separate pools do the work, each with its own control and its own rule
about when that control still bites:

- the **interval kernels**, through `ruranges` and rayon. rayon builds its
  global pool the first time it is used and reads `RAYON_NUM_THREADS` then;
  `ruranges` exposes no setter, so this can only be chosen *before* the first
  operation runs.
- **pyarrow's CPU pool**, which parses BED, GTF and GFF3 when pyarrow is
  installed. `pa.set_cpu_count` changes it at any point.
- **pyarrow's IO pool**, which reads the bytes. Separate from the CPU pool,
  defaults to 8, and is missed by every environment variable people usually
  reach for.

Setting the environment variables alone is not enough: pyarrow reads them when
it builds its pools at import, and ignores them afterwards. So both are done —
the variables for a pyarrow that has not been imported yet and for
subprocesses, and the setters for the pools already running.

The pools are reached through `sys.modules` rather than by importing pyarrow.
Importing it here to resize it would pull an optional dependency into every
call, and would be pointless when it has not been imported: the variables are
already set, and pyarrow reads them when it does get imported. In practice
pandas imports pyarrow itself, so the setters are usually the branch that runs.
"""

import os
import sys

__all__ = ["set_num_threads"]


def set_num_threads(num_threads: int | None = None) -> int:
    """Set how many threads the interval kernels and the readers use.

    Parameters
    ----------
    num_threads : int | None, optional
        The number of threads to use. None, the default, restores every pool to
        one thread per available core.

    Returns
    -------
    int
        The thread count that was applied.

    Notes
    -----
    The interval kernels are the one pool this cannot resize after the fact.
    rayon builds its global pool on first use and `ruranges` exposes no setter,
    so `RAYON_NUM_THREADS` is read once and ignored from then on: call this
    before the first PyRanges operation, or that half keeps the count it
    started with. pyarrow's two pools have no such rule.

    Nothing here requires pyarrow. When it is absent the environment variables
    are still set — harmless, and they apply if pyarrow is imported later or by
    a subprocess.

    Examples
    --------
    >>> import pyranges1 as pr
    >>> pr.set_num_threads(4)
    4

    >>> import os
    >>> os.environ["RAYON_NUM_THREADS"]
    '4'

    """
    resolved = (os.cpu_count() or 1) if num_threads is None else int(num_threads)
    if resolved < 1:
        msg = f"num_threads must be at least 1, got {resolved}."
        raise ValueError(msg)

    # Read by rayon when it builds the global pool, and by a pyarrow that has
    # not been imported yet. Also inherited by subprocesses, which is what a
    # benchmark harness relies on.
    os.environ["RAYON_NUM_THREADS"] = str(resolved)
    os.environ["OMP_NUM_THREADS"] = str(resolved)
    os.environ["ARROW_IO_THREADS"] = str(resolved)

    # Only if it is already loaded: an unimported pyarrow will read the
    # variables set above, and importing it here just to resize it would drag
    # an optional dependency into a call that does not need it.
    pyarrow = sys.modules.get("pyarrow")
    if pyarrow is not None:
        # The pools that already exist ignore the variables above.
        pyarrow.set_cpu_count(resolved)
        pyarrow.set_io_thread_count(resolved)

    return resolved
