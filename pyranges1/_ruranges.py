from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any

MIN_RURANGES_VERSION = "0.1.5"


def _version_tuple(v: str) -> tuple[int, ...]:
    nums = [int(x) for x in re.findall(r"\d+", v)]
    return tuple(nums) if nums else (0,)


def require_ruranges() -> Any:
    import ruranges

    try:
        installed = version("ruranges")
    except PackageNotFoundError as exc:
        msg = "pyranges1 requires the `ruranges` package."
        raise ImportError(msg) from exc

    if _version_tuple(installed) < _version_tuple(MIN_RURANGES_VERSION):
        msg = f"pyranges1 requires ruranges>={MIN_RURANGES_VERSION}; found {installed}."
        raise ImportError(msg)

    if not hasattr(ruranges, "numpy"):
        msg = (
            "Installed `ruranges` is missing the `ruranges.numpy` namespace. "
            f"Please install ruranges>={MIN_RURANGES_VERSION}."
        )
        raise ImportError(msg)

    return ruranges
