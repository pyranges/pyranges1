import importlib.metadata
import json
import logging
import sys

import numpy as np
import pandas as pd

import pyranges1 as pr

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

__version__ = importlib.metadata.version("pyranges1")  # note: update when pyranges1 becomes default


def version_info() -> None:
    """Print version info for pyranges and its dependencies.

    Used for debugging.
    """
    import importlib.util

    def update_version_info(_version_info: dict[str, str], library: str) -> None:
        version = importlib.import_module(library).__version__ if importlib.util.find_spec(library) else "not installed"

        _version_info[library] = version

    version_info = {
        "pyranges version": pr.__version__,
        "pandas version": pd.__version__,
        "numpy version": np.__version__,
        "python version": ".".join([str(s) for s in sys.version_info]),
    }

    update_version_info(version_info, "ncls")
    update_version_info(version_info, "sorted_nearest")
    update_version_info(version_info, "runlength")
    update_version_info(version_info, "bamread")
    update_version_info(version_info, "pybigwig")
    update_version_info(version_info, "hypothesis")
    update_version_info(version_info, "pyfaidx")

    LOGGER.info(json.dumps(version_info, indent=4))
