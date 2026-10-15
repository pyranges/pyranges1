#!/usr/bin/env python3

import argparse
import logging
import shlex
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description="This script runs various checks and tests.")
parser.add_argument("-v", "--verbose", action="store_true", help="Show output from programs run.")
parser.add_argument("-l", "--list", action="store_true", help="List all checks and tests and exits.")
args = parser.parse_args()
verbose: bool = args.verbose
show_checks: bool = args.list


def run_command(command, *, show_output: bool) -> int:
    """Run a shell command and return its exit status."""
    result = subprocess.run(command, capture_output=not show_output, text=True, check=False)  # noqa: S603
    return result.returncode


def main() -> int:
    """Run all continuous integration checks."""
    commands = {
        "ruff format": ["ruff", "format", "--check", "--diff", "pyranges1"],
        "ruff check": ["ruff", "check", "pyranges1"],
        "ty": ["ty", "check"],
        "pytest": ["pytest", "--doctest-modules", "pyranges1/", "tests/unit/"],
        # "more doctest": ["python", "tests/run_doctest_tutorial_howto.py"],
    }
    if show_checks:
        LOGGER.info("Available checks and tests:")
        for task, command in commands.items():
            LOGGER.info("    %s: %s", task, shlex.join(command))
        sys.exit(0)

    failed_tasks = []

    for task, command in commands.items():
        LOGGER.debug("Running: %s", shlex.join(command))
        retcode = run_command(command, show_output=verbose)
        if retcode != 0:
            LOGGER.debug("Task %s failed!", task)
            failed_tasks.append(task)

    if failed_tasks:
        LOGGER.warning("The following tasks failed:")
        for task in failed_tasks:
            LOGGER.warning("%s:\n    %s", task, shlex.join(commands[task]))
        return 1

    LOGGER.debug("All tasks completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
