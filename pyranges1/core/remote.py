"""Readers' access to files at a URL."""

import shutil
import tempfile
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, cast

# Fetched with the standard library; any other scheme goes through fsspec.
_STDLIB_SCHEMES = ("http", "https", "ftp")


def _is_url(f: object) -> bool:
    """Tell whether `f` names a file at a URL rather than a local path."""
    scheme, separator, _ = str(f).partition("://")
    return bool(separator) and scheme.isidentifier() and scheme != "file"


@contextmanager
def _local_copy(url: str, storage_options: dict | None) -> Iterator[Path]:
    """Copy the file at `url` to a temporary file, removed on exit.

    The copy keeps the URL's suffixes, so compression is recognised as for a local
    file. http(s) and ftp need nothing extra; s3://, gs://, az:// and the rest need
    fsspec and its filesystem package (s3fs, gcsfs, adlfs, ...).
    """
    scheme = url.partition("://")[0]
    suffix = "".join(Path(url.split("?")[0]).suffixes)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as copy:
        if scheme in _STDLIB_SCHEMES and not storage_options:
            with urllib.request.urlopen(url) as source:  # noqa: S310 -- the scheme is checked above
                shutil.copyfileobj(source, copy, 1 << 23)
        else:
            try:
                import fsspec  # type: ignore[import]
            except ImportError:
                msg = f"Reading {scheme}:// paths requires fsspec: `pip install fsspec` and the package for {scheme}."
                raise ImportError(msg) from None
            with fsspec.open(url, "rb", **(storage_options or {})) as source:
                # fsspec is untyped; its OpenFile yields a binary file object.
                shutil.copyfileobj(cast("BinaryIO", source), copy, 1 << 23)
    try:
        yield Path(copy.name)
    finally:
        Path(copy.name).unlink(missing_ok=True)
