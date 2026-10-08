import gzip
import http.server
import shutil
import sys
import threading
from functools import partial

import pandas as pd
import pytest

import pyranges1 as pr


@pytest.fixture
def served(tmp_path):
    """Serve tmp_path over http on 127.0.0.1, yielding the base URL."""
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    handler.log_message = lambda *_: None  # type: ignore[method-assign]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_readers_read_http_urls(tmp_path, served) -> None:
    for name in ("aorta.bed", "ensembl.gtf"):
        shutil.copy(pr.example_data.files[name], tmp_path / name)
    (tmp_path / "aorta.bed.gz").write_bytes(gzip.compress((tmp_path / "aorta.bed").read_bytes()))

    pd.testing.assert_frame_equal(pr.read_bed(f"{served}/aorta.bed"), pr.read_bed(tmp_path / "aorta.bed"))
    pd.testing.assert_frame_equal(pr.read_bed(f"{served}/aorta.bed.gz"), pr.read_bed(tmp_path / "aorta.bed"))
    pd.testing.assert_frame_equal(pr.read_gtf(f"{served}/ensembl.gtf"), pr.read_gtf(tmp_path / "ensembl.gtf"))


def test_readers_read_fsspec_urls(tmp_path) -> None:
    fsspec = pytest.importorskip("fsspec")
    with fsspec.open("memory://data/aorta.bed", "wb") as fh:
        fh.write(pr.example_data.files["aorta.bed"].read_bytes())
    pd.testing.assert_frame_equal(pr.read_bed("memory://data/aorta.bed"), pr.read_bed(pr.example_data.files["aorta.bed"]))


def test_a_url_scheme_without_fsspec_says_what_to_install(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(ImportError, match="fsspec"):
        pr.read_bed("s3://bucket/peaks.bed")
