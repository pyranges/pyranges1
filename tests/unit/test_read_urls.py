import gzip
import http.server
import shutil
import socket
import socketserver
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
    # socketserver, not http.server's HTTPServer: its server_bind looks up the host's
    # fully qualified name, which takes half a minute on some machines.
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
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


def test_read_bed_region_over_http(tmp_path, served, monkeypatch) -> None:
    pysam = pytest.importorskip("pysam")
    shutil.copy(pr.example_data.files["aorta.bed"], tmp_path / "a.bed")
    pysam.tabix_index(str(tmp_path / "a.bed"), preset="bed")
    # htslib saves a copy of a remote index in the working directory.
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    remote = pr.read_bed(f"{served}/a.bed.gz", region="chr1:9930-9960")
    pd.testing.assert_frame_equal(remote, pr.read_bed(tmp_path / "a.bed.gz", region="chr1:9930-9960"))


def test_a_stalled_server_times_out(monkeypatch) -> None:
    from pyranges1.core import remote

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()  # accepts connections, never answers
        monkeypatch.setattr(remote, "_TIMEOUT", 0.5)
        with pytest.raises(TimeoutError):
            pr.read_bed(f"http://127.0.0.1:{listener.getsockname()[1]}/peaks.bed")
