import pyranges1 as pr
from pyranges1 import cli


def test_cli_offers_every_reader() -> None:
    readers = {name for name in dir(pr) if name.startswith("read_")}
    assert readers <= set(cli.READERS)
