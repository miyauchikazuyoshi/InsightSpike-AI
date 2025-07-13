from pathlib import Path

from insightspike import utils


def test_clean_text():
    assert utils.clean_text("a  b\n c") == "a b c"


def test_iter_text(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("x")
    files = list(utils.iter_text(tmp_path))
    assert files == [p]
