from pathlib import Path
from insightspike.loader import load_corpus


def test_load_corpus_file(tmp_path):
    f = tmp_path / 'sample.txt'
    f.write_text('a\nb')
    docs = load_corpus(f)
    assert docs == ['a', 'b']
