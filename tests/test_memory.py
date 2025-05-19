import pytest
from insightspike.layer2_memory_manager import Memory
from unittest.mock import patch

def test_memory_build_and_save(tmp_path):
    # ファイルから10文を読み込む
    with open("data/raw/test_sentences.txt") as f:
        docs = [line.strip() for line in f if line.strip()]
    with patch("insightspike.layer2_memory_manager.Memory.train_index"):
        mem = Memory.build(docs)
    assert len(mem.episodes) == len(docs)
    save_path = tmp_path / "memory_test.pkl"
    mem.save(save_path)
    loaded = Memory.load(save_path)
    assert loaded.episodes[0].text == docs[0]