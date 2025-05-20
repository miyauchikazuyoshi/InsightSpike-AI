import pytest
from insightspike.layer2_memory_manager import Memory
from unittest.mock import patch
from datasets import load_dataset

def test_memory_with_test_sentences(tmp_path):
    with open("data/raw/test_sentences.txt") as f:
        docs = [line.strip() for line in f if line.strip()]
    mem = Memory.build(docs)
    assert len(mem.episodes) == len(docs)
    save_path = tmp_path / "memory_test.pkl"
    mem.save(save_path)
    loaded = Memory.load(save_path)
    assert len(loaded.episodes) == len(docs)
    assert loaded.episodes[0].text == docs[0]