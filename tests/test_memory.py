import pytest
from insightspike.layer2_memory_manager import Memory

def test_memory_build_and_save(tmp_path):
    docs = ["test sentence 1", "test sentence 2"]
    mem = Memory.build(docs)
    assert len(mem.episodes) == 2
    save_path = tmp_path / "memory_test.pkl"
    mem.save(save_path)
    loaded = Memory.load(save_path)
    assert loaded.episodes[0].text == "test sentence 1"