import pytest
from insightspike.layer2_memory_manager import Memory
from unittest.mock import patch
from datasets import load_dataset

def test_memory_with_databake(tmp_path):
    # Wikipediaなどから取得した多様な文500件を仮に用意（ここでは例としてダミー文を生成）
    docs = [f"Sample sentence about topic {i}. This is a unique Wikipedia-like fact number {i}." for i in range(500)]
    with patch("insightspike.layer2_memory_manager.Memory.train_index"):
        mem = Memory.build(docs)
    assert len(mem.episodes) == len(docs)
    save_path = tmp_path / "memory_databake.pkl"
    mem.save(save_path)
    loaded = Memory.load(save_path)
    assert len(loaded.episodes) == len(docs)
    assert loaded.episodes[0].text == docs[0]

def test_memory_with_real_wiki(tmp_path):
    # Wikipedia英語コーパスから500件取得
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:500]")
    docs = [item["text"] for item in dataset if item["text"].strip()]
    docs = docs[:500]  # 念のため500件に制限
    with patch("insightspike.layer2_memory_manager.Memory.train_index"):
        mem = Memory.build(docs)
    assert len(mem.episodes) == len(docs)
    save_path = tmp_path / "memory_databake.pkl"
    mem.save(save_path)
    loaded = Memory.load(save_path)
    assert len(loaded.episodes) == len(docs)
    assert loaded.episodes[0].text == docs[0]