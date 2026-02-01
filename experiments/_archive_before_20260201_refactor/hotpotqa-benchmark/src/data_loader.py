"""HotpotQA data loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class HotpotQAExample:
    """Single HotpotQA example."""

    id: str
    question: str
    answer: str
    supporting_facts_titles: list[str]
    supporting_facts_sent_ids: list[int]
    context_titles: list[str]
    context_sentences: list[list[str]]
    question_type: str  # "comparison" or "bridge"
    level: str  # "easy", "medium", "hard"

    @property
    def supporting_facts(self) -> list[tuple[str, int]]:
        """Get supporting facts as list of (title, sent_id) tuples."""
        return list(zip(self.supporting_facts_titles, self.supporting_facts_sent_ids))

    @property
    def context(self) -> list[tuple[str, list[str]]]:
        """Get context as list of (title, sentences) tuples."""
        return list(zip(self.context_titles, self.context_sentences))

    def get_all_sentences(self) -> list[tuple[str, int, str]]:
        """Get all sentences with their (title, sent_id, text)."""
        result = []
        for title, sentences in self.context:
            for sent_id, sent_text in enumerate(sentences):
                result.append((title, sent_id, sent_text))
        return result

    def get_supporting_sentences(self) -> list[str]:
        """Get the actual supporting fact sentences."""
        title_to_sents = dict(self.context)
        result = []
        for title, sent_id in self.supporting_facts:
            if title in title_to_sents and sent_id < len(title_to_sents[title]):
                result.append(title_to_sents[title][sent_id])
        return result


class HotpotQALoader:
    """Load HotpotQA dataset from JSONL files."""

    def __init__(self, data_path: Path | str):
        self.data_path = Path(data_path)
        self._examples: list[HotpotQAExample] | None = None

    def load(self) -> list[HotpotQAExample]:
        """Load all examples from the JSONL file."""
        if self._examples is not None:
            return self._examples

        self._examples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                example = HotpotQAExample(
                    id=data["id"],
                    question=data["question"],
                    answer=data["answer"],
                    supporting_facts_titles=data["supporting_facts"]["title"],
                    supporting_facts_sent_ids=data["supporting_facts"]["sent_id"],
                    context_titles=data["context"]["title"],
                    context_sentences=data["context"]["sentences"],
                    question_type=data["type"],
                    level=data["level"],
                )
                self._examples.append(example)

        return self._examples

    def __iter__(self) -> Iterator[HotpotQAExample]:
        return iter(self.load())

    def __len__(self) -> int:
        return len(self.load())

    def __getitem__(self, idx: int) -> HotpotQAExample:
        return self.load()[idx]

    def filter_by_type(self, question_type: str) -> list[HotpotQAExample]:
        """Filter examples by question type (comparison/bridge)."""
        return [ex for ex in self.load() if ex.question_type == question_type]

    def filter_by_level(self, level: str) -> list[HotpotQAExample]:
        """Filter examples by difficulty level (easy/medium/hard)."""
        return [ex for ex in self.load() if ex.level == level]
