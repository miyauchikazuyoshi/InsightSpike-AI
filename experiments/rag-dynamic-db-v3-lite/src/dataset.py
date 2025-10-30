"""Dataset loader for RAG v3-lite experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Sequence, Dict, Any
import json


@dataclass
class DocumentExample:
    doc_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class QueryExample:
    query: str
    ground_truth: str
    documents: Sequence[DocumentExample]


def load_dataset(path: Path, limit: int | None = None) -> List[QueryExample]:
    """Load JSONL dataset."""

    examples: List[QueryExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            docs: List[DocumentExample] = []
            if "episodes" in payload:
                episodes = payload.get("episodes", [])
                for idx, episode in enumerate(episodes):
                    episode_id = episode.get("id", f"ep_{len(examples)}_{idx}")
                    text = episode.get("text")
                    if not text:
                        context = episode.get("context", "")
                        operation = episode.get("operation", "")
                        affordance = episode.get("affordance", "")
                        salience = episode.get("salience", "")
                        outcome = episode.get("outcome", "")
                        goal = episode.get("goal", "")
                        text = (
                            f"Context: {context}. Operation: {operation}. "
                            f"Affordance: {affordance}. Salience: {salience}. "
                            f"Outcome: {outcome}. Goal: {goal}."
                        )
                    metadata = {
                        "context": str(episode.get("context", "")),
                        "operation": str(episode.get("operation", "")),
                        "affordance": str(episode.get("affordance", "")),
                        "salience": str(episode.get("salience", "")),
                        "outcome": str(episode.get("outcome", "")),
                        "goal": str(episode.get("goal", "")),
                        "domain": str(episode.get("domain", payload.get("domain", ""))),
                    }
                    role = episode.get("role")
                    if role:
                        metadata["role"] = str(role)
                    else:
                        metadata["role"] = "support" if episode.get("is_support", False) else "distractor"
                    if "type" in episode:
                        metadata["type"] = str(episode["type"])
                    docs.append(DocumentExample(episode_id, text, metadata))
            else:
                docs = [
                    DocumentExample(
                        doc["id"],
                        doc["text"],
                        {k: str(v) for k, v in doc.get("metadata", {}).items()},
                    )
                    for doc in payload.get("documents", [])
                ]
            examples.append(
                QueryExample(
                    query=payload["query"],
                    ground_truth=payload.get("ground_truth", ""),
                    documents=docs,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples
