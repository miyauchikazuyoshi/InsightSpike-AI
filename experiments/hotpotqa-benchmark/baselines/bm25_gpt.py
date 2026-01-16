"""BM25 + GPT baseline for HotpotQA."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from .base import BaseRAG, RAGResult

if TYPE_CHECKING:
    from src.data_loader import HotpotQAExample


class BM25GPTBaseline(BaseRAG):
    """BM25 retrieval + GPT-4o-mini answer generation."""

    name = "BM25 + GPT-4o-mini"

    def __init__(
        self,
        top_k: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.top_k = top_k
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._bm25_cls = None
        self.client = None

    def setup(self, examples: list[HotpotQAExample]) -> None:
        """Validate dependencies."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank_bm25: pip install rank-bm25")
        self._bm25_cls = BM25Okapi

    def _get_llm_client(self):
        """Lazy-load OpenAI client."""
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Please install openai: pip install openai")

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self.client = OpenAI(api_key=api_key)

        return self.client

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_bm25_for_example(
        self, example: HotpotQAExample
    ) -> tuple[list[tuple[str, int, str]], "BM25Okapi"]:
        if self._bm25_cls is None:
            raise RuntimeError("BM25 class not initialized. Call setup() first.")
        corpus = example.get_all_sentences()
        tokenized_corpus = [self._tokenize(text) for _, _, text in corpus]
        return corpus, self._bm25_cls(tokenized_corpus)

    def _retrieve(
        self,
        query: str,
        corpus: list[tuple[str, int, str]],
        bm25: "BM25Okapi",
        top_k: int,
    ) -> list[tuple[str, int, str]]:
        """Retrieve top-k sentences using BM25 (closed-world per example)."""
        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [corpus[i] for i in top_indices]

    def _generate_answer(self, question: str, context: list[str]) -> str:
        """Generate answer using GPT-4o-mini."""
        client = self._get_llm_client()

        context_str = "\n".join(f"- {sent}" for sent in context)
        prompt = f"""Based on the following context, answer the question concisely.

Context:
{context_str}

Question: {question}

Answer (be concise, just give the answer):"""

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content.strip()

    def process(self, example: HotpotQAExample) -> RAGResult:
        """Process a single question."""
        start_time = time.time()

        # Retrieve relevant sentences (closed-world per example)
        corpus, bm25 = self._build_bm25_for_example(example)
        retrieved = self._retrieve(example.question, corpus, bm25, self.top_k)

        # Generate answer
        context_texts = [text for _, _, text in retrieved]
        answer = self._generate_answer(example.question, context_texts)

        # Extract retrieved facts
        retrieved_facts = [(title, sent_id) for title, sent_id, _ in retrieved]

        latency_ms = (time.time() - start_time) * 1000

        return RAGResult(
            answer=answer,
            retrieved_facts=retrieved_facts,
            latency_ms=latency_ms,
            metadata={
                "model": self.model,
                "top_k": self.top_k,
                "retrieved_texts": context_texts,
            },
        )
