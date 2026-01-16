"""Closed-book GPT baseline for HotpotQA (no context)."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from .base import BaseRAG, RAGResult

if TYPE_CHECKING:
    from src.data_loader import HotpotQAExample


class ClosedBookGPTBaseline(BaseRAG):
    """LLM-only baseline that answers without any retrieved context."""

    name = "Closed-book GPT-4o-mini"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

    def setup(self, examples: list[HotpotQAExample]) -> None:
        """No setup needed for closed-book baseline."""
        return None

    def _get_llm_client(self):
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

    def _generate_answer(self, question: str) -> str:
        client = self._get_llm_client()

        prompt = f"""Answer the following question concisely.

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
        start_time = time.time()

        answer = self._generate_answer(example.question)
        latency_ms = (time.time() - start_time) * 1000

        return RAGResult(
            answer=answer,
            retrieved_facts=[],
            latency_ms=latency_ms,
            metadata={
                "model": self.model,
            },
        )
