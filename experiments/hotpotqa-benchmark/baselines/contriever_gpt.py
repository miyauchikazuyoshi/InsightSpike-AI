"""Contriever + GPT baseline for HotpotQA."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np

from .base import BaseRAG, RAGResult

if TYPE_CHECKING:
    from src.data_loader import HotpotQAExample


class ContrieverGPTBaseline(BaseRAG):
    """Dense retrieval (Contriever) + GPT-4o-mini answer generation."""

    name = "Contriever + GPT-4o-mini"

    def __init__(
        self,
        retriever_model: str = "facebook/contriever",
        top_k: int = 5,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
        device: str = "auto",
        max_length: int = 256,
        batch_size: int = 16,
    ):
        self.retriever_model = retriever_model
        self.top_k = top_k
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = None
        self.model = None
        self.client = None
        self._torch = None

    def setup(self, examples: list[HotpotQAExample]) -> None:
        """Load the Contriever model."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers and torch: pip install transformers torch")

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.retriever_model)
        try:
            self.model = AutoModel.from_pretrained(
                self.retriever_model, use_safetensors=True
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load retriever weights. Use a model with safetensors "
                "or upgrade torch to >=2.6."
            ) from exc
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch

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

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.tokenizer is None or self.model is None or self._torch is None:
            raise RuntimeError("Contriever model not initialized. Call setup() first.")

        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                outputs = self.model(**inputs)
                pooled = outputs.last_hidden_state.mean(dim=1)
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
        if not embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(embeddings)

    def _retrieve(self, query: str, example: HotpotQAExample) -> list[tuple[str, int, str]]:
        corpus = example.get_all_sentences()
        if not corpus:
            return []
        texts = [text for _, _, text in corpus]
        query_emb = self._encode_texts([query])
        doc_embs = self._encode_texts(texts)
        if query_emb.size == 0 or doc_embs.size == 0:
            return []
        scores = doc_embs @ query_emb[0]
        top_indices = np.argsort(-scores)[: self.top_k]
        return [corpus[i] for i in top_indices]

    def _generate_answer(self, question: str, context: list[str]) -> str:
        client = self._get_llm_client()

        context_str = "\n".join(f"- {sent}" for sent in context)
        prompt = f"""Based on the following context, answer the question concisely.

Context:
{context_str}

Question: {question}

Answer (be concise, just give the answer):"""

        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content.strip()

    def process(self, example: HotpotQAExample) -> RAGResult:
        start_time = time.time()

        retrieved = self._retrieve(example.question, example)
        context_texts = [text for _, _, text in retrieved]
        answer = self._generate_answer(example.question, context_texts)

        retrieved_facts = [(title, sent_id) for title, sent_id, _ in retrieved]
        latency_ms = (time.time() - start_time) * 1000

        return RAGResult(
            answer=answer,
            retrieved_facts=retrieved_facts,
            latency_ms=latency_ms,
            metadata={
                "retriever_model": self.retriever_model,
                "llm_model": self.llm_model,
                "top_k": self.top_k,
            },
        )
