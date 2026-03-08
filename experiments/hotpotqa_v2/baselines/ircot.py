"""IRCoT (Interleaving Retrieval with Chain-of-Thought) baseline.

Based on Trivedi et al., ACL 2023:
"Interleaving Retrieval with Chain-of-Thought Reasoning for
Knowledge-Intensive Multi-Step Questions"

Adapted for HotPotQA distractor setting (closed-world, 10 paragraphs).
"""

from __future__ import annotations

import os
import re
import time
from typing import TYPE_CHECKING

from experiments.hotpotqa_v2.baselines.base import BaseRAG, RAGResult
from experiments.hotpotqa_v2.src.retriever import BM25Retriever

if TYPE_CHECKING:
    from experiments.hotpotqa_v2.src.data_loader import HotpotQAExample

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #

_COT_STEP_PROMPT = """\
Given the following context paragraphs and question, continue the chain-of-thought reasoning.
Generate ONLY the next single reasoning sentence. Do NOT repeat previous reasoning.
When you have enough information to answer, write: "So the answer is: <answer>."

Context:
{context}

Question: {question}

Reasoning so far:
{cot_so_far}

Next reasoning sentence:"""

_EXTRACT_ANSWER_PROMPT = """\
Based on the following reasoning chain, extract the final concise answer.

Question: {question}

Reasoning:
{cot}

Answer (be concise, just give the answer):"""

# Regex to capture "answer is: ..." pattern
_ANSWER_RE = re.compile(
    r"(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)", re.IGNORECASE
)


class IRCoTBaseline(BaseRAG):
    """IRCoT: Interleaving Retrieval with Chain-of-Thought.

    Algorithm (adapted for HotPotQA distractor setting):
      1. Build BM25 index from example's 10 paragraphs
      2. Initial retrieval with the question → seed paragraphs
      3. Loop (max_steps iterations):
         a. Generate next CoT sentence given question + collected context + CoT
         b. If CoT contains "answer is:" → break
         c. Use CoT sentence as new BM25 query → add results to context
      4. Extract final answer from CoT
    """

    name = "ircot"

    def __init__(
        self,
        top_k: int = 3,
        max_steps: int = 8,
        max_sentences: int = 15,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 150,
    ):
        self.top_k = top_k
        self.max_steps = max_steps
        self.max_sentences = max_sentences
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._retriever = BM25Retriever()
        self._client = None

    def setup(self, examples: list[HotpotQAExample]) -> None:
        pass  # per-example index, no global setup needed

    # ------------------------------------------------------------------ #
    # LLM helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_mock() -> bool:
        provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        return provider in {"mock", "offline", "none"}

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _llm_call(self, prompt: str) -> str:
        """Single LLM call with retry."""
        if self._is_mock():
            # Mock: return a generic reasoning sentence
            return "The answer is: Unknown."

        client = self._get_client()
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                msg = str(exc).lower()
                retryable = any(
                    m in msg
                    for m in ("rate limit", "429", "timeout", "service unavailable")
                )
                if not retryable or attempt >= 2:
                    raise
                wait = 10.0 * (2 ** attempt)
                print(f"[ircot] LLM retry {attempt+1}/3 in {wait:.0f}s: {exc}")
                time.sleep(wait)
        return ""

    # ------------------------------------------------------------------ #
    # Main process
    # ------------------------------------------------------------------ #

    def process(self, example: HotpotQAExample) -> RAGResult:
        t0 = time.time()

        # 1. Build BM25 index
        corpus, bm25 = self._retriever.build_index(example)

        # 2. Initial retrieval with the question
        initial_facts = self._retriever.retrieve(
            example.question, corpus, bm25, self.top_k
        )

        # Collected sentences: {(title, sent_id): text}
        collected: dict[tuple[str, int], str] = {}
        for fact in initial_facts:
            collected[(fact.title, fact.sent_id)] = fact.text

        # 3. IRCoT loop
        cot_sentences: list[str] = []
        answer = None

        for step in range(self.max_steps):
            # Build context string from collected sentences
            context_str = "\n".join(
                f"[{title}] {text}" for (title, _), text in collected.items()
            )
            cot_so_far = " ".join(cot_sentences) if cot_sentences else "(none yet)"

            # Generate next CoT sentence
            prompt = _COT_STEP_PROMPT.format(
                context=context_str,
                question=example.question,
                cot_so_far=cot_so_far,
            )
            next_sentence = self._llm_call(prompt)
            cot_sentences.append(next_sentence)

            # Check if answer found
            match = _ANSWER_RE.search(next_sentence)
            if match:
                answer = match.group(1).strip().rstrip(".")
                break

            # Use the generated sentence as a new query for BM25
            if len(collected) < self.max_sentences:
                new_facts = self._retriever.retrieve(
                    next_sentence, corpus, bm25, self.top_k
                )
                for fact in new_facts:
                    key = (fact.title, fact.sent_id)
                    if key not in collected and len(collected) < self.max_sentences:
                        collected[key] = fact.text

        # 4. Extract answer if not found in CoT
        if answer is None:
            full_cot = " ".join(cot_sentences)
            # Try regex first
            match = _ANSWER_RE.search(full_cot)
            if match:
                answer = match.group(1).strip().rstrip(".")
            else:
                # Fallback: ask LLM to extract answer
                extract_prompt = _EXTRACT_ANSWER_PROMPT.format(
                    question=example.question, cot=full_cot
                )
                answer = self._llm_call(extract_prompt)

        latency_ms = (time.time() - t0) * 1000

        return RAGResult(
            answer=answer or "Unknown",
            retrieved_facts=list(collected.keys()),
            latency_ms=latency_ms,
            metadata={
                "cot_steps": len(cot_sentences),
                "sentences_collected": len(collected),
                "cot": " ".join(cot_sentences),
            },
        )
