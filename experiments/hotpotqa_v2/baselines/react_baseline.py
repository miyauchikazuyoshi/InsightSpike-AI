"""ReAct (Reason + Act) baseline for HotPotQA.

Based on Yao et al., ICLR 2023:
"ReAct: Synergizing Reasoning and Acting in Language Models"

Adapted for HotPotQA distractor setting (closed-world, 10 paragraphs).
Uses BM25 search over the given paragraphs instead of Wikipedia API.
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


# --------------------------------------------------------------------------- #
# Few-shot examples for ReAct on HotPotQA
# --------------------------------------------------------------------------- #

_REACT_FEW_SHOT = """\
Solve a multi-hop question by interleaving Thought, Action, and Observation steps.

Available actions:
- search[query]: Search the given paragraphs for information related to the query. Returns the most relevant sentences.
- finish[answer]: Return the final answer. Be concise.

Here are some examples:

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin and find their types of work.
Action 1: search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn was a mathematician. Now I need to find what Leonid Levin is known for.
Action 2: search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
Thought 3: Both are mathematicians, so they are known for the same type of work.
Action 3: finish[yes]

Question: The creator of "Wallace and Gromit" also created what other clay animation series?
Thought 1: I need to find who created Wallace and Gromit.
Action 1: search[Wallace and Gromit creator]
Observation 1: Wallace and Gromit is a British clay animation series created by Nick Park.
Thought 2: Nick Park created Wallace and Gromit. Now I need to find what other clay animation series Nick Park created.
Action 2: search[Nick Park clay animation]
Observation 2: Nick Park has also created the animated series Shaun the Sheep and the feature film Chicken Run.
Thought 3: Nick Park also created Shaun the Sheep.
Action 3: finish[Shaun the Sheep]

Now solve the following question:

"""

# Parsing patterns
_ACTION_RE = re.compile(r"Action\s*\d*\s*:\s*(search|finish)\[(.+?)\]", re.IGNORECASE)
_THOUGHT_RE = re.compile(r"(Thought\s*\d*\s*:.*?)(?=Action|$)", re.DOTALL | re.IGNORECASE)


class ReActBaseline(BaseRAG):
    """ReAct: Reason + Act with BM25 search.

    Algorithm (adapted for HotPotQA distractor setting):
      1. Build BM25 index from example's 10 paragraphs
      2. Few-shot prompt with Thought/Action/Observation format
      3. Loop (max_steps iterations):
         a. Generate Thought + Action
         b. Parse action:
            - search[query]: BM25 search → Observation
            - finish[answer]: return answer
         c. Append Observation to trajectory
      4. If max_steps reached: force answer extraction
    """

    name = "react"

    def __init__(
        self,
        top_k: int = 3,
        max_steps: int = 7,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.top_k = top_k
        self.max_steps = max_steps
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
            return "Thought 1: I need to search for the answer.\nAction 1: finish[Unknown]"

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
                print(f"[react] LLM retry {attempt+1}/3 in {wait:.0f}s: {exc}")
                time.sleep(wait)
        return ""

    # ------------------------------------------------------------------ #
    # BM25 search action
    # ------------------------------------------------------------------ #

    def _search_action(
        self,
        query: str,
        corpus: list[tuple[str, int, str]],
        bm25: object,
    ) -> tuple[str, list[tuple[str, int]]]:
        """Execute a search action and return observation text + facts."""
        facts = self._retriever.retrieve(query, corpus, bm25, self.top_k)
        if not facts:
            return "No relevant information found.", []

        obs_parts = []
        fact_keys = []
        for f in facts:
            obs_parts.append(f"[{f.title}] {f.text}")
            fact_keys.append((f.title, f.sent_id))

        return "\n".join(obs_parts), fact_keys

    # ------------------------------------------------------------------ #
    # Main process
    # ------------------------------------------------------------------ #

    def process(self, example: HotpotQAExample) -> RAGResult:
        t0 = time.time()

        # 1. Build BM25 index
        corpus, bm25 = self._retriever.build_index(example)

        # 2. Initialize trajectory
        trajectory = _REACT_FEW_SHOT + f"Question: {example.question}\n"
        all_facts: dict[tuple[str, int], bool] = {}
        answer = None
        steps = 0

        # 3. ReAct loop
        for step in range(1, self.max_steps + 1):
            steps = step

            # Generate Thought + Action
            response = self._llm_call(trajectory)
            trajectory += response + "\n"

            # Parse action
            action_match = _ACTION_RE.search(response)
            if not action_match:
                # No action found — try to continue
                # Append a hint to generate an action
                trajectory += f"Action {step}: "
                action_resp = self._llm_call(trajectory)
                trajectory += action_resp + "\n"
                action_match = _ACTION_RE.search(action_resp)

            if action_match:
                action_type = action_match.group(1).lower()
                action_arg = action_match.group(2).strip()

                if action_type == "finish":
                    answer = action_arg
                    break
                elif action_type == "search":
                    obs_text, fact_keys = self._search_action(
                        action_arg, corpus, bm25
                    )
                    for fk in fact_keys:
                        all_facts[fk] = True
                    trajectory += f"Observation {step}: {obs_text}\n"
            else:
                # Still no action — add generic observation
                trajectory += f"Observation {step}: Could not parse action. Please use search[query] or finish[answer].\n"

        # 4. Force answer if not found
        if answer is None:
            # Last resort: ask for direct answer
            force_prompt = (
                trajectory
                + "\nBased on the above reasoning, provide the final answer.\n"
                + f"Action {steps+1}: finish["
            )
            force_resp = self._llm_call(force_prompt)
            # Try to extract from finish[...]
            finish_match = re.search(r"^([^\]]+)", force_resp)
            if finish_match:
                answer = finish_match.group(1).strip()
            else:
                answer = force_resp.strip()

        latency_ms = (time.time() - t0) * 1000

        return RAGResult(
            answer=answer or "Unknown",
            retrieved_facts=list(all_facts.keys()),
            latency_ms=latency_ms,
            metadata={
                "steps": steps,
                "facts_collected": len(all_facts),
                "trajectory": trajectory,
            },
        )
