"""Shared LLM answer generation for HotpotQA v2.

Consolidates the LLM calling logic that was duplicated across v1
adapter, bm25_gpt, and static_graphrag into a single module.
Supports GPT-4o-mini, retry with backoff, and mock/offline mode.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

# .env 自動読み込み（プロジェクトルートの .env を探す）
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[3] / ".env"  # src -> hotpotqa_v2 -> experiments -> root
    if _env_path.is_file():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass  # python-dotenv が無くても動作する


_RETRY_AFTER_RE = re.compile(r"try again in ([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Default prompt template
_PROMPT_TEMPLATE = """Based on the following context, answer the question concisely.

Context:
{context}

Question: {question}

Answer (be concise, just give the answer):"""


class LLMAnswerer:
    """Unified LLM answer generator with retry and mock support.

    Usage::

        answerer = LLMAnswerer(model="gpt-4o-mini")
        answer = answerer.generate(question, context_sentences)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
        retry_max: int = 3,
        retry_wait: float = 10.0,
        retry_backoff: float = 2.0,
        retry_max_wait: float = 120.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_max = max(0, int(retry_max))
        self.retry_wait = max(0.0, float(retry_wait))
        self.retry_backoff = max(1.0, float(retry_backoff))
        self.retry_max_wait = max(0.0, float(retry_max_wait))
        self._client = None

    # ------------------------------------------------------------------ #
    # Mock / offline mode
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_mock_enabled() -> bool:
        """Check if LLM_PROVIDER is set to mock/offline/none."""
        provider = (
            os.getenv("LLM_PROVIDER")
            or os.getenv("INSIGHTSPIKE_LLM_PROVIDER")
            or ""
        ).strip().lower()
        return provider in {"mock", "offline", "none"}

    @staticmethod
    def _mock_answer(question: str, context: list[str]) -> str:
        """Return a simple heuristic answer without calling any LLM."""
        if not context:
            return "Unknown"
        q_tokens = set(_TOKEN_RE.findall(question.lower()))
        best_sentence = context[0]
        best_score = -1
        for sent in context:
            tokens = set(_TOKEN_RE.findall(sent.lower()))
            score = len(q_tokens & tokens)
            if score > best_score:
                best_score = score
                best_sentence = sent
        if best_score <= 0:
            return "Unknown"
        return best_sentence.strip()

    # ------------------------------------------------------------------ #
    # OpenAI client
    # ------------------------------------------------------------------ #

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Please install openai: pip install openai")

            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_API_BASE")  # Groq/Ollama/Together互換
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            kwargs: dict = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------ #
    # Retry helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_status_code(exc: Exception) -> int | None:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        status = getattr(exc, "status", None)
        if isinstance(status, int):
            return status
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if isinstance(status, int):
                return status
        return None

    @staticmethod
    def _extract_retry_after(exc: Exception) -> float | None:
        message = str(exc)
        match = _RETRY_AFTER_RE.search(message)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
        if headers:
            for key in ("retry-after", "Retry-After"):
                value = headers.get(key)
                if value:
                    try:
                        return float(value)
                    except ValueError:
                        return None
        return None

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        status = cls._get_status_code(exc)
        if status in (408, 429) or (status is not None and 500 <= status < 600):
            return True
        message = str(exc).lower()
        markers = (
            "rate limit", "rate_limit", "rpd", "429", "too many requests",
            "connection error", "timeout", "timed out",
            "temporarily unavailable", "service unavailable",
        )
        return any(m in message for m in markers)

    def _wait_seconds(self, exc: Exception, attempt: int) -> float:
        retry_after = self._extract_retry_after(exc)
        if retry_after is not None:
            return max(0.0, retry_after)
        wait_s = self.retry_wait * (self.retry_backoff ** max(attempt - 1, 0))
        if self.retry_max_wait > 0:
            wait_s = min(wait_s, self.retry_max_wait)
        return max(0.0, wait_s)

    # ------------------------------------------------------------------ #
    # Raw LLM call (for CoT / hybrid mode)
    # ------------------------------------------------------------------ #

    def _llm_call_raw(self, prompt: str, max_tokens: int | None = None) -> str:
        """Send a raw prompt to the LLM with retry.  Used by hybrid CoT.

        Parameters
        ----------
        prompt : str
            The complete prompt to send.
        max_tokens : int | None
            Override for max_tokens (defaults to 150 for CoT sentences).
        """
        if self.is_mock_enabled():
            return "The answer is: Unknown."

        client = self._get_client()
        tok = max_tokens or 150

        attempts = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=tok,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                if (
                    self.retry_max <= 0
                    or not self._is_retryable(exc)
                    or attempts >= self.retry_max
                ):
                    raise
                attempts += 1
                wait_s = self._wait_seconds(exc, attempts)
                print(
                    f"\n[warn] LLM error (raw), retry {attempts}/{self.retry_max} "
                    f"in {wait_s:.1f}s: {exc}"
                )
                if wait_s:
                    time.sleep(wait_s)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def generate(
        self,
        question: str,
        context: list[str],
        prompt_template: str | None = None,
    ) -> str:
        """Generate an answer from *question* and *context* sentences.

        Parameters
        ----------
        question : str
            The question to answer.
        context : list[str]
            Sentences forming the context.
        prompt_template : str | None
            Optional custom prompt template with ``{context}`` and
            ``{question}`` placeholders.

        Returns
        -------
        str
            The generated answer string.
        """
        # Mock mode — no LLM call
        if self.is_mock_enabled():
            return self._mock_answer(question, context)

        client = self._get_client()

        template = prompt_template or _PROMPT_TEMPLATE
        context_str = "\n".join(f"- {sent}" for sent in context)
        prompt = template.format(context=context_str, question=question)

        attempts = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                if (
                    self.retry_max <= 0
                    or not self._is_retryable(exc)
                    or attempts >= self.retry_max
                ):
                    raise
                attempts += 1
                wait_s = self._wait_seconds(exc, attempts)
                print(
                    f"\n[warn] LLM error, retry {attempts}/{self.retry_max} "
                    f"in {wait_s:.1f}s: {exc}"
                )
                if wait_s:
                    time.sleep(wait_s)
