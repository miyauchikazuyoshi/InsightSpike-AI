"""BM25 retrieval module for HotpotQA v2.

Extracted from v1 ``hotpotqa_adapter.py`` into an independent module so
that both the geDIG adapter and baselines can share the same retrieval
logic without code duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.hotpotqa_v2.src.data_loader import HotpotQAExample


@dataclass
class RetrievedFact:
    """Single retrieved sentence with its metadata and BM25 score."""

    title: str
    sent_id: int
    text: str
    score: float


class BM25Retriever:
    """BM25-based closed-world retriever for HotpotQA examples.

    Each HotpotQA example is its own closed-world corpus (10 context
    paragraphs).  The retriever builds a BM25 index per example and
    returns top-k sentences by BM25 score.

    Usage::

        retriever = BM25Retriever()
        corpus, index = retriever.build_index(example)
        results = retriever.retrieve("who directed...", corpus, index, top_k=5)
    """

    def __init__(self):
        self._bm25_cls = None

    def _ensure_bm25(self):
        """Lazy-load BM25Okapi class."""
        if self._bm25_cls is None:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError:
                raise ImportError("Please install rank_bm25: pip install rank-bm25")
            self._bm25_cls = BM25Okapi

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def build_index(
        self, example: HotpotQAExample
    ) -> tuple[list[tuple[str, int, str]], object]:
        """Build a BM25 index for a single example.

        Parameters
        ----------
        example : HotpotQAExample
            The example whose context provides the corpus.

        Returns
        -------
        corpus : list of (title, sent_id, text)
        bm25 : BM25Okapi instance
        """
        self._ensure_bm25()
        corpus = example.get_all_sentences()
        tokenized_corpus = [self._tokenize(text) for _, _, text in corpus]
        return corpus, self._bm25_cls(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        corpus: list[tuple[str, int, str]],
        bm25: object,
        top_k: int = 5,
    ) -> list[RetrievedFact]:
        """Retrieve top-k sentences by BM25 score.

        Parameters
        ----------
        query : str
            The query string.
        corpus : list of (title, sent_id, text)
            The corpus built by ``build_index``.
        bm25 : BM25Okapi
            The BM25 index built by ``build_index``.
        top_k : int
            Number of top results to return.

        Returns
        -------
        list of RetrievedFact
        """
        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            RetrievedFact(
                title=corpus[i][0],
                sent_id=corpus[i][1],
                text=corpus[i][2],
                score=float(scores[i]),
            )
            for i in top_indices
        ]

    def retrieve_multi_query(
        self,
        queries: list[str],
        corpus: list[tuple[str, int, str]],
        bm25: object,
        top_k: int = 5,
    ) -> list[RetrievedFact]:
        """Retrieve and merge results from multiple queries.

        De-duplicates by (title, sent_id), keeping the highest score.
        """
        merged: dict[tuple[str, int], RetrievedFact] = {}
        for query in queries:
            for fact in self.retrieve(query, corpus, bm25, top_k):
                key = (fact.title, fact.sent_id)
                if key not in merged or fact.score > merged[key].score:
                    merged[key] = fact
        return sorted(merged.values(), key=lambda f: -f.score)
