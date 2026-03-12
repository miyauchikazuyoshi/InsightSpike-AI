"""Wikipedia API-based article retriever for open-world multi-hop QA.

Provides search and text fetching via the MediaWiki API, with:
  - Article search (opensearch API)
  - Full text extraction (extracts API)
  - Rate-limit handling and retry logic
  - Result deduplication across multiple queries

Based on fetch logic from prepare_frames.py.

Usage::

    retriever = WikipediaRetriever(max_results=10)
    articles = retriever.search_and_fetch("James Buchanan president", top_k=5)
    for title, sentences in articles:
        print(f"{title}: {len(sentences)} sentences")
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Sentence splitting (from prepare_frames.py)
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str, max_sentences: int = 150) -> list[str]:
    """Split Wikipedia extract into sentences, filtering noise."""
    if not text:
        return []

    # Remove section headers (== Header ==)
    text = re.sub(r"={2,}[^=]+=+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Simple sentence splitting
    raw = re.split(r"(?<=[.!?])\s+", text)

    sentences: list[str] = []
    for s in raw:
        s = s.strip()
        if len(s) < 10:
            continue
        if len(s) > 500:
            # Split long sentences at semicolons or commas
            parts = re.split(r"[;,]\s+", s)
            for p in parts:
                p = p.strip()
                if len(p) >= 10:
                    sentences.append(p)
        else:
            sentences.append(s)

    return sentences[:max_sentences]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WikiArticle:
    """A retrieved Wikipedia article."""
    title: str
    sentences: list[str]
    source_query: str = ""           # which query found this article
    fetch_latency_ms: float = 0.0


@dataclass
class SearchResult:
    """Result of a Wikipedia search operation."""
    articles: list[WikiArticle]
    queries_used: list[str]
    total_api_calls: int = 0
    total_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Wikipedia Retriever
# ---------------------------------------------------------------------------

class WikipediaRetriever:
    """Search and fetch Wikipedia articles via the MediaWiki API.

    Parameters
    ----------
    max_results : int
        Maximum results per search query (opensearch limit).
    max_sentences : int
        Maximum sentences to extract per article.
    max_retries : int
        Number of retries for failed API calls.
    request_delay : float
        Seconds to wait between API calls (rate limiting).
    user_agent : str
        User-Agent header for API requests.
    """

    def __init__(
        self,
        max_results: int = 10,
        max_sentences: int = 150,
        max_retries: int = 3,
        request_delay: float = 0.1,
        user_agent: str = "InsightSpike-AI/1.0 (research)",
    ):
        self.max_results = max_results
        self.max_sentences = max_sentences
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.user_agent = user_agent
        self._cache: dict[str, WikiArticle | None] = {}

    # ------------------------------------------------------------------ #
    # Search API
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int | None = None) -> list[str]:
        """Search Wikipedia for article titles matching *query*.

        Uses the opensearch API for fast title suggestions.

        Parameters
        ----------
        query : str
            Search query string.
        top_k : int | None
            Maximum number of titles to return (default: self.max_results).

        Returns
        -------
        list[str]
            List of article titles found.
        """
        k = top_k or self.max_results
        api_url = (
            "https://en.wikipedia.org/w/api.php?"
            "action=opensearch&format=json&namespace=0"
            f"&limit={k}"
            f"&search={urllib.parse.quote(query)}"
        )

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    api_url,
                    headers={"User-Agent": self.user_agent},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                # opensearch returns [query, [titles], [descriptions], [urls]]
                if len(data) >= 2 and isinstance(data[1], list):
                    return data[1][:k]
                return []

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    print(f"  WARN: Search failed for '{query}': {e}")
                    return []

        return []

    # ------------------------------------------------------------------ #
    # Fetch article text
    # ------------------------------------------------------------------ #

    def fetch_article(self, title: str) -> WikiArticle | None:
        """Fetch full text of a Wikipedia article by title.

        Uses the MediaWiki extracts API for plain-text content.
        Results are cached to avoid duplicate API calls.

        Parameters
        ----------
        title : str
            Wikipedia article title (e.g. "James Buchanan").

        Returns
        -------
        WikiArticle | None
            Article with sentences, or None if not found.
        """
        # Check cache
        cache_key = title.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        t0 = time.time()

        api_url = (
            "https://en.wikipedia.org/w/api.php?"
            "action=query&format=json&prop=extracts&explaintext=1"
            f"&titles={urllib.parse.quote(title)}"
        )

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    api_url,
                    headers={"User-Agent": self.user_agent},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    if page_id == "-1":
                        # Page not found
                        self._cache[cache_key] = None
                        return None

                    page_title = page_data.get("title", title)
                    extract = page_data.get("extract", "")
                    sentences = _split_into_sentences(
                        extract, self.max_sentences
                    )

                    if not sentences:
                        self._cache[cache_key] = None
                        return None

                    article = WikiArticle(
                        title=page_title,
                        sentences=sentences,
                        fetch_latency_ms=(time.time() - t0) * 1000,
                    )
                    self._cache[cache_key] = article
                    return article

                self._cache[cache_key] = None
                return None

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    print(f"  WARN: Fetch failed for '{title}': {e}")
                    self._cache[cache_key] = None
                    return None

        self._cache[cache_key] = None
        return None

    # ------------------------------------------------------------------ #
    # Combined search + fetch
    # ------------------------------------------------------------------ #

    def search_and_fetch(
        self,
        query: str,
        top_k: int = 5,
        exclude_titles: set[str] | None = None,
    ) -> SearchResult:
        """Search Wikipedia and fetch article texts.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int
            Number of articles to retrieve.
        exclude_titles : set[str] | None
            Titles to exclude (case-insensitive).

        Returns
        -------
        SearchResult
            Retrieved articles with metadata.
        """
        t0 = time.time()
        exclude = {t.lower() for t in (exclude_titles or set())}

        # Search for titles
        titles = self.search(query, top_k=top_k + len(exclude))
        api_calls = 1

        # Fetch each article
        articles: list[WikiArticle] = []
        for title in titles:
            if title.lower() in exclude:
                continue
            if len(articles) >= top_k:
                break

            time.sleep(self.request_delay)
            article = self.fetch_article(title)
            api_calls += 1

            if article is not None:
                article.source_query = query
                articles.append(article)

        return SearchResult(
            articles=articles,
            queries_used=[query],
            total_api_calls=api_calls,
            total_latency_ms=(time.time() - t0) * 1000,
        )

    def multi_query_search(
        self,
        queries: list[str],
        top_k_per_query: int = 3,
        max_total: int = 10,
        exclude_titles: set[str] | None = None,
    ) -> SearchResult:
        """Search with multiple queries and merge results.

        Deduplicates by title (case-insensitive).

        Parameters
        ----------
        queries : list[str]
            List of search queries.
        top_k_per_query : int
            Articles per query.
        max_total : int
            Maximum total articles to return.
        exclude_titles : set[str] | None
            Titles to exclude.

        Returns
        -------
        SearchResult
            Merged articles from all queries.
        """
        t0 = time.time()
        seen_titles: set[str] = {t.lower() for t in (exclude_titles or set())}
        all_articles: list[WikiArticle] = []
        all_queries: list[str] = []
        total_api_calls = 0

        for query in queries:
            if len(all_articles) >= max_total:
                break

            result = self.search_and_fetch(
                query,
                top_k=top_k_per_query,
                exclude_titles=seen_titles,
            )
            total_api_calls += result.total_api_calls

            for article in result.articles:
                if article.title.lower() not in seen_titles:
                    seen_titles.add(article.title.lower())
                    all_articles.append(article)
                    if len(all_articles) >= max_total:
                        break

            all_queries.append(query)

        return SearchResult(
            articles=all_articles,
            queries_used=all_queries,
            total_api_calls=total_api_calls,
            total_latency_ms=(time.time() - t0) * 1000,
        )

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def clear_cache(self) -> None:
        """Clear the article cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached articles."""
        return len(self._cache)
