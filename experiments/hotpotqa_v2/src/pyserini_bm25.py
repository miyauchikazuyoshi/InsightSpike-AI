"""Pyserini (Lucene) BM25 wrapper with rank_bm25-compatible interface.

Provides a drop-in replacement for rank_bm25.BM25Okapi that uses
Pyserini's Lucene-based BM25 scoring — the same engine used in the
BRIGHT benchmark paper (ICLR 2025).

Usage::

    from pyserini_bm25 import build_pyserini_index, PyseriniBM25

    # Step 1: Build Lucene index (one-time)
    build_pyserini_index("data/bright/biology_docs.jsonl",
                         "data/bright/biology_lucene_index")

    # Step 2: Use as drop-in replacement
    bm25 = PyseriniBM25("data/bright/biology_lucene_index", docs)
    scores = bm25.get_scores(query_tokens)  # same interface as BM25Okapi
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _ensure_java():
    """Set JAVA_HOME to Java 21 (required for Pyserini 0.25.0 Anserini JAR).

    Pyserini 0.25.0 Anserini JAR is compiled with Java 21 (class file v65).
    It also needs --add-modules=jdk.incubator.vector (set by pyserini/setup.py).
    """
    java21_path = "/usr/local/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
    if os.path.exists(java21_path):
        os.environ["JAVA_HOME"] = java21_path
        os.environ["PATH"] = f"/usr/local/opt/openjdk@21/bin:{os.environ.get('PATH', '')}"


def build_pyserini_index(
    docs_path: str,
    index_path: str,
    threads: int = 4,
    storeRaw: bool = True,
) -> None:
    """Build a Lucene index from a BRIGHT domain docs JSONL file.

    Converts the BRIGHT format to Pyserini's expected JSONL format,
    then invokes Pyserini's indexer.

    Parameters
    ----------
    docs_path : str
        Path to domain_docs.jsonl (BRIGHT format: {"id": ..., "content": ...}).
    index_path : str
        Output directory for the Lucene index.
    threads : int
        Number of indexing threads.
    storeRaw : bool
        Whether to store raw document text in the index.
    """
    _ensure_java()  # Must be before any pyserini import (sets JAVA_HOME for JVM)

    index_dir = Path(index_path)
    if index_dir.exists() and any(index_dir.iterdir()):
        print(f"  Lucene index already exists at {index_path}, skipping build.")
        return

    index_dir.mkdir(parents=True, exist_ok=True)

    # Convert to Pyserini JSONL format in a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        pyserini_jsonl = os.path.join(tmp_dir, "docs.jsonl")
        n_docs = 0
        with open(docs_path) as fin, open(pyserini_jsonl, "w") as fout:
            for line in fin:
                doc = json.loads(line)
                # Pyserini expects: {"id": str, "contents": str}
                pyserini_doc = {
                    "id": str(doc["id"]),
                    "contents": doc["content"],
                }
                fout.write(json.dumps(pyserini_doc) + "\n")
                n_docs += 1

        print(f"  Converted {n_docs} docs to Pyserini format.")
        print(f"  Building Lucene index at {index_path}...")

        # Use Pyserini's command-line indexer via subprocess
        # (more reliable than Python API for large collections)
        import subprocess
        import sys

        # Find the pyserini JAR path
        import pyserini
        pyserini_dir = os.path.dirname(pyserini.__file__)
        jar_path = os.path.join(pyserini_dir, "resources", "jars")

        # Use the Java indexer directly
        cmd = [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", tmp_dir,
            "--index", index_path,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads),
            "--storeRaw",
        ]

        # Ensure JAVA_HOME is set for the subprocess
        env = os.environ.copy()
        # Use Java 21 for Pyserini 0.25.0 (Anserini JAR compiled with Java 21)
        java21_path = "/usr/local/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
        if os.path.exists(java21_path):
            env["JAVA_HOME"] = java21_path
            env["PATH"] = f"/usr/local/opt/openjdk@21/bin:{env.get('PATH', '')}"

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env,
        )

        if result.returncode != 0:
            print(f"  Indexer stderr: {result.stderr[:500]}")
            raise RuntimeError(f"Lucene indexing failed (exit {result.returncode})")

        print(f"  Lucene index built: {n_docs} docs.")


class PyseriniBM25:
    """Pyserini Lucene BM25 with rank_bm25.BM25Okapi-compatible interface.

    Parameters
    ----------
    index_path : str
        Path to pre-built Lucene index.
    docs : list[dict]
        Documents list (aligned with original JSONL order).
        Used to map doc IDs back to array indices.
    k1 : float
        BM25 k1 parameter. BRIGHT paper uses 0.9.
    b : float
        BM25 b parameter. BRIGHT paper uses 0.4.
    hits_k : int
        Max number of hits to retrieve per query.
    """

    def __init__(
        self,
        index_path: str,
        docs: list[dict],
        k1: float = 0.9,
        b: float = 0.4,
        hits_k: int = 1000,
    ):
        _ensure_java()
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=k1, b=b)

        # Build reverse mapping: doc_id -> index in docs array
        self.doc_id_to_idx: dict[str, int] = {}
        for i, doc in enumerate(docs):
            self.doc_id_to_idx[str(doc["id"])] = i

        self.n_docs = len(docs)
        self.hits_k = hits_k

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Score all documents for a query (rank_bm25 compatible).

        Parameters
        ----------
        query_tokens : list[str]
            Tokenized query (will be joined with spaces for Lucene).

        Returns
        -------
        np.ndarray
            Array of BM25 scores, length == n_docs.
            Non-retrieved docs get score 0.0.
        """
        scores = np.zeros(self.n_docs, dtype=np.float64)

        if not query_tokens:
            return scores

        # Lucene expects a string query, not tokens
        query_str = " ".join(query_tokens)

        try:
            hits = self.searcher.search(query_str, k=self.hits_k)
        except Exception as e:
            print(f"  [PyseriniBM25] Search error: {e}")
            return scores

        for hit in hits:
            idx = self.doc_id_to_idx.get(hit.docid)
            if idx is not None:
                scores[idx] = hit.score

        return scores
