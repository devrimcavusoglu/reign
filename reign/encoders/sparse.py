"""
Sparse / classical retrieval baselines.

These exist so the doc-to-doc setting has a non-neural floor: BM25 and TF-IDF
are still strong on long inputs because they don't suffer the chunking
information loss REIGN is engineered around. Skipping them makes REIGN's
neural-only lift look misleadingly large.

Both retrievers implement `BaseRetriever`. They share a simple regex
tokenizer (`\\w+` lowercase) — adequate for the English-Wikipedia corpora
in scope; swap in via `tokenizer=` for domain-specific runs.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np

_DEFAULT_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _default_tokenize(text: str) -> list[str]:
    return _DEFAULT_TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over full documents.

    Uses the ``bm25s`` library (numpy-vectorized BM25 over a sparse
    term-document matrix) with k1=1.5, b=0.75 — the BEIR-conventional
    parameters. The previous implementation (``rank_bm25.BM25Okapi``) was a
    pure-Python loop over (query terms × corpus docs) and ran for >19 hours
    on this corpus before being killed; ``bm25s`` does the same scoring in
    minutes.

    For very large corpora (>1M docs) consider Pyserini / Anserini instead.
    """

    name = "bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = _default_tokenize,
    ):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self._bm25 = None
        self._corpus_size = 0

    def index(self, corpus: Sequence[str]) -> None:
        import bm25s

        tokenized = [self.tokenizer(doc) for doc in corpus]
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b)
        self._bm25.index(tokenized)
        self._corpus_size = len(tokenized)

    def retrieve(self, queries: Sequence[str], top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if self._bm25 is None:
            raise RuntimeError("Call index(corpus) before retrieve().")
        top_k = min(top_k, self._corpus_size)
        tokenized_q = [self.tokenizer(q) for q in queries]
        results, scores = self._bm25.retrieve(tokenized_q, k=top_k)
        return np.asarray(results, dtype=np.int64), np.asarray(scores, dtype=np.float32)


class TfidfRetriever:
    """TF-IDF + cosine similarity over full documents.

    Uses sklearn's `TfidfVectorizer` with sublinear TF, lowercase, and the
    same `\\w+` tokenizer as BM25 for direct comparability.
    """

    name = "tfidf"

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self._vectorizer = None
        self._corpus_matrix = None  # (n_docs, vocab) sparse, L2-normalised

    def index(self, corpus: Sequence[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self._vectorizer = TfidfVectorizer(
            tokenizer=_default_tokenize,
            lowercase=False,  # tokenizer already lowercases
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            token_pattern=None,
        )
        matrix = self._vectorizer.fit_transform(corpus)
        self._corpus_matrix = normalize(matrix, norm="l2", copy=False)

    def retrieve(self, queries: Sequence[str], top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.preprocessing import normalize

        if self._vectorizer is None or self._corpus_matrix is None:
            raise RuntimeError("Call index(corpus) before retrieve().")
        q_matrix = normalize(self._vectorizer.transform(queries), norm="l2", copy=False)
        # cosine = dot product of L2-normalised vectors
        scores = (q_matrix @ self._corpus_matrix.T).toarray()  # (n_queries, n_docs)
        top_k = min(top_k, scores.shape[1])
        idx_out = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        for i in range(idx_out.shape[0]):
            order = np.argsort(-scores[i, idx_out[i]])
            idx_out[i] = idx_out[i, order]
        score_out = np.take_along_axis(scores, idx_out, axis=1).astype(np.float32)
        return idx_out, score_out


__all__ = ["BM25Retriever", "TfidfRetriever"]
