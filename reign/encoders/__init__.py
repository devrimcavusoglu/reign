"""
Pluggable encoders for evaluation baselines.

The `BaseEncoder` protocol defines the minimal interface every baseline
(native long-context dense retrievers, classical IR, REIGN itself) must
implement so that the retrieval evaluation harness in `reign/eval.py` can
treat them uniformly.

Concrete implementations live in sibling modules:
    reign/encoders/dense.py     — dense retrievers via HF (BGE-M3, Jina, Nomic, ...)
    reign/encoders/sparse.py    — BM25 / TF-IDF over full documents
    reign/encoders/reign.py     — REIGN as a BaseEncoder for symmetric eval
    reign/encoders/loco.py      — LoCo benchmark loader/adapters

This file is intentionally lightweight: it only declares the protocols that
the concrete baseline modules implement.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class BaseEncoder(Protocol):
    """Minimal interface for a retrieval encoder used in baseline comparisons."""

    name: str

    def encode(self, texts: Iterable[str], batch_size: int | None = None) -> np.ndarray:
        """Encode a batch of documents into a 2-D matrix of shape (n_texts, embed_dim).

        Implementations should:
        - return a numpy array (caller may move to torch / GPU as needed)
        - normalise embeddings if cosine similarity is the intended metric
        - be deterministic given fixed weights and inputs
        """
        ...


@runtime_checkable
class BaseRetriever(Protocol):
    """Index-then-score retriever interface for baselines that don't fit the encoder protocol.

    Use this for BM25, learned sparse retrievers, and any baseline whose scoring
    function is not cosine-over-fixed-dim-vectors. Dense retrievers and TF-IDF can
    also be expressed here when the index/query asymmetry matters.
    """

    name: str

    def index(self, corpus: Sequence[str]) -> None:
        """Build the index over a corpus. Idempotent within an instance lifetime."""
        ...

    def retrieve(self, queries: Sequence[str], top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Score `queries` against the indexed corpus.

        Returns:
            (indices, scores), each of shape (n_queries, top_k). `indices[i, j]` is the
            corpus position of the j-th hit for query i; `scores[i, j]` is its score.
        """
        ...


__all__ = ["BaseEncoder", "BaseRetriever"]
