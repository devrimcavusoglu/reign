"""Sanity tests for the IR-format dataset loader in ``reign.encoders.eval_utils``.

These tests use a small in-memory dataset that mirrors the BEIR/MTEB layout
(corpus / queries / default configs) — they don't require the published HF
dataset to be reachable. The contract under test is:

* ``build_query_corpus`` restricts queries to those that appear in the chosen
  qrels split, but always returns the full corpus.
* ``build_relevance`` projects (query-id, corpus-id, score) triples onto the
  positional indices of the returned query/corpus arrays.
* ``compute_metrics`` computes the standard suite from those arrays.

The fixtures are tiny enough to verify by hand.
"""

from __future__ import annotations

from unittest import mock

import datasets
import numpy as np
import pytest

from reign.encoders.eval_utils import build_query_corpus, build_relevance, compute_metrics


def _fake_load_dataset(_dataset, config, split=None):
    """Stand-in for ``datasets.load_dataset`` exercising the IR layout.

    Three queries, six corpus docs, train+test qrels splits. Each query has 3
    relevant corpus docs (1 with score=2, 2 with score=1) — the same
    1-pair + 2-distractor convention REIGN uses.
    """
    if config == "corpus":
        return datasets.Dataset.from_dict(
            {
                "_id": ["c0", "c1", "c2", "c3", "c4", "c5"],
                "text": ["doc0", "doc1", "doc2", "doc3", "doc4", "doc5"],
            }
        )
    if config == "queries":
        return datasets.Dataset.from_dict(
            {
                "_id": ["q0", "q1", "q2"],
                "text": ["query0", "query1", "query2"],
            }
        )
    if config == "default":
        if split == "train":
            return datasets.Dataset.from_dict(
                {
                    "query-id": ["q0", "q0", "q0"],
                    "corpus-id": ["c0", "c1", "c2"],
                    "score": [2, 1, 1],
                }
            )
        if split == "test":
            return datasets.Dataset.from_dict(
                {
                    "query-id": ["q1", "q1", "q1", "q2", "q2", "q2"],
                    "corpus-id": ["c3", "c4", "c5", "c5", "c0", "c1"],
                    "score": [2, 1, 1, 2, 1, 1],
                }
            )
    raise AssertionError(f"unexpected config/split: {config}/{split}")


@pytest.fixture
def patched_loader():
    with mock.patch.object(datasets, "load_dataset", side_effect=_fake_load_dataset):
        yield


def test_build_query_corpus_restricts_queries_but_keeps_full_corpus(patched_loader):
    q_texts, q_meta, c_texts, c_meta, qrels = build_query_corpus("fake/repo", "test")

    # Test split has only q1 and q2 in qrels → q0 must be excluded from queries.
    assert [m["_id"] for m in q_meta] == ["q1", "q2"]
    assert q_texts == ["query1", "query2"]

    # Corpus is always the full pool.
    assert [m["_id"] for m in c_meta] == ["c0", "c1", "c2", "c3", "c4", "c5"]
    assert len(c_texts) == 6

    # qrels passthrough preserves all rows (used by build_relevance).
    assert len(qrels) == 6


def test_build_query_corpus_train_split_picks_train_queries(patched_loader):
    q_texts, q_meta, _, _, qrels = build_query_corpus("fake/repo", "train")
    assert [m["_id"] for m in q_meta] == ["q0"]
    assert q_texts == ["query0"]
    assert len(qrels) == 3


def test_build_relevance_projects_qrels_to_positional_matrix(patched_loader):
    _, q_meta, _, c_meta, qrels = build_query_corpus("fake/repo", "test")
    rel = build_relevance(q_meta, c_meta, qrels)

    # q1 → c3 (2), c4 (1), c5 (1) ; q2 → c5 (2), c0 (1), c1 (1)
    expected = np.array(
        [
            [0, 0, 0, 2, 1, 1],
            [1, 1, 0, 0, 0, 2],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(rel, expected)


def test_build_relevance_skips_unknown_ids(patched_loader, caplog):
    """Stray qrel rows pointing at ids outside the pool should warn but not crash."""
    _, q_meta, _, c_meta, _ = build_query_corpus("fake/repo", "test")
    bad_qrels = [
        {"query-id": "q1", "corpus-id": "c3", "score": 2},
        {"query-id": "ghost", "corpus-id": "c0", "score": 1},
        {"query-id": "q2", "corpus-id": "phantom", "score": 1},
    ]
    with caplog.at_level("WARNING"):
        rel = build_relevance(q_meta, c_meta, bad_qrels)
    assert "2 qrel rows skipped" in caplog.text
    assert rel[0, 3] == 2  # the only valid row landed
    assert rel.sum() == 2


def test_compute_metrics_on_perfect_ranking(patched_loader):
    """A perfect ranking should hit 1.0 on every metric in the suite."""
    _, q_meta, _, c_meta, qrels = build_query_corpus("fake/repo", "test")
    rel = build_relevance(q_meta, c_meta, qrels)

    perfect_top = np.argsort(-rel, axis=1, kind="stable")[:, :3]
    metrics = compute_metrics(perfect_top, rel, k=3)
    # P@1: pair (rel=2) at rank 1 → graded gain 1.0
    assert metrics["P@1"] == 1.0
    # R@3: all 3 relevants found → graded recall = total_gain / total_gain = 1.0
    assert metrics["R@3"] == 1.0
    # MAP@3: binary AP on perfect ordering → 1.0
    assert metrics["MAP@3"] == 1.0
    # nDCG@3: ideal DCG with exponential gain matches actual → 1.0
    assert metrics["nDCG@3"] == 1.0


def test_graded_p_r_at_k_honour_2_to_1_weighting():
    """A query where the only top-3 hit is a distractor (rel=1) should land at
    half-credit on P@k; a query where only the pair (rel=2) is at rank 1
    should be at full credit. Confirms the 2 : 1 : 0 weighting passed
    through P@k and R@k."""
    # 2 queries × 5 corpus docs.
    # query 0: rel = [2, 1, 1, 0, 0], retriever pulls them perfectly
    # query 1: rel = [2, 1, 1, 0, 0], retriever pulls only the distractors first
    relevance = np.array(
        [
            [2, 1, 1, 0, 0],
            [2, 1, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    top_indices = np.array(
        [
            [0, 1, 2, 3, 4],  # perfect ordering — pair, distractor, distractor
            [1, 2, 3, 0, 4],  # distractor, distractor, irrelevant, pair, irrelevant
        ],
        dtype=np.int64,
    )
    m = compute_metrics(top_indices, relevance, k=3)

    # P@1, query 0: pair (rel=2) → gain 1.0; query 1: distractor (rel=1) → gain 0.5
    # → mean = (1.0 + 0.5) / 2 = 0.75
    assert abs(m["P@1"] - 0.75) < 1e-9

    # P@3, query 0: gains [1.0, 0.5, 0.5] → mean 2.0/3 ≈ 0.667
    # P@3, query 1: gains [0.5, 0.5, 0.0] → mean 1.0/3 ≈ 0.333
    # → overall mean ≈ 0.5
    assert abs(m["P@3"] - 0.5) < 1e-9

    # R@3, query 0: found gain 2.0 / total 2.0 = 1.0
    # R@3, query 1: found gain 1.0 / total 2.0 = 0.5
    # → mean 0.75
    assert abs(m["R@3"] - 0.75) < 1e-9


def test_ndcg_uses_exponential_gain_beir_convention():
    """nDCG@k stays BEIR-comparable: gain = 2^rel - 1. With this gain pair
    contributes 3 and distractor contributes 1 (3:1 ratio), regardless of
    the 2:1 ratio P/R/MAP use."""
    # Single query, top-2: [pair, distractor], ideal = same.
    relevance = np.array([[2, 1, 1, 0, 0]], dtype=np.int64)
    top_indices = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)  # perfect
    m = compute_metrics(top_indices, relevance, k=3)
    # Perfect ordering → nDCG = 1.0
    assert m["nDCG@3"] == 1.0

    # Now flip pair and distractor: top-3 = [distractor, pair, distractor]
    top_indices = np.array([[1, 0, 2, 3, 4]], dtype=np.int64)
    m = compute_metrics(top_indices, relevance, k=3)
    # DCG = (2^1-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
    #     = 1.0       + 3 / log2(3) + 1/2
    discounts = 1.0 / np.log2(np.arange(2, 5))  # ranks 1..3
    expected_dcg = (1.0 * discounts[0]) + (3.0 * discounts[1]) + (1.0 * discounts[2])
    expected_idcg = (3.0 * discounts[0]) + (1.0 * discounts[1]) + (1.0 * discounts[2])
    assert abs(m["nDCG@3"] - expected_dcg / expected_idcg) < 1e-9
