"""Hand-checked unit tests for the per-query metric path in
``reign/encoders/eval_utils.py``.

Every reported number in the paper flows through these two functions:
``per_query_metrics`` computes one value per query, ``compute_metrics`` is
defined as its mean, and ``build_per_query_payload`` is what gets written into
the result JSONs and later consumed by ``scripts/paired_bootstrap.py``. If the
per-query path and the aggregate path could drift, the significance tests in
Table 5 would be testing something other than the numbers in Tables 3-4.

The fixture below is small enough to work out with a pencil. Every expected
value is derived in a comment from the metric definitions, not copied from a
previous run, so a change in behaviour shows up as a failure rather than as a
silently updated golden file.

Fixture (n_queries = 4, n_corpus = 6, k = 3), grades 2 = true pair,
1 = distractor, 0 = irrelevant:

    relevance                       ranking (top-3 corpus indices)
    Q0  [0, 2, 1, 1, 0, 0]          [1, 4, 2]  -> grades [2, 0, 1]
    Q1  [2, 0, 0, 0, 0, 0]          [3, 0, 5]  -> grades [0, 2, 0]
    Q2  [0, 0, 0, 0, 1, 0]          [4, 1, 2]  -> grades [1, 0, 0]
    Q3  [0, 0, 0, 0, 0, 0]          [0, 1, 2]  -> grades [0, 0, 0]

Q3 has no relevant document at all, which is the degenerate case where the
ideal DCG is zero; it must score 0 rather than divide by zero.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reign.encoders.eval_utils import (
    build_per_query_payload,
    compute_metrics,
    per_query_metrics,
)

K = 3

# Discount at rank i is 1 / log2(i + 2).
D1 = 1.0                      # 1 / log2(2)
D2 = 1.0 / math.log2(3.0)     # 0.6309297535714574
D3 = 0.5                      # 1 / log2(4)


@pytest.fixture
def fixture():
    relevance = np.array(
        [
            [0, 2, 1, 1, 0, 0],
            [2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    top_indices = np.array(
        [
            [1, 4, 2],
            [3, 0, 5],
            [4, 1, 2],
            [0, 1, 2],
        ],
        dtype=np.int64,
    )
    return top_indices, relevance


# --------------------------------------------------------------------------
# nDCG@3 — exponential gain (2^rel - 1), the BEIR/MTEB convention
# --------------------------------------------------------------------------


def test_ndcg_at_k_per_query_matches_hand_derivation(fixture):
    top_indices, relevance = fixture
    pq = per_query_metrics(top_indices, relevance, k=K)

    # Q0: retrieved grades [2, 0, 1] -> gains [3, 0, 1]
    #     DCG  = 3*D1 + 0*D2 + 1*D3          = 3 + 0 + 0.5      = 3.5
    #     ideal grades (sorted row, top 3) = [2, 1, 1] -> gains [3, 1, 1]
    #     IDCG = 3*D1 + 1*D2 + 1*D3          = 3 + 0.63093 + 0.5 = 4.13093
    #     nDCG = 3.5 / 4.13093               ~ 0.847267
    q0 = (3 * D1 + 1 * D3) / (3 * D1 + 1 * D2 + 1 * D3)

    # Q1: retrieved grades [0, 2, 0] -> gains [0, 3, 0]
    #     DCG  = 3*D2 = 1.89279 ; ideal [2, 0, 0] -> IDCG = 3*D1 = 3
    #     nDCG = D2 = 0.63093  (the single relevant doc sits at rank 2)
    q1 = (3 * D2) / (3 * D1)

    # Q2: retrieved grades [1, 0, 0] -> gains [1, 0, 0]
    #     DCG = 1*D1 = 1 ; ideal [1, 0, 0] -> IDCG = 1 -> perfect ranking
    q2 = 1.0

    # Q3: no relevant document -> DCG = IDCG = 0 -> defined as 0, not NaN
    q3 = 0.0

    assert pq[f"nDCG@{K}"] == pytest.approx([q0, q1, q2, q3])
    # Spot-check the decimals quoted in the comments above.
    assert q0 == pytest.approx(0.847267, abs=1e-6)
    assert q1 == pytest.approx(0.630930, abs=1e-6)


def test_ndcg_degenerate_query_is_zero_not_nan(fixture):
    top_indices, relevance = fixture
    pq = per_query_metrics(top_indices, relevance, k=K)
    ndcg = pq[f"nDCG@{K}"]
    assert np.isfinite(ndcg).all()
    assert ndcg[3] == 0.0


# --------------------------------------------------------------------------
# Graded precision / recall — linear gain (rel / max_gain), max_gain = 2
# --------------------------------------------------------------------------


def test_precision_per_query_matches_hand_derivation(fixture):
    top_indices, relevance = fixture
    pq = per_query_metrics(top_indices, relevance, k=K)

    # P@1 is the gain of the rank-1 document: rel/2.
    #   Q0 grade 2 -> 1.0 ; Q1 grade 0 -> 0.0 ; Q2 grade 1 -> 0.5 ; Q3 -> 0.0
    assert pq["P@1"] == pytest.approx([1.0, 0.0, 0.5, 0.0])

    # P@3 = (sum of gains in the top 3) / 3.
    #   Q0 (1.0 + 0 + 0.5)/3 = 0.5
    #   Q1 (0 + 1.0 + 0)/3   = 1/3
    #   Q2 (0.5 + 0 + 0)/3   = 1/6
    #   Q3 0
    assert pq[f"P@{K}"] == pytest.approx([0.5, 1 / 3, 1 / 6, 0.0])


def test_recall_per_query_matches_hand_derivation(fixture):
    top_indices, relevance = fixture
    pq = per_query_metrics(top_indices, relevance, k=K)

    # Total available gain per query = sum(rel)/2:
    #   Q0 (2+1+1)/2 = 2.0 ; Q1 1.0 ; Q2 0.5 ; Q3 0.0
    # Gain found in the top 3:
    #   Q0 1.0 + 0.5 = 1.5 -> 1.5/2.0 = 0.75  (pair + one of two distractors)
    #   Q1 1.0             -> 1.0/1.0 = 1.0
    #   Q2 0.5             -> 0.5/0.5 = 1.0
    #   Q3 nothing to find -> 0.0 by convention (no division by zero)
    assert pq[f"R@{K}"] == pytest.approx([0.75, 1.0, 1.0, 0.0])


def test_distractor_counts_as_half_a_hit_not_a_full_one():
    """The 2 : 1 : 0 weighting is the point of the graded metrics."""
    relevance = np.array([[2, 1, 0]], dtype=np.int64)
    pair_first = per_query_metrics(np.array([[0]]), relevance, k=1)
    distractor_first = per_query_metrics(np.array([[1]]), relevance, k=1)
    assert pair_first["P@1"][0] == pytest.approx(1.0)
    assert distractor_first["P@1"][0] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# MAP@3 — deliberately binary, for BEIR comparability
# --------------------------------------------------------------------------


def test_map_per_query_matches_hand_derivation(fixture):
    top_indices, relevance = fixture
    pq = per_query_metrics(top_indices, relevance, k=K)

    # AP is binary (any grade >= 1 is a hit), unlike P/R above.
    #   Q0 hits at ranks 1 and 3: (1/1 + 2/3) / 2 = 0.83333
    #   Q1 hit at rank 2:         (1/2) / 1       = 0.5
    #   Q2 hit at rank 1:         (1/1) / 1       = 1.0
    #   Q3 no hits:                                 0.0
    assert pq[f"MAP@{K}"] == pytest.approx([(1.0 + 2 / 3) / 2, 0.5, 1.0, 0.0])


# --------------------------------------------------------------------------
# The invariant the significance tests depend on
# --------------------------------------------------------------------------


def test_aggregate_is_exactly_the_mean_of_the_per_query_values(fixture):
    top_indices, relevance = fixture
    pq = per_query_metrics(top_indices, relevance, k=K)
    agg = compute_metrics(top_indices, relevance, k=K)

    assert set(agg) == set(pq)
    for name, values in pq.items():
        assert agg[name] == pytest.approx(float(np.mean(values)), abs=1e-15), name


def test_payload_carries_the_same_values_and_a_real_ranking(fixture):
    top_indices, relevance = fixture
    top_scores = np.array(
        [
            [0.90, 0.40, 0.35],
            [0.80, 0.75, 0.10],
            [0.70, 0.20, 0.15],
            [0.60, 0.50, 0.05],
        ],
        dtype=np.float32,
    )
    q_meta = [{"_id": f"Q{i}"} for i in range(4)]
    c_meta = [{"_id": f"C{j}"} for j in range(6)]

    payload = build_per_query_payload(
        top_indices, top_scores, relevance, q_meta, c_meta, k=K
    )
    pq = per_query_metrics(top_indices, relevance, k=K)
    agg = compute_metrics(top_indices, relevance, k=K)

    assert list(payload) == ["Q0", "Q1", "Q2", "Q3"]
    for i, qid in enumerate(payload):
        entry = payload[qid]
        for name, values in pq.items():
            assert entry[name] == pytest.approx(float(values[i])), f"{qid} {name}"
        # The stored ranking must name the actual corpus ids that were retrieved.
        assert entry["retrieved"] == [f"C{j}" for j in top_indices[i]]
        assert entry["scores"] == pytest.approx(list(top_scores[i]), abs=1e-6)

    # The self-check paired_bootstrap.load_per_query performs on every file it
    # reads: the aggregate must be the mean of the per-query values.
    for name, value in agg.items():
        mean = float(np.mean([payload[q][name] for q in payload]))
        assert mean == pytest.approx(value, abs=1e-12), name


def test_payload_can_omit_the_ranking(fixture):
    top_indices, relevance = fixture
    q_meta = [{"_id": f"Q{i}"} for i in range(4)]
    c_meta = [{"_id": f"C{j}"} for j in range(6)]
    payload = build_per_query_payload(
        top_indices, None, relevance, q_meta, c_meta, k=K, include_ranking=False
    )
    for entry in payload.values():
        assert "retrieved" not in entry and "scores" not in entry
        assert f"nDCG@{K}" in entry


# --------------------------------------------------------------------------
# Sentinel handling (drop_self_matches pads short rows with -1)
# --------------------------------------------------------------------------


def test_sentinel_slots_contribute_no_gain(fixture):
    """-1 means "nothing retrieved here" and must score as zero, not as index -1.

    Indexing with a raw -1 would silently wrap to the last corpus document and
    could award credit for a document that was never returned.
    """
    _, relevance = fixture
    padded = np.array([[1, -1, -1]], dtype=np.int64)      # only the pair, then padding
    only_pair = np.array([[1]], dtype=np.int64)
    rel0 = relevance[:1]

    padded_pq = per_query_metrics(padded, rel0, k=K)
    # Last corpus doc (index -1 would wrap to column 5) is irrelevant for Q0, so
    # a wrap would not change the grade — pin the arithmetic instead:
    # gains [3, 0, 0] -> DCG = 3 ; IDCG unchanged at 4.13093.
    expected_ndcg = (3 * D1) / (3 * D1 + 1 * D2 + 1 * D3)
    assert padded_pq[f"nDCG@{K}"][0] == pytest.approx(expected_ndcg)
    # P@3 divides by k=3 even though only one slot was filled.
    assert padded_pq[f"P@{K}"][0] == pytest.approx(1.0 / 3)
    # Rank 1 is unaffected by what follows it.
    assert padded_pq["P@1"][0] == pytest.approx(per_query_metrics(only_pair, rel0, k=1)["P@1"][0])


def test_sentinel_never_awards_credit_for_a_wrapped_index():
    """A -1 must not pick up the last corpus document's relevance."""
    relevance = np.array([[0, 0, 2]], dtype=np.int64)  # the ONLY relevant doc is last
    padded = per_query_metrics(np.array([[0, -1]]), relevance, k=2)
    wrapped = per_query_metrics(np.array([[0, 2]]), relevance, k=2)
    assert padded[f"nDCG@2"][0] == 0.0          # nothing relevant was retrieved
    assert wrapped[f"nDCG@2"][0] > 0.0          # the real index does score
