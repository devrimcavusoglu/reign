"""Tests for the DAPFAM -> BEIR-format adapter (paper Section 5.3).

Offline and sub-second: exercises the view-text assembly, the domain bucketing,
and an end-to-end ``build_beir`` over the synthetic smoke records (no network).
"""

from __future__ import annotations

import numpy as np

from reign.dapfam.build_dataset import (
    POSITIVE_SCORE,
    _domain_bucket,
    build_beir,
    build_text,
    load_smoke_records,
)
from reign.dapfam.split_qrels import _stratified_split


def test_build_text_views_are_progressive():
    rec = {
        "title_en": "T",
        "abstract_en": "A",
        "claims_text": "C",
        "description_en": "D",
    }
    assert build_text(rec, "ta") == "T\n\nA"
    assert build_text(rec, "tac") == "T\n\nA\n\nC"
    assert build_text(rec, "fulltext") == "T\n\nA\n\nC\n\nD"


def test_build_text_skips_empty_and_missing_fields():
    rec = {"title_en": "", "abstract_en": None, "claims_text": "  C  ", "description_en": "D"}
    assert build_text(rec, "fulltext") == "C\n\nD"


def test_domain_bucket_normalisation():
    assert _domain_bucket("IN") == "in"
    assert _domain_bucket(" out ") == "out"
    assert _domain_bucket("Out") == "out"
    assert _domain_bucket("unknown") is None
    assert _domain_bucket(None) is None


def test_build_beir_schema_and_grades():
    # Smoke fixture: 4 positives (relevance_score 1.0 → score 2) + 1 negative
    # (C5, relevance_score 0.0). Every positive is IN xor OUT.
    corpus, queries, relations = load_smoke_records()

    # Default: keep_negatives=True → negative kept as score=0 in `test` only.
    beir, stats = build_beir(corpus, queries, relations, "tac")
    corpus_ds = beir["corpus"]["corpus"]
    queries_ds = beir["queries"]["queries"]
    default = beir["default"]

    assert set(corpus_ds.column_names) >= {"_id", "text"}
    assert set(queries_ds.column_names) >= {"_id", "text"}
    for split in ("test", "test_in", "test_out"):
        assert default[split].column_names == ["query-id", "corpus-id", "score"]

    assert len(default["test"]) == 5  # 4 positives + 1 kept negative
    assert set(default["test"]["score"]) == {0, POSITIVE_SCORE}
    assert stats["kept_negatives"] == 1
    assert stats["dropped_negatives"] == 0
    assert stats["qrels"]["test_positives"] == 4
    # test_in/test_out are POSITIVES-ONLY and sum to the positives.
    assert len(default["test_in"]) + len(default["test_out"]) == 4
    assert set(default["test_in"]["score"]) | set(default["test_out"]["score"]) == {POSITIVE_SCORE}

    # No dangling endpoints.
    cids, qids = set(corpus_ds["_id"]), set(queries_ds["_id"])
    assert set(default["test"]["corpus-id"]) <= cids
    assert set(default["test"]["query-id"]) <= qids

    # --no-keep-negatives → positives-only qrels (legacy behavior).
    beir2, stats2 = build_beir(corpus, queries, relations, "tac", keep_negatives=False)
    assert len(beir2["default"]["test"]) == 4
    assert set(beir2["default"]["test"]["score"]) == {POSITIVE_SCORE}
    assert stats2["dropped_negatives"] == 1
    assert stats2["kept_negatives"] == 0


def test_stratified_split_is_disjoint_deterministic_and_ratioed():
    qids = [f"Q{i}" for i in range(400)]
    keys = np.array([i % 20 for i in range(400)], dtype=float)  # spread difficulty
    a = _stratified_split(qids, keys, 0.70, 0.15, seed=42)
    b = _stratified_split(qids, keys, 0.70, 0.15, seed=42)
    tr, va, te = a

    assert (tr, va, te) == b  # deterministic for fixed seed
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert tr | va | te == set(qids)  # exhaustive
    # ~70/15/15 within rounding of the quartile slices.
    assert abs(len(tr) - 280) <= 4
    assert abs(len(va) - 60) <= 4
    assert abs(len(te) - 60) <= 4


def test_smoke_scale_1_is_unchanged_and_scale_n_is_splittable():
    # scale<=1 must stay byte-identical to the fixture the other tests assert on.
    c1, q1, r1 = load_smoke_records()
    assert (len(c1), len(q1)) == (6, 3)
    assert sum(1 for x in r1 if x["relevance_score"] >= 0.5) == 4

    c, q, r = load_smoke_records(16)
    assert (len(c), len(q)) == (96, 48)
    pos_qids = {x["query_id"] for x in r if x["relevance_score"] >= 0.5}
    assert pos_qids == {f"Q{i}" for i in range(48)}  # every query has a positive
    doms = {x["domain_rel"] for x in r}
    assert doms == {"IN", "OUT"}  # IN/OUT both present for the split's test_in/out


def test_build_beir_drops_dangling_qrels():
    corpus, queries, relations = load_smoke_records()
    relations = relations + [
        {
            "query_id": "Q0",
            "relevant_id": "DOES_NOT_EXIST",
            "relevance_score": 1.0,
            "domain_rel": "IN",
        }
    ]
    beir, stats = build_beir(corpus, queries, relations, "ta")
    assert stats["dropped_dangling"] == 1
    assert "DOES_NOT_EXIST" not in set(beir["default"]["test"]["corpus-id"])
