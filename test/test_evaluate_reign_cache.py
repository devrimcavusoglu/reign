"""Correctness tests for the opt-in corpus-embed cache in scripts/evaluate_reign.py.

Fast, offline (no GPU / no checkpoint). Covers exactly the invariants the
cache's self-check relies on:

* ``_corpus_cache_key`` is deterministic and sensitive to every input that
  changes the embeddings (checkpoint, GN, chunk, stride, corpus ids/texts/order).
* ``_retrieve_and_score`` is a deterministic pure function — identical (and
  bit-identical) corpus embeddings yield an identical metrics dict + drop count
  (this is *why* the cache may be trusted across query splits).
* ``np.savez``/``np.load`` round-trips float32 embeddings bit-identically
  (underpins the ``np.array_equal(reloaded, fresh)`` self-check guard).
* **The fingerprint guard.** The cache key contains the checkpoint *path*, which
  is not a stable identity: retraining into an existing checkpoint directory
  keeps the key while replacing the weights. Every entry therefore records a
  fingerprint of the checkpoint's weights files, and ``load_corpus_cache``
  refuses any entry whose fingerprint does not match — including entries written
  before fingerprinting existed. The tests below cover the four outcomes that
  matter: hit, retrain-in-place refusal, legacy-entry refusal, and the
  ``--refresh-corpus-cache`` override.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_reign",
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_reign.py",
)
er = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(er)


def _key(ckpt="ck/best", gn="thenlper/gte-base", chunk=512, stride=512, ids=None, txt=None):
    ids = ids if ids is not None else [f"C{i}" for i in range(5)]
    txt = txt if txt is not None else [f"doc {i}" for i in range(5)]
    return er._corpus_cache_key(ckpt, gn, chunk, stride, ids, txt)


def test_cache_key_deterministic_and_sensitive():
    base = _key()
    assert base == _key()  # deterministic
    assert len(base) == 32
    # Every embedding-affecting input must change the key.
    assert _key(ckpt="other/best") != base
    assert _key(gn="thenlper/gte-large") != base
    assert _key(chunk=256) != base
    assert _key(stride=384) != base
    assert _key(ids=[f"C{i}" for i in range(6)] + []) != base  # different corpus
    assert _key(txt=[f"DOC {i}" for i in range(5)]) != base  # different content
    # Order matters: encode order == cache row order.
    rev_ids = [f"C{i}" for i in range(5)][::-1]
    rev_txt = [f"doc {i}" for i in range(5)][::-1]
    assert _key(ids=rev_ids, txt=rev_txt) != base
    # chunk/stride compared as ints, not strings.
    assert _key(chunk=512) == _key(chunk=512)


def _toy_eval(seed=0):
    rng = np.random.default_rng(seed)
    n_q, n_c, d = 4, 10, 16
    q = rng.standard_normal((n_q, d)).astype(np.float32)
    c = rng.standard_normal((n_c, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    q_meta = [{"_id": f"Q{i}"} for i in range(n_q)]
    c_meta = [{"_id": f"C{j}"} for j in range(n_c)]
    qrels = [
        {"query-id": "Q0", "corpus-id": "C1", "score": 2},
        {"query-id": "Q0", "corpus-id": "C2", "score": 1},
        {"query-id": "Q1", "corpus-id": "C3", "score": 2},
        {"query-id": "Q2", "corpus-id": "C5", "score": 2},
        {"query-id": "Q3", "corpus-id": "C9", "score": 2},
    ]
    rel = er.build_relevance(q_meta, c_meta, qrels)
    return q, c, q_meta, c_meta, rel


def test_retrieve_and_score_deterministic_and_cache_invariant():
    q, c, q_meta, c_meta, rel = _toy_eval()
    m1, nd1, pq1 = er._retrieve_and_score(q, c, q_meta, c_meta, rel, top_k=3)
    m2, nd2, pq2 = er._retrieve_and_score(q, c, q_meta, c_meta, rel, top_k=3)
    assert m1 == m2 and nd1 == nd2 and pq1 == pq2  # pure / deterministic
    assert isinstance(m1, dict) and m1  # produced real metrics

    # The cache invariant: a bit-identical reload of corpus_emb must yield the
    # exact same metrics dict + drop count (this is what lets test_in/test_out
    # trust the cache the miss-split wrote).
    c_reloaded = c.copy()
    assert np.array_equal(c_reloaded, c)
    m3, nd3, pq3 = er._retrieve_and_score(q, c_reloaded, q_meta, c_meta, rel, top_k=3)
    assert m3 == m1 and nd3 == nd1 and pq3 == pq1


def test_per_query_payload_matches_aggregate():
    """The per-query dump is what the DAPFAM significance test consumes, so the
    aggregate must be exactly the mean of it, and every ranking entry must name
    a real corpus id."""
    q, c, q_meta, c_meta, rel = _toy_eval()
    metrics, _, per_query = er._retrieve_and_score(q, c, q_meta, c_meta, rel, top_k=3)

    assert set(per_query) == {m["_id"] for m in q_meta}
    corpus_ids = {m["_id"] for m in c_meta}
    for qid, entry in per_query.items():
        assert len(entry["retrieved"]) == 3
        assert all(r is None or r in corpus_ids for r in entry["retrieved"])
        assert len(entry["scores"]) == 3
    for name, agg in metrics.items():
        mean = sum(e[name] for e in per_query.values()) / len(per_query)
        assert abs(mean - agg) < 1e-12, f"{name}: per-query mean {mean} != aggregate {agg}"


def test_npz_roundtrip_is_bit_identical(tmp_path):
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((123, 64)).astype(np.float32)
    ids = np.array([f"C{i}" for i in range(123)])
    import os

    p = tmp_path / "k.npz"
    # Mirror evaluate_reign.py's atomic write: temp name must end in .npz so
    # np.savez doesn't silently append it and break os.replace.
    tmp = p.parent / f"{p.stem}.tmp{os.getpid()}.npz"
    np.savez(tmp, emb=emb, ids=ids)
    os.replace(tmp, p)
    data = np.load(p, allow_pickle=False)
    assert np.array_equal(data["emb"], emb)  # the self-check's core guard
    assert [str(x) for x in data["ids"].tolist()] == list(ids)
    assert data["emb"].dtype == np.float32


# ---------------------------------------------------------------------------
# Checkpoint fingerprint — the guard against retraining in place
# ---------------------------------------------------------------------------


def _make_checkpoint(root, weights=b"W" * 4096, name="model.safetensors"):
    """A minimal checkpoint directory: a config plus one weights file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "reign"}')
    (root / name).write_bytes(weights)
    return root


def _cache_entry(path, emb, ids, fingerprint):
    """Write a cache entry the way evaluate_reign.py writes one."""
    return er.save_corpus_cache(path, emb, ids, fingerprint)


def _legacy_cache_entry(path, emb, ids):
    """A cache file in the pre-fingerprint format: emb + ids only."""
    np.savez(path, emb=emb, ids=np.array([str(x) for x in ids]))


def test_fingerprint_changes_when_the_weights_change(tmp_path):
    ck = _make_checkpoint(tmp_path / "run")
    before = er._checkpoint_fingerprint(ck)

    # Same path, new training run written over it — the exact failure mode.
    (ck / "model.safetensors").write_bytes(b"X" * 4096)
    after = er._checkpoint_fingerprint(ck)
    assert after != before

    # Same bytes again -> same fingerprint (stable, not time- or path-random).
    (ck / "model.safetensors").write_bytes(b"W" * 4096)
    assert er._checkpoint_fingerprint(ck) == before
    assert er._checkpoint_fingerprint(ck) == er._checkpoint_fingerprint(ck)


def test_fingerprint_detects_truncation_and_growth(tmp_path):
    ck = _make_checkpoint(tmp_path / "run", weights=b"W" * 8192)
    full = er._checkpoint_fingerprint(ck)
    (ck / "model.safetensors").write_bytes(b"W" * 4096)  # partial write
    assert er._checkpoint_fingerprint(ck) != full


def test_fingerprint_ignores_trainer_state(tmp_path):
    """Optimizer/scheduler state changes on every resume without changing what
    the model computes, so it must not invalidate a still-valid cache."""
    ck = _make_checkpoint(tmp_path / "run")
    (ck / "training_artifacts.pt").write_bytes(b"optimizer-state-epoch-1")
    before = er._checkpoint_fingerprint(ck)
    (ck / "training_artifacts.pt").write_bytes(b"optimizer-state-epoch-2-longer")
    assert er._checkpoint_fingerprint(ck) == before


def test_fingerprint_of_a_non_local_checkpoint_is_its_identifier():
    """A Hub id has no local weights file; it is pinned by the id itself."""
    fp = er._checkpoint_fingerprint("some-org/reign-base-l3")
    assert fp == "remote:some-org/reign-base-l3"
    assert fp != er._checkpoint_fingerprint("some-org/reign-large-l4")


def test_cache_hit_when_the_fingerprint_matches(tmp_path):
    ck = _make_checkpoint(tmp_path / "run")
    fp = er._checkpoint_fingerprint(ck)
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    ids = ["C0", "C1", "C2"]
    cache = tmp_path / "cache.npz"
    _cache_entry(cache, emb, ids, fp)

    loaded = er.load_corpus_cache(cache, ck, ids)
    assert loaded is not None, "an untouched checkpoint must produce a cache hit"
    assert np.array_equal(loaded, emb)


def test_cache_refused_after_retraining_into_the_same_directory(tmp_path):
    ck = _make_checkpoint(tmp_path / "run")
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    ids = ["C0", "C1", "C2"]
    cache = tmp_path / "cache.npz"
    _cache_entry(cache, emb, ids, er._checkpoint_fingerprint(ck))

    # A second training run overwrites the checkpoint at the same path.
    (ck / "model.safetensors").write_bytes(b"RETRAINED" * 512)

    with pytest.raises(SystemExit) as excinfo:
        er.load_corpus_cache(cache, ck, ids)
    msg = str(excinfo.value)
    assert str(cache) in msg          # names the cache file
    assert str(ck) in msg             # names the checkpoint
    assert "--refresh-corpus-cache" in msg  # names the remedy
    assert "weights have changed" in msg


def test_legacy_cache_without_a_fingerprint_is_refused(tmp_path):
    ck = _make_checkpoint(tmp_path / "run")
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    ids = ["C0", "C1", "C2"]
    cache = tmp_path / "legacy.npz"
    _legacy_cache_entry(cache, emb, ids)

    with pytest.raises(SystemExit) as excinfo:
        er.load_corpus_cache(cache, ck, ids)
    msg = str(excinfo.value)
    assert "predates" in msg          # says why, specifically
    assert str(cache) in msg
    assert "--refresh-corpus-cache" in msg


def test_refresh_flag_ignores_and_overwrites_a_stale_entry(tmp_path):
    ck = _make_checkpoint(tmp_path / "run")
    stale = np.zeros((3, 4), dtype=np.float32)
    ids = ["C0", "C1", "C2"]
    cache = tmp_path / "cache.npz"
    _cache_entry(cache, stale, ids, er._checkpoint_fingerprint(ck))
    (ck / "model.safetensors").write_bytes(b"RETRAINED" * 512)

    # With --refresh-corpus-cache the stale entry is neither read nor fatal:
    # the caller re-encodes (load returns None) ...
    assert er.load_corpus_cache(cache, ck, ids, refresh=True) is None

    # ... and the overwrite rebinds the entry to the new weights, so the very
    # next run hits the cache instead of failing.
    fresh = np.ones((3, 4), dtype=np.float32)
    new_fp = er._checkpoint_fingerprint(ck)
    written = _cache_entry(cache, fresh, ids, new_fp)
    assert np.array_equal(written, fresh)
    reloaded = er.load_corpus_cache(cache, ck, ids)
    assert reloaded is not None and np.array_equal(reloaded, fresh)
    assert not np.array_equal(reloaded, stale)


def test_refresh_flag_also_works_on_a_legacy_entry(tmp_path):
    ck = _make_checkpoint(tmp_path / "run")
    ids = ["C0", "C1", "C2"]
    cache = tmp_path / "legacy.npz"
    _legacy_cache_entry(cache, np.zeros((3, 4), dtype=np.float32), ids)
    assert er.load_corpus_cache(cache, ck, ids, refresh=True) is None


def test_missing_cache_file_is_a_plain_miss_not_an_error(tmp_path):
    ck = _make_checkpoint(tmp_path / "run")
    assert er.load_corpus_cache(tmp_path / "absent.npz", ck, ["C0"]) is None


def test_corpus_id_mismatch_recomputes_rather_than_failing(tmp_path):
    """Same weights, different corpus: a key collision, not a stale model."""
    ck = _make_checkpoint(tmp_path / "run")
    fp = er._checkpoint_fingerprint(ck)
    cache = tmp_path / "cache.npz"
    _cache_entry(cache, np.zeros((3, 4), dtype=np.float32), ["C0", "C1", "C2"], fp)
    assert er.load_corpus_cache(cache, ck, ["C0", "C1", "C9"]) is None


def test_saved_entry_round_trips_through_the_guard(tmp_path):
    """save_corpus_cache writes exactly what load_corpus_cache will accept."""
    ck = _make_checkpoint(tmp_path / "run")
    fp = er._checkpoint_fingerprint(ck)
    rng = np.random.default_rng(5)
    emb = rng.standard_normal((40, 16)).astype(np.float32)
    ids = [f"C{i}" for i in range(40)]
    cache = tmp_path / "cache.npz"

    written = er.save_corpus_cache(cache, emb, ids, fp)
    assert np.array_equal(written, emb)  # bit-identical round-trip
    loaded = er.load_corpus_cache(cache, ck, ids, fingerprint=fp)
    assert np.array_equal(loaded, emb)
    assert loaded.dtype == np.float32
    # No temp file left behind by the atomic write.
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["cache.npz"]


def test_two_checkpoints_do_not_share_a_cache_entry(tmp_path):
    """Distinct runs must never validate against each other's cache."""
    a = _make_checkpoint(tmp_path / "run-a", weights=b"A" * 4096)
    b = _make_checkpoint(tmp_path / "run-b", weights=b"B" * 4096)
    ids = ["C0", "C1", "C2"]
    cache = tmp_path / "cache.npz"
    _cache_entry(cache, np.zeros((3, 4), dtype=np.float32), ids,
                 er._checkpoint_fingerprint(a))
    with pytest.raises(SystemExit):
        er.load_corpus_cache(cache, b, ids)


def test_drop_self_match_path():
    # Query shares an id with a corpus doc → must be dropped, metrics still sane.
    q = np.eye(3, 8, dtype=np.float32)
    c = np.eye(3, 8, dtype=np.float32)  # C k aligns with Q k
    q_meta = [{"_id": f"S{i}"} for i in range(3)]
    c_meta = [{"_id": f"S{i}"} for i in range(3)]  # shared id space
    rel = er.build_relevance(q_meta, c_meta, [{"query-id": "S0", "corpus-id": "S1", "score": 2}])
    metrics, n_dropped, per_query = er._retrieve_and_score(q, c, q_meta, c_meta, rel, top_k=2)
    assert n_dropped == 3  # each query's identical self-doc removed
    assert isinstance(metrics, dict) and metrics
    # A dropped self-match must never reappear in the persisted ranking.
    for qid, entry in per_query.items():
        assert qid not in entry["retrieved"]
