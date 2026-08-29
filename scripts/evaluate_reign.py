#!/usr/bin/env python3
"""Run a trained REIGN checkpoint on the IR-format dataset, symmetrically with
the dense baselines.

Produces the "REIGN-on-X" rows of paper Table 3-4 alongside the X-alone rows
emitted by `scripts/evaluate_dense_baselines.py`.

Usage:
    python scripts/evaluate_reign.py \\
        --checkpoint path/to/reign-base-on-gte-large \\
        --gn-model thenlper/gte-large \\
        --chunk-size 512 \\
        --dataset devrim/goodwiki_long_synthetic_ir \\
        --split test \\
        --top_k 10 \\
        --batch_size 8 \\
        --output_path results/reign_gte-large_test.json

.. warning::
   **Never train a new run into an existing checkpoint directory. One run, one
   new directory.**

   The optional ``--corpus-embed-cache`` is keyed on the checkpoint *path*, so
   that the three DAPFAM query splits (test / test_in / test_out), which share
   one corpus pool, encode that corpus once instead of three times. A path is a
   stable key only as long as the weights behind it do not change. Overwriting a
   checkpoint directory with a retrained model keeps the key and invalidates the
   contents, which would silently score the new model with the old model's
   document embeddings.

   Every cache entry therefore stores a fingerprint of the checkpoint's weights
   files, and loading an entry whose fingerprint does not match the checkpoint on
   disk is a hard error rather than a warning. If you hit that error, either
   delete the named cache file or re-run with ``--refresh-corpus-cache``, which
   re-encodes the corpus and overwrites the entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from reign.encoders.eval_utils import (
    build_per_query_payload,
    build_query_corpus,
    build_relevance,
    compute_metrics,
    drop_self_matches,
    topk_from_similarity,
)
from reign.encoders.reign import ReignBaselineEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_reign")


def _corpus_cache_key(checkpoint, gn_model, chunk_size, stride, corpus_ids, corpus_texts):
    """Content-addressed key for the post-REIGN corpus doc-embedding matrix.

    Depends only on (encoder identity, corpus content) — NOT on the query split.
    test/test_in/test_out share one full corpus pool, so all three resolve to the
    same key and reuse a single encode. Two different strides/checkpoints get
    distinct keys (no cross-contamination).
    """
    h = hashlib.sha256()
    for part in (str(checkpoint), str(gn_model), str(int(chunk_size)), str(int(stride))):
        h.update(part.encode())
        h.update(b"\x00")
    h.update(f"n={len(corpus_ids)}".encode())
    h.update(b"\x00")
    for cid, txt in zip(corpus_ids, corpus_texts):
        h.update(str(cid).encode())
        h.update(b"\x01")
        h.update(txt.encode())
        h.update(b"\x02")
    return h.hexdigest()[:32]


# Glob patterns for the *model weights* inside a checkpoint directory. Trainer
# state (``training_artifacts.pt``: optimizer, scheduler, RNG) is deliberately
# excluded — it changes on every resume without changing what the model computes.
_WEIGHT_FILE_PATTERNS = ("*.safetensors", "pytorch_model*.bin", "model*.bin")
_FINGERPRINT_EDGE_BYTES = 1 << 20  # hash the first and last 1 MiB of each file
_FINGERPRINT_SCHEMA = b"reign-corpus-embed-cache-fingerprint/v1"


def _checkpoint_fingerprint(checkpoint) -> str:
    """Identify the *weights* behind ``checkpoint``, not merely its path.

    :func:`_corpus_cache_key` keys the on-disk corpus-embedding cache on the
    checkpoint path. That is only a valid identity while the weights at that
    path stay fixed; retraining into an existing directory would reuse the key
    for a different model. This fingerprint is stored in the cache entry and
    re-checked on load so that case fails loudly instead of silently serving the
    previous model's document embeddings.

    For a local checkpoint directory, hashes for every weights file (sorted by
    name): the file name, its size in bytes, and its first and last
    ``_FINGERPRINT_EDGE_BYTES``. This is O(1) in checkpoint size — it runs on
    every evaluation — and detects retraining, truncation and partial writes,
    which is the whole job. Full-file hashing would also work but costs seconds
    per gigabyte for no extra protection here.

    A checkpoint that is not a local directory (e.g. a Hugging Face Hub id) has
    no local file to hash and is fingerprinted by its identifier, which is what
    pins it.
    """
    path = Path(checkpoint)
    if not path.is_dir():
        return f"remote:{checkpoint}"

    files = sorted(
        {p for pattern in _WEIGHT_FILE_PATTERNS for p in path.glob(pattern) if p.is_file()},
        key=lambda p: p.name,
    )
    if not files:
        # Nothing to hash. Still distinguishable from a populated checkpoint, so
        # a cache written before the weights appeared cannot be reused after.
        return f"noweights:{path.resolve().name}"

    h = hashlib.sha256()
    h.update(_FINGERPRINT_SCHEMA)
    h.update(b"\x00")
    for f in files:
        size = f.stat().st_size
        h.update(f.name.encode())
        h.update(b"\x00")
        h.update(str(size).encode())
        h.update(b"\x00")
        with f.open("rb") as fh:
            h.update(fh.read(_FINGERPRINT_EDGE_BYTES))
            if size > _FINGERPRINT_EDGE_BYTES:
                fh.seek(max(0, size - _FINGERPRINT_EDGE_BYTES))
                h.update(fh.read(_FINGERPRINT_EDGE_BYTES))
        h.update(b"\x01")
    return h.hexdigest()


def _read_cached_fingerprint(data) -> str | None:
    """Fingerprint stored in a loaded ``.npz``, or None if the entry predates it."""
    if "_fingerprint" not in getattr(data, "files", []):
        return None
    return str(data["_fingerprint"].item() if data["_fingerprint"].ndim == 0
               else data["_fingerprint"].tolist()[0])


def _stale_cache_error(cache_path, checkpoint, stored: str | None, actual: str) -> str:
    """The message a user gets when a cache entry does not match its checkpoint."""
    if stored is None:
        cause = (
            "this cache file predates corpus-cache fingerprinting, so there is no "
            "record of which checkpoint's weights produced it and it cannot be "
            "trusted"
        )
    else:
        cause = (
            "the checkpoint's weights have changed since this cache file was "
            f"written (cached fingerprint {stored[:16]}..., checkpoint now "
            f"{actual[:16]}...) — the usual cause is a new training run writing "
            "into an existing checkpoint directory"
        )
    return (
        f"corpus-embed cache is stale and was NOT used: {cause}.\n"
        f"  cache file: {cache_path}\n"
        f"  checkpoint: {checkpoint}\n"
        "  Remedy: delete the cache file above, or re-run with "
        "--refresh-corpus-cache to re-encode the corpus and overwrite it.\n"
        "  (Reminder: never train a new run into an existing checkpoint "
        "directory — give each run its own output directory.)"
    )


def load_corpus_cache(cache_path, checkpoint, corpus_ids, *, refresh=False, fingerprint=None):
    """Corpus embeddings from ``cache_path``, or None if they must be re-encoded.

    Returns None (re-encode) when the entry is absent, when ``refresh`` is set,
    or when the entry's corpus ids do not match ``corpus_ids`` — the last case
    is a content-key collision, not a stale model, and recomputing is correct.

    Raises ``SystemExit`` when the entry exists but does not belong to the
    weights currently at ``checkpoint``: either it predates fingerprinting, or
    the checkpoint has been retrained in place. Both mean the stored embeddings
    were produced by a different model, and serving them would silently attach
    one model's document embeddings to another model's reported metrics.
    """
    cache_path = Path(cache_path)
    if fingerprint is None:
        fingerprint = _checkpoint_fingerprint(checkpoint)
    if not cache_path.exists():
        return None
    if refresh:
        logger.info(
            "--refresh-corpus-cache: ignoring existing %s; re-encoding and overwriting",
            cache_path,
        )
        return None

    data = np.load(cache_path, allow_pickle=False)
    stored_fp = _read_cached_fingerprint(data)
    if stored_fp != fingerprint:
        raise SystemExit(_stale_cache_error(cache_path, checkpoint, stored_fp, fingerprint))
    if [str(x) for x in data["ids"].tolist()] != [str(x) for x in corpus_ids]:
        logger.warning("Corpus-embed cache id mismatch at %s — recomputing", cache_path)
        return None

    corpus_emb = data["emb"]
    logger.info(
        "Corpus-embed cache HIT %s (%d docs, fingerprint %s) — skipped corpus encode",
        cache_path,
        corpus_emb.shape[0],
        fingerprint[:16],
    )
    return corpus_emb


def save_corpus_cache(cache_path, corpus_emb, corpus_ids, fingerprint):
    """Atomically write a cache entry; return the embeddings read back from it.

    The caller compares the returned array against the in-memory one, so a
    serialization that is not bit-identical is caught before any later split
    reuses the entry.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez auto-appends ".npz" unless the name already ends in it, so the
    # temp name must end in ".npz" for os.replace to find the real file.
    tmp = cache_path.parent / f"{cache_path.stem}.tmp{os.getpid()}.npz"
    np.savez(
        tmp,
        emb=corpus_emb,
        ids=np.array([str(x) for x in corpus_ids]),
        # Binds the entry to the exact weights that produced it, so a later run
        # against a retrained checkpoint at the same path is refused.
        _fingerprint=np.array(fingerprint),
    )
    os.replace(tmp, cache_path)
    written = np.load(cache_path, allow_pickle=False)
    if _read_cached_fingerprint(written) != fingerprint:
        raise RuntimeError(
            f"corpus-embed cache self-check FAILED at {cache_path}: the checkpoint "
            "fingerprint did not survive the write — refusing to leave an entry "
            "that cannot be validated on reload"
        )
    return written["emb"]


def _retrieve_and_score(query_emb, corpus_emb, q_meta, c_meta, relevance, top_k):
    """Pure: similarity -> top-k -> self-match drop -> metrics. Deterministic.

    Also returns the per-query metric/ranking payload; the aggregate is the mean
    of the per-query values, so the cache self-check on ``metrics`` still covers
    both.
    """
    fetch_k = top_k + 1
    sims = query_emb @ corpus_emb.T
    top_indices, top_scores = topk_from_similarity(sims, k=fetch_k)
    top_indices, top_scores, n_dropped = drop_self_matches(
        top_indices, top_scores, q_meta, c_meta, target_k=top_k
    )
    metrics = compute_metrics(top_indices, relevance, k=top_k)
    per_query = build_per_query_payload(
        top_indices, top_scores, relevance, q_meta, c_meta, k=top_k
    )
    return metrics, int(n_dropped), per_query


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path or HF id of the trained REIGN model")
    p.add_argument("--gn-model", required=True, help="HF id of the guidance network used at training")
    p.add_argument(
        "--gn-chunk-size",
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=512,
        help="Number of tokens per chunk fed to the Guidance Network (default: 512).",
    )
    p.add_argument(
        "--gn-stride",
        dest="stride",
        type=int,
        default=384,
        help=(
            "Stride between successive GN chunks (default: 384, i.e. 25%% overlap at "
            "chunk_size=512). Set equal to --gn-chunk-size for non-overlapping chunking."
        ),
    )
    p.add_argument(
        "--dataset",
        default="devrim/goodwiki_long_synthetic_ir",
        help="HF dataset id in BEIR/MTEB layout (corpus/queries/default configs)",
    )
    p.add_argument("--split", default="test", help="qrels split: train | val | test")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gn-batch-size", type=int, default=12)
    p.add_argument("--device", default=None, help="cuda / cpu (auto-detect if unset)")
    p.add_argument("--name", default=None, help="Override the row name in output JSON")
    p.add_argument("--output_path", required=True)
    p.add_argument(
        "--corpus-embed-cache",
        dest="corpus_embed_cache",
        default=None,
        help=(
            "Optional dir for an on-disk post-REIGN corpus doc-embedding cache. "
            "When set, the full corpus is encoded once per (checkpoint, GN, chunk, "
            "stride, corpus) and reused across query splits (test/test_in/test_out), "
            "skipping the redundant 45K-doc REIGN forward. On the cache-MISS split "
            "the fresh embeddings are saved, reloaded, and the entire "
            "retrieval+metrics path is recomputed and asserted bit-identical before "
            "any later split trusts the cache. Default off → byte-identical to the "
            "no-cache path."
        ),
    )
    p.add_argument(
        "--refresh-corpus-cache",
        dest="refresh_corpus_cache",
        action="store_true",
        help=(
            "Ignore any existing --corpus-embed-cache entry for this "
            "(checkpoint, GN, chunk, stride, corpus), re-encode the corpus, and "
            "overwrite the entry. Use this after the fingerprint guard refuses a "
            "stale entry, instead of deleting the file by hand."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    queries, q_meta, corpus, c_meta, qrels = build_query_corpus(args.dataset, args.split)
    logger.info("Building relevance matrix...")
    relevance = build_relevance(q_meta, c_meta, qrels)

    encoder = ReignBaselineEncoder(
        checkpoint_path=args.checkpoint,
        gn_model=args.gn_model,
        chunk_size=args.chunk_size,
        stride=args.stride,
        device=args.device,
        gn_batch_size=args.gn_batch_size,
        name=args.name,
    )
    logger.info(
        "Encoder %s | params=%d | hidden=%d | gn=%s | chunk_size=%d | stride=%d",
        encoder.name,
        encoder.n_params,
        encoder.hidden_size,
        args.gn_model,
        args.chunk_size,
        args.stride,
    )

    corpus_ids = [m["_id"] for m in c_meta]
    cache_path = None
    fresh_compute = True
    fingerprint = None
    if args.corpus_embed_cache:
        cdir = Path(args.corpus_embed_cache)
        cdir.mkdir(parents=True, exist_ok=True)
        key = _corpus_cache_key(
            args.checkpoint, args.gn_model, args.chunk_size, args.stride, corpus_ids, corpus
        )
        cache_path = cdir / f"{key}.npz"
        # The path-based key cannot see a retrain-in-place; the fingerprint can.
        fingerprint = _checkpoint_fingerprint(args.checkpoint)
        cached = load_corpus_cache(
            cache_path,
            args.checkpoint,
            corpus_ids,
            refresh=args.refresh_corpus_cache,
            fingerprint=fingerprint,
        )
        if cached is not None:
            corpus_emb = cached
            fresh_compute = False

    if fresh_compute:
        logger.info("Encoding %d corpus docs...", len(corpus))
        corpus_emb = encoder.encode(corpus, batch_size=args.batch_size, side="document")
    logger.info("Encoding %d queries...", len(queries))
    query_emb = encoder.encode(queries, batch_size=args.batch_size, side="query")

    # Retrieve top_k+1: each query has at most one self-match in the corpus
    # (RELISH shares PMIDs between queries and corpus). On GoodWiki the id
    # spaces are disjoint so the filter is a no-op.
    logger.info("Computing similarity + top-%d ...", args.top_k + 1)
    metrics, n_dropped, per_query = _retrieve_and_score(
        query_emb, corpus_emb, q_meta, c_meta, relevance, args.top_k
    )
    if n_dropped:
        logger.info("Dropped %d self-matches across %d queries", n_dropped, len(queries))

    # Persist + verify the cache on the MISS split, before any later split trusts
    # it: bit-identical round-trip AND the full retrieval+metrics path recomputed
    # from the reloaded array must match. Any drift → hard error (never emit
    # untrusted numbers on the authoritative path).
    if args.corpus_embed_cache and fresh_compute:
        reloaded = save_corpus_cache(cache_path, corpus_emb, corpus_ids, fingerprint)
        if not np.array_equal(reloaded, corpus_emb):
            raise RuntimeError(
                f"corpus-embed cache self-check FAILED at {cache_path}: reloaded != "
                "fresh (serialization not bit-identical) — refusing untrusted metrics"
            )
        m2, nd2, _ = _retrieve_and_score(query_emb, reloaded, q_meta, c_meta, relevance, args.top_k)
        if m2 != metrics or nd2 != n_dropped:
            raise RuntimeError(
                "corpus-embed cache self-check FAILED: metrics differ via reload "
                f"(fresh={metrics} reload={m2}) — refusing to trust cache"
            )
        logger.info(
            "Corpus-embed cache self-check PASSED (bit-identical round-trip + "
            "metrics identical via reload); wrote %s with checkpoint fingerprint %s",
            cache_path,
            fingerprint[:16],
        )

    logger.info("Metrics: %s", metrics)

    out = {
        "encoder": encoder.name,
        "checkpoint": args.checkpoint,
        "gn_model": args.gn_model,
        "chunk_size": args.chunk_size,
        "stride": args.stride,
        "n_params": encoder.n_params,
        "hidden_size": encoder.hidden_size,
        "dataset": args.dataset,
        "split": args.split,
        "top_k": args.top_k,
        "n_queries": len(queries),
        "n_corpus": len(corpus),
        "self_matches_dropped": int(n_dropped),
        "corpus_embed_cache": (str(cache_path) if cache_path is not None else None),
        "corpus_embed_cache_hit": (bool(args.corpus_embed_cache) and not fresh_compute),
        "checkpoint_fingerprint": fingerprint,
        "metrics": metrics,
        "per_query": per_query,
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
