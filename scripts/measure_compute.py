#!/usr/bin/env python3
"""
Wall-clock + memory measurements for retrieval baselines.

Drives the *compute axis* the EMNLP'26 reframing wants alongside the
accuracy axis: fine-tune wall-clock (REIGN only), inference μs/query,
peak GPU memory, index size on disk. Outputs structured JSON so the
Pareto plot (`scripts/plot_pareto.py`, future) can read every row
uniformly.

The script measures *one* baseline per invocation. Each measurement
runs a configurable number of warm-up + measured iterations on a
sample of queries / corpus drawn from the GoodWiki-Long test split.

Usage:
    # Dense baseline
    python scripts/measure_compute.py \\
        --kind dense --baseline bge-m3 \\
        --n_corpus 200 --n_queries 50 --top_k 10 \\
        --output_path results/compute_bge-m3.json

    # Sparse retriever
    python scripts/measure_compute.py \\
        --kind sparse --retriever bm25 \\
        --n_corpus 1000 --n_queries 100 \\
        --output_path results/compute_bm25.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from reign.encoders.eval_utils import build_query_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("measure_compute")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["dense", "sparse", "reign"], required=True)
    p.add_argument("--baseline", help="dense baseline id (when --kind dense)")
    p.add_argument("--retriever", help="sparse retriever id: bm25 | tfidf (when --kind sparse)")
    p.add_argument("--protocol", choices=["truncate", "chunk_pool"], default=None)
    p.add_argument(
        "--torch-dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help=(
            "Override the baseline's compute dtype so latency is measured under the "
            "same configuration that produced the reported accuracy numbers."
        ),
    )
    # REIGN-specific:
    p.add_argument("--checkpoint", help="path to REIGN best/ checkpoint (when --kind reign)")
    p.add_argument("--gn-model", help="HF id of the guidance network used (when --kind reign)")
    p.add_argument("--chunk-size", type=int, default=512, help="REIGN GN chunk size")
    p.add_argument(
        "--gn-stride",
        dest="stride",
        type=int,
        default=None,
        help="GN stride (default: equal to --chunk-size, i.e. non-overlapping chunks)",
    )
    p.add_argument(
        "--gn-cache-mode",
        choices=["uncached", "cached", "build"],
        default="uncached",
        help=(
            "Which side of the cached/uncached efficiency axis to measure. "
            "'uncached' = GN forward + REIGN forward; 'build' = time the one-time "
            "GN forward + HDF5 write; 'cached' = HDF5 read + REIGN forward."
        ),
    )
    p.add_argument(
        "--cache-tag",
        default=None,
        help="Identifier prefix for the measurement cache (default: derived from --output_path)",
    )
    p.add_argument("--name", help="row label for this baseline in the output JSON")
    p.add_argument(
        "--dataset",
        default="devrim/goodwiki_long_synthetic_ir",
        help="HF dataset id in BEIR/MTEB layout (corpus/queries/default configs)",
    )
    p.add_argument("--split", default="test", help="qrels split: train | val | test")
    p.add_argument("--n_corpus", type=int, default=200, help="Subsample of corpus for the measurement")
    p.add_argument("--n_queries", type=int, default=50)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_warmup", type=int, default=2)
    p.add_argument("--n_repeat", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--output_path", required=True)
    return p.parse_args()


@contextmanager
def gpu_memory_tracker(device: str | None):
    """Reset peak-mem counter on entry; expose .peak_bytes after exit."""
    import torch

    use_cuda = device == "cuda" or (device is None and torch.cuda.is_available())
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    class _Stats:
        peak_bytes: int = 0

    stats = _Stats()
    try:
        yield stats
    finally:
        if use_cuda:
            torch.cuda.synchronize()
            stats.peak_bytes = int(torch.cuda.max_memory_allocated())


def time_call(fn, *args, n_warmup: int = 1, n_repeat: int = 3, **kwargs) -> tuple[float, float, Any]:
    """Run fn warmup+repeat times, return (mean_seconds, std_seconds, last_result)."""
    last = None
    for _ in range(n_warmup):
        last = fn(*args, **kwargs)
    times: list[float] = []
    for _ in range(n_repeat):
        gc.collect()
        t0 = time.perf_counter()
        last = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), last


def index_size_bytes(corpus_emb: np.ndarray | None = None, retriever_obj=None) -> int:
    """Approximate disk footprint of the index."""
    if corpus_emb is not None:
        return int(corpus_emb.nbytes)
    # For sparse retrievers, fall back to pickle-size approximation.
    if retriever_obj is not None:
        import pickle

        try:
            return len(pickle.dumps(retriever_obj))
        except Exception:
            return -1
    return -1


def measure_dense(args):
    from reign.encoders.dense import BASELINES, DenseEncoder, DenseEncoderConfig

    if args.baseline is None:
        raise SystemExit("--baseline required when --kind dense")
    config: DenseEncoderConfig = BASELINES[args.baseline]
    overrides: dict = {}
    if args.protocol is not None:
        overrides["inference_protocol"] = args.protocol
    if args.torch_dtype is not None:
        overrides["torch_dtype"] = args.torch_dtype
    if overrides:
        config = DenseEncoderConfig(**{**config.__dict__, **overrides})

    queries, _, corpus, _, _ = build_query_corpus(args.dataset, args.split)
    queries = queries[: args.n_queries]
    corpus = corpus[: args.n_corpus]

    encoder = DenseEncoder(config, device=args.device)

    with gpu_memory_tracker(args.device) as gpu_index:
        t_index_mean, t_index_std, corpus_emb = time_call(
            encoder.encode,
            corpus,
            batch_size=args.batch_size,
            side="document",
            n_warmup=args.n_warmup,
            n_repeat=args.n_repeat,
        )

    with gpu_memory_tracker(args.device) as gpu_query:
        t_query_mean, t_query_std, query_emb = time_call(
            encoder.encode,
            queries,
            batch_size=args.batch_size,
            side="query",
            n_warmup=args.n_warmup,
            n_repeat=args.n_repeat,
        )

    # Top-k retrieval timing (cosine + argsort, CPU)
    def _retrieve():
        sims = query_emb @ corpus_emb.T
        return np.argpartition(-sims, args.top_k - 1, axis=1)[:, : args.top_k]

    t_retrieve_mean, t_retrieve_std, _ = time_call(
        _retrieve, n_warmup=args.n_warmup, n_repeat=args.n_repeat
    )

    return {
        "kind": "dense",
        "baseline": encoder.name,
        "model_name_or_path": config.model_name_or_path,
        "n_params": encoder.n_params,
        "hidden_size": encoder.hidden_size,
        "inference_protocol": config.inference_protocol,
        "max_length": config.max_length,
        "torch_dtype": config.torch_dtype,
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "n_warmup": args.n_warmup,
        "n_repeat": args.n_repeat,
        "index_seconds": {"mean": t_index_mean, "std": t_index_std},
        "query_encode_seconds": {"mean": t_query_mean, "std": t_query_std},
        "retrieve_seconds": {"mean": t_retrieve_mean, "std": t_retrieve_std},
        "per_query_total_us": (t_query_mean + t_retrieve_mean) / max(len(queries), 1) * 1e6,
        "peak_gpu_bytes_index": gpu_index.peak_bytes,
        "peak_gpu_bytes_query": gpu_query.peak_bytes,
        "index_bytes": index_size_bytes(corpus_emb=corpus_emb),
        "gn_cache_mode": "n/a",
    }


def measure_sparse(args):
    from reign.encoders.sparse import BM25Retriever, TfidfRetriever

    retrievers = {"bm25": BM25Retriever, "tfidf": TfidfRetriever}
    if args.retriever is None or args.retriever not in retrievers:
        raise SystemExit(f"--retriever must be one of {sorted(retrievers)}")

    queries, _, corpus, _, _ = build_query_corpus(args.dataset, args.split)
    queries = queries[: args.n_queries]
    corpus = corpus[: args.n_corpus]

    retriever = retrievers[args.retriever]()

    t_index_mean, t_index_std, _ = time_call(
        retriever.index, corpus, n_warmup=args.n_warmup, n_repeat=args.n_repeat
    )

    t_retrieve_mean, t_retrieve_std, _ = time_call(
        retriever.retrieve,
        queries,
        top_k=args.top_k,
        n_warmup=args.n_warmup,
        n_repeat=args.n_repeat,
    )

    return {
        "kind": "sparse",
        "retriever": retriever.name,
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "top_k": args.top_k,
        "n_warmup": args.n_warmup,
        "n_repeat": args.n_repeat,
        "index_seconds": {"mean": t_index_mean, "std": t_index_std},
        "retrieve_seconds": {"mean": t_retrieve_mean, "std": t_retrieve_std},
        "per_query_total_us": t_retrieve_mean / max(len(queries), 1) * 1e6,
        "index_bytes": index_size_bytes(retriever_obj=retriever),
        "gn_cache_mode": "n/a",
    }


def _cached_encode_factory(encoder, dataset_identifier: str, stride: int):
    """Build an ``encode``-shaped callable that reads GN chunk embeddings from the
    on-disk HDF5 cache instead of running the Guidance Network.

    This is the *cached* half of the commitment's "cached and uncached on
    identical hardware". The uncached path runs GN-forward + REIGN-forward; this
    path runs HDF5-read + REIGN-forward, which is what a deployment with a warm
    cache actually pays.

    Indices are resolved per batch against the caller's global offset rather than
    always reloading rows 0..bs-1 (the behaviour of the extractor's own
    convenience loader). Re-reading one small row set would sit in the OS page
    cache and report an optimistically fast number; walking distinct rows is the
    honest cost.
    """
    import torch

    fe = encoder.feature_extractor

    def encode_cached(texts, batch_size=None, side="document"):
        texts = list(texts)
        bs = batch_size or 8
        out = []
        for i in range(0, len(texts), bs):
            n = len(texts[i : i + bs])
            embeddings, masks, _ = fe.cache.load_cache(
                fe.model_name_or_path,
                fe.chunk_size,
                dataset_identifier,
                text_indices=list(range(i, i + n)),
                stride=stride,
            )
            if embeddings is None:
                raise SystemExit(
                    f"cached measurement requested but no usable cache for "
                    f"{fe.model_name_or_path} / {dataset_identifier} (stride={stride}). "
                    "Run --gn-cache-mode build first."
                )
            features = fe._reconstruct_batch_encoding(embeddings, masks)
            features = {k: v.to(encoder.device) for k, v in features.items() if hasattr(v, "to")}
            with torch.no_grad():
                result = encoder.model(**features)
            pooled = getattr(result, "pooler_output", None)
            if pooled is None:
                pooled = result[1] if isinstance(result, tuple) and len(result) > 1 else result[0]
            if encoder.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    return encode_cached


def measure_reign(args):
    """REIGN compute: same shape as measure_dense, but with the REIGN BaseEncoder
    wrapper (GN forward + REIGN aggregation). Memory peaks include both models.

    ``--gn-cache-mode`` selects which half of the efficiency story is measured:
      uncached  GN forward + REIGN forward (cold cache / first pass over a corpus)
      build     one-time GN forward + HDF5 write, i.e. the cost being amortised
      cached    HDF5 read + REIGN forward (warm cache, the steady state)
    """
    from reign.encoders.reign import ReignBaselineEncoder

    if args.checkpoint is None or args.gn_model is None:
        raise SystemExit("--checkpoint and --gn-model are required when --kind reign")

    queries, _, corpus, _, _ = build_query_corpus(args.dataset, args.split)
    queries = queries[: args.n_queries]
    corpus = corpus[: args.n_corpus]

    encoder = ReignBaselineEncoder(
        checkpoint_path=args.checkpoint,
        gn_model=args.gn_model,
        chunk_size=args.chunk_size,
        stride=args.stride,
        device=args.device,
        name=args.name,
    )

    if args.gn_cache_mode != "uncached":
        return _measure_reign_cached(args, encoder, queries, corpus)

    with gpu_memory_tracker(args.device) as gpu_index:
        t_index_mean, t_index_std, corpus_emb = time_call(
            encoder.encode,
            corpus,
            batch_size=args.batch_size,
            side="document",
            n_warmup=args.n_warmup,
            n_repeat=args.n_repeat,
        )

    with gpu_memory_tracker(args.device) as gpu_query:
        t_query_mean, t_query_std, query_emb = time_call(
            encoder.encode,
            queries,
            batch_size=args.batch_size,
            side="query",
            n_warmup=args.n_warmup,
            n_repeat=args.n_repeat,
        )

    def _retrieve():
        sims = query_emb @ corpus_emb.T
        return np.argpartition(-sims, args.top_k - 1, axis=1)[:, : args.top_k]

    t_retrieve_mean, t_retrieve_std, _ = time_call(
        _retrieve, n_warmup=args.n_warmup, n_repeat=args.n_repeat
    )

    return {
        "kind": "reign",
        "baseline": encoder.name,
        "checkpoint": args.checkpoint,
        "gn_model": args.gn_model,
        "chunk_size": args.chunk_size,
        "n_params": encoder.n_params,
        "hidden_size": encoder.hidden_size,
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "n_warmup": args.n_warmup,
        "n_repeat": args.n_repeat,
        "index_seconds": {"mean": t_index_mean, "std": t_index_std},
        "query_encode_seconds": {"mean": t_query_mean, "std": t_query_std},
        "retrieve_seconds": {"mean": t_retrieve_mean, "std": t_retrieve_std},
        "per_query_total_us": (t_query_mean + t_retrieve_mean) / max(len(queries), 1) * 1e6,
        "peak_gpu_bytes_index": gpu_index.peak_bytes,
        "peak_gpu_bytes_query": gpu_query.peak_bytes,
        "index_bytes": index_size_bytes(corpus_emb=corpus_emb),
        "gn_cache_mode": "uncached",
    }


def _measure_reign_cached(args, encoder, queries, corpus):
    """`build` (populate the GN cache, timing the amortised one-time cost) and
    `cached` (steady-state HDF5-read + REIGN-forward) measurement paths."""
    import torch

    from reign.feature_extractor import ReignFeatureExtractor

    stride = args.stride or args.chunk_size
    tag = args.cache_tag or f"measure_{Path(args.output_path).stem}"
    ident_docs = f"{tag}_docs_cs_{args.chunk_size}_st_{stride}_n{len(corpus)}"
    ident_queries = f"{tag}_queries_cs_{args.chunk_size}_st_{stride}_n{len(queries)}"

    # A cache-enabled extractor writing under the project's normal cache root.
    cached_fe = ReignFeatureExtractor(
        batch_size=12,
        model_name_or_path=args.gn_model,
        chunk_size=args.chunk_size,
        stride=stride,
        device=encoder.device,
        enable_cache=True,
    )
    encoder.feature_extractor = cached_fe

    def _build(ident, texts):
        """One-time GN forward + HDF5 write for `texts` under `ident`."""
        return cached_fe.compute_and_cache_dataset_embeddings(
            texts=texts, dataset_identifier=ident
        )

    record: dict[str, Any] = {
        "kind": "reign",
        "baseline": encoder.name,
        "checkpoint": args.checkpoint,
        "gn_model": args.gn_model,
        "chunk_size": args.chunk_size,
        "stride": stride,
        "n_params": encoder.n_params,
        "hidden_size": encoder.hidden_size,
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "n_warmup": args.n_warmup,
        "n_repeat": args.n_repeat,
        "gn_cache_mode": args.gn_cache_mode,
        "cache_identifier_docs": ident_docs,
        "cache_identifier_queries": ident_queries,
    }

    have_docs = cached_fe.cache.has_cache(args.gn_model, args.chunk_size, ident_docs, stride=stride)
    have_q = cached_fe.cache.has_cache(args.gn_model, args.chunk_size, ident_queries, stride=stride)

    if args.gn_cache_mode == "build" or not (have_docs and have_q):
        # Time the build once (no warmup/repeat: it is inherently a one-shot cost).
        with gpu_memory_tracker(args.device) as gpu_build:
            t0 = time.perf_counter()
            if not have_docs:
                _build(ident_docs, corpus)
            if not have_q:
                _build(ident_queries, queries)
            t_build = time.perf_counter() - t0
        record["cache_build_seconds"] = t_build
        record["cache_build_seconds_per_doc"] = t_build / max(len(corpus) + len(queries), 1)
        record["peak_gpu_bytes_cache_build"] = gpu_build.peak_bytes
        cache_file = cached_fe.cache._get_cache_path(
            args.gn_model, args.chunk_size, ident_docs, stride=stride
        )
        record["cache_bytes_docs"] = (
            int(Path(cache_file).stat().st_size) if cache_file and Path(cache_file).exists() else -1
        )
        logger.info("Cache build took %.1fs (%d docs + %d queries)", t_build, len(corpus), len(queries))

    if args.gn_cache_mode == "build":
        return record

    # ---- cached steady state: HDF5 read + REIGN forward ----
    enc_docs = _cached_encode_factory(encoder, ident_docs, stride)
    enc_queries = _cached_encode_factory(encoder, ident_queries, stride)

    with gpu_memory_tracker(args.device) as gpu_index:
        t_index_mean, t_index_std, corpus_emb = time_call(
            enc_docs, corpus, batch_size=args.batch_size,
            n_warmup=args.n_warmup, n_repeat=args.n_repeat,
        )
    with gpu_memory_tracker(args.device) as gpu_query:
        t_query_mean, t_query_std, query_emb = time_call(
            enc_queries, queries, batch_size=args.batch_size,
            n_warmup=args.n_warmup, n_repeat=args.n_repeat,
        )

    def _retrieve():
        sims = query_emb @ corpus_emb.T
        return np.argpartition(-sims, args.top_k - 1, axis=1)[:, : args.top_k]

    t_retrieve_mean, t_retrieve_std, _ = time_call(
        _retrieve, n_warmup=args.n_warmup, n_repeat=args.n_repeat
    )

    record.update({
        "index_seconds": {"mean": t_index_mean, "std": t_index_std},
        "query_encode_seconds": {"mean": t_query_mean, "std": t_query_std},
        "retrieve_seconds": {"mean": t_retrieve_mean, "std": t_retrieve_std},
        "per_query_total_us": (t_query_mean + t_retrieve_mean) / max(len(queries), 1) * 1e6,
        "peak_gpu_bytes_index": gpu_index.peak_bytes,
        "peak_gpu_bytes_query": gpu_query.peak_bytes,
        "index_bytes": index_size_bytes(corpus_emb=corpus_emb),
    })
    return record


def main():
    args = parse_args()
    if getattr(args, "stride", None) is None:
        args.stride = args.chunk_size
    if args.kind == "dense":
        record = measure_dense(args)
    elif args.kind == "reign":
        record = measure_reign(args)
    else:
        record = measure_sparse(args)

    record["host"] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(record, f, indent=2, default=str)
    logger.info("Wrote %s", out_path)
    logger.info("Summary: %s", {k: v for k, v in record.items() if k != "host"})


if __name__ == "__main__":
    main()
