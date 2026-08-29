#!/usr/bin/env python3
"""
Run native long-context dense retrieval baselines on REIGN's GoodWiki-Long setting.

Native long-context dense encoders (BGE-M3, Jina-v3, Nomic-v1.5, Stella, ...)
are the strongest point of comparison for REIGN: they read the whole document
in one forward pass instead of aggregating chunk embeddings. This script emits
the corresponding rows of paper Tables 3-4, plus per-baseline metadata (params,
embedding dim, inference protocol) used by the efficiency analysis.

Usage:
    python scripts/evaluate_dense_baselines.py \\
        --baseline bge-m3 \\
        --dataset devrim/goodwiki_long_synthetic_ir \\
        --split test \\
        --top_k 10 \\
        --batch_size 8 \\
        --output_path results/dense_bge-m3_test.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from reign.encoders.dense import BASELINES, DenseEncoder, DenseEncoderConfig
from reign.encoders.eval_utils import (
    build_per_query_payload,
    build_query_corpus,
    build_relevance,
    compute_metrics,
    drop_self_matches,
    topk_from_similarity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_dense_baselines")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline",
        required=True,
        choices=sorted(BASELINES),
        help="Curated baseline id from reign.encoders.dense.BASELINES",
    )
    p.add_argument(
        "--dataset",
        default="devrim/goodwiki_long_synthetic_ir",
        help="HF dataset id in BEIR/MTEB layout (corpus/queries/default configs)",
    )
    p.add_argument("--split", default="test", help="qrels split: train | val | test")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default=None, help="cuda / cpu (auto-detect if unset)")
    p.add_argument(
        "--protocol",
        choices=["truncate", "chunk_pool"],
        default=None,
        help="Override the baseline's default inference protocol",
    )
    p.add_argument(
        "--torch-dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help=(
            "Override the baseline's compute dtype. fp16/bf16 is standard for "
            "retrieval-embedding inference (matches MTEB/BEIR and the model "
            "authors' setup) and ~2x faster on long docs; cosine rankings are "
            "unaffected to ~4 decimals. Default: the registry value (fp32)."
        ),
    )
    p.add_argument("--output_path", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    config: DenseEncoderConfig = BASELINES[args.baseline]
    overrides: dict = {}
    if args.protocol is not None:
        overrides["inference_protocol"] = args.protocol
    if args.torch_dtype is not None:
        overrides["torch_dtype"] = args.torch_dtype
    if overrides:
        config = DenseEncoderConfig(**{**config.__dict__, **overrides})

    queries, q_meta, corpus, c_meta, qrels = build_query_corpus(args.dataset, args.split)
    logger.info("Building relevance matrix...")
    relevance = build_relevance(q_meta, c_meta, qrels)

    encoder = DenseEncoder(config, device=args.device)
    logger.info(
        "Encoder %s | params=%d | hidden=%d | protocol=%s | max_len=%d",
        encoder.name,
        encoder.n_params,
        encoder.hidden_size,
        config.inference_protocol,
        config.max_length,
    )

    logger.info("Encoding %d corpus docs...", len(corpus))
    corpus_emb = encoder.encode(corpus, batch_size=args.batch_size, side="document")
    logger.info("Encoding %d queries...", len(queries))
    query_emb = encoder.encode(queries, batch_size=args.batch_size, side="query")

    # Retrieve top_k+1: each query has at most one self-match in the corpus
    # (RELISH shares PMIDs between queries and corpus). On GoodWiki the id
    # spaces are disjoint so the filter is a no-op.
    fetch_k = args.top_k + 1
    logger.info("Computing similarity + top-%d ...", fetch_k)
    # Cosine sim — embeddings are L2-normalised by default in DenseEncoder
    sims = query_emb @ corpus_emb.T
    top_indices, top_scores = topk_from_similarity(sims, k=fetch_k)

    top_indices, top_scores, n_dropped = drop_self_matches(
        top_indices, top_scores, q_meta, c_meta, target_k=args.top_k
    )
    if n_dropped:
        logger.info("Dropped %d self-matches across %d queries", n_dropped, len(queries))

    metrics = compute_metrics(top_indices, relevance, k=args.top_k)
    per_query = build_per_query_payload(
        top_indices, top_scores, relevance, q_meta, c_meta, k=args.top_k
    )
    logger.info("Metrics: %s", metrics)

    out = {
        "baseline": encoder.name,
        "model_name_or_path": config.model_name_or_path,
        "n_params": encoder.n_params,
        "hidden_size": encoder.hidden_size,
        "inference_protocol": config.inference_protocol,
        "max_length": config.max_length,
        "chunk_size": config.chunk_size if config.inference_protocol == "chunk_pool" else None,
        "pooling": config.pooling,
        "torch_dtype": config.torch_dtype,
        "dataset": args.dataset,
        "split": args.split,
        "top_k": args.top_k,
        "n_queries": len(queries),
        "n_corpus": len(corpus),
        "self_matches_dropped": int(n_dropped),
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
