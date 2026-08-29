#!/usr/bin/env python3
"""
Run BM25 / TF-IDF baselines on REIGN's GoodWiki-Long-style retrieval setting.

Doc-to-doc retrieval has a long pre-neural history; skipping classical
baselines makes any neural lift look misleadingly large. This script produces
the BM25 / TF-IDF rows that sit alongside Tables 3 and 4 in the paper.

Usage:
    python scripts/evaluate_sparse_baselines.py \\
        --retriever bm25 \\
        --dataset devrim/goodwiki_long_synthetic_ir \\
        --split test \\
        --top_k 10 \\
        --output_path results/sparse_bm25_goodwiki_long_test.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from reign.encoders import BaseRetriever
from reign.encoders.eval_utils import (
    build_query_corpus,
    build_relevance,
    build_per_query_payload,
    compute_metrics,
    drop_self_matches,
)
from reign.encoders.sparse import BM25Retriever, TfidfRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_sparse_baselines")


RETRIEVERS: dict[str, type[BaseRetriever]] = {
    "bm25": BM25Retriever,
    "tfidf": TfidfRetriever,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=sorted(RETRIEVERS), required=True)
    parser.add_argument(
        "--dataset",
        default="devrim/goodwiki_long_synthetic_ir",
        help="HF dataset id in BEIR/MTEB layout (corpus/queries/default configs)",
    )
    parser.add_argument("--split", default="test", help="qrels split: train | val | test")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    queries, q_meta, corpus, c_meta, qrels = build_query_corpus(args.dataset, args.split)
    logger.info("Building relevance matrix...")
    relevance = build_relevance(q_meta, c_meta, qrels)

    retriever = RETRIEVERS[args.retriever]()
    logger.info("Indexing corpus with %s...", retriever.name)
    retriever.index(corpus)

    # Retrieve top_k+1 so drop_self_matches can fill ``top_k`` even when a
    # query's own id appears in the corpus (RELISH: corpus and queries share
    # the same PMID pool, and each query has exactly one self-match). On
    # GoodWiki the id spaces are disjoint so the filter is a no-op.
    fetch_k = args.top_k + 1
    logger.info("Retrieving top-%d for %d queries...", fetch_k, len(queries))
    top_indices, top_scores = retriever.retrieve(queries, top_k=fetch_k)

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
        "retriever": retriever.name,
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
