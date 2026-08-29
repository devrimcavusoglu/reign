#!/usr/bin/env python3
"""Evaluate a retriever on the LoCo benchmark (Saad-Falcon et al., 2024).

LoCo is a 12-subtask long-document retrieval benchmark, used here as the
out-of-distribution counterpart to the in-distribution GoodWiki-Long results.
Supports both modes the eval harness understands:

  --baseline <name>           dense baselines from reign.encoders.dense.BASELINES
  --reign-checkpoint <path>   trained REIGN model + --gn-model

For each LoCo subtask, the runner downloads the query/document JSONLs from
``hazyresearch/LoCoV1-{Queries,Documents}``, encodes both sides, computes
cosine similarity, and reports the standard graded metric suite from
``reign.encoders.eval_utils.compute_metrics`` at the requested top-k. Subtask
metric is nDCG@10 by convention (the LoCo paper headline).

Outputs a single JSON per (encoder, subtask) under ``--output_dir`` and
prints a small final table. Each JSON file is content-identical in shape to
the dense-baseline and REIGN result JSONs, so downstream aggregators (e.g.
``_print_e4_table.py``, ``_print_e5_table.py``) need no schema changes.

Usage:
    # Dense baseline (e.g. one quick subtask)
    python scripts/evaluate_loco.py \\
        --baseline gte-base-chunked --subtask qasper_title \\
        --output-dir results/loco --tag baseline-smoke

    # REIGN over a fine-tuned checkpoint, all subtasks
    python scripts/evaluate_loco.py \\
        --reign-checkpoint models/reign-base-l3_gn-gte-small_s512_val-selected/best \\
        --gn-model thenlper/gte-small --gn-stride 512 \\
        --subtask all --output-dir results/loco --tag reign-gte-small-s512
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

from reign.encoders.eval_utils import (
    build_per_query_payload,
    compute_metrics,
    topk_from_similarity,
)
from reign.encoders.loco import LOCO_SUBTASKS, load_loco_subtask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_loco")


def parse_args():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--baseline",
        help="Dense baseline id from reign.encoders.dense.BASELINES (mutually exclusive with --reign-checkpoint)",
    )
    mode.add_argument(
        "--reign-checkpoint",
        help="Path or HF id of a trained REIGN model (mutually exclusive with --baseline)",
    )
    p.add_argument("--gn-model", help="Required with --reign-checkpoint (e.g. thenlper/gte-small)")
    p.add_argument("--gn-chunk-size", type=int, default=512)
    p.add_argument("--gn-stride", type=int, default=512)
    p.add_argument(
        "--subtask",
        default="all",
        help=f"One of {LOCO_SUBTASKS}, or 'all' to loop every subtask.",
    )
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gn-batch-size", type=int, default=12)
    p.add_argument(
        "--torch-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="Dense-baseline compute dtype. fp16 is the MTEB/BEIR convention.",
    )
    p.add_argument(
        "--protocol",
        choices=["truncate", "chunk_pool"],
        default=None,
        help="Override the dense baseline's default inference protocol.",
    )
    p.add_argument("--device", default=None, help="cuda / cpu (auto-detect if unset)")
    p.add_argument("--output-dir", default="results/loco")
    p.add_argument(
        "--tag",
        required=True,
        help="Short tag used in the output filename (e.g. 'reign-gte-small-s512').",
    )
    p.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Smoke-test cap on query count per subtask (omit for full eval).",
    )
    return p.parse_args()


def _build_encoder(args):
    if args.baseline:
        from reign.encoders.dense import BASELINES, DenseEncoder

        if args.baseline not in BASELINES:
            raise SystemExit(f"unknown baseline {args.baseline!r}; pick from {sorted(BASELINES)}")
        cfg = BASELINES[args.baseline]
        if args.protocol:
            from dataclasses import replace

            cfg = replace(cfg, inference_protocol=args.protocol)
        if args.torch_dtype:
            from dataclasses import replace

            cfg = replace(cfg, torch_dtype=args.torch_dtype)
        encoder = DenseEncoder(cfg, device=args.device)
        return encoder, cfg.display_name, {
            "kind": "dense_baseline",
            "model_name_or_path": cfg.model_name_or_path,
            "inference_protocol": cfg.inference_protocol,
            "torch_dtype": cfg.torch_dtype,
        }

    if not args.gn_model:
        raise SystemExit("--gn-model is required with --reign-checkpoint")
    from reign.encoders.reign import ReignBaselineEncoder

    encoder = ReignBaselineEncoder(
        checkpoint_path=args.reign_checkpoint,
        gn_model=args.gn_model,
        chunk_size=args.gn_chunk_size,
        stride=args.gn_stride,
        device=args.device,
        gn_batch_size=args.gn_batch_size,
    )
    return encoder, encoder.name, {
        "kind": "reign",
        "checkpoint": args.reign_checkpoint,
        "gn_model": args.gn_model,
        "chunk_size": args.gn_chunk_size,
        "stride": args.gn_stride,
        "n_params": int(encoder.n_params),
        "hidden_size": int(encoder.hidden_size),
    }


def _evaluate_subtask(encoder, subtask_name: str, top_k: int, batch_size: int,
                      max_queries: int | None):
    sub = load_loco_subtask(subtask_name)
    queries = sub.queries
    relevance = sub.relevance
    if max_queries is not None and max_queries < len(queries):
        logger.warning("Capping queries to %d (was %d) — smoke run", max_queries, len(queries))
        queries = queries[:max_queries]
        relevance = relevance[:max_queries]

    logger.info("Encoding %d corpus docs (%s) ...", len(sub.corpus), subtask_name)
    corpus_emb = encoder.encode(sub.corpus, batch_size=batch_size)
    logger.info("Encoding %d queries (%s) ...", len(queries), subtask_name)
    query_emb = encoder.encode(queries, batch_size=batch_size)

    sims = query_emb @ corpus_emb.T
    top_indices, top_scores = topk_from_similarity(sims, k=top_k)
    metrics = compute_metrics(top_indices, relevance, k=top_k)
    # LoCo indexes queries/corpus positionally rather than by id, so synthesise
    # ids that carry the position: downstream analyses (e.g. binning a query by
    # the chunk count of its relevant document) need to map back into the corpus.
    q_meta = [{"_id": f"{subtask_name}:q{i}"} for i in range(relevance.shape[0])]
    c_meta = [{"_id": f"{subtask_name}:d{j}"} for j in range(relevance.shape[1])]
    per_query = build_per_query_payload(
        top_indices, top_scores, relevance, q_meta, c_meta, k=top_k
    )
    for i, qid in enumerate(m["_id"] for m in q_meta):
        rel_positions = np.flatnonzero(relevance[i] > 0)
        per_query[qid]["relevant"] = [f"{subtask_name}:d{int(j)}" for j in rel_positions]
    return {
        "subtask": subtask_name,
        "n_queries": int(relevance.shape[0]),
        "n_corpus": int(relevance.shape[1]),
        "n_positive_pairs": int(relevance.sum()),
        "top_k": int(top_k),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "per_query": per_query,
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder, encoder_name, encoder_meta = _build_encoder(args)
    logger.info("Encoder: %s (%s)", encoder_name, encoder_meta.get("kind"))

    subtasks = list(LOCO_SUBTASKS) if args.subtask == "all" else [args.subtask]
    summary = []
    for sub_name in subtasks:
        out_path = out_dir / f"loco_{args.tag}_{sub_name}.json"
        if out_path.exists():
            logger.info("[skip] %s already present", out_path)
            with open(out_path) as fh:
                summary.append(json.load(fh))
            continue
        result = _evaluate_subtask(
            encoder, sub_name,
            top_k=args.top_k,
            batch_size=args.batch_size,
            max_queries=args.max_queries,
        )
        result["encoder"] = encoder_name
        result["encoder_meta"] = encoder_meta
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        logger.info("Wrote %s", out_path)
        summary.append(result)

    # Brief table at the end. nDCG@k is the LoCo headline.
    k = args.top_k
    ndcg_key = f"nDCG@{k}"
    print(f"\n# LoCo summary (encoder={encoder_name}, top_k={k})")
    print(f"| subtask                          | n_q  | n_corp | {ndcg_key:>10} |")
    print(f"|----------------------------------|------|--------|------------|")
    ndcg_vals = []
    for r in summary:
        v = r["metrics"].get(ndcg_key, 0.0)
        ndcg_vals.append(v)
        print(f"| {r['subtask']:32s} | {r['n_queries']:>4} | {r['n_corpus']:>6} | {v*100:>10.2f} |")
    if ndcg_vals:
        avg = sum(ndcg_vals) / len(ndcg_vals)
        print(f"| {'AVERAGE':32s} | {'':>4} | {'':>6} | {avg*100:>10.2f} |")


if __name__ == "__main__":
    main()
