"""
Shared dataset-loading + metric helpers for baseline evaluation runners.

Used by `scripts/evaluate_sparse_baselines.py` and
`scripts/evaluate_dense_baselines.py` (and any future runner) so the
data shape, relevance convention, and metric definitions stay in one
place.

Consumes the BEIR/MTEB-style IR dataset (e.g.
`devrim/goodwiki_long_synthetic_ir`) which exposes three configs:

    corpus   — single split "corpus"  (all docs)
    queries  — single split "queries" (all queries)
    default  — splits "train" / "val" / "test"  (qrels: query-id, corpus-id, score)

The returned `query_meta` is restricted to queries that appear in the chosen
qrels split; the corpus is always the full pool (a query in any split competes
against every corpus doc, with cross-split synthetics serving as hard negatives).

Relevance scores follow BEIR convention and are passed through unchanged:
    score = 2  → fully relevant (a paraphrase / "pair" of the query)
    score = 1  → partially relevant ("distractor" — topical overlap only)
    score = 0  → irrelevant (default for any (query, doc) not in qrels)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def build_query_corpus(
    dataset: str,
    qrels_split: str,
) -> tuple[list[str], list[dict], list[str], list[dict]]:
    """Load the IR dataset and return text + metadata for queries (in `qrels_split`)
    and the full corpus.

    Returns: (query_texts, query_meta, corpus_texts, corpus_meta).

    `query_meta[i]` carries `{"_id": <query-id>}` so the relevance matrix can be
    aligned to the corpus by id rather than positional `dataset_idx` (the old
    convention, which doesn't survive the IR layout's shared pools).

    `corpus_meta[j]` carries `{"_id": <corpus-id>}` for the same reason.
    """
    import os

    import datasets  # local import: keeps test collection cheap

    def _load_split(config: str, split: str):
        """Resolve a (config, split) pair across two on-disk layouts.

        Hub-style / parquet-backed datasets (e.g. ``devrim/goodwiki_long_synthetic_ir``)
        load via ``datasets.load_dataset(dataset, config, split=split)``. Local
        dataset adapters write each config as an HF ``DatasetDict.save_to_disk``
        directory (a ``dataset_dict.json`` plus per-split arrow shards) which
        ``load_dataset`` does not parse; for that case fall back to
        ``load_from_disk``. We detect the directory layout by looking for the
        sentinel file rather than trying to parse and catch exceptions, so the
        primary path stays fast for Hub datasets.
        """
        if isinstance(dataset, str) and os.path.isdir(os.path.join(dataset, config)) and os.path.isfile(
            os.path.join(dataset, config, "dataset_dict.json")
        ):
            return datasets.load_from_disk(os.path.join(dataset, config))[split]
        return datasets.load_dataset(dataset, config, split=split)

    logger.info("Loading %s queries (full pool)", dataset)
    queries_full = _load_split("queries", "queries")
    logger.info("Loading %s corpus (full pool)", dataset)
    corpus_full = _load_split("corpus", "corpus")
    logger.info("Loading %s qrels [%s]", dataset, qrels_split)
    qrels = _load_split("default", qrels_split)

    # Restrict queries to the ids that have at least one qrel in this split.
    split_query_ids = set(qrels["query-id"])
    logger.info(
        "qrels[%s] covers %d unique queries (out of %d in pool)",
        qrels_split,
        len(split_query_ids),
        len(queries_full),
    )

    query_texts: list[str] = []
    query_meta: list[dict] = []
    for q in queries_full:
        if q["_id"] not in split_query_ids:
            continue
        text = q.get("text", "")
        if not text:
            continue
        query_texts.append(text)
        query_meta.append({"_id": q["_id"]})

    corpus_texts: list[str] = []
    corpus_meta: list[dict] = []
    for c in corpus_full:
        text = c.get("text", "")
        if not text:
            continue
        corpus_texts.append(text)
        corpus_meta.append({"_id": c["_id"]})

    logger.info("Built %d queries and %d corpus docs", len(query_texts), len(corpus_texts))
    return query_texts, query_meta, corpus_texts, corpus_meta, qrels


def build_relevance(
    query_meta: Sequence[dict],
    corpus_meta: Sequence[dict],
    qrels,
) -> np.ndarray:
    """Returns (n_queries, n_corpus) graded relevance matrix.

    Reads (query-id, corpus-id, score) triples from `qrels` (a HF Dataset or any
    iterable of dicts) and projects them onto the positional indices defined by
    `query_meta` / `corpus_meta`. Any (q, c) pair not in qrels is implicitly 0.
    """
    qid_to_row = {m["_id"]: i for i, m in enumerate(query_meta)}
    cid_to_col = {m["_id"]: j for j, m in enumerate(corpus_meta)}

    n_q, n_c = len(query_meta), len(corpus_meta)
    rel = np.zeros((n_q, n_c), dtype=np.int8)
    skipped = 0
    for r in qrels:
        i = qid_to_row.get(r["query-id"])
        j = cid_to_col.get(r["corpus-id"])
        if i is None or j is None:
            skipped += 1
            continue
        rel[i, j] = int(r["score"])
    if skipped:
        logger.warning("build_relevance: %d qrel rows skipped (id missing from pool)", skipped)
    return rel


# Graded relevance convention used across REIGN baselines:
#
#     score = 2  → fully relevant (a paraphrase / "pair" of the query)
#     score = 1  → partially relevant ("distractor" — topical overlap only)
#     score = 0  → irrelevant
#
# All metrics below honour this 2 : 1 : 0 weighting. Concretely we map score
# to a linear gain in [0, 1] via gain = score / MAX_GAIN where MAX_GAIN is the
# largest relevance grade observed in the qrels (here 2). Distractors therefore
# count for *half* a hit, not a full one — fixing the silent overstatement that
# binary metrics produce on this dataset.

MAX_GAIN_DEFAULT = 2  # mirrors the qrels schema (2 = pair, 1 = distractor)


def _gains(rel: np.ndarray, max_gain: int) -> np.ndarray:
    """Linear gain in [0, 1]: rel/max_gain. score=2 → 1.0, score=1 → 0.5, 0 → 0."""
    return rel.astype(np.float64) / float(max_gain)


def precision_at_k(rel_topk: np.ndarray, k: int, max_gain: int = MAX_GAIN_DEFAULT) -> float:
    """Graded precision@k: mean over queries of (sum of gains in top-k) / k."""
    g = _gains(rel_topk[:, :k], max_gain)
    return float(np.mean(g.sum(axis=1) / k))


def recall_at_k(
    rel_topk: np.ndarray,
    total_gain: np.ndarray,
    k: int,
    max_gain: int = MAX_GAIN_DEFAULT,
) -> float:
    """Graded recall@k: gains found in top-k divided by total possible gain.

    `total_gain[i]` is the sum of (rel/max_gain) across every relevant doc for
    query i. With 1 pair (rel=2) + 2 distractors (rel=1) it equals 1.0 + 0.5
    + 0.5 = 2.0, so a perfect retriever (all 3 in top-k) scores 1.0 and
    finding only the pair scores 0.5.
    """
    found = _gains(rel_topk[:, :k], max_gain).sum(axis=1)
    safe = np.where(total_gain == 0, 1.0, total_gain)
    return float(np.mean(np.minimum(found, total_gain) / safe))


def average_precision(rel_row: np.ndarray, k: int, threshold: int = 1) -> float:
    """Standard binary AP@k: any (rel >= threshold) is treated as a hit.

    Kept binary for BEIR/MTEB comparability. The 2 : 1 : 0 weighting REIGN
    cares about already shows up in P@k and R@k; bringing it into AP creates
    a metric that doesn't reach 1.0 on perfect ordering whenever the qrels
    are non-uniform, which is more confusing than it is informative.
    """
    rel = (rel_row[:k] >= threshold).astype(np.float64)
    if rel.sum() == 0:
        return 0.0
    cum_hits = np.cumsum(rel)
    precision_at_each = cum_hits / np.arange(1, k + 1)
    return float((precision_at_each * rel).sum() / rel.sum())


def map_at_k(rel_topk: np.ndarray, k: int) -> float:
    """Mean (binary) AP@k across queries."""
    return float(np.mean([average_precision(rel_topk[i], k) for i in range(rel_topk.shape[0])]))


def ndcg_at_k(rel_topk: np.ndarray, ideal_rel: list[np.ndarray], k: int) -> float:
    """nDCG@k with **exponential** gain (gain = 2^rel − 1), matching the
    BEIR/MTEB convention (pytrec_eval ``ndcg_cut.k``). This deliberately uses
    a different weighting than the other metrics: the others honour REIGN's
    explicit 2 : 1 : 0 design, while nDCG stays BEIR-comparable so the column
    can stand next to numbers reported by other papers. The implicit
    pair-vs-distractor ratio under exponential gain is 3 : 1, not 2 : 1.
    """
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    gains = (np.power(2.0, rel_topk[:, :k]) - 1.0) * discounts
    dcg = gains.sum(axis=1)

    ideal = np.zeros((rel_topk.shape[0], k), dtype=np.float64)
    for i, row in enumerate(ideal_rel):
        take = row[:k]
        ideal[i, : len(take)] = take
    igains = (np.power(2.0, ideal) - 1.0) * discounts
    idcg = igains.sum(axis=1)
    return float(np.mean(np.where(idcg == 0, 0, dcg / np.where(idcg == 0, 1, idcg))))


def per_query_metrics(
    top_indices: np.ndarray,
    relevance: np.ndarray,
    k: int,
    max_gain: int | None = None,
) -> dict[str, np.ndarray]:
    """Same metric suite as :func:`compute_metrics`, but **unaggregated**.

    Returns one float array of length ``n_queries`` per metric name. Averaging
    each array reproduces :func:`compute_metrics` exactly — that function is
    implemented as the mean of this one, so the two can never drift apart.

    Per-query scores are what significance testing needs: a paired bootstrap
    over two systems' per-query nDCG requires the individual values, which the
    aggregate throws away.
    """
    n_q = top_indices.shape[0]
    # Sentinel index -1 (from drop_self_matches when a query has fewer than
    # target_k non-self hits) means "no doc here, zero gain". Replace with 0
    # for the gather and mask the result back to zero.
    safe_top = np.where(top_indices >= 0, top_indices, 0)
    rel_topk = np.take_along_axis(relevance, safe_top, axis=1)
    rel_topk = np.where(top_indices >= 0, rel_topk, 0)
    if max_gain is None:
        max_gain = int(max(MAX_GAIN_DEFAULT, relevance.max())) if relevance.size else MAX_GAIN_DEFAULT
    total_gain = _gains(relevance, max_gain).sum(axis=1)

    p1 = _gains(rel_topk[:, :1], max_gain).sum(axis=1) / 1.0
    pk = _gains(rel_topk[:, :k], max_gain).sum(axis=1) / k

    found = _gains(rel_topk[:, :k], max_gain).sum(axis=1)
    safe = np.where(total_gain == 0, 1.0, total_gain)
    rk = np.minimum(found, total_gain) / safe

    ap = np.array([average_precision(rel_topk[i], k) for i in range(n_q)], dtype=np.float64)

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = ((np.power(2.0, rel_topk[:, :k]) - 1.0) * discounts).sum(axis=1)
    ideal = np.zeros((n_q, k), dtype=np.float64)
    for i in range(n_q):
        take = np.sort(relevance[i])[::-1][:k]
        ideal[i, : len(take)] = take
    idcg = ((np.power(2.0, ideal) - 1.0) * discounts).sum(axis=1)
    ndcg = np.where(idcg == 0, 0.0, dcg / np.where(idcg == 0, 1.0, idcg))

    return {
        "P@1": p1,
        f"P@{k}": pk,
        f"R@{k}": rk,
        f"MAP@{k}": ap,
        f"nDCG@{k}": ndcg,
    }


def compute_metrics(
    top_indices: np.ndarray,
    relevance: np.ndarray,
    k: int,
    max_gain: int | None = None,
) -> dict[str, float]:
    """Graded retrieval metric suite for REIGN.

    All metrics honour the 2 : 1 : 0 (pair : distractor : irrelevant) weighting:
    distractors are counted as half a hit, never as a full hit. ``max_gain``
    defaults to the highest relevance grade observed in ``relevance``.
    """
    return {
        name: float(values.mean())
        for name, values in per_query_metrics(top_indices, relevance, k, max_gain).items()
    }


def build_per_query_payload(
    top_indices: np.ndarray,
    top_scores: np.ndarray | None,
    relevance: np.ndarray,
    query_meta: Sequence[dict],
    corpus_meta: Sequence[dict],
    k: int,
    max_gain: int | None = None,
    include_ranking: bool = True,
) -> dict[str, dict]:
    """Per-query metrics keyed by query id, optionally with the retrieved ranking.

    Storing the ranking (corpus ids + scores) alongside the scores makes every
    downstream statistic recomputable without re-encoding the corpus, which is
    the expensive half of an evaluation.
    """
    pq = per_query_metrics(top_indices, relevance, k, max_gain)
    payload: dict[str, dict] = {}
    for i, qmeta in enumerate(query_meta):
        entry: dict = {name: float(values[i]) for name, values in pq.items()}
        if include_ranking:
            ids = []
            for slot in range(top_indices.shape[1]):
                j = int(top_indices[i, slot])
                ids.append(corpus_meta[j]["_id"] if j >= 0 else None)
            entry["retrieved"] = ids
            if top_scores is not None:
                entry["scores"] = [round(float(s), 6) for s in top_scores[i]]
        payload[qmeta["_id"]] = entry
    return payload


def drop_self_matches(
    top_indices: np.ndarray,
    top_scores: np.ndarray | None,
    query_meta: Sequence[dict],
    corpus_meta: Sequence[dict],
    target_k: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Remove rank positions where ``corpus_meta[j]._id == query_meta[i]._id`` and
    shift the remaining hits up to fill the gap. Output shape is
    ``(n_queries, target_k)`` regardless of input width.

    Use this on retrievers run with ``top_k = target_k + buffer`` so there is
    headroom to drop a self-match at any rank. Positions that can't be filled
    (vanishingly rare: a query has fewer than ``target_k`` non-self matches in
    its retrieved set) are sentinel-padded with ``-1`` — ``compute_metrics``
    treats negative indices as zero-gain.

    Returns ``(new_indices, new_scores, n_dropped)`` where ``n_dropped`` is
    the total number of rank cells removed across all queries.
    """
    cid_to_idx: dict[str, set[int]] = {}
    for j, cmeta in enumerate(corpus_meta):
        cid_to_idx.setdefault(cmeta["_id"], set()).add(j)

    n_q = top_indices.shape[0]
    out_idx = np.full((n_q, target_k), -1, dtype=np.int64)
    out_sc = (
        np.zeros((n_q, target_k), dtype=top_scores.dtype) if top_scores is not None else None
    )
    n_dropped = 0
    for i, qmeta in enumerate(query_meta):
        self_positions = cid_to_idx.get(qmeta["_id"], set())
        kept = 0
        for slot in range(top_indices.shape[1]):
            if kept >= target_k:
                break
            j = int(top_indices[i, slot])
            if j in self_positions:
                n_dropped += 1
                continue
            out_idx[i, kept] = j
            if out_sc is not None:
                out_sc[i, kept] = top_scores[i, slot]
            kept += 1
    return out_idx, out_sc, n_dropped


def topk_from_similarity(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Given a (n_q, n_c) similarity matrix, return top-k indices and scores per row."""
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    for i in range(idx.shape[0]):
        order = np.argsort(-scores[i, idx[i]])
        idx[i] = idx[i, order]
    sc = np.take_along_axis(scores, idx, axis=1).astype(np.float32)
    return idx, sc
