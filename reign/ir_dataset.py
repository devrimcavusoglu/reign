"""Standard-IR contrastive fine-tuning dataset (query / positive / negative).

Replaces the GoodWiki-Long-Synthetic ``(original, synthetic_pair, distractors)``
shim for standard ``(query, corpus, qrels)`` IR datasets (e.g. DAPFAM,
GoodWiki-IR). Principles:

* **No stored-score refactor.** Datasets keep standard BEIR ``{0,1,2}`` qrels.
  Only the *training-time interpretation* is configurable, because ``score=1``
  is dataset-dependent:
    - ``score>=2``  → **positive** (always)
    - ``score==1``  → ``partial_policy``: ``soft_positive`` (graded soft
      positive at loss ``partial_weight``, e.g. a partially-relevant doc in a
      graded-relevance corpus) | ``negative`` (added to the negative pool,
      e.g. GoodWiki-IR topical distractor) | ``ignore``
    - ``score==0``  → **explicit provided negative** (e.g. DAPFAM's ~20/query)
    - absent        → implicit irrelevant → in-batch negative (mask-protected)
* **Use the dataset's own provided negatives** (``score==0``) as explicit
  negatives — not only in-batch over a positives-only qrels.
* **False-negative masking**: a doc that is relevant to an anchor (positive, or
  a ``soft_positive`` partial) must never be that anchor's negative. The
  per-anchor relevant-id set is carried to the loss via the collate metadata.

GN-cache roles are IR-meaningful: ``query`` / ``positive`` / ``partial`` /
``negative``. The frozen-GN HDF5 cache only round-trips the essential metadata
fields ``{article_id, reference_article_id, other_article_id, article_type,
dataset_idx}`` — so each doc's **corpus id is stored in ``article_id``** (queries
store the qid there) and the false-negative mask is built from ``article_id``
membership, never row/roll arithmetic.

Two modes:

* ``mode="train"`` → per ``__getitem__``: (query, sampled positive, K partials,
  M negatives, anchor_relevant_idset); collate via ``collate_cached_ir_data``;
  consumed by ``ReignIRLitModel``.
* ``mode="eval"``  → (query, sampled positive, K partials) **only** (no
  negatives), with GoodWiki-compatible metadata so the existing
  ``Evaluator``/``_create_relevance_matrix`` (an in-batch proxy for checkpoint
  selection — the authoritative metric is ``scripts/evaluate_reign.py``) works
  unchanged via the existing ``collate_cached_data``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from reign.dataset import ReignDataset, _cache_id

logger = logging.getLogger(__name__)

PARTIAL_POLICIES = ("soft_positive", "negative", "ignore")


class IRContrastiveDataset:
    """Per-query view of a ``(query, corpus, qrels)`` IR dataset.

    Groups qrels per query into positives (score>=2), partials (score==1) and
    explicit negatives (score==0), routing score==1 per ``partial_policy``.
    Queries with no positive (score>=2) are skipped. Exposes flat per-role
    text/metadata lists (for GN cache building) plus per-qid slot ranges and
    ``relevant_ids`` (positives ∪ soft-positive partials) for false-neg masking.
    """

    def __init__(
        self,
        dataset_name: str,
        qrels_split: str,
        partial_policy: str = "soft_positive",
        max_samples: Optional[int] = None,
    ):
        if partial_policy not in PARTIAL_POLICIES:
            raise ValueError(f"partial_policy must be one of {PARTIAL_POLICIES}")
        self.dataset_name = dataset_name
        self.qrels_split = qrels_split
        self.partial_policy = partial_policy

        queries = ReignDataset._load_split(dataset_name, "queries", "queries")
        corpus = ReignDataset._load_split(dataset_name, "corpus", "corpus")
        qrels = ReignDataset._load_split(dataset_name, "default", qrels_split)

        q_text = {str(r["_id"]): r.get("text", "") for r in queries}
        c_text = {str(r["_id"]): r.get("text", "") for r in corpus}

        # Group qrels per qid by grade.
        grouped: Dict[str, Dict[str, List[str]]] = {}
        for r in qrels:
            qid, cid = str(r["query-id"]), str(r["corpus-id"])
            score = int(r["score"])
            g = grouped.setdefault(qid, {"pos": [], "one": [], "neg": []})
            if score >= 2:
                g["pos"].append(cid)
            elif score == 1:
                g["one"].append(cid)
            else:
                g["neg"].append(cid)

        # Flat per-role lists + per-qid slot ranges.
        self.query_texts: List[str] = []
        self.query_meta: List[Dict[str, Any]] = []
        self.pos_texts: List[str] = []
        self.pos_meta: List[Dict[str, Any]] = []
        self.partial_texts: List[str] = []
        self.partial_meta: List[Dict[str, Any]] = []
        self.neg_texts: List[str] = []
        self.neg_meta: List[Dict[str, Any]] = []

        self.qids: List[str] = []
        self.pos_range: Dict[str, Tuple[int, int]] = {}
        self.partial_range: Dict[str, Tuple[int, int]] = {}
        self.neg_range: Dict[str, Tuple[int, int]] = {}
        self.relevant_ids: Dict[str, set] = {}

        n_skip_no_pos = n_skip_no_text = 0
        for qid in sorted(grouped):
            if max_samples is not None and len(self.qids) >= max_samples:
                break
            g = grouped[qid]
            positives = [c for c in g["pos"] if c_text.get(c)]
            if not positives:
                n_skip_no_pos += 1
                continue
            qtext = q_text.get(qid, "")
            if not qtext:
                n_skip_no_text += 1
                continue

            # Route score==1 per policy.
            if self.partial_policy == "soft_positive":
                partials, extra_neg = g["one"], []
            elif self.partial_policy == "negative":
                partials, extra_neg = [], g["one"]
            else:  # ignore
                partials, extra_neg = [], []
            partials = [c for c in partials if c_text.get(c)]
            negatives = [c for c in (g["neg"] + extra_neg) if c_text.get(c)]

            q_idx = len(self.query_texts)
            self.query_texts.append(qtext)
            self.query_meta.append(
                {
                    "article_id": qid,
                    "reference_article_id": "",
                    "other_article_id": "",
                    "article_type": "query",
                    "dataset_idx": q_idx,
                }
            )

            ps = len(self.pos_texts)
            for cid in positives:
                self.pos_texts.append(c_text[cid])
                self.pos_meta.append(
                    {
                        "article_id": cid,
                        "reference_article_id": qid,
                        "other_article_id": "",
                        "article_type": "pair",
                        "dataset_idx": len(self.pos_texts) - 1,
                    }
                )
            self.pos_range[qid] = (ps, len(self.pos_texts))

            prs = len(self.partial_texts)
            for cid in partials:
                self.partial_texts.append(c_text[cid])
                self.partial_meta.append(
                    {
                        "article_id": cid,
                        "reference_article_id": qid,
                        "other_article_id": "",
                        "article_type": "distractor",  # != "pair" → partial in Evaluator
                        "dataset_idx": len(self.partial_texts) - 1,
                    }
                )
            self.partial_range[qid] = (prs, len(self.partial_texts))

            ns = len(self.neg_texts)
            for cid in negatives:
                self.neg_texts.append(c_text[cid])
                self.neg_meta.append(
                    {
                        "article_id": cid,
                        "reference_article_id": "",  # invisible to Evaluator
                        "other_article_id": "",
                        "article_type": "negative",
                        "dataset_idx": len(self.neg_texts) - 1,
                    }
                )
            self.neg_range[qid] = (ns, len(self.neg_texts))

            # Relevant set for false-neg masking: positives always; partials
            # only when treated as soft positives (negative/ignore-policy
            # score==1 docs are legitimately allowed in the negative pool).
            rel = set(positives)
            if self.partial_policy == "soft_positive":
                rel |= set(partials)
            self.relevant_ids[qid] = rel
            self.qids.append(qid)

        logger.info(
            "IRContrastiveDataset[%s/%s] policy=%s: %d queries "
            "(pos=%d partial=%d neg=%d); skipped %d no-positive, %d no-text",
            dataset_name,
            qrels_split,
            partial_policy,
            len(self.qids),
            len(self.pos_texts),
            len(self.partial_texts),
            len(self.neg_texts),
            n_skip_no_pos,
            n_skip_no_text,
        )

    def __len__(self) -> int:
        return len(self.qids)


def _pad_and_stack(emb_list, mask_list):
    """Pad variable-chunk embeddings to a dense (B, max_chunks, H) batch.

    Identical semantics to the nested helper in ``dataset.collate_cached_data``.
    """
    if not emb_list:
        return torch.empty(0), torch.empty(0)
    valid_emb, valid_mask = [], []
    for emb, mask in zip(emb_list, mask_list):
        if emb.shape[0] > 0:
            valid_emb.append(emb)
            if mask.shape[0] != emb.shape[0]:
                mask = torch.ones(emb.shape[0], dtype=torch.int64, device=emb.device)
            valid_mask.append(mask)
    if not valid_emb:
        ref = next((e for e in emb_list if e.numel() > 0), None)
        device = ref.device if ref is not None else torch.device("cpu")
        hidden = ref.shape[1] if (ref is not None and ref.dim() > 1) else 768
        return (
            torch.empty((1, 0, hidden), device=device),
            torch.empty((1, 0), dtype=torch.int64, device=device),
        )
    max_len = max(e.shape[0] for e in valid_emb)
    hidden = valid_emb[0].shape[1]
    bs = len(valid_emb)
    device = valid_emb[0].device
    padded_emb = torch.zeros(bs, max_len, hidden, device=device, dtype=valid_emb[0].dtype)
    padded_mask = torch.zeros(bs, max_len, device=device, dtype=torch.int64)
    for i, (e, m) in enumerate(zip(valid_emb, valid_mask)):
        n = e.shape[0]
        padded_emb[i, :n, :] = e
        padded_mask[i, :n] = m
    return padded_emb, padded_mask


class IRContrastiveCachedDataset(Dataset):
    """Cached per-query IR dataset (see module docstring).

    ``mode="train"`` → 5-tuple ``(query, positive, [partials], [negatives],
    relevant_idset)`` for ``collate_cached_ir_data`` + ``ReignIRLitModel``.
    ``mode="eval"`` → 3-tuple ``(query, positive, [partials])`` (no negatives)
    for the existing ``collate_cached_data`` + ``Evaluator``.
    """

    def __init__(
        self,
        dataset_name: str,
        qrels_split: str,
        feature_extractor,
        partial_policy: str = "soft_positive",
        n_partials_per_sample: int = 0,
        n_negatives_per_sample: int = 0,
        mode: str = "train",
        max_samples: Optional[int] = None,
    ):
        if feature_extractor is None:
            raise ValueError("feature_extractor is required for IRContrastiveCachedDataset")
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.fe = feature_extractor
        self.mode = mode
        self.n_partials = n_partials_per_sample
        self.n_negatives = n_negatives_per_sample
        self.base = IRContrastiveDataset(
            dataset_name, qrels_split, partial_policy=partial_policy, max_samples=max_samples
        )

        cs = getattr(feature_extractor, "chunk_size", None)
        st = getattr(feature_extractor, "stride", None)
        self._cid = {
            role: _cache_id(dataset_name, qrels_split, role, max_samples, cs, st)
            for role in ("query", "positive", "partial", "negative")
        }

        self._build_cache("query", self.base.query_texts, self.base.query_meta)
        self._build_cache("positive", self.base.pos_texts, self.base.pos_meta)
        if self.base.partial_texts:
            self._build_cache("partial", self.base.partial_texts, self.base.partial_meta)
        if mode == "train" and self.base.neg_texts:
            self._build_cache("negative", self.base.neg_texts, self.base.neg_meta)

        # Every emitted row must carry exactly K partials / M provided negatives
        # so the loss can reshape ``(B·K)`` / pool ``(B, B_neg)`` uniformly. Drop
        # qids that cannot supply a required role (e.g. drop zero-partial
        # queries). ``n_partials``/``n_negatives`` == 0 ⇒ no filter
        # for that role.
        items, drop_p, drop_n = [], 0, 0
        for qid in self.base.qids:
            if self.n_partials > 0:
                a, b = self.base.partial_range.get(qid, (0, 0))
                if b - a == 0:
                    drop_p += 1
                    continue
            if mode == "train" and self.n_negatives > 0:
                a, b = self.base.neg_range.get(qid, (0, 0))
                if b - a == 0:
                    drop_n += 1
                    continue
            items.append(qid)
        self._items = items
        # query cache slot = position of qid in base.qids (build order).
        self._qid_to_qslot = {qid: i for i, qid in enumerate(self.base.qids)}
        if drop_p or drop_n:
            logger.info(
                "IRContrastiveCachedDataset[%s]: dropped %d qids w/o partials, "
                "%d qids w/o provided negatives (uniform-row requirement)",
                mode,
                drop_p,
                drop_n,
            )

    def _build_cache(self, role: str, texts: List[str], meta: List[Dict]) -> None:
        if not texts:
            return
        cid = self._cid[role]
        if self.fe.cache.has_cache(
            self.fe.model_name_or_path, self.fe.chunk_size, cid, stride=self.fe.stride
        ):
            return
        logger.info("Computing %s-side embeddings (%s)", role, cid)
        self.fe.compute_and_cache_dataset_embeddings_with_metadata(texts, cid, meta)

    def _fetch(self, role: str, slots: List[int]):
        got = self.fe.get_cached_embeddings_with_metadata(self._cid[role], slots)
        if got is None:
            return []
        emb_list, meta_list = got
        return [(e, m, md) for (e, m), md in zip(emb_list, meta_list)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        qid = self._items[idx]
        rng = np.random.default_rng(seed=idx)

        qslot = self._qid_to_qslot[qid]
        query = self._fetch("query", [qslot])
        if not query:
            raise ValueError(f"No cached query embedding at slot {qslot} (qid={qid})")
        query_item = query[0]

        ps, pe = self.base.pos_range[qid]
        pos_slot = ps + int(rng.integers(0, pe - ps))
        positive = self._fetch("positive", [pos_slot])
        if not positive:
            raise ValueError(f"No cached positive at slot {pos_slot} (qid={qid})")
        positive_item = positive[0]

        partial_items: List[Tuple] = []
        if self.n_partials > 0:
            a, b = self.base.partial_range.get(qid, (0, 0))
            navail = b - a
            if navail > 0:
                K = self.n_partials
                repl = navail < K
                offs = rng.choice(navail, size=K, replace=repl).tolist()
                partial_items = self._fetch("partial", [a + int(o) for o in offs])

        if self.mode == "eval":
            return query_item, positive_item, partial_items

        negative_items: List[Tuple] = []
        if self.n_negatives > 0:
            a, b = self.base.neg_range.get(qid, (0, 0))
            navail = b - a
            if navail > 0:
                M = self.n_negatives
                repl = navail < M
                offs = rng.choice(navail, size=M, replace=repl).tolist()
                negative_items = self._fetch("negative", [a + int(o) for o in offs])

        return (
            query_item,
            positive_item,
            partial_items,
            negative_items,
            self.base.relevant_ids[qid],
        )


def collate_cached_ir_data(batch):
    """Collate ``mode="train"`` 5-tuples into padded role tensors + metadata.

    Returns ``(q_emb, q_mask, p_emb, p_mask, par_emb, par_mask, neg_emb,
    neg_mask, metadata)`` where ``metadata`` carries per-role metadata lists
    plus ``anchor_qids`` and ``anchor_relevant_idsets`` (length B, aligned to
    the query rows) for the loss's false-negative mask.
    """
    q_emb, q_mask, q_meta = [], [], []
    p_emb, p_mask, p_meta = [], [], []
    par_emb, par_mask, par_meta = [], [], []
    neg_emb, neg_mask, neg_meta = [], [], []
    anchor_qids, anchor_relevant_idsets = [], []

    for (qe, qm, qmd), (pe, pm, pmd), partials, negatives, rel in batch:
        q_emb.append(qe)
        q_mask.append(qm)
        q_meta.append(qmd)
        p_emb.append(pe)
        p_mask.append(pm)
        p_meta.append(pmd)
        for e, m, md in partials:
            par_emb.append(e)
            par_mask.append(m)
            par_meta.append(md)
        for e, m, md in negatives:
            neg_emb.append(e)
            neg_mask.append(m)
            neg_meta.append(md)
        anchor_qids.append(qmd.get("article_id", ""))
        anchor_relevant_idsets.append(set(rel))

    bq_emb, bq_mask = _pad_and_stack(q_emb, q_mask)
    bp_emb, bp_mask = _pad_and_stack(p_emb, p_mask)
    bpar_emb, bpar_mask = _pad_and_stack(par_emb, par_mask)
    bneg_emb, bneg_mask = _pad_and_stack(neg_emb, neg_mask)

    metadata = {
        "original_metadata": q_meta,
        "synthetic_metadata": p_meta,
        "partial_metadata": par_meta,
        "negative_metadata": neg_meta,
        "anchor_qids": anchor_qids,
        "anchor_relevant_idsets": anchor_relevant_idsets,
    }
    return bq_emb, bq_mask, bp_emb, bp_mask, bpar_emb, bpar_mask, bneg_emb, bneg_mask, metadata
