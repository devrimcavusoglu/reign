"""Dataset module for REIGN training, consuming the BEIR/MTEB-style IR layout
(e.g. ``devrim/goodwiki_long_synthetic_ir``).

The repo exposes three configs:

* ``corpus``   — single split ``corpus``  with all synthetic docs
* ``queries``  — single split ``queries`` with all original (Wikipedia) articles
* ``default``  — splits ``train`` / ``val`` / ``test`` of qrels

Each query has exactly three relevant corpus docs in its qrels split: one
``score=2`` (a paraphrase / "pair") and two ``score=1`` ("distractor"s — topical
overlap only). The training contract therefore yields one instance per query in
the chosen qrels split, formatted as
``(original_article, synthetic_pair, [distractor_1, distractor_2])``.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

try:
    from reign.utils import profile_function
except ImportError:

    def profile_function(name=None):
        def decorator(func):
            return func

        return decorator


def _cache_id(
    dataset_name: str,
    qrels_split: str,
    role: str,
    max_samples: Optional[int],
    chunk_size: Optional[int] = None,
    stride: Optional[int] = None,
) -> str:
    """Stable per-(dataset, split, role, chunk_size, stride) cache identifier used
    by the feature extractor's HDF5 cache. ``role`` is one of ``original`` /
    ``synthetic`` / ``distractors``.

    chunk_size and stride must be in the key: changing either alters the chunk
    boundaries the GN sees, so embeddings are different per-text and the cache
    must be regenerated. The previous version omitted them and silently reused
    stale caches across stride values — fatal for the stride ablation.
    """
    parts = [dataset_name, qrels_split, role]
    if chunk_size is not None:
        parts.append(f"cs_{chunk_size}")
    if stride is not None:
        parts.append(f"st_{stride}")
    if max_samples is not None:
        parts.append(f"max_{max_samples}")
    base = "_".join(parts)
    digest = hashlib.md5(base.encode()).hexdigest()[:16]
    return f"{role}_{digest}_{base.replace('/', '_')}"


class ReignDataset(Dataset):
    """One training instance per query in the chosen qrels split.

    Args:
        dataset_name: HF dataset id in BEIR/MTEB layout.
        qrels_split: ``train`` / ``val`` / ``test`` — selects which queries to use.
        max_samples: Optional cap on the number of training instances (debug).
        transform: Optional text transform applied to all texts.
    """

    @staticmethod
    def _load_split(dataset_name: str, config: str, split: str):
        """Load (config, split) supporting both HF Hub-style and
        ``DatasetDict.save_to_disk`` on-disk layouts.

        Hub datasets (e.g. ``devrim/goodwiki_long_synthetic_ir``) use
        ``datasets.load_dataset``. Locally-built datasets typically save each
        config as a ``DatasetDict`` directory, which ``load_dataset`` cannot
        parse — fall back to ``load_from_disk`` when we see the sentinel file.
        Mirrors the helper in ``reign.encoders.eval_utils``.
        """
        import os

        candidate = os.path.join(dataset_name, config)
        if (
            isinstance(dataset_name, str)
            and os.path.isdir(candidate)
            and os.path.isfile(os.path.join(candidate, "dataset_dict.json"))
        ):
            return datasets.load_from_disk(candidate)[split]
        return datasets.load_dataset(dataset_name, config, split=split)

    def __init__(
        self,
        dataset_name: str = "devrim/goodwiki_long_synthetic_ir",
        qrels_split: str = "train",
        max_samples: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        self.dataset_name = dataset_name
        self.qrels_split = qrels_split
        self.max_samples = max_samples
        self.transform = transform

        logger.info("Loading queries (full pool) from %s", dataset_name)
        queries_full = self._load_split(dataset_name, "queries", "queries")
        logger.info("Loading corpus (full pool) from %s", dataset_name)
        corpus_full = self._load_split(dataset_name, "corpus", "corpus")
        logger.info("Loading qrels[%s] from %s", qrels_split, dataset_name)
        qrels = self._load_split(dataset_name, "default", qrels_split)

        self._queries_by_id: Dict[str, Dict[str, Any]] = {q["_id"]: q for q in queries_full}
        self._corpus_by_id: Dict[str, Dict[str, Any]] = {c["_id"]: c for c in corpus_full}

        # Group qrels per query-id, separating positives (score=2) and partials (score=1).
        # We materialise one training instance per (qid, positive) — for GoodWiki this
        # is a 1:1 mapping (one positive per query), but for multi-positive datasets where
        # a query has K positives we emit K instances per query. Other positives are
        # excluded from each instance's distractor list (treating them as distractors
        # would teach the model to push real relevant docs apart, the bug the original
        # "first positive wins, rest become distractors" branch introduced).
        groups: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: {"positives": [], "partials": []}
        )
        for r in qrels:
            qid, cid, score = r["query-id"], r["corpus-id"], int(r["score"])
            if score >= 2:
                groups[qid]["positives"].append(cid)
            else:
                groups[qid]["partials"].append(cid)

        self._instances: List[Tuple[str, str, List[str]]] = []
        n_multi_positive = 0
        for qid in sorted(groups):
            grp = groups[qid]
            positives = grp["positives"]
            partials = grp["partials"]
            if not positives:
                if not partials:
                    raise ValueError(f"Query {qid} has no qrels at all")
                logger.warning(
                    "Query %s has no score=2 pair; falling back to first partial", qid
                )
                self._instances.append((qid, partials[0], partials[1:]))
                continue
            if len(positives) > 1:
                n_multi_positive += 1
            for pos_cid in positives:
                self._instances.append((qid, pos_cid, list(partials)))
        if n_multi_positive:
            logger.info(
                "%d queries had multiple positives — expanded into one instance per (qid, positive)",
                n_multi_positive,
            )

        if max_samples is not None and max_samples < len(self._instances):
            logger.info("Limiting dataset to %d instances", max_samples)
            self._instances = self._instances[:max_samples]

        logger.info(
            "ReignDataset[%s/%s]: %d training instances (each = 1 original + 1 pair + N distractors)",
            dataset_name,
            qrels_split,
            len(self._instances),
        )

    def get_dataset_identifiers(self) -> Tuple[str, str, str]:
        """Cache identifiers for original / synthetic / distractor embeddings."""
        return (
            _cache_id(self.dataset_name, self.qrels_split, "original", self.max_samples),
            _cache_id(self.dataset_name, self.qrels_split, "synthetic", self.max_samples),
            _cache_id(self.dataset_name, self.qrels_split, "distractors", self.max_samples),
        )

    def get_all_texts(self) -> Tuple[List[str], List[str], List[str]]:
        """Materialise (originals, pairs, distractors) for cache pre-population.

        Distractors are returned as a flat list across all instances; the order
        matches an instance-by-instance traversal so positional indexing into
        the cache is well-defined.
        """
        originals: List[str] = []
        pairs: List[str] = []
        distractors: List[str] = []
        for qid, pair_cid, dist_cids in self._instances:
            originals.append(self._queries_by_id[qid].get("text", ""))
            pairs.append(self._corpus_by_id[pair_cid].get("text", ""))
            for cid in dist_cids:
                distractors.append(self._corpus_by_id[cid].get("text", ""))
        return originals, pairs, distractors

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        qid, pair_cid, dist_cids = self._instances[idx]
        query_row = self._queries_by_id[qid]
        pair_row = self._corpus_by_id[pair_cid]

        original_text = query_row.get("text", "")
        pair_text = pair_row.get("text", "")

        original_metadata = {
            "dataset_idx": idx,
            "_id": qid,
            "article_id": qid,  # legacy key used by reign/eval.py._create_relevance_matrix
            "title": query_row.get("title", ""),
        }
        # ``reference_article_id`` links a search-side doc back to its query for
        # the trainer-side relevance matrix in ``reign/eval.py``. GoodWiki-synthetic
        # corpus docs carry this field natively (each synthetic was generated as
        # a pair/distractor of a specific original article); for IR-format
        # datasets where the corpus is unannotated and the
        # relationship lives only in qrels, we synthesise it here from the qrel
        # row's qid. Always preferring the qrels-derived qid keeps both code
        # paths consistent — corpus-stored ``reference_article_id`` would only
        # disagree if the same corpus doc appeared as a pair/distractor for
        # multiple queries, which the qrels-derived value handles correctly.
        pair_metadata = {
            "dataset_idx": idx,
            "_id": pair_cid,
            "article_type": "pair",
            "reference_article_id": qid,
            "other_article_id": "",
        }

        distractor_items: List[Dict[str, Any]] = []
        for cid in dist_cids:
            row = self._corpus_by_id[cid]
            text = row.get("text", "")
            if self.transform:
                text = self.transform(text)
            distractor_items.append(
                {
                    "text": text,
                    "metadata": {
                        "dataset_idx": idx,
                        "_id": cid,
                        "article_type": "distractor",
                        "reference_article_id": qid,
                        "other_article_id": "",
                    },
                }
            )

        if self.transform:
            original_text = self.transform(original_text)
            pair_text = self.transform(pair_text)

        return (
            {"text": original_text, "metadata": original_metadata},
            {"text": pair_text, "metadata": pair_metadata},
            distractor_items,
        )


class ReignCachedDataset(Dataset):
    """Drop-in replacement for ``ReignDataset`` that returns pre-computed
    embeddings instead of texts. Cache misses are filled by the supplied
    ``feature_extractor``.
    """

    def __init__(
        self,
        dataset_name: str = "devrim/goodwiki_long_synthetic_ir",
        qrels_split: str = "train",
        max_samples: Optional[int] = None,
        feature_extractor=None,
        transform: Optional[Callable] = None,
        n_distractors_per_sample: Optional[int] = None,
    ):
        if feature_extractor is None:
            raise ValueError("feature_extractor is required for ReignCachedDataset")
        self.transform = transform
        self.feature_extractor = feature_extractor
        # If set, each instance returns exactly this many distractor embeddings
        # via per-instance deterministic subsampling. Required for IR-format
        # datasets where partials/query is variable — the collate
        # otherwise flattens to a non-multiple of batch_size and crashes
        # ``get_combined_batch``. Leave ``None`` for GoodWiki where every query
        # has exactly 2 distractors by construction.
        self.n_distractors_per_sample = n_distractors_per_sample

        self.base_dataset = ReignDataset(
            dataset_name=dataset_name,
            qrels_split=qrels_split,
            max_samples=max_samples,
            transform=transform,
        )

        # Include chunk_size + stride in the cache id so different chunking
        # configs (especially under the chunking/stride ablation) don't silently
        # reuse stale caches.
        chunk_size = getattr(feature_extractor, "chunk_size", None)
        stride = getattr(feature_extractor, "stride", None)
        self.original_cache_id = _cache_id(
            dataset_name, qrels_split, "original", max_samples, chunk_size, stride
        )
        self.synthetic_cache_id = _cache_id(
            dataset_name, qrels_split, "synthetic", max_samples, chunk_size, stride
        )
        self.distractor_cache_id = _cache_id(
            dataset_name, qrels_split, "distractors", max_samples, chunk_size, stride
        )

        self._prepare_cached_embeddings()
        self._precompute_distractor_indices()

    def _precompute_distractor_indices(self) -> None:
        """Build the per-instance index list into the flat distractor cache.

        Two responsibilities:

        1. Maps each (kept) instance index to a list of flat positions in the
           distractor cache file. The cache was written with one entry per
           ``(instance, distractor)`` pair in instance-order, so we replay the
           offset arithmetic to recover the base offset per instance.
        2. If ``n_distractors_per_sample`` is set, deterministically subsamples
           each instance's distractor list down to ``K`` positions and (when an
           instance has fewer than K available) oversamples with replacement.
           Instances with zero distractors can't form partial pairs, so they
           are filtered out via ``self._valid_indices``. Cache offsets stay
           consistent because the filter operates on top of the unchanged
           ``base_dataset._instances`` ordering.
        """
        import numpy as np

        # Compute cumulative flat-cache offsets for every base-dataset instance
        # (cache write order). full_offsets[i] is the flat-cache base for
        # instance i; the per-instance distractor occupies [base, base+n_i).
        full_offsets: List[int] = [0]
        for idx in range(len(self.base_dataset)):
            _, _, dist_cids = self.base_dataset._instances[idx]
            full_offsets.append(full_offsets[-1] + len(dist_cids))

        K = self.n_distractors_per_sample
        # Drop instances with 0 distractors when a K cap is in effect — they
        # can't contribute partials and would force the batch to a non-multiple
        # of batch_size. Without a cap (GoodWiki path) keep every instance.
        self._valid_indices: List[int] = []
        for idx in range(len(self.base_dataset)):
            n = len(self.base_dataset._instances[idx][2])
            if K is not None and n == 0:
                continue
            self._valid_indices.append(idx)
        dropped = len(self.base_dataset) - len(self._valid_indices)
        if dropped:
            logger.info(
                "ReignCachedDataset: dropping %d/%d instances with 0 distractors",
                dropped,
                len(self.base_dataset),
            )

        self.distractor_index_mapping: Dict[int, List[int]] = {}
        for virtual_idx, real_idx in enumerate(self._valid_indices):
            _, _, dist_cids = self.base_dataset._instances[real_idx]
            base = full_offsets[real_idx]
            n = len(dist_cids)
            if K is None:
                self.distractor_index_mapping[virtual_idx] = (
                    list(range(base, base + n)) if n else []
                )
                continue
            inst_rng = np.random.default_rng(seed=real_idx)
            if n >= K:
                offsets = inst_rng.choice(n, size=K, replace=False).tolist()
            else:
                offsets = inst_rng.choice(n, size=K, replace=True).tolist()
            self.distractor_index_mapping[virtual_idx] = [base + int(o) for o in offsets]

    def get_all_texts_and_metadata(
        self,
    ) -> Tuple[List[str], List[str], List[str], List[Dict], List[Dict], List[Dict]]:
        original_texts: List[str] = []
        synthetic_texts: List[str] = []
        distractor_texts: List[str] = []
        original_metadata: List[Dict] = []
        synthetic_metadata: List[Dict] = []
        distractor_metadata: List[Dict] = []

        for idx in range(len(self.base_dataset)):
            original_data, synthetic_data, distractor_data_list = self.base_dataset[idx]
            original_texts.append(original_data["text"])
            original_metadata.append(original_data["metadata"])
            synthetic_texts.append(synthetic_data["text"])
            synthetic_metadata.append(synthetic_data["metadata"])
            for d in distractor_data_list:
                distractor_texts.append(d["text"])
                distractor_metadata.append(d["metadata"])

        return (
            original_texts,
            synthetic_texts,
            distractor_texts,
            original_metadata,
            synthetic_metadata,
            distractor_metadata,
        )

    def _prepare_cached_embeddings(self) -> None:
        (
            original_texts,
            synthetic_texts,
            distractor_texts,
            original_metadata,
            synthetic_metadata,
            distractor_metadata,
        ) = self.get_all_texts_and_metadata()

        if not self.feature_extractor.cache.has_cache(
            self.feature_extractor.model_name_or_path,
            self.feature_extractor.chunk_size,
            self.original_cache_id,
            stride=self.feature_extractor.stride,
        ):
            logger.info("Computing original-side embeddings (%s)", self.original_cache_id)
            self.feature_extractor.compute_and_cache_dataset_embeddings_with_metadata(
                original_texts, self.original_cache_id, original_metadata
            )
        if not self.feature_extractor.cache.has_cache(
            self.feature_extractor.model_name_or_path,
            self.feature_extractor.chunk_size,
            self.synthetic_cache_id,
            stride=self.feature_extractor.stride,
        ):
            logger.info("Computing synthetic-side embeddings (%s)", self.synthetic_cache_id)
            self.feature_extractor.compute_and_cache_dataset_embeddings_with_metadata(
                synthetic_texts, self.synthetic_cache_id, synthetic_metadata
            )
        if distractor_texts and not self.feature_extractor.cache.has_cache(
            self.feature_extractor.model_name_or_path,
            self.feature_extractor.chunk_size,
            self.distractor_cache_id,
            stride=self.feature_extractor.stride,
        ):
            logger.info("Computing distractor embeddings (%s)", self.distractor_cache_id)
            self.feature_extractor.compute_and_cache_dataset_embeddings_with_metadata(
                distractor_texts, self.distractor_cache_id, distractor_metadata
            )

    def __len__(self) -> int:
        return len(self._valid_indices)

    @profile_function("dataset.cached_getitem")
    def __getitem__(self, virtual_idx: int):
        real_idx = self._valid_indices[virtual_idx]
        orig = self.feature_extractor.get_cached_embeddings_with_metadata(
            self.original_cache_id, [real_idx]
        )
        if orig is None or len(orig[0]) == 0:
            raise ValueError(f"No cached original embedding at index {real_idx}")
        original_embeddings, original_metadata_list = orig
        original_embedding, original_attention_mask = original_embeddings[0]
        original_metadata = original_metadata_list[0]

        synth = self.feature_extractor.get_cached_embeddings_with_metadata(
            self.synthetic_cache_id, [real_idx]
        )
        if synth is None or len(synth[0]) == 0:
            raise ValueError(f"No cached synthetic embedding at index {real_idx}")
        synthetic_embeddings, synthetic_metadata_list = synth
        synthetic_embedding, synthetic_attention_mask = synthetic_embeddings[0]
        synthetic_metadata = synthetic_metadata_list[0]

        distractor_embeddings: List[Tuple[torch.Tensor, torch.Tensor, Dict]] = []
        dist_indices = self.distractor_index_mapping.get(virtual_idx, [])
        if dist_indices:
            cached = self.feature_extractor.get_cached_embeddings_with_metadata(
                self.distractor_cache_id, dist_indices
            )
            if cached is not None:
                emb_list, meta_list = cached
                distractor_embeddings = [
                    (emb, mask, meta) for (emb, mask), meta in zip(emb_list, meta_list)
                ]

        return (
            (original_embedding, original_attention_mask, original_metadata),
            (synthetic_embedding, synthetic_attention_mask, synthetic_metadata),
            distractor_embeddings,
        )


def collate_data(
    batch: List[Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]],
) -> Tuple[List[str], List[str], List[List[str]]]:
    """Text-only collate: returns (originals, pairs, [[distractor_texts]])."""
    originals, pairs, distractors_per_sample = [], [], []
    for o, p, dists in batch:
        originals.append(o["text"])
        pairs.append(p["text"])
        distractors_per_sample.append([d["text"] for d in dists])
    return originals, pairs, distractors_per_sample


def collate_data_with_metadata(
    batch: List[Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Metadata-preserving collate (default for non-cached training)."""
    originals, pairs, distractors_per_sample = [], [], []
    for o, p, dists in batch:
        originals.append(o)
        pairs.append(p)
        distractors_per_sample.append(dists)
    return originals, pairs, distractors_per_sample


@profile_function("dataset.collate_cached_data")
def collate_cached_data(
    batch: List[
        Tuple[
            Tuple[torch.Tensor, torch.Tensor, Dict],
            Tuple[torch.Tensor, torch.Tensor, Dict],
            List[Tuple[torch.Tensor, torch.Tensor, Dict]],
        ]
    ],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, List],
]:
    """Pad and stack cached embeddings + masks across a batch."""
    original_embeddings, original_masks = [], []
    synthetic_embeddings, synthetic_masks = [], []
    distractor_embeddings, distractor_masks = [], []
    original_metadata, synthetic_metadata, distractor_metadata = [], [], []

    for (orig_emb, orig_mask, orig_meta), (synth_emb, synth_mask, synth_meta), dists in batch:
        original_embeddings.append(orig_emb)
        original_masks.append(orig_mask)
        original_metadata.append(orig_meta)

        synthetic_embeddings.append(synth_emb)
        synthetic_masks.append(synth_mask)
        synthetic_metadata.append(synth_meta)

        for d_emb, d_mask, d_meta in dists:
            distractor_embeddings.append(d_emb)
            distractor_masks.append(d_mask)
            distractor_metadata.append(d_meta)

    def pad_and_stack(emb_list, mask_list):
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

    bo_emb, bo_mask = pad_and_stack(original_embeddings, original_masks)
    bs_emb, bs_mask = pad_and_stack(synthetic_embeddings, synthetic_masks)
    bd_emb, bd_mask = pad_and_stack(distractor_embeddings, distractor_masks)

    metadata = {
        "original_metadata": original_metadata,
        "synthetic_metadata": synthetic_metadata,
        "distractor_metadata": distractor_metadata,
    }
    return bo_emb, bo_mask, bs_emb, bs_mask, bd_emb, bd_mask, metadata


def create_data_loaders(
    dataset_name: str = "devrim/goodwiki_long_synthetic_ir",
    train_split: str = "train",
    eval_split: str = "test",
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    transform: Optional[Callable] = None,
    collate_fn: Optional[Callable] = "default",
    use_cached_dataset: bool = False,
    feature_extractor=None,
    n_distractors_per_sample: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and eval data loaders against the IR-format dataset.

    ``train_split`` and ``eval_split`` index into the ``default`` qrels config
    (``train`` / ``val`` / ``test``). The corpus and queries are always the full
    pool — splits only restrict which queries we generate training instances for.
    """
    if collate_fn == "default":
        collate_fn = collate_cached_data if use_cached_dataset else collate_data_with_metadata
    elif collate_fn == "text_only":
        collate_fn = collate_data
    elif collate_fn is not None and not callable(collate_fn):
        raise ValueError(f"Unknown collate_fn: {collate_fn}")

    if use_cached_dataset:
        if feature_extractor is None:
            raise ValueError("feature_extractor is required when use_cached_dataset=True")
        dataset_cls = ReignCachedDataset
        extra = {
            "feature_extractor": feature_extractor,
            "n_distractors_per_sample": n_distractors_per_sample,
        }
    else:
        if n_distractors_per_sample is not None:
            logger.warning(
                "n_distractors_per_sample=%d ignored: non-cached ReignDataset does not "
                "yet implement per-sample K capping. Pass --enable-cache to use this.",
                n_distractors_per_sample,
            )
        dataset_cls = ReignDataset
        extra = {}

    train_dataset = dataset_cls(
        dataset_name=dataset_name,
        qrels_split=train_split,
        max_samples=max_samples,
        transform=transform,
        **extra,
    )
    eval_dataset = dataset_cls(
        dataset_name=dataset_name,
        qrels_split=eval_split,
        max_samples=max_samples,
        transform=transform,
        **extra,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, eval_loader
