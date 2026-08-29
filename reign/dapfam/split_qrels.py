"""Stratified query-disjoint train/val/test splitter for DAPFAM.

DAPFAM ships eval-only (``build_dataset.py`` emits ``default/`` with ``test`` /
``test_in`` / ``test_out``, all positives). To **train** REIGN on DAPFAM we need
disjoint train/val/test query sets so the eval is not contaminated by training
labels:

* Stratify queries by quartiles of ``n_positives`` (DAPFAM relevance is binary, so
  there is no partial-relevance term in the stratification key).
* Query-disjoint, seed=42, ~70/15/15.
* **Overwrite** ``default/`` so the IR-format readers (``ReignDataset``,
  ``build_query_corpus``) pick up the new splits — the same convention as the
  synthetic long-document dataset.

DAPFAM also carries an IN/OUT cross-domain partition. We preserve it **scoped to
the held-out test queries**: the rewritten
``test_in`` / ``test_out`` are the IN/OUT subsets of the new ``test`` split, so
the cross-domain breakdown is reported on exactly the held-out set that the
fine-tuned model and the zero-shot baselines are scored on (a fair, consistent
table).

Resulting ``default/`` splits: ``train`` / ``val`` / ``test`` /
``test_in`` / ``test_out``. Protocol: train on ``train``, select on ``val``,
report every method (zero-shot baselines and fine-tuned REIGN) on the held-out
``test``; IN/OUT via ``test_in`` / ``test_out``.

Corpus and queries configs are untouched (full pools shared across splits, the same
convention as the synthetic long-document dataset).

Usage::

    python -m reign.dapfam.split_qrels \\
        --data-dir data/dapfam_ir_fulltext \\
        --train-ratio 0.70 --val-ratio 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import datasets
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dapfam.split_qrels")


def _stratified_split(
    qids: list[str],
    keys: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Quartile-stratified query-disjoint split (each difficulty bucket sliced by the same ratios)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(keys, kind="stable")
    qids_sorted = [qids[i] for i in order]

    n = len(qids_sorted)
    edges = [int(round(n * p)) for p in (0.25, 0.5, 0.75)]
    quartiles = [
        qids_sorted[: edges[0]],
        qids_sorted[edges[0] : edges[1]],
        qids_sorted[edges[1] : edges[2]],
        qids_sorted[edges[2] :],
    ]

    train_set: set[str] = set()
    val_set: set[str] = set()
    test_set: set[str] = set()
    for q, bucket in enumerate(quartiles):
        bucket_arr = np.array(bucket)
        rng.shuffle(bucket_arr)
        nq = len(bucket_arr)
        n_train = int(round(nq * train_ratio))
        n_val = int(round(nq * val_ratio))
        train_set.update(bucket_arr[:n_train].tolist())
        val_set.update(bucket_arr[n_train : n_train + n_val].tolist())
        test_set.update(bucket_arr[n_train + n_val :].tolist())
        logger.info(
            "quartile %d: %d total -> %d train / %d val / %d test",
            q,
            nq,
            n_train,
            n_val,
            nq - n_train - n_val,
        )
    return train_set, val_set, test_set


def _rows_for(src_pd, qids: set[str]):
    """Decoupled (in-memory) Dataset of the source rows whose query-id is in ``qids``."""
    rows = src_pd[src_pd["query-id"].isin(qids)].reset_index(drop=True)
    return datasets.Dataset.from_pandas(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/dapfam_ir_fulltext")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-split",
        default="test",
        help="Existing full-positives split in default/ to partition (default: test).",
    )
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 (leave room for test)")

    default_dir = Path(args.data_dir) / "default"
    logger.info("Loading default/ from %s", default_dir)
    ddict = datasets.load_from_disk(str(default_dir))
    if args.source_split not in ddict:
        raise KeyError(f"source split {args.source_split!r} not in default/ (have {list(ddict)})")
    # Decouple from the memory-mapped arrow shards before we overwrite default/.
    src_pd = ddict[args.source_split].to_pandas()
    has_in = "test_in" in ddict
    has_out = "test_out" in ddict
    in_pd = ddict["test_in"].to_pandas() if has_in else None
    out_pd = ddict["test_out"].to_pandas() if has_out else None
    logger.info("Source qrels [%s]: %d rows", args.source_split, len(src_pd))

    # Stratify on n_POSITIVES (score>=2) per query, NOT total qrel rows. This
    # (a) is the right difficulty signal and (b) makes the query→split
    # assignment invariant to whether provided negatives (score=0,
    # --keep-negatives) are present — so a rebuild with negatives reproduces the
    # exact same held-out test queries as a positives-only build (keeps the
    # baseline/zero-shot table comparable). Every query is still partitioned
    # with all its rows (positives + its score=0 negatives ride into its split).
    pos_mask = src_pd["score"] >= 2
    pos_per_q: Counter = Counter(src_pd.loc[pos_mask, "query-id"].tolist())
    all_qids = sorted(set(src_pd["query-id"]))
    keys = np.array([pos_per_q.get(q, 0) for q in all_qids], dtype=float)
    logger.info(
        "%d unique queries; n_positives/query min=%d mean=%.2f max=%d",
        len(all_qids),
        int(keys.min()),
        float(keys.mean()),
        int(keys.max()),
    )

    train_q, val_q, test_q = _stratified_split(
        all_qids, keys, args.train_ratio, args.val_ratio, args.seed
    )
    assert train_q.isdisjoint(val_q) and train_q.isdisjoint(test_q) and val_q.isdisjoint(test_q)
    logger.info(
        "split sizes (queries): train=%d val=%d test=%d (total=%d / %d)",
        len(train_q),
        len(val_q),
        len(test_q),
        len(train_q) + len(val_q) + len(test_q),
        len(all_qids),
    )

    out = {
        "train": _rows_for(src_pd, train_q),
        "val": _rows_for(src_pd, val_q),
        "test": _rows_for(src_pd, test_q),
    }
    # IN/OUT scoped to the held-out test queries (consistent reporting set).
    if in_pd is not None:
        out["test_in"] = _rows_for(in_pd, test_q)
    if out_pd is not None:
        out["test_out"] = _rows_for(out_pd, test_q)

    for name, ds in out.items():
        nq = len(set(ds["query-id"])) if len(ds) else 0
        logger.info("  %8s: %d qrel rows, %d unique queries", name, len(ds), nq)
    if "test_in" in out and "test_out" in out:
        s = len(out["test_in"]) + len(out["test_out"])
        if s != len(out["test"]):
            logger.warning(
                "test_in + test_out (%d) != test (%d) — domain partition not exhaustive",
                s,
                len(out["test"]),
            )

    logger.info("Overwriting %s (train/val/test [+ test_in/test_out])", default_dir)
    datasets.DatasetDict(out).save_to_disk(str(default_dir))
    logger.info("Done.")


if __name__ == "__main__":
    main()
