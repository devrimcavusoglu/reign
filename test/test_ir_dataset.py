"""Tests for the standard-IR contrastive path (paper Section 5.3).

Offline, fast: builds a tiny on-disk BEIR dataset with controlled grades
{0,1,2}, exercises ``IRContrastiveDataset`` grade routing / ``relevant_ids`` /
``partial_policy``, and unit-tests the ``InfoNCELoss`` false-negative mask
(parity when ``None``; a masked true-positive column drops from the denominator).
"""

from __future__ import annotations

import datasets
import pytest
import torch

from reign.ir_dataset import IRContrastiveDataset
from reign.loss import InfoNCELoss


def _write_ir(root, qrels_rows):
    """corpus C0..C5, queries Q0..Q2, default/train = qrels_rows (q,c,score)."""
    corpus = datasets.Dataset.from_dict(
        {
            "_id": [f"C{i}" for i in range(6)],
            "title": [""] * 6,
            "text": [f"doc {i} text" for i in range(6)],
        }
    )
    queries = datasets.Dataset.from_dict(
        {"_id": [f"Q{i}" for i in range(3)], "text": [f"query {i} text" for i in range(3)]}
    )
    qrels = datasets.Dataset.from_dict(
        {
            "query-id": [q for q, _, _ in qrels_rows],
            "corpus-id": [c for _, c, _ in qrels_rows],
            "score": [s for _, _, s in qrels_rows],
        }
    )
    datasets.DatasetDict({"corpus": corpus}).save_to_disk(f"{root}/corpus")
    datasets.DatasetDict({"queries": queries}).save_to_disk(f"{root}/queries")
    datasets.DatasetDict({"train": qrels}).save_to_disk(f"{root}/default")
    return str(root)


# Q0: 1 pos (C0), 1 partial (C1), 1 neg (C2). Q1: 1 pos (C3), 1 partial (C4).
# Q2: only a partial (C5) + a neg — NO positive → must be skipped.
QRELS = [
    ("Q0", "C0", 2),
    ("Q0", "C1", 1),
    ("Q0", "C2", 0),
    ("Q1", "C3", 2),
    ("Q1", "C4", 1),
    ("Q2", "C5", 1),
    ("Q2", "C0", 0),
]


def test_soft_positive_policy(tmp_path):
    root = _write_ir(tmp_path, QRELS)
    ds = IRContrastiveDataset(root, "train", partial_policy="soft_positive")

    assert ds.qids == ["Q0", "Q1"]  # Q2 skipped (no score>=2)
    # score==1 → partials; score==0 → negatives.
    assert sorted(m["article_id"] for m in ds.partial_meta) == ["C1", "C4"]
    assert sorted(m["article_id"] for m in ds.neg_meta) == ["C2"]
    # relevant_ids = positives ∪ soft-positive partials.
    assert ds.relevant_ids["Q0"] == {"C0", "C1"}
    assert ds.relevant_ids["Q1"] == {"C3", "C4"}
    # metadata wiring for the Evaluator proxy.
    assert all(m["article_type"] == "pair" for m in ds.pos_meta)
    assert all(m["reference_article_id"] in {"Q0", "Q1"} for m in ds.pos_meta)


def test_negative_policy(tmp_path):
    root = _write_ir(tmp_path, QRELS)
    ds = IRContrastiveDataset(root, "train", partial_policy="negative")

    assert ds.qids == ["Q0", "Q1"]
    assert ds.partial_meta == []  # no partials emitted
    # score==1 joins score==0 in the negative pool.
    assert sorted(m["article_id"] for m in ds.neg_meta) == ["C1", "C2", "C4"]
    # partials are NOT relevant under negative policy → allowed as negatives.
    assert ds.relevant_ids["Q0"] == {"C0"}
    assert ds.relevant_ids["Q1"] == {"C3"}


def test_ignore_policy(tmp_path):
    root = _write_ir(tmp_path, QRELS)
    ds = IRContrastiveDataset(root, "train", partial_policy="ignore")

    assert ds.qids == ["Q0", "Q1"]
    assert ds.partial_meta == []
    # score==1 dropped entirely; only score==0 are negatives.
    assert sorted(m["article_id"] for m in ds.neg_meta) == ["C2"]
    assert ds.relevant_ids["Q0"] == {"C0"}


def _toy_loss_inputs(n_neg, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, D, generator=g)
    pos = torch.randn(1, D, generator=g)
    negs = torch.randn(n_neg, D, generator=g)
    input1 = torch.cat([q, torch.zeros(n_neg, D)], dim=0)
    input2 = torch.cat([pos, negs], dim=0)
    target = torch.cat([torch.ones(1), torch.full((n_neg,), -1.0)])
    return input1, input2, target


def test_false_neg_mask_none_is_identity():
    loss = InfoNCELoss(temperature=0.07)
    i1, i2, t = _toy_loss_inputs(n_neg=4)
    a = loss(i1, i2, t)
    b = loss(i1, i2, t, false_neg_mask=None)
    assert torch.allclose(a, b)


def test_false_neg_mask_drops_column_and_lowers_loss():
    loss = InfoNCELoss(temperature=0.07)
    i1, i2, t = _toy_loss_inputs(n_neg=4)
    # Make negative column 0 identical to the positive (a "shared" false
    # negative). Unmasked → it inflates the denominator → higher loss.
    i2 = i2.clone()
    i2[1] = i2[0]  # row 1 == positive (first negative col)
    base = loss(i1, i2, t)

    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[0, 0] = True  # drop that false negative for the anchor
    masked = loss(i1, i2, t, false_neg_mask=mask)

    assert torch.isfinite(masked)
    assert masked < base  # removing a (false) hard negative reduces the loss

    with pytest.raises(ValueError):
        loss(i1, i2, t, false_neg_mask=torch.zeros(2, 9, dtype=torch.bool))
