"""LoCo benchmark (Saad-Falcon et al., 2024) loader for the REIGN eval harness.

LoCo is a 12-subtask long-document retrieval benchmark hosted as two HF
datasets, ``hazyresearch/LoCoV1-Queries`` and ``hazyresearch/LoCoV1-Documents``,
each storing one JSONL per subtask under ``documents/<subtask>_test.jsonl``.

Schema (queries side)::

    {"qid": "<dataset>_Query_<i>", "query": "...", "answer_pids": [...],
     "dataset": "<subtask>"}

Schema (docs side)::

    {"pid": "<dataset>_Passage_<j>", "passage": "...", "dataset": "<subtask>"}

Relevance is *binary* (a query/document is relevant iff its ``pid`` is in
``answer_pids``); this is how the LoCo paper reports nDCG@10. The loader exposes
that as a dense ``(n_queries, n_corpus)`` int8 matrix to plug into the existing
``reign.encoders.eval_utils.compute_metrics`` pipeline without any LoCo-specific
metric logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

LOCO_SUBTASKS: tuple[str, ...] = (
    "2wikimqa",
    "courtlistener_HTML",
    "courtlistener_Plain_Text",
    "gov_report",
    "legal_case_reports",
    "multifieldqa",
    "passage_retrieval",
    "qasper_abstract",
    "qasper_title",
    "qmsum",
    "stackoverflow",
    "summ_screen_fd",
)

_QUERIES_REPO = "hazyresearch/LoCoV1-Queries"
_DOCS_REPO = "hazyresearch/LoCoV1-Documents"


@dataclass(frozen=True)
class LocoSubtask:
    """A single LoCo subtask in BEIR-shaped form, ready for retrieval eval."""

    name: str
    query_ids: list[str]
    queries: list[str]
    corpus_ids: list[str]
    corpus: list[str]
    relevance: np.ndarray  # (n_queries, n_corpus) int8, 1 == relevant


def _download_jsonl(repo: str, subtask: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo,
        filename=f"documents/{subtask}_test.jsonl",
        repo_type="dataset",
    )


def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_loco_subtask(name: str) -> LocoSubtask:
    """Load a single LoCo subtask. Caches downloads under the HF hub cache."""
    if name not in LOCO_SUBTASKS:
        raise ValueError(f"Unknown LoCo subtask {name!r}; pick from {LOCO_SUBTASKS}")

    logger.info("Loading LoCo subtask %s", name)
    q_path = _download_jsonl(_QUERIES_REPO, name)
    d_path = _download_jsonl(_DOCS_REPO, name)

    query_rows = list(_read_jsonl(q_path))
    doc_rows = list(_read_jsonl(d_path))

    query_ids = [r["qid"] for r in query_rows]
    queries = [r["query"] for r in query_rows]
    corpus_ids = [r["pid"] for r in doc_rows]
    corpus = [r["passage"] for r in doc_rows]

    pid_to_col = {pid: i for i, pid in enumerate(corpus_ids)}
    relevance = np.zeros((len(query_ids), len(corpus_ids)), dtype=np.int8)
    missing = 0
    for i, q in enumerate(query_rows):
        for pid in q.get("answer_pids", []):
            j = pid_to_col.get(pid)
            if j is None:
                missing += 1
                continue
            relevance[i, j] = 1
    if missing:
        logger.warning("LoCo %s: %d answer_pids absent from corpus (skipped)", name, missing)

    logger.info(
        "LoCo %s: %d queries, %d corpus docs, %d positive pairs",
        name,
        len(query_ids),
        len(corpus_ids),
        int(relevance.sum()),
    )
    return LocoSubtask(
        name=name,
        query_ids=query_ids,
        queries=queries,
        corpus_ids=corpus_ids,
        corpus=corpus,
        relevance=relevance,
    )


__all__ = ["LOCO_SUBTASKS", "LocoSubtask", "load_loco_subtask"]
