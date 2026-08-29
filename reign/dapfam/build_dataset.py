"""DAPFAM -> BEIR-format dataset adapter (real-world long-document case study).

DAPFAM (``datalyes/DAPFAM_patent``, arXiv 2506.22141) is a patent prior-art
retrieval benchmark where the *query is a full patent family* (~20K tokens mean)
and targets are ~11K tokens — the many-chunk long-context regime REIGN targets.
This adapter converts the three upstream HF configs into the BEIR layout that
``reign/dataset.py`` / ``reign/encoders/eval_utils.py`` consume (the same layout as
the synthetic long-document dataset):

* ``corpus``  config / ``corpus``  split: ``{_id, title, text}``
* ``queries`` config / ``queries`` split: ``{_id, text}``
* ``default`` config / splits ``test`` / ``test_in`` / ``test_out``:
  ``{query-id, corpus-id, score}``

Upstream schema (datasets-server confirmed):

* config ``corpus``    : ``relevant_id`` (key), ``title_en``, ``abstract_en``,
  ``claims_text``, ``description_en``, ``classifications_ipcr_list_first_three_chars_list``.
* config ``queries``   : ``query_id`` (key), same text fields (+ ``abstract_keywords``).
* config ``relations`` : ``query_id``, ``relevant_id``, ``relevance_score`` (float,
  binary 1.0/0.0), ``domain_rel`` (``IN``/``OUT`` — IPC3-overlap partition).

Text views (``--text-view``):

* ``ta``       : ``title_en`` + ``abstract_en``
* ``tac``      : + ``claims_text``
* ``fulltext`` : + ``description_en``   (default; the long regime)

Relevance mapping (BEIR convention): a positive (``relevance_score`` >= 0.5)
becomes ``score = 2``. The dataset's provided negatives (``relevance_score`` ==
0.0) are, by default (``--keep-negatives``), retained as ``score = 0`` rows in
the ``test`` split so the standard-IR fine-tune can use them as explicit
negatives. This is **eval-invariant**: ``build_relevance`` writes
``rel = int(score)``, so a ``score=0`` row is identical to an absent pair for
every retrieval metric, and ``build_query_corpus`` already included the query
via its positives. Pass ``--no-keep-negatives`` for a positives-only qrels.
``test_in`` / ``test_out`` stay POSITIVES-ONLY — they partition positives by
``domain_rel`` for the IN/OUT cross-domain breakdown, zero harness change
(``build_query_corpus`` loads ``default`` via ``load_from_disk(...)[split]``).

DAPFAM is eval-only (no official train split); a query-disjoint train/val/test
split for the optional fine-tune is produced separately by ``split_qrels.py``.

Source: HF ``datalyes/DAPFAM_patent``, license CC-BY-NC-SA-4.0 (non-commercial;
attribute arXiv 2506.22141). The module has no import-time side effects and ships an
offline fixture path (``--smoke``), so the adapter can be unit-tested without network.

Usage::

    python -m reign.dapfam.build_dataset \\
        --text-view fulltext --out-dir data/dapfam_ir_fulltext --download
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("dapfam.build_dataset")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Upstream dataset identifier on the Hugging Face Hub (overridable via --hf-dataset).
DEFAULT_HF_DATASET = "datalyes/DAPFAM_patent"

# Upstream id / text / label fields (datasets-server confirmed).
CORPUS_ID_FIELD = "relevant_id"
QUERY_ID_FIELD = "query_id"
REL_QUERY_FIELD = "query_id"
REL_CORPUS_FIELD = "relevant_id"
REL_SCORE_FIELD = "relevance_score"
REL_DOMAIN_FIELD = "domain_rel"

# Ordered text fields per view; missing/empty parts are skipped at join time.
TEXT_VIEWS: Dict[str, List[str]] = {
    "ta": ["title_en", "abstract_en"],
    "tac": ["title_en", "abstract_en", "claims_text"],
    "fulltext": ["title_en", "abstract_en", "claims_text", "description_en"],
}

# A positive judgement maps to the BEIR "fully relevant" grade. DAPFAM relevance
# is binary; with only score=2 present, compute_metrics' graded nDCG (exponential
# gain, max_gain inferred = 2) degenerates to standard binary nDCG.
POSITIVE_SCORE = 2
POSITIVE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Text assembly
# ---------------------------------------------------------------------------


def build_text(record: Dict[str, object], view: str) -> str:
    """Concatenate the view's text fields (blank-line separated), skipping empties."""
    fields = TEXT_VIEWS[view]
    parts: List[str] = []
    for f in fields:
        val = record.get(f)
        if val is None:
            continue
        s = str(val).strip()
        if s:
            parts.append(s)
    return "\n\n".join(parts).strip()


def _domain_bucket(domain_rel: object) -> Optional[str]:
    """Normalise ``domain_rel`` to ``in`` / ``out`` (case-insensitive), else None."""
    if domain_rel is None:
        return None
    d = str(domain_rel).strip().lower()
    if d == "in":
        return "in"
    if d == "out":
        return "out"
    return None


# ---------------------------------------------------------------------------
# BEIR assembly
# ---------------------------------------------------------------------------


def build_beir(corpus_recs, query_recs, relation_recs, view: str, keep_negatives: bool = True):
    """Assemble the three BEIR configs from upstream records.

    ``corpus_recs`` / ``query_recs`` / ``relation_recs`` are iterables of dicts
    (HF ``Dataset`` rows or plain dicts for smoke). Returns a dict of three
    ``datasets.DatasetDict``s keyed by config name plus a stats dict.

    ``keep_negatives`` (default True): retain the dataset's provided negatives
    (``relevance_score`` 0) as ``score=0`` rows in the ``test`` split for the
    standard-IR fine-tune. Eval-invariant (``score=0`` ≡ absent in
    ``build_relevance``/``compute_metrics``). Pass False for a positives-only
    qrels.
    """
    import datasets

    corpus_rows = {"_id": [], "title": [], "text": []}
    corpus_ids = set()
    for rec in corpus_recs:
        cid = str(rec[CORPUS_ID_FIELD])
        corpus_rows["_id"].append(cid)
        corpus_rows["title"].append(str(rec.get("title_en") or ""))
        corpus_rows["text"].append(build_text(rec, view))
        corpus_ids.add(cid)

    query_rows = {"_id": [], "text": []}
    query_ids = set()
    for rec in query_recs:
        qid = str(rec[QUERY_ID_FIELD])
        query_rows["_id"].append(qid)
        query_rows["text"].append(build_text(rec, view))
        query_ids.add(qid)

    # `test` holds positives (score=2); when ``keep_negatives`` is on it also
    # carries the dataset's own provided negatives as ``score=0`` rows so the
    # standard-IR fine-tune can use them as explicit negatives. ``score=0`` rows
    # are eval-invariant: ``build_relevance`` writes ``rel=int(score)`` so a 0
    # row is identical to an absent pair for all retrieval metrics. ``test_in`` /
    # ``test_out`` stay POSITIVES-ONLY (the IPC3 domain breakdown is defined over
    # relevant pairs). Skip rows whose endpoints fell out of a subsampled pool
    # (--max-samples) so qrels never dangle.
    splits: Dict[str, Dict[str, list]] = {
        name: {"query-id": [], "corpus-id": [], "score": []} for name in ("test", "test_in", "test_out")
    }
    n_neg_dropped = n_neg_kept = n_dangling = 0
    unknown_domain: Counter = Counter()
    for rec in relation_recs:
        score = float(rec[REL_SCORE_FIELD])
        qid = str(rec[REL_QUERY_FIELD])
        cid = str(rec[REL_CORPUS_FIELD])
        if score < POSITIVE_THRESHOLD:
            if not keep_negatives:
                n_neg_dropped += 1
                continue
            if qid not in query_ids or cid not in corpus_ids:
                n_dangling += 1
                continue
            splits["test"]["query-id"].append(qid)
            splits["test"]["corpus-id"].append(cid)
            splits["test"]["score"].append(0)
            n_neg_kept += 1
            continue
        if qid not in query_ids or cid not in corpus_ids:
            n_dangling += 1
            continue
        splits["test"]["query-id"].append(qid)
        splits["test"]["corpus-id"].append(cid)
        splits["test"]["score"].append(POSITIVE_SCORE)
        bucket = _domain_bucket(rec.get(REL_DOMAIN_FIELD))
        if bucket == "in":
            tgt = splits["test_in"]
        elif bucket == "out":
            tgt = splits["test_out"]
        else:
            unknown_domain[str(rec.get(REL_DOMAIN_FIELD))] += 1
            continue
        tgt["query-id"].append(qid)
        tgt["corpus-id"].append(cid)
        tgt["score"].append(POSITIVE_SCORE)

    if unknown_domain:
        logger.warning(
            "domain_rel values not in {IN,OUT} (kept in `test`, absent from in/out): %s",
            dict(unknown_domain),
        )
    n_pos = sum(1 for s in splits["test"]["score"] if s == POSITIVE_SCORE)
    logger.info(
        "qrels: %d positives + %d kept negatives (test=%d) | %d in | %d out | "
        "dropped %d negatives, %d dangling",
        n_pos,
        n_neg_kept,
        len(splits["test"]["query-id"]),
        len(splits["test_in"]["query-id"]),
        len(splits["test_out"]["query-id"]),
        n_neg_dropped,
        n_dangling,
    )

    beir = {
        "corpus": datasets.DatasetDict({"corpus": datasets.Dataset.from_dict(corpus_rows)}),
        "queries": datasets.DatasetDict({"queries": datasets.Dataset.from_dict(query_rows)}),
        "default": datasets.DatasetDict(
            {name: datasets.Dataset.from_dict(rows) for name, rows in splits.items()}
        ),
    }
    stats = compute_stats(corpus_rows, query_rows, splits, view, n_neg_dropped, n_neg_kept, n_dangling)
    return beir, stats


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _whitespace_tokens(text: str) -> int:
    return len(text.split())


def _token_distribution(texts: Sequence[str], sample_cap: int = 4000) -> Dict[str, int]:
    """Whitespace-token percentiles over a deterministic head sample (fast on 45k docs)."""
    sample = texts[:sample_cap]
    counts = sorted(_whitespace_tokens(t) for t in sample)
    if not counts:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "p95": 0, "max": 0}

    def _pct(p: float) -> int:
        i = max(0, min(len(counts) - 1, int(p * (len(counts) - 1))))
        return counts[i]

    return {
        "min": counts[0],
        "p25": _pct(0.25),
        "median": _pct(0.50),
        "p75": _pct(0.75),
        "p95": _pct(0.95),
        "max": counts[-1],
        "sampled": len(sample),
    }


def compute_stats(
    corpus_rows, query_rows, splits, view, n_neg_dropped, n_neg_kept, n_dangling
) -> Dict[str, object]:
    t = splits["test"]
    pos_per_q: Counter = Counter(q for q, s in zip(t["query-id"], t["score"]) if s == POSITIVE_SCORE)
    return {
        "text_view": view,
        "n_corpus": len(corpus_rows["_id"]),
        "n_queries": len(query_rows["_id"]),
        "n_queries_with_positives": len(pos_per_q),
        "qrels": {
            "test": len(t["query-id"]),
            "test_positives": sum(1 for s in t["score"] if s == POSITIVE_SCORE),
            "test_negatives": n_neg_kept,
            "test_in": len(splits["test_in"]["query-id"]),
            "test_out": len(splits["test_out"]["query-id"]),
        },
        "positives_per_query": {
            "min": min(pos_per_q.values()) if pos_per_q else 0,
            "mean": round(sum(pos_per_q.values()) / len(pos_per_q), 2) if pos_per_q else 0.0,
            "max": max(pos_per_q.values()) if pos_per_q else 0,
        },
        "kept_negatives": n_neg_kept,
        "dropped_negatives": n_neg_dropped,
        "dropped_dangling": n_dangling,
        "query_token_distribution_whitespace": _token_distribution(query_rows["text"]),
        "corpus_token_distribution_whitespace": _token_distribution(corpus_rows["text"]),
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_config(hf_dataset: str, config: str):
    """Load a DAPFAM config; tolerate either a single ``train`` split or a bare Dataset."""
    import datasets

    obj = datasets.load_dataset(hf_dataset, config)
    if isinstance(obj, datasets.DatasetDict):
        # Upstream ships one split (``train``); take it whatever it is named.
        split = "train" if "train" in obj else next(iter(obj))
        return obj[split]
    return obj


def load_dapfam(hf_dataset: str, max_samples: Optional[int] = None):
    """Return ``(corpus_recs, query_recs, relation_recs)`` lists of dicts.

    With ``max_samples`` the queries are capped to the first N; corpus is
    restricted to (referenced positives) ∪ (a bounded head of the corpus) so a
    sanity run stays small but keeps a realistic retrieval pool.
    """
    logger.info("Loading %s configs (corpus, queries, relations)", hf_dataset)
    corpus = _load_config(hf_dataset, "corpus")
    queries = _load_config(hf_dataset, "queries")
    relations = _load_config(hf_dataset, "relations")
    logger.info(
        "Upstream sizes: corpus=%d queries=%d relations=%d",
        len(corpus),
        len(queries),
        len(relations),
    )

    if max_samples is None:
        return list(corpus), list(queries), list(relations)

    keep_qids = {str(queries[i][QUERY_ID_FIELD]) for i in range(min(max_samples, len(queries)))}
    rel = [r for r in relations if str(r[REL_QUERY_FIELD]) in keep_qids]
    referenced = {
        str(r[REL_CORPUS_FIELD]) for r in rel if float(r[REL_SCORE_FIELD]) >= POSITIVE_THRESHOLD
    }
    corpus_cap = max(2000, 10 * len(keep_qids))
    q_recs = [q for q in queries if str(q[QUERY_ID_FIELD]) in keep_qids]
    c_recs, seen = [], set()
    for i, c in enumerate(corpus):
        cid = str(c[CORPUS_ID_FIELD])
        if cid in referenced or i < corpus_cap:
            if cid not in seen:
                c_recs.append(c)
                seen.add(cid)
    logger.info(
        "--max-samples %d → queries=%d corpus=%d relations=%d",
        max_samples,
        len(q_recs),
        len(c_recs),
        len(rel),
    )
    return c_recs, q_recs, rel


# ---------------------------------------------------------------------------
# Smoke fixtures (synthetic, no network)
# ---------------------------------------------------------------------------


def load_smoke_records(scale: int = 1):
    """Tiny deterministic in-memory DAPFAM-shaped dataset (no network).

    ``scale=1`` returns the minimal fixture (6 corpus / 3 queries / 4 positives
    + 1 negative) that the adapter unit tests assert on — do not change it.
    ``scale>1`` generates ``3*scale`` queries / ``6*scale`` corpus with 2
    positives (IN + OUT) per query, large enough that a 70/15/15 query-disjoint
    split is non-degenerate — used by the CPU training smoke.
    """
    if scale <= 1:
        corpus = [
            {
                CORPUS_ID_FIELD: f"C{i}",
                "title_en": f"Patent title {i}",
                "abstract_en": f"Abstract of patent {i} about widget {i % 3}.",
                "claims_text": f"Claim 1. A widget {i % 3} comprising parts." * 2,
                "description_en": f"Detailed description of widget {i % 3}. " * 8,
            }
            for i in range(6)
        ]
        queries = [
            {
                QUERY_ID_FIELD: f"Q{i}",
                "title_en": f"Query patent {i}",
                "abstract_en": f"Query abstract {i} on widget {i % 3}.",
                "claims_text": f"Claim 1. Improved widget {i % 3}." * 2,
                "description_en": f"Query description of widget {i % 3}. " * 8,
            }
            for i in range(3)
        ]
        relations = [
            {
                REL_QUERY_FIELD: "Q0",
                REL_CORPUS_FIELD: "C0",
                REL_SCORE_FIELD: 1.0,
                REL_DOMAIN_FIELD: "IN",
            },
            {
                REL_QUERY_FIELD: "Q0",
                REL_CORPUS_FIELD: "C3",
                REL_SCORE_FIELD: 1.0,
                REL_DOMAIN_FIELD: "OUT",
            },
            {
                REL_QUERY_FIELD: "Q0",
                REL_CORPUS_FIELD: "C5",
                REL_SCORE_FIELD: 0.0,
                REL_DOMAIN_FIELD: "OUT",
            },
            {
                REL_QUERY_FIELD: "Q1",
                REL_CORPUS_FIELD: "C1",
                REL_SCORE_FIELD: 1.0,
                REL_DOMAIN_FIELD: "IN",
            },
            {
                REL_QUERY_FIELD: "Q2",
                REL_CORPUS_FIELD: "C2",
                REL_SCORE_FIELD: 1.0,
                REL_DOMAIN_FIELD: "OUT",
            },
        ]
        return corpus, queries, relations

    nq, nc = 3 * scale, 6 * scale
    corpus = [
        {
            CORPUS_ID_FIELD: f"C{i}",
            "title_en": f"Patent title {i}",
            "abstract_en": f"Abstract of patent {i} about widget {i % 5}.",
            "claims_text": f"Claim 1. A widget {i % 5} comprising parts {i}." * 2,
            "description_en": f"Detailed description of widget {i % 5} variant {i}. " * 8,
        }
        for i in range(nc)
    ]
    queries = [
        {
            QUERY_ID_FIELD: f"Q{i}",
            "title_en": f"Query patent {i}",
            "abstract_en": f"Query abstract {i} on widget {i % 5}.",
            "claims_text": f"Claim 1. Improved widget {i % 5} aspect {i}." * 2,
            "description_en": f"Query description of widget {i % 5} aspect {i}. " * 8,
        }
        for i in range(nq)
    ]
    relations = []
    for i in range(nq):
        p1, p2, neg = i % nc, (i + 3) % nc, (i + 5) % nc
        d1, d2 = ("IN", "OUT") if i % 2 == 0 else ("OUT", "IN")
        relations.append(
            {
                REL_QUERY_FIELD: f"Q{i}",
                REL_CORPUS_FIELD: f"C{p1}",
                REL_SCORE_FIELD: 1.0,
                REL_DOMAIN_FIELD: d1,
            }
        )
        relations.append(
            {
                REL_QUERY_FIELD: f"Q{i}",
                REL_CORPUS_FIELD: f"C{p2}",
                REL_SCORE_FIELD: 1.0,
                REL_DOMAIN_FIELD: d2,
            }
        )
        if neg not in (p1, p2):
            relations.append(
                {
                    REL_QUERY_FIELD: f"Q{i}",
                    REL_CORPUS_FIELD: f"C{neg}",
                    REL_SCORE_FIELD: 0.0,
                    REL_DOMAIN_FIELD: d1,
                }
            )
    return corpus, queries, relations


# ---------------------------------------------------------------------------
# Save / push
# ---------------------------------------------------------------------------


def save_local(beir_configs: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ddict in beir_configs.items():
        ddict.save_to_disk(str(out_dir / name))
    logger.info("Wrote BEIR-format DAPFAM dataset to %s", out_dir)


def push_to_hub(beir_configs: Dict, repo_id: str) -> None:
    for name, ddict in beir_configs.items():
        for split, ds in ddict.items():
            logger.info("Pushing %s/%s to %s", name, split, repo_id)
            ds.push_to_hub(repo_id, config_name=name, split=split)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument(
        "--text-view",
        choices=sorted(TEXT_VIEWS),
        default="fulltext",
        help="Which patent text fields to concatenate (default: fulltext).",
    )
    p.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        help="Upstream HF dataset id (default: datalyes/DAPFAM_patent).",
    )
    p.add_argument(
        "--out-dir",
        default="data/dapfam_ir_fulltext",
        help="Output dir for the BEIR-format dataset (default: data/dapfam_ir_fulltext).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap to the first N queries (with a bounded corpus) for a fast real-load sanity.",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Permit network download from the HF Hub (off by default; explicit opt-in).",
    )
    p.add_argument(
        "--push-to-hub",
        default=None,
        metavar="REPO_ID",
        help="Optional HF Hub repo id to push the three configs to.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run end-to-end against a tiny synthetic in-memory dataset (no network).",
    )
    p.add_argument(
        "--smoke-scale",
        type=int,
        default=1,
        help="Synthetic dataset scale for --smoke (1 = minimal fixture; >1 for the "
        "training smoke so a 70/15/15 split is non-degenerate).",
    )
    p.add_argument(
        "--keep-negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain the dataset's provided negatives as score=0 rows in the "
        "`test` split (for the standard-IR fine-tune). Eval-invariant. "
        "Use --no-keep-negatives for a positives-only qrels.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    out_dir = Path(args.out_dir)

    if args.smoke:
        logger.info("Smoke mode: synthetic in-memory dataset (scale=%d), no network.", args.smoke_scale)
        corpus_recs, query_recs, relation_recs = load_smoke_records(args.smoke_scale)
    else:
        if not args.download and args.max_samples is None:
            raise SystemExit(
                "Refusing to pull the full DAPFAM dataset without --download. "
                "Pass --download for the full build, or --max-samples N for a "
                "bounded real-load sanity, or --smoke for the offline path."
            )
        corpus_recs, query_recs, relation_recs = load_dapfam(
            args.hf_dataset, max_samples=args.max_samples
        )

    beir_configs, stats = build_beir(
        corpus_recs, query_recs, relation_recs, args.text_view, keep_negatives=args.keep_negatives
    )
    save_local(beir_configs, out_dir)

    stats_path = out_dir / "stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Wrote stats to %s", stats_path)
    print(json.dumps(stats, indent=2))

    if args.push_to_hub:
        push_to_hub(beir_configs, args.push_to_hub)

    return 0


if __name__ == "__main__":
    sys.exit(main())
