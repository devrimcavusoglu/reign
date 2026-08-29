#!/usr/bin/env python3
"""Aggregate the E1 compute sweep into the measured-efficiency table
(paper App. G, Table 11).

Reads results/e1/compute_*.json and emits markdown (for review) and LaTeX (for
the appendix, beside the analytic FLOPs table). The point of the table is the
pairing:

  * REIGN *uncached* sits next to its own chunked-GN baseline. The two should
    cost about the same, because over k chunk embeddings the REIGN encoder adds
    almost nothing to the Guidance Network forward pass — this table is what
    makes that checkable rather than asserted.
  * REIGN *cached* is the steady state that the training-cost claim relies on,
    with the one-time cache-build cost reported separately so the amortisation
    is visible rather than assumed.

Usage:
    python scripts/_print_e1_table.py [--results-dir results/e1] [--latex out.tex]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# Display order and grouping. (row-key, label, group)
LAYOUT = [
    ("sparse_bm25", "BM25", "Sparse lexical"),
    ("sparse_tfidf", "TF-IDF", "Sparse lexical"),
    ("dense_bge-m3", "BGE-M3", "Native long-context dense (truncate 8K)"),
    ("dense_jina-v3", "Jina-Embeddings-v3", "Native long-context dense (truncate 8K)"),
    ("dense_stella-1.5b", "Stella-en-1.5B-v5", "Native long-context dense (truncate 8K)"),
    ("dense_nomic-v1.5", "Nomic-Embed-v1.5", "Native long-context dense (truncate 8K)"),
    ("dense_gte-small-chunked", "GTE-small (chunked)", "Bare GN (chunked mean-pool, 512)"),
    ("dense_gte-base-chunked", "GTE-base (chunked)", "Bare GN (chunked mean-pool, 512)"),
    ("dense_gte-large-chunked", "GTE-large (chunked)", "Bare GN (chunked mean-pool, 512)"),
    ("dense_bge-base-chunked", "BGE-base (chunked)", "Bare GN (chunked mean-pool, 512)"),
    ("dense_bge-large-chunked", "BGE-large (chunked)", "Bare GN (chunked mean-pool, 512)"),
    ("reign_gte-small_uncached", "REIGN + GTE-small", "REIGN, uncached GN (cold cache)"),
    ("reign_gte-base_uncached", "REIGN + GTE-base", "REIGN, uncached GN (cold cache)"),
    ("reign_gte-large_uncached", "REIGN + GTE-large", "REIGN, uncached GN (cold cache)"),
    ("reign_gte-small_cached", "REIGN + GTE-small", "REIGN, cached GN embeddings (warm)"),
    ("reign_gte-base_cached", "REIGN + GTE-base", "REIGN, cached GN embeddings (warm)"),
    ("reign_gte-large_cached", "REIGN + GTE-large", "REIGN, cached GN embeddings (warm)"),
]


def load(results_dir: str) -> dict[str, dict]:
    out = {}
    for path in glob.glob(os.path.join(results_dir, "compute_*.json")):
        key = os.path.basename(path)[len("compute_") : -len(".json")]
        with open(path) as f:
            out[key] = json.load(f)
    return out


def _ms_per_query(d: dict) -> float | None:
    v = d.get("per_query_total_us")
    return v / 1000.0 if v is not None else None


def _index_s(d: dict) -> float | None:
    return (d.get("index_seconds") or {}).get("mean")


def _peak_gb(d: dict) -> float | None:
    peaks = [d.get("peak_gpu_bytes_index"), d.get("peak_gpu_bytes_query")]
    peaks = [p for p in peaks if p]
    return max(peaks) / 1e9 if peaks else None


def _fmt(v, spec="7.1f", none="--"):
    return none if v is None else f"{v:{spec}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/e1")
    ap.add_argument("--latex", default=None, help="also write a LaTeX table here")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    rows = load(args.results_dir)
    if not rows:
        raise SystemExit(f"no compute_*.json under {args.results_dir}")

    any_row = next(iter(rows.values()))
    n_corpus, n_queries = any_row.get("n_corpus"), any_row.get("n_queries")

    lines = []
    lines.append(
        f"Protocol: {n_corpus} corpus docs, {n_queries} queries, GoodWiki-Long test, "
        f"batch 8 (Jina-v3 at batch 4 — batch 8 exceeds 24 GB on this card), "
        f"{any_row.get('n_warmup')} warmup + {any_row.get('n_repeat')} timed repeats, "
        "single RTX 4090 (24 GB), otherwise-idle GPU.\n"
    )
    # No Params column: DenseEncoder.n_params is unreliable for adapter models
    # (it reports 12.9M for Jina-v3, which the paper lists at 572M), and the
    # accuracy tables already carry authoritative parameter counts. This table's
    # job is latency and memory.
    header = (f"| {'System':26s} | {'Index (s)':>9s} | {'ms/query':>9s} "
              f"| {'Peak GPU (GB)':>13s} | {'Index (MB)':>10s} |")
    sep = "|" + "-" * 28 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 15 + "|" + "-" * 12 + "|"
    lines += [header, sep]

    current_group = None
    missing = []
    for key, label, group in LAYOUT:
        d = rows.get(key)
        if d is None:
            missing.append(key)
            continue
        if group != current_group:
            lines.append(f"| *{group}* | | | | |")
            current_group = group
        lines.append(
            f"| {label:26s} "
            f"| {_fmt(_index_s(d), '9.1f')} | {_fmt(_ms_per_query(d), '9.1f')} "
            f"| {_fmt(_peak_gb(d), '13.2f')} "
            f"| {_fmt((d.get('index_bytes') or 0) / 1e6, '10.1f')} |"
        )

    # Cache-build rows: the amortised one-time cost.
    build_lines = []
    for key, d in sorted(rows.items()):
        if not key.endswith("_build"):
            continue
        short = key[len("reign_") : -len("_build")]
        t = d.get("cache_build_seconds")
        per = d.get("cache_build_seconds_per_doc")
        size = d.get("cache_bytes_docs")
        build_lines.append(
            f"| REIGN + {short:12s} | {_fmt(t, '8.1f')} s | {_fmt(per and per * 1000, '8.1f')} ms/doc "
            f"| {_fmt(size and size / 1e6, '8.1f')} MB |"
        )
    if build_lines:
        lines += ["", "**One-time GN cache build** (the cost being amortised):", "",
                  "| System | Build time | Per document | Cache size |",
                  "|---|---:|---:|---:|"] + build_lines

    # The headline pairing: uncached REIGN vs its own chunked GN.
    pair_lines = []
    for short in ("gte-small", "gte-base", "gte-large"):
        u = rows.get(f"reign_{short}_uncached")
        c = rows.get(f"reign_{short}_cached")
        b = rows.get(f"dense_{short}-chunked")
        if not (u and b):
            continue
        ur, br = _ms_per_query(u), _ms_per_query(b)
        cr = _ms_per_query(c) if c else None
        ratio = ur / br if (ur and br) else None
        speedup = ur / cr if (ur and cr) else None
        pair_lines.append(
            f"| {short:10s} | {_fmt(br, '8.1f')} | {_fmt(ur, '8.1f')} | "
            f"{_fmt(ratio, '6.2f')}x | {_fmt(cr, '8.1f')} | {_fmt(speedup, '6.1f')}x |"
        )
    if pair_lines:
        lines += ["", "**Uncached REIGN vs its own chunked GN**, "
                  "**and the cached speed-up**:", "",
                  "| GN | GN chunked (ms/q) | REIGN uncached (ms/q) | ratio | REIGN cached (ms/q) | cached speed-up |",
                  "|---|---:|---:|---:|---:|---:|"] + pair_lines

    if missing:
        lines += ["", f"_Missing rows (not yet run or failed): {', '.join(missing)}_"]

    text = "\n".join(lines)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(text + "\n")

    if args.latex:
        tex = [
            r"\begin{table}[t]", r"\centering", r"\small",
            r"\caption{\textbf{Measured inference cost} on GoodWiki-Long "
            f"({n_corpus} documents, {n_queries} queries), single RTX~4090 (24\\,GB), "
            r"otherwise-idle GPU. Latency is end-to-end per query (encode + retrieve). "
            r"REIGN is shown with the Guidance Network run per query (\emph{uncached}) "
            r"and with its chunk embeddings served from disk (\emph{cached}). "
            r"Jina-v3 measured at batch 4 (batch 8 exceeds 24\,GB); all other dense "
            r"rows at batch 8.}",
            r"\label{tab:measured_efficiency}",
            r"\begin{tabular}{lrrr}", r"\toprule",
            r"\textbf{System} & \textbf{Index (s)} & \textbf{ms/query} & \textbf{Peak GPU (GB)} \\",
            r"\midrule",
        ]
        current_group = None
        for key, label, group in LAYOUT:
            d = rows.get(key)
            if d is None:
                continue
            if group != current_group:
                tex.append(r"\multicolumn{4}{l}{\emph{" + group + r"}}\\")
                current_group = group
            tex.append(
                f"{label} & {_fmt(_index_s(d), '.1f')} & {_fmt(_ms_per_query(d), '.1f')} "
                f"& {_fmt(_peak_gb(d), '.2f')} \\\\"
            )
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        with open(args.latex, "w") as f:
            f.write("\n".join(tex) + "\n")
        print(f"\n[wrote LaTeX to {args.latex}]")


if __name__ == "__main__":
    main()
