#!/usr/bin/env python3
"""Aggregate the E4 positional-encoding ablation (paper App. E).

Three arms (none / absolute / sinusoidal) trained under one identical protocol,
each evaluated in-distribution (GoodWiki-Long) and zero-shot (LoCo macro,
DAPFAM). Section 4.1's design justification follows this table, so the table
reports the arms side by side with the delta against the published `none`
design rather than declaring a winner.

Usage: python scripts/_print_e4_table.py [--results-dir results/e4]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

ARMS = ["none", "absolute", "sinusoidal"]
LABEL = {
    "none": "None (published design)",
    "absolute": "Learned absolute",
    "sinusoidal": "Sinusoidal",
}


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def loco_macro(results_dir: str, arm: str) -> tuple[float | None, int]:
    """Macro-average nDCG@10 across the LoCo subtasks present for this arm."""
    paths = sorted(glob.glob(os.path.join(results_dir, f"loco_pe-{arm}", "loco_*.json")))
    vals = []
    for p in paths:
        d = _load(p)
        if not d:
            continue
        m = d.get("metrics", {})
        key = next((k for k in m if k.startswith("nDCG@")), None)
        if key:
            vals.append(float(m[key]))
    return (100 * sum(vals) / len(vals) if vals else None), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/e4")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    rows = []
    for arm in ARMS:
        gw = _load(os.path.join(args.results_dir, f"goodwiki_pe-{arm}.json"))
        dp = _load(os.path.join(args.results_dir, f"dapfam_pe-{arm}.json"))
        lc, n_sub = loco_macro(args.results_dir, arm)
        rows.append({
            "arm": arm,
            "goodwiki": 100 * gw["metrics"]["nDCG@10"] if gw else None,
            "loco": lc,
            "loco_n": n_sub,
            "dapfam": 100 * dp["metrics"]["nDCG@100"] if dp else None,
        })

    base = next((r for r in rows if r["arm"] == "none"), None)

    def cell(v, ref):
        if v is None:
            return f"{'--':>16s}"
        if ref is None or ref is v:
            return f"{v:16.2f}"
        return f"{v:9.2f} ({v - ref:+.2f})"

    print("\nE4 — positional-encoding ablation (base-l3 + GTE-small, chunk 512 / stride 512)")
    print("All arms trained under one identical protocol; delta is against the published `none` arm.\n")
    print(f"{'Chunk position signal':26s} | {'GoodWiki nDCG@10':>16s} | {'LoCo macro nDCG@10':>18s} | {'DAPFAM nDCG@100':>16s}")
    print("-" * 26 + "-|-" + "-" * 16 + "-|-" + "-" * 18 + "-|-" + "-" * 16)
    for r in rows:
        loco_txt = cell(r["loco"], base["loco"] if base else None)
        if r["loco"] is not None and r["loco_n"] < 12:
            loco_txt += f" [{r['loco_n']}/12]"
        print(f"{LABEL[r['arm']]:26s} | {cell(r['goodwiki'], base['goodwiki'] if base else None)} "
              f"| {loco_txt:>18s} | {cell(r['dapfam'], base['dapfam'] if base else None)}")

    done = sum(1 for r in rows if r["goodwiki"] is not None)
    print(f"\narms with in-distribution results: {done}/3")
    missing = [r["arm"] for r in rows if r["goodwiki"] is None]
    if missing:
        print(f"still pending: {', '.join(missing)}")

    if args.markdown:
        lines = ["| Chunk position signal | GoodWiki nDCG@10 | LoCo macro nDCG@10 | DAPFAM nDCG@100 |",
                 "|---|---:|---:|---:|"]
        for r in rows:
            def md(v, ref):
                if v is None:
                    return "--"
                return f"{v:.2f}" if (ref is None or v is ref) else f"{v:.2f} ({v - ref:+.2f})"
            lines.append(
                f"| {LABEL[r['arm']]} | {md(r['goodwiki'], base['goodwiki'] if base else None)} "
                f"| {md(r['loco'], base['loco'] if base else None)} "
                f"| {md(r['dapfam'], base['dapfam'] if base else None)} |"
            )
        with open(args.markdown, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
