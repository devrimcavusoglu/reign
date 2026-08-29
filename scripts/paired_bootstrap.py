#!/usr/bin/env python3
"""Paired significance testing over per-query retrieval scores.

Produces paper Table 5: a paired test over the 187 per-query nDCG@100 scores on
DAPFAM, comparing REIGN against named baselines, with confidence intervals.
Both a bootstrap and a randomization test are reported, because they answer
different questions and disagreeing results would themselves be informative:

  * **Paired bootstrap** gives the confidence interval on the mean difference.
    Queries are resampled with replacement; the two systems are always scored
    on the same resampled query set, which is what makes it *paired* and what
    removes per-query difficulty as a source of variance.
  * **Randomization (permutation)** gives the p-value. Under the null the two
    systems are interchangeable on each query, so each query's difference is
    independently sign-flipped; p is the share of shuffles whose |mean| is at
    least the observed |mean|. This is the standard IR significance test
    (Smucker et al., 2007).

With several comparisons against one system, Holm-Bonferroni adjusted p-values
are reported alongside the raw ones.

Usage:
    python scripts/paired_bootstrap.py \\
        --system results/e2/reign_gte-large_s512.json \\
        --baseline results/e2/dense_jina-v3.json \\
        --baseline results/e2/dense_stella-1.5b.json \\
        --baseline results/e2/dense_gte-large-chunked.json \\
        --metric nDCG@100 --output results/e2/significance.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("paired_bootstrap")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True, help="result JSON for the system under test")
    p.add_argument("--baseline", action="append", required=True, help="repeatable baseline JSON")
    p.add_argument("--metric", default="nDCG@100")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output", default=None, help="write full results as JSON here")
    p.add_argument("--markdown", default=None, help="also write a paper-ready markdown table")
    return p.parse_args()


def load_per_query(path: str, metric: str) -> tuple[str, dict[str, float]]:
    with open(path) as f:
        d = json.load(f)
    pq = d.get("per_query")
    if not pq:
        raise SystemExit(
            f"{path} has no 'per_query' block — re-run the evaluation with the "
            "per-query dump enabled (reign.encoders.eval_utils.build_per_query_payload)."
        )
    missing = [q for q, v in pq.items() if metric not in v]
    if missing:
        raise SystemExit(f"{path}: metric {metric!r} missing for {len(missing)} queries")
    name = d.get("encoder") or d.get("baseline") or d.get("retriever") or Path(path).stem
    # Self-check: the stored aggregate must be the mean of the per-query values.
    scores = {q: float(v[metric]) for q, v in pq.items()}
    agg = d.get("metrics", {}).get(metric)
    if agg is not None:
        mean = float(np.mean(list(scores.values())))
        if abs(mean - float(agg)) > 1e-6:
            raise SystemExit(
                f"{path}: per-query mean {mean:.8f} != reported aggregate {float(agg):.8f}"
            )
    return name, scores


def paired_test(sys_scores: np.ndarray, base_scores: np.ndarray, *, n_boot: int, n_perm: int,
                seed: int, alpha: float) -> dict:
    rng = np.random.default_rng(seed)
    diff = sys_scores - base_scores
    n = len(diff)
    observed = float(diff.mean())

    # --- paired bootstrap CI: resample queries, keep systems aligned ---
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # --- randomization test: sign-flip each query's difference ---
    signs = rng.choice((-1.0, 1.0), size=(n_perm, n))
    perm = (signs * diff).mean(axis=1)
    p_perm = float((np.abs(perm) >= abs(observed) - 1e-15).mean())

    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    return {
        "n_queries": n,
        "mean_system": float(sys_scores.mean()),
        "mean_baseline": float(base_scores.mean()),
        "mean_difference": observed,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "ci_level": 1 - alpha,
        "p_randomization": p_perm,
        "significant": bool(p_perm < alpha),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / n,
        "n_boot": n_boot,
        "n_perm": n_perm,
        "seed": seed,
    }


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def main():
    args = parse_args()
    sys_name, sys_scores = load_per_query(args.system, args.metric)
    logger.info("System: %s (%d queries)", sys_name, len(sys_scores))

    rows = []
    for bpath in args.baseline:
        b_name, b_scores = load_per_query(bpath, args.metric)
        common = sorted(set(sys_scores) & set(b_scores))
        if len(common) != len(sys_scores) or len(common) != len(b_scores):
            logger.warning(
                "%s: query sets differ (system=%d baseline=%d common=%d) — testing on the "
                "common subset", b_name, len(sys_scores), len(b_scores), len(common),
            )
        s = np.array([sys_scores[q] for q in common], dtype=np.float64)
        b = np.array([b_scores[q] for q in common], dtype=np.float64)
        res = paired_test(s, b, n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed,
                          alpha=args.alpha)
        res["system"] = sys_name
        res["baseline"] = b_name
        res["baseline_path"] = bpath
        rows.append(res)
        logger.info(
            "%s vs %s: d=%+.4f 95%% CI [%+.4f, %+.4f] p=%.4f (%s)",
            sys_name, b_name, res["mean_difference"], res["ci_lower"], res["ci_upper"],
            res["p_randomization"], "significant" if res["significant"] else "n.s.",
        )

    for row, adj in zip(rows, holm([r["p_randomization"] for r in rows])):
        row["p_holm"] = adj
        row["significant_holm"] = bool(adj < args.alpha)

    out = {
        "metric": args.metric,
        "system_path": args.system,
        "alpha": args.alpha,
        "comparisons": rows,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Wrote %s", args.output)

    # Human-readable summary (percentage points, the paper's unit).
    print(f"\nPaired comparison on {args.metric} — {sys_name}")
    print(f"{'baseline':22s} {'sys':>7s} {'base':>7s} {'delta':>8s} "
          f"{'95% CI':>18s} {'p':>8s} {'p_holm':>8s} {'W/L/T':>13s}")
    for r in rows:
        ci = f"[{100*r['ci_lower']:+.2f}, {100*r['ci_upper']:+.2f}]"
        print(f"{r['baseline'][:22]:22s} {100*r['mean_system']:7.2f} {100*r['mean_baseline']:7.2f} "
              f"{100*r['mean_difference']:+8.2f} {ci:>18s} {r['p_randomization']:8.4f} "
              f"{r['p_holm']:8.4f} {r['wins']:4d}/{r['losses']:3d}/{r['ties']:3d}")
    print("\nVerdict per comparison (alpha = %.2f, Holm-adjusted):" % args.alpha)
    for r in rows:
        verdict = ("REIGN significantly better" if r["significant_holm"] and r["mean_difference"] > 0
                   else "REIGN significantly worse" if r["significant_holm"]
                   else "statistical parity")
        print(f"  vs {r['baseline']}: {verdict}")

    if args.markdown:
        lines = [
            f"| Comparison | {args.metric} (sys) | {args.metric} (base) | Δ | 95% CI | p | p (Holm) | W/L/T |",
            "|---|---:|---:|---:|:---:|---:|---:|:---:|",
        ]
        for r in rows:
            lines.append(
                f"| {sys_name} vs {r['baseline']} | {100*r['mean_system']:.2f} | "
                f"{100*r['mean_baseline']:.2f} | {100*r['mean_difference']:+.2f} | "
                f"[{100*r['ci_lower']:+.2f}, {100*r['ci_upper']:+.2f}] | "
                f"{r['p_randomization']:.3f} | {r['p_holm']:.3f} | "
                f"{r['wins']}/{r['losses']}/{r['ties']} |"
            )
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text("\n".join(lines) + "\n")
        logger.info("Wrote %s", args.markdown)


if __name__ == "__main__":
    main()
