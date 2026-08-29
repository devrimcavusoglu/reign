#!/usr/bin/env python3
"""Aggregate the E5 GoodWiki-Long training-objective ablation (paper App. I).

Every arm shares one protocol (base-l3 + GTE-small, chunk/stride 512, 20 epochs,
lr 1e-5, wd 1e-4, seed 42, best-val on nDCG@10). What varies is the loss, the
InfoNCE temperature, the distractor weight alpha, the batch size (which sets the
in-batch negative count), and warm vs cold start. One arm extends the epoch
budget to 50 to check that the 20-epoch cut is not what separates the losses.
"""
from __future__ import annotations
import argparse, glob, json, os

# (arm-key, human label). The arm key is the suffix the E5 stage writes into
# every result filename: results/e5/{goodwiki,dapfam}_<arm>.json,
# results/e5/loco_<arm>/, results/e5/mteb_<arm>/. Keep this list in sync with
# the arms the stage trains — a key with no files on disk simply renders "--".
ARMS = [
    # --- ThreeWayCosine (the released objective) ---
    ("cosine-bs18",            "ThreeWayCosine        distractors   bs18 (published recipe)"),
    ("cosine-bs48",            "ThreeWayCosine        distractors   bs48"),
    # --- InfoNCE at tau=0.07 ---
    ("infonce-pw05-bs18",      "InfoNCE t=.07  a=0.5  distractors   bs18"),
    ("infonce-pw05-bs48",      "InfoNCE t=.07  a=0.5  distractors   bs48"),
    ("infonce-pw00-bs48",      "InfoNCE t=.07  a=0    in-batch only  bs48"),
    ("infonce-pw05-warm",      "InfoNCE t=.07  a=0.5  distractors   bs48, warm-start"),
    # --- InfoNCE at tau=0.1: completes the tau x distractors grid at bs48 ---
    ("infonce-pw05-t01-bs48",  "InfoNCE t=.10  a=0.5  distractors   bs48"),
    ("infonce-pw00-t01-bs48",  "InfoNCE t=.10  a=0    in-batch only  bs48"),
    # --- epoch-budget extension of the best tau=0.07 arm ---
    ("infonce-pw00-bs48-e50",  "InfoNCE t=.07  a=0    in-batch only  bs48, 50 epochs"),
]

# Label column width for the fixed-width text table; widest label above + slack.
_LABEL_W = 62

def _load(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

def loco_macro(d, arm):
    vals = []
    for p in sorted(glob.glob(os.path.join(d, f"loco_{arm}", "loco_*.json"))):
        j = _load(p)
        if not j: continue
        m = j.get("metrics", {})
        k = next((x for x in m if x.startswith("nDCG@")), None)
        if k: vals.append(100 * m[k])
    return (sum(vals) / len(vals) if vals else None), len(vals)

def mteb_scores(d, arm):
    out = {}
    for p in glob.glob(os.path.join(d, f"mteb_{arm}", "**", "*.json"), recursive=True):
        j = _load(p)
        if not isinstance(j, dict): continue
        task = j.get("task_name") or os.path.basename(p).replace(".json", "")
        sc = j.get("scores") or {}
        for split in ("test", "dev"):
            for entry in (sc.get(split) or []):
                v = entry.get("ndcg_at_10") or entry.get("main_score")
                if v is not None:
                    out[task] = 100 * v
                    break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/e5")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    d = args.results_dir

    rows = []
    for arm, label in ARMS:
        gw = _load(os.path.join(d, f"goodwiki_{arm}.json"))
        dp = _load(os.path.join(d, f"dapfam_{arm}.json"))
        lc, n = loco_macro(d, arm)
        rows.append(dict(arm=arm, label=label,
                         gw=100 * gw["metrics"]["nDCG@10"] if gw else None,
                         loco=lc, loco_n=n,
                         dapfam=100 * dp["metrics"]["nDCG@100"] if dp else None,
                         mteb=mteb_scores(d, arm)))

    def f(v, w=8):
        return f"{'--':>{w}}" if v is None else f"{v:{w}.2f}"

    print("\nE5 — GoodWiki-Long training-objective ablation "
          "(base-l3 + GTE-small, chunk/stride 512, 20 epochs, seed 42)")
    print("Published reference (cosine, bs18, 50 epochs): GoodWiki 67.09 @s512 / 67.31 @s384, "
          "LoCo 68.92, DAPFAM 31.69\n")
    hdr = f"{'Arm':{_LABEL_W}s} {'GoodWiki':>9s} {'LoCo':>9s} {'DAPFAM':>9s}   MTEB"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        mt = "  ".join(f"{k}={v:.2f}" for k, v in sorted(r["mteb"].items())) or "--"
        loco = f(r["loco"], 9) + (f" [{r['loco_n']}/12]" if r["loco"] is not None and r["loco_n"] < 12 else "")
        print(f"{r['label']:{_LABEL_W}s} {f(r['gw'],9)} {loco:>9s} {f(r['dapfam'],9)}   {mt}")

    done = sum(1 for r in rows if r["gw"] is not None)
    print(f"\narms with GoodWiki results: {done}/{len(ARMS)}")

    if args.markdown:
        L = ["| Arm | GoodWiki nDCG@10 | LoCo macro | DAPFAM nDCG@100 | MTEB |", "|---|---:|---:|---:|---|"]
        for r in rows:
            mt = ", ".join(f"{k} {v:.2f}" for k, v in sorted(r["mteb"].items())) or "--"
            # Labels are padded for the fixed-width text table above; markdown
            # lays the columns out itself, so collapse the alignment spacing.
            label = " ".join(r["label"].split())
            L.append(f"| {label} | {f(r['gw'],0).strip()} | {f(r['loco'],0).strip()} | "
                     f"{f(r['dapfam'],0).strip()} | {mt} |")
        with open(args.markdown, "w") as fh: fh.write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
