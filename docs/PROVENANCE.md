# Provenance

Which protocol produced which artefact, where the reference numbers live, and the integrity guarantees around cached embeddings.

## Released checkpoints

Every released checkpoint — the 23 `_val-selected` names in the [model zoo](../README.md#model-zoo) — was trained under **Protocol A**, the three-way cosine embedding recipe: λ = 0.5, batch 18 (360 pairs per step), AdamW at lr 1e-5 with weight decay 1e-4 and cosine annealing, 50 epochs, 16-mixed precision, seed 42, over cached frozen-GN chunk embeddings.

The `_val-selected` suffix is literal: the released weights are the best-validation snapshot, selected on nDCG@10 over the `val` qrels split with validation run every 4 epochs. Each checkpoint directory carries both the selected `best/` snapshot and the final `last/` snapshot; every paper number uses `best/`.

The DAPFAM fine-tuned family is separate. Those checkpoints use InfoNCE at temperature 0.07 over DAPFAM's provided negatives, because that dataset's relevance labels are binary. Full details in [TRAINING.md](TRAINING.md).

## Ablation arms

The Appendix E (positional-encoding) and Appendix I (training-objective) ablation arms were **not** trained under Protocol A. They use **Protocol B**: InfoNCE at τ = 0.07, batch 48, 20 epochs, `base-l3` on GTE-small at chunk/stride 512.

This is deliberate. Within each ablation, all arms share one protocol, so the comparison between arms is internally valid. It also means the arms sit below the released operating point of Table 2, and an ablation arm's absolute number is not comparable to a headline table. The paper states this in both appendices.

## Per-table pointers

| Paper artifact | Reference output | Produced by |
| --- | --- | --- |
| Table 1 — dataset statistics | — (read off the dataset card) | — |
| Table 2 — GoodWiki-Long | `results/reference/goodwiki/goodwiki_test.json` | `main-goodwiki` |
| Table 3 — LoCo | `results/reference/loco/loco_subtask_ndcg.json` | `main-loco` |
| Table 4 — DAPFAM | `results/reference/dapfam/dapfam_test.json` | `main-dapfam` |
| Table 5 — significance | `results/reference/e2/significance_dapfam_5way.{json,md}` | `e2-significance` |
| Table 6 — MTEB | `results/reference/mteb/mteb_arguana_fiqa.json` | `mteb` |
| Table 7 — capacity sweep, in-distribution | `results/reference/goodwiki/goodwiki_test.json` | `main-goodwiki` |
| Table 8 — capacity head-to-head, OOD | `results/reference/dapfam/dapfam_ood_head_to_head.json`, `results/reference/loco/loco_subtask_ndcg.json` | `main-dapfam`, `main-loco` |
| Table 9 — positional encoding | `results/reference/e4/e4_dapfam_pe-{none,absolute,sinusoidal}.json` | `e4-pe-ablation` |
| Table 10 — analytic FLOPs | — (analytic, not measured) | — |
| Table 11 — measured cost | `results/reference/e1/e1_table.{md,tex}` | `e1-efficiency` |
| Table 12 — training objective | `results/reference/e5/e5_table.md` | `e5-objective-ablation` |
| Section 5.3 / App. J — DAPFAM fine-tunes | `results/reference/dapfam/dapfam_finetune_family.json` | `main-dapfam` |
| Figures 3, 5 | plotted from the result JSONs above | figure generator |

That is the whole of `results/reference/` — fourteen files. The reference files carry aggregate metrics and run-identifying configuration, not raw run outputs; the one exception is the three Table 9 arms, which retain per-query DAPFAM scores because the ablation's verdict is a within-arm per-query comparison. In particular the Table 5 per-query dumps are **not** shipped: the significance output is, and `main-dapfam` regenerates the dumps behind it. [REPRODUCING.md](REPRODUCING.md#what-ships-in-resultsreference) enumerates every file, what it lets you diff, and what deliberately does not ship, alongside the stage-by-stage commands and budgets.

Two rows of Table 5 are re-measurements rather than bit-exact reproductions of Table 4 and are labelled as such in the paper caption: Jina-v3 at 32.98 against the published 32.97 (fp16 non-determinism) and Stella-1.5B at 33.00 against 32.91 (re-measured at batch 8 rather than batch 4). The other rows match Table 4 exactly.

## Cache integrity

Two caches sit on the critical path, and both are content-addressed.

**GN chunk-embedding cache** (`--cache-root`, default `~/.reign_cache`). The frozen GN's outputs are deterministic for a fixed GN, chunk size, stride, and input, so they are hashed and stored to HDF5 keyed on exactly those. Different strides and different GN models resolve to different keys, which is what makes the stride sweep safe to run against one cache root.

**Corpus document-embedding cache** (`--corpus-embed-cache`). The post-REIGN corpus embedding matrix is keyed by a SHA-256 fingerprint over the checkpoint identity, the GN model, the chunk size, the stride, and the full corpus content — document ids and texts. Two different checkpoints or two different strides therefore get distinct keys and cannot cross-contaminate. The key deliberately excludes the query split, so `test`, `test_in`, and `test_out` share one corpus pool, resolve to a single entry, and only the first pays the encode.

The evaluator refuses stale caches rather than silently trusting them:

- **On read**, every entry carries a SHA-256 fingerprint of the checkpoint's weight files, stored in the `.npz` beside the embeddings. If it does not match the checkpoint being evaluated — or if the entry predates fingerprinting and carries none — the evaluator aborts with `SystemExit` rather than quietly falling back; `--refresh-corpus-cache` re-encodes and overwrites instead. A stored document-id list that disagrees with the corpus actually loaded is a content-key collision rather than a stale model, so that case is logged and recomputed.
- **On write**, the cache is verified before any later split is allowed to trust it. The file must round-trip bit-identically, and the full retrieval-and-metrics path recomputed from the reloaded array must produce identical metrics. Either check failing raises a hard error rather than emitting numbers from an unverified cache.

**Never retrain into an existing checkpoint directory — a new run gets a new directory.** Because the corpus-embedding cache *key* is derived from checkpoint identity rather than from the weight bytes, replacing the weights inside a directory a cache entry already references would otherwise leave that entry pointing at superseded weights; the weights fingerprint stored in the entry is what turns that situation into a hard failure instead of a silently wrong number. The trainer enforces the rule directly: it refuses to create an output directory that already exists.

For completeness: an earlier internal ablation evaluation was invalidated by stale cache reuse and was re-run with fresh encodings. The fingerprint and self-check guards described above make this class of error impossible.

## Dataset provenance

`devrim/goodwiki_long_synthetic_ir` derives from GoodWiki, a cleaned release of English Wikipedia in structured markdown, filtered to articles exceeding 16,000 characters. Queries are the original articles. Corpus documents are LLM rephrasals generated with GPT-4o-mini plus topical distractors, giving graded relevance — score 2 for the rephrasal, score 1 for a distractor.

The rephrased corpus documents are machine-generated and are marked as synthetic in the dataset card. The dataset is released under CC BY-SA 4.0, preserving GoodWiki's and Wikipedia's share-alike licensing and attribution.

DAPFAM is built from public USPTO patent records and ships eval-only; the query-disjoint stratified 70/15/15 train/val/test partition used here is constructed locally at seed 42 and is deterministic, so re-running the split step reproduces the same partition. LoCo is used zero-shot with no LoCo-specific training.
