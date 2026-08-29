# Reproducing the paper

Every table and figure in the paper maps to a stage of `scripts/reproduce.sh`. This document gives, per stage, what it produces, the command underneath it, the reference output to compare against, and the runtime and hardware budget.

```bash
bash scripts/reproduce.sh <stage>
```

Stages take no positional arguments; configuration is by environment variable — `MODELS_DIR` (default `./models`), `RESULTS_DIR` (default `./results`), `PYTHON`, `DAPFAM_DATA`, and `GN_STRIDE`; `bash scripts/reproduce.sh help` lists them. Every stage writes under `RESULTS_DIR` and is re-runnable: a row whose output file already exists is skipped, so an interrupted queue resumes by re-invoking the same command (`main-loco` writes one file per subtask, so its check is "all twelve subtask files present"). `bash scripts/reproduce.sh all` runs every stage below in order.

The default row set of each stage is the headline table it belongs to. The appendix sweeps run through row-set overrides — `REIGN_ROWS_OVERRIDE`, `DAPFAM_ZS_ROWS_OVERRIDE`, `DAPFAM_SPLITS`, `DENSE_NATIVE`, `DENSE_CHUNKED`, `DENSE_DAPFAM`, `MTEB_TASKS`, `MTEB_CKPT`, `MTEB_GN`, `FT_RUNS_OVERRIDE` — with the exact values for each paper table listed under [Rows the default stages do not run](#rows-the-default-stages-do-not-run).

## Before you start

**Checkpoints.** Stages load REIGN checkpoints from the models directory — `./models` by default, overridable with `MODELS_DIR`. (`reproduce.sh` exports `REIGN_MODEL_DIR` from `MODELS_DIR` so the training stages write where the evaluation steps look; set `MODELS_DIR`, not `REIGN_MODEL_DIR`, or the two will disagree.) A checkpoint is a directory containing `config.json` and `model.safetensors`; the `best/` snapshot is the val-selected one and is what every stage uses. Automatic download from the Hugging Face Hub lands together with the weights upload; until then, place checkpoint directories under the models directory yourself. Checkpoint names are listed in the model zoo in the [README](../README.md).

**Guidance networks.** The GN backbones (`thenlper/gte-{small,base,large}`, `BAAI/bge-{base,large}-en-v1.5`) are pulled from the Hub on first use and cached by `transformers`.

**Datasets.** `devrim/goodwiki_long_synthetic_ir` is pulled from the Hub. LoCo is fetched by the LoCo runner. DAPFAM ships eval-only and is built and split locally before first use; the `main-dapfam` stage does this for you (build the FullText view, then a query-disjoint stratified 70/15/15 split at seed 42, matching Appendix J).

**Embedding caches.** GN chunk embeddings are cached under `~/.reign_cache` (`--cache-root`, the code default). Post-REIGN corpus document-embedding matrices go wherever `--corpus-embed-cache` points: `reproduce.sh` puts them in `$RESULTS_DIR/.corpus_emb_cache`, and `dapfam_finetune.sh` defaults to `.cache/reign/corpus_emb` (override with `CORPUS_EMB_CACHE`). Both caches are content-addressed; see [PROVENANCE.md](PROVENANCE.md) for the integrity guarantees.

**Hardware.** Everything in the paper runs on a single 24 GB consumer GPU (RTX 4090). Budgets below assume that card.

## Stage map

| Paper artifact | Stage | Notes |
| --- | --- | --- |
| Table 1 (dataset statistics) | — | Read off the dataset card; no run required. |
| Table 2 — GoodWiki-Long test | `main-goodwiki` | |
| Table 3 — LoCoV1 per-subtask | `main-loco` | |
| Table 4 — DAPFAM patent retrieval | `main-dapfam` | |
| Table 5 — DAPFAM paired significance | `e2-significance` | Depends on `main-dapfam` re-run with per-query dumps. |
| Table 6 — MTEB short-context | `mteb` | |
| Table 7 — encoder-capacity sweep, in-distribution | `main-goodwiki` | Same stage under `REIGN_ROWS_OVERRIDE`; values [below](#table-7--encoder-capacity-sweep-in-distribution). |
| Table 8 — encoder-capacity head-to-head, OOD | `main-dapfam`, `main-loco` | Same stages under the row overrides; values [below](#table-8--encoder-capacity-head-to-head-ood). |
| Table 9 — chunk-position ablation | `e4-pe-ablation` | Trains three arms from scratch. |
| Table 10 — analytic FLOPs | — | Analytic, from the cost model in Appendix G; not a measurement. |
| Table 11 — measured inference cost | `e1-efficiency` | Requires an otherwise-idle GPU. |
| Table 12 — training-objective ablation | `e5-objective-ablation` | Trains nine arms from scratch. |
| Figure 1 — token-length distributions | — | Dataset histogram helper; reads the dataset, writes the PDF. |
| Figure 2 — architecture schematic | — | Diagram, not generated from a run. |
| Figure 3 — parameter-efficiency Pareto | — | Figure generator over `main-*` result JSONs. |
| Figure 4 — training-pipeline schematic | — | Diagram, not generated from a run. |
| Figure 5 — capacity sweet-spot reversal | — | Figure generator over the Table 7 and Table 8 result JSONs. |

Figures 3 and 5 are plotted from result files, so they need the relevant stages to have run first; they add no new computation.

---

## `main-goodwiki` — Table 2 (and Table 7)

**Produces.** nDCG@10 on the GoodWiki-Long `test` split (5,854 queries against the full 53,562-document corpus, self-matches removed) for sparse baselines, native long-context dense baselines, bare-GN truncated and chunked mean-pool baselines, and REIGN. One shared evaluation implementation scores every system over identical qrels, with exponential graded gains (2^rel − 1).

**Underneath.** Three runners, one per system class:

```bash
python scripts/evaluate_sparse_baselines.py --retriever bm25 \
  --dataset devrim/goodwiki_long_synthetic_ir --split test --top_k 10 \
  --output_path results/sparse_bm25_test.json

python scripts/evaluate_dense_baselines.py --baseline jina-v3 \
  --dataset devrim/goodwiki_long_synthetic_ir --split test --top_k 10 \
  --batch_size 8 --torch-dtype float16 \
  --output_path results/dense_jina-v3_test.json

python scripts/evaluate_reign.py \
  --checkpoint models/reign-base-l3_gn-gte-small_s512_val-selected/best \
  --gn-model thenlper/gte-small --gn-chunk-size 512 --gn-stride 512 \
  --dataset devrim/goodwiki_long_synthetic_ir --split test \
  --top_k 10 --batch_size 8 --output_path results/reign_gte-small_goodwiki_test.json
```

Table 7 is the same REIGN command over the `tiny-l1`, `small-l2`, `base-l3` and `large-l4` checkpoints for each GTE backbone; Table 2's REIGN rows are the `base-l3` ones at the best stride per GN. The stage reaches those checkpoints through `REIGN_ROWS_OVERRIDE` — see [Table 7](#table-7--encoder-capacity-sweep-in-distribution).

**Reference.** `results/reference/goodwiki/goodwiki_test.json` — one entry per system, carrying the run's checkpoint, GN, chunk size and stride, the full aggregate `metrics_pct` block, and `paper_value`, the nDCG@10 printed in the paper. It covers all twenty-one rows of Table 2, the stride-512 companion of the Table 2 GTE-small row, and all nine non-`base-l3` cells of Table 7. Diff the `metrics` object of your `results/<name>.json` against the `metrics_pct` of the matching entry (the reference is in percent; stage outputs are fractions).

**Budget.** Sparse rows are CPU-only and finish in minutes. Dense baselines each re-encode the full 53,562-document corpus; this is materially cheaper than the same work on DAPFAM, whose documents are several times longer (corpus median 1,805 tokens here against 7,346 whitespace tokens on DAPFAM), and the 1.5B-parameter row is the slowest. REIGN rows are dominated by the GN pass on a cold cache and are fast once the chunk-embedding cache is warm.

## `main-loco` — Table 3 (and part of Table 8)

**Produces.** Per-subtask nDCG@10 across the twelve LoCoV1 subtasks plus the macro-average, zero-shot: the GoodWiki-Long-trained checkpoint with no LoCo-specific training. Baseline rows in the paper's upper block are reproduced from the LoCo paper rather than re-run.

**Underneath.**

```bash
python scripts/evaluate_loco.py \
  --reign-checkpoint models/reign-base-l3_gn-gte-small_s512_val-selected/best \
  --gn-model thenlper/gte-small --gn-chunk-size 512 --gn-stride 512 \
  --subtask all --output-dir results/loco_gte-small_s512 --tag gte-small-s512
```

The bare-GN chunked rows use the same runner with `--baseline` instead of `--reign-checkpoint`. Table 3 reports both stride 384 and stride 512 for each GN, so each configuration is run twice — once per `GN_STRIDE` value, since the stride is part of the `--tag` and therefore of every output filename.

**Reference.** `results/reference/loco/loco_subtask_ndcg.json` — one entry per evaluated system with nDCG@10 for each of the twelve subtasks and the macro-average, for all fifteen runs the paper reports: the three bare-GN chunked baselines and the `base-l3` and `small-l2` REIGN rows on each GTE backbone at both strides (Table 3 plus the LoCo half of Table 8). Each entry names its `reproduce_tag`, which is the `--tag` the stage passes, so the run you are diffing is `results/loco/loco_<tag>_<subtask>.json`. The runner writes one JSON per subtask and no completion marker; the macro-average is printed at the end of the run and is not persisted, so recompute it as the mean over the twelve subtask files. The Table 3 rows above the GTE block (BGE-Large, Ada-002, Jina-v2, E5-Mistral, M2-BERT) are reproduced from the LoCo paper, were never re-run here, and therefore ship no reference.

**Budget.** LoCo is the longest-running of the three evaluation benchmarks: a single arm evaluated across GoodWiki-Long, DAPFAM, and LoCo takes about 2h15m sequentially, and LoCo dominates that. Table 3 needs the three GN scales at two strides each, so budget accordingly; the subtasks within one configuration run sequentially, while independent configurations can run in parallel.

## `main-dapfam` — Table 4 (and part of Table 8)

**Produces.** nDCG@100 on the DAPFAM `test` split (187 queries) against the full 45,336-document FullText corpus with self-matches removed, for the six systems the Table 5 paired tests compare, plus the in-domain fine-tune. Table 4's `test_in` (180) and `test_out` (138) positives subsets come from the same stage with `DAPFAM_SPLITS="test test_in test_out"`; the three splits share one corpus encode, so the extra columns are cheap once `test` has run. Table 4's remaining dense rows are reached through `DENSE_DAPFAM` and its remaining REIGN rows through `DAPFAM_ZS_ROWS_OVERRIDE` — see [Table 4](#table-4--dapfam-rows-outside-the-default-set).

**Underneath.** First build and split the dataset, then evaluate:

```bash
python -m reign.dapfam.build_dataset --text-view fulltext \
  --out-dir data/dapfam_ir_fulltext --download
python -m reign.dapfam.split_qrels --data-dir data/dapfam_ir_fulltext \
  --train-ratio 0.70 --val-ratio 0.15 --seed 42

python scripts/evaluate_reign.py \
  --checkpoint models/reign-base-l3_gn-gte-large_val-selected/best \
  --gn-model thenlper/gte-large --gn-chunk-size 512 --gn-stride 512 \
  --dataset data/dapfam_ir_fulltext --split test \
  --top_k 100 --batch_size 4 --gn-batch-size 8 \
  --corpus-embed-cache ~/.reign_cache/corpus_emb \
  --output_path results/reign_gte-large_dapfam_test.json
```

The split step is deterministic at seed 42 and idempotent — re-running it reproduces the same partition. The stage then runs the in-domain fine-tune through `scripts/dapfam_finetune.sh`, whose default is the headline run `reign-base-l3_gn-gte-base_dapfam-ft-c512s512` — cold start, lr 1e-5, wd 1e-4, 15 epochs, chunk/stride 512. The `-c<chunk>s<stride>` suffix is derived from `GN_CHUNK`/`GN_STRIDE`, so the directory name always states the chunking the weights were trained at.

**Reference.** Two files.

- `results/reference/dapfam/dapfam_test.json` — the thirteen Table 4 systems, each with its `test` aggregate and, for the chunked-GN and REIGN rows, `test_in` and `test_out`. Entries carry the checkpoint, GN, stride, the number of self-matches dropped, the full `metrics_pct` block, and `paper_value`.
- `results/reference/dapfam/dapfam_finetune_family.json` — the eleven released DAPFAM fine-tuned checkpoints plus the cold-start control, each with the exact training configuration that produced it and its three-split aggregates.

The three splits share one corpus pool, so all three resolve to the same corpus-embedding cache entry and only the first pays the encode. Table 5's paired analysis has its own reference under `results/reference/e2/` and is not duplicated in either file.

**Budget.** This is the expensive stage. Each dense baseline re-encodes all 45,336 documents, about **2.5 hours per model** on a 24 GB RTX 4090; Jina-v3 does not fit at larger batch sizes and requires batch 4. REIGN rows are far cheaper once the corpus-embedding cache is warm, since the three splits reuse one encode.

## `e1-efficiency` — Table 11 (Appendix G)

**Produces.** End-to-end index time, per-query latency, and peak GPU memory for sparse, native long-context dense, bare-GN chunked, and REIGN rows. REIGN is measured three ways: `uncached` (GN run per query), `build` (the one-time cache write), and `cached` (chunk embeddings served from disk).

**Underneath.**

```bash
python scripts/measure_compute.py --kind reign \
  --checkpoint models/reign-base-l3_gn-gte-small_s512_val-selected/best \
  --gn-model thenlper/gte-small --chunk-size 512 --gn-stride 512 \
  --dataset devrim/goodwiki_long_synthetic_ir --split test \
  --n_corpus 500 --n_queries 100 --batch_size 8 --n_warmup 1 --n_repeat 3 \
  --top_k 10 --gn-cache-mode cached --name reign-on-gte-small \
  --cache-tag e1_gte-small --output_path results/e1/compute_reign_gte-small_cached.json
```

The protocol is uniform across every row and must not be mixed: 500 corpus documents, 100 queries, batch 8, one warm-up plus three timed repeats. Dense rows use the dtype that produced their accuracy numbers (fp16 for Jina-v3 and Stella-1.5B). Jina-v3 is measured at batch 4; all other dense rows at batch 8.

**These are timing measurements — run them on an otherwise-idle GPU.** A co-scheduled job invalidates every row, so the stage runs strictly sequentially and never backgrounds a run. Check the card is idle before starting.

**Reference.** `results/reference/e1/e1_table.md` and `e1_table.tex` — the rendered Table 11, exactly as `scripts/_print_e1_table.py` writes it at the end of the stage, so diff your `results/e1/e1_table.md` against it. The per-row `compute_*.json` timing dumps are machine- and driver-specific and are not shipped; the table is the artifact to compare against.

**Budget.** Modest — the sample sizes are small by design (500 documents, 100 queries). The sweep is dominated by the native long-context dense rows, and by the requirement that nothing else touches the GPU while it runs.

## `e2-significance` — Table 5

**Produces.** Paired significance for REIGN+GTE-large @s512 against five baselines on DAPFAM `test` (187 queries): paired bootstrap 95% confidence intervals (B = 10,000) and paired randomization *p*-values (B = 10,000), Holm-adjusted across the five comparisons, with per-query win/loss/tie counts. The five comparators are Jina-v3, Stella-1.5B, GTE-large chunked, BM25, and TF-IDF.

**Underneath.** Two steps. First re-run the DAPFAM `test` evaluations with per-query nDCG@100 persisted — the same protocol that produced Table 4, with per-query scores dumped rather than only the aggregate. Each row is checked against its published aggregate before it is used. Then:

```bash
python scripts/paired_bootstrap.py \
  --system results/e2/reign_gte-large_s512.json \
  --baseline results/e2/dense_jina-v3.json \
  --baseline results/e2/dense_stella-1.5b.json \
  --baseline results/e2/dense_gte-large-chunked.json \
  --baseline results/e2/sparse_bm25.json \
  --baseline results/e2/sparse_tfidf.json \
  --metric nDCG@100 --n-boot 10000 --n-perm 10000 --seed 42 --alpha 0.05 \
  --output results/e2/significance.json \
  --markdown results/e2/significance.md
```

**Reference.** `results/reference/e2/significance_dapfam_5way.json` and `.md` hold the shipped significance output — the five comparisons with their per-query means, bootstrap CIs, randomization *p*-values, Holm-adjusted *p*-values and win/loss/tie counts — to diff against your `results/e2/significance.{json,md}`. This is the authoritative source for the Table 5 numbers, including the two re-measured rows noted below. The six per-query dumps it was computed from are not shipped — regenerate them with `main-dapfam`.

**Budget.** The per-query re-runs dominate — the same ≈2.5 hours per dense model as `main-dapfam`. The bootstrap and randomization tests themselves take seconds.

**Note on the deltas.** Two rows in Table 5 are re-measurements rather than bit-exact reproductions of Table 4: Jina-v3 scores 32.98 against the published 32.97 (fp16 non-determinism), and Stella-1.5B 33.00 against 32.91 (re-measured at batch 8 rather than batch 4). Both are far inside the reported confidence intervals, and shifting Stella to its published mean leaves the verdict unchanged (*p* = .843).

## `e4-pe-ablation` — Table 9 (Appendix E)

**Produces.** The chunk-position ablation: no positional encoding (the published design), learned absolute chunk positions, and fixed sinusoidal encodings, each trained from scratch under one identical protocol and evaluated identically on GoodWiki-Long, LoCo, and DAPFAM.

**Underneath.** Three training arms differing in exactly one argument:

```bash
python -m reign.train \
  --dataset devrim/goodwiki_long_synthetic_ir --train-split train --eval-split val \
  --model-config base-l3 --gn-model thenlper/gte-small \
  --gn-chunk-size 512 --gn-stride 512 \
  --batch-size 48 --eval-batch-size 48 --negative-batch-size-multiplier 47 \
  --loss-function infonce --temperature 0.07 --weight-partial 0.5 \
  --max-epochs 20 --lr 1e-5 --weight-decay 1e-4 \
  --enable-cache --precision 16-mixed \
  --metric-to-monitor ndcg@10 --check-val-every-n-epoch 2 --seed 42 --device cuda \
  --position-embedding-type none \
  --output-dir reign-base-l3_gn-gte-small_pe-none
```

then `--position-embedding-type absolute` and `--position-embedding-type sinusoidal` into their own output directories, followed by the three standard evaluations per arm.

**This is the ablation protocol, not the released recipe.** All three arms sit below the released operating point of Table 2 because the released checkpoints use the cosine recipe. The comparison is internally valid and answers the positional-encoding question; it is not comparable to Table 2. See [TRAINING.md](TRAINING.md).

**Reference.** `results/reference/e4/e4_dapfam_pe-none.json`, `e4_dapfam_pe-absolute.json` and `e4_dapfam_pe-sinusoidal.json` — the DAPFAM evaluation of each arm, to diff against your `results/e4/dapfam_pe-<arm>.json`. These three are the only reference files in this repository that retain a `per_query` block, and it is the DAPFAM test split's 187 queries: the positional-encoding verdict is a within-arm comparison over identical queries, so the per-query scores are what makes it checkable. The GoodWiki and LoCo evaluations of the three arms are not shipped; regenerate them with the stage and read the verdict off `results/e4/e4_table.md`.

**Budget.** Arms train in parallel at roughly 2–3 GB VRAM each and fit comfortably on one 24 GB card. Evaluation, not training, dominates: the three benchmarks for one arm take about 2h15m sequentially, so running the three arms in parallel bounds the stage by its slowest single arm.

## `e5-objective-ablation` — Table 12 (Appendix I)

**Produces.** The training-objective ablation: the three-way cosine recipe against InfoNCE variants under matched conditions (`base-l3` on GTE-small, chunk/stride 512, 20 epochs unless noted, seed 42, best-validation selection). The InfoNCE grid covers τ ∈ {0.07, 0.1}, distractors included as graded soft positives (α = 0.5) or excluded (α = 0), batch 18 against batch 48, a warm start from the released checkpoint, and a 50-epoch run.

**Underneath.** Same trainer, one arm per configuration; the batch-48 arms run in one wave and the batch-18 and warm-start arms in a second.

```bash
python -m reign.train \
  --dataset devrim/goodwiki_long_synthetic_ir --train-split train --eval-split val \
  --model-config base-l3 --gn-model thenlper/gte-small \
  --gn-chunk-size 512 --gn-stride 512 \
  --loss-function infonce --temperature 0.07 --weight-partial 0.0 \
  --batch-size 48 --eval-batch-size 48 --negative-batch-size-multiplier 47 \
  --max-epochs 20 --lr 1e-5 --weight-decay 1e-4 \
  --enable-cache --precision 16-mixed \
  --metric-to-monitor ndcg@10 --check-val-every-n-epoch 2 --seed 42 --device cuda \
  --output-dir reign-base-l3_gte-small_e5-infonce-pw00-bs48
```

Each arm is then evaluated on GoodWiki-Long, LoCo, DAPFAM, and MTEB.

**Reference.** `results/reference/e5/e5_table.md` — the rendered Table 12, one row per arm with its GoodWiki nDCG@10, LoCo macro-average, DAPFAM nDCG@100 and MTEB scores, as `scripts/_print_e5_table.py` writes it at the end of the stage. Diff your `results/e5/e5_table.md` against it. The thirty-six underlying per-arm evaluation outputs are not shipped.

**Budget.** Nine training arms plus four evaluations each: roughly **60–90 hours** sequential on one 24 GB card. The arms are independent and parallelise well.

**Expect the InfoNCE batch-18 arm to collapse.** Validation nDCG@10 peaks at the first validation epoch and decays monotonically; that row is an epoch-1 snapshot by construction, not a training failure on your side. Appendix I explains why, and [TRAINING.md](TRAINING.md) restates the practical consequence.

## `mteb` — Table 6 (Appendix B)

**Produces.** MAP@1, MAP@10, R@10, P@10 and nDCG@10 on the two short-context MTEB retrieval benchmarks Appendix B reports — ArguAna and FiQA-2018 — for the two systems Table 6 compares: REIGN and the truncated GTE-small baseline. Nothing else; the appendix does not report a wider task set.

**Underneath.** Two commands, because the two rows come from two different runners. The REIGN row uses this repository's harness:

```bash
python scripts/evaluate_mteb.py \
  --model-path models/reign-base-l3_gn-gte-small_s384_val-selected/best \
  --gn-model thenlper/gte-small \
  --task-names ArguAna FiQA2018 --eval-splits test \
  --batch-size 8 --max-seq-length 512 \
  --output-dir results/mteb/reign-base-l3_gte-small
```

The baseline row is the bare Guidance Network with no REIGN encoder, so it is the stock `mteb` runner over the Hub model — `scripts/evaluate_mteb.py` always builds a REIGN encoder and cannot produce it:

```bash
mteb run -m thenlper/gte-small -t ArguAna FiQA2018 \
  --eval_splits test --batch_size 8 \
  --output_folder results/mteb/baseline_gte-small
```

`mteb` arrives with `pip install -e ".[eval]"`; the stage skips the baseline row with a message if the CLI is not on `PATH`. No truncation flag is needed — 512 tokens is GTE-small's native window, which is exactly what "truncated (512 tokens)" means in the Table 6 caption.

The checkpoint is the **stride-384** one: that is what produced Table 6. Override it with `MTEB_CKPT` (and `MTEB_GN`, `MTEB_TASKS`) to score any other released checkpoint.

**Reference.** `results/reference/mteb/mteb_arguana_fiqa.json` — both systems, both tasks, the five reported metrics each, plus the `mteb` version and dataset revisions the run pinned. Every value in it is a value printed in Table 6.

**Budget.** The cheapest stage: these are short-passage corpora, so the chunked path degenerates to a single chunk per input and there is little work to do.

**Expect REIGN to lose here.** It trails the truncated GTE-small baseline by 5.1 nDCG@10 on ArguAna and 6.8 on FiQA-2018. Inputs shorter than the chunk size collapse to a single chunk embedding, so the cross-chunk encoder has nothing to aggregate. This is the operating-regime boundary described in the Limitations, and reproducing the gap is the expected outcome.

---

## What ships in `results/reference/`

Fourteen files, one directory per stage. This is the complete list; nothing else is shipped, and every file below is an artifact of the run that produced the corresponding paper table.

| File | Size | Covers | Diff it against |
| --- | ---: | --- | --- |
| `goodwiki/goodwiki_test.json` | 13 KB | Table 2 (all 21 rows), the stride-512 companion of its GTE-small row, and the 9 non-`base-l3` cells of Table 7 | `results/*_goodwiki_test.json` |
| `loco/loco_subtask_ndcg.json` | 13 KB | Table 3 and the LoCo half of Table 8 — 15 runs × 12 subtasks + macro-average | `results/loco/loco_<tag>_<subtask>.json` |
| `dapfam/dapfam_test.json` | 11 KB | Table 4 — 13 systems, `test` plus `test_in`/`test_out` for the chunked-GN and REIGN rows | `results/e2/*.json` |
| `dapfam/dapfam_ood_head_to_head.json` | 12 KB | 9 of the 12 cells of Table 8's DAPFAM half (the other 3 are Table 4 rows, in `dapfam_test.json`) | `results/e2/reign_*.json` |
| `dapfam/dapfam_finetune_family.json` | 18 KB | Section 5.3 / Appendix J — the 11 released fine-tuned checkpoints plus the cold-start control, with the training configuration of each | `results/reign-*-ft_<split>_<tag>.json` |
| `mteb/mteb_arguana_fiqa.json` | 3 KB | Table 6 — both systems, both tasks, five metrics each | `results/mteb/**/{ArguAna,FiQA2018}.json` |
| `e1/e1_table.md`, `e1/e1_table.tex` | 4 KB, 2 KB | Table 11, rendered | `results/e1/e1_table.{md,tex}` |
| `e2/significance_dapfam_5way.json`, `.md` | 4 KB, 1 KB | Table 5 — five comparisons with CIs, *p*-values, Holm adjustment, W/L/T | `results/e2/significance.{json,md}` |
| `e4/e4_dapfam_pe-{none,absolute,sinusoidal}.json` | 32 KB each | Table 9 — the DAPFAM evaluation of each arm, with per-query scores | `results/e4/dapfam_pe-<arm>.json` |
| `e5/e5_table.md` | 1 KB | Table 12, rendered | `results/e5/e5_table.md` |

The six aggregate files (`goodwiki/`, `loco/`, `dapfam/`, `mteb/`) hold aggregate `metrics` blocks and run-identifying configuration only — checkpoint, GN, chunk size, stride — plus a `paper_value` field carrying the number printed in the paper, so a row can be checked against the table without opening the PDF. Each file opens with `_what` and `_protocol` keys stating what it covers and the protocol every row in it shares. Metrics in them are percentages; stage outputs are fractions. The three `e4/` files are the only shipped references that retain per-query scores.

**What does not ship.** Per-query dumps for Table 5 (regenerate with `main-dapfam`); the per-row `compute_*.json` timing dumps behind Table 11, which are machine-specific; the GoodWiki and LoCo evaluations of the Table 9 arms; the 36 per-arm evaluation outputs behind Table 12. Tables 1 and 10 are not runs at all — Table 1 is dataset statistics and Table 10 is analytic FLOPs. The Table 3 baseline block above the GTE rows is quoted from the LoCo paper and was never re-run here.

## Rows the default stages do not run

Each stage's default row set is the headline table. The appendix sweeps come from the same stages under a row-set override — one `short|checkpoint-dir|gn-model` spec per line, exactly as the array in `scripts/reproduce.sh` is written. `short` is the key in every output filename, so each row needs a distinct one; `GN_STRIDE` is global to an invocation, so a table with two strides needs two invocations.

### Table 7 — encoder-capacity sweep, in-distribution

Twelve cells: four encoder sizes against three GTE backbones. The `base-l3` row is the GTE part of Table 2, but the default row set evaluates `base-l3` + GTE-small at stride 512 (67.09) while Table 7 reports the stride-384 run (67.31), so name it explicitly. Nine of the twelve cells were evaluated at stride 384:

```bash
GN_STRIDE=384 REIGN_ROWS_OVERRIDE="tiny-l1-gte-small|reign-tiny-l1_gn-gte-small_s384_val-selected|thenlper/gte-small
tiny-l1-gte-base|reign-tiny-l1_gn-gte-base_val-selected|thenlper/gte-base
tiny-l1-gte-large|reign-tiny-l1_gn-gte-large_val-selected|thenlper/gte-large
small-l2-gte-small|reign-small-l2_gn-gte-small_s384_val-selected|thenlper/gte-small
base-l3-gte-small|reign-base-l3_gn-gte-small_s384_val-selected|thenlper/gte-small
base-l3-gte-base|reign-base-l3_gn-gte-base_val-selected|thenlper/gte-base
large-l4-gte-small|reign-large-l4_gn-gte-small_s384_val-selected|thenlper/gte-small
large-l4-gte-base|reign-large-l4_gn-gte-base_s384_val-selected|thenlper/gte-base
large-l4-gte-large|reign-large-l4_gn-gte-large_s384_val-selected|thenlper/gte-large" \
  bash scripts/reproduce.sh main-goodwiki
```

and one at stride 512:

```bash
GN_STRIDE=512 REIGN_ROWS_OVERRIDE="small-l2-gte-large|reign-small-l2_gn-gte-large_st-512_val-selected|thenlper/gte-large" \
  bash scripts/reproduce.sh main-goodwiki
```

**The remaining two cells cannot be pinned to a stride.** The result artifacts behind `small-l2` + GTE-base (66.02) and `base-l3` + GTE-large (66.73) predate the runner recording the evaluation stride, and so does `base-l3` + BGE-large in Table 2 (66.45). The shipped reference carries `"stride": null` for those three rows rather than a guess. Their checkpoints are `reign-small-l2_gn-gte-base_val-selected`, `reign-base-l3_gn-gte-large_val-selected` and `reign-base-l3_gn-bge-large_val-selected`; reproducing them means trying both strides and reading which one lands on the published value.

### Table 8 — encoder-capacity head-to-head, OOD

Twelve cells per benchmark: `small-l2` against `base-l3`, on each GTE backbone, at both strides. Six configurations, the same ones for the DAPFAM and LoCo halves. Five of them are evaluated at both strides from one set of weights; `base-l3` + GTE-small is the exception, with a separate checkpoint per stride.

| REIGN encoder | GN | checkpoint |
| --- | --- | --- |
| `small-l2` | GTE-small | `reign-small-l2_gn-gte-small_s384_val-selected` |
| `small-l2` | GTE-base | `reign-small-l2_gn-gte-base_val-selected` |
| `small-l2` | GTE-large | `reign-small-l2_gn-gte-large_val-selected` |
| `base-l3` | GTE-small | `reign-base-l3_gn-gte-small_s384_val-selected` at stride 384, `reign-base-l3_gn-gte-small_s512_val-selected` at stride 512 |
| `base-l3` | GTE-base | `reign-base-l3_gn-gte-base_val-selected` |
| `base-l3` | GTE-large | `reign-base-l3_gn-gte-large_val-selected` |

LoCo half, once per stride:

```bash
GN_STRIDE=384 REIGN_ROWS_OVERRIDE="small-l2-gte-small|reign-small-l2_gn-gte-small_s384_val-selected|thenlper/gte-small
small-l2-gte-base|reign-small-l2_gn-gte-base_val-selected|thenlper/gte-base
small-l2-gte-large|reign-small-l2_gn-gte-large_val-selected|thenlper/gte-large
base-l3-gte-small|reign-base-l3_gn-gte-small_s384_val-selected|thenlper/gte-small
base-l3-gte-base|reign-base-l3_gn-gte-base_val-selected|thenlper/gte-base
base-l3-gte-large|reign-base-l3_gn-gte-large_val-selected|thenlper/gte-large" \
  DENSE_NATIVE="" DENSE_CHUNKED="" bash scripts/reproduce.sh main-loco
```

Repeat with `GN_STRIDE=512`, swapping the `base-l3-gte-small` row's checkpoint to `reign-base-l3_gn-gte-small_s512_val-selected`. `DENSE_NATIVE=""` and `DENSE_CHUNKED=""` suppress the baseline block, which Table 8 does not use.

DAPFAM half: the same six rows through `DAPFAM_ZS_ROWS_OVERRIDE`, once per stride.

```bash
GN_STRIDE=384 DAPFAM_ZS_ROWS_OVERRIDE="<the same six specs>" \
  DENSE_DAPFAM="" bash scripts/reproduce.sh main-dapfam
```

`e2-significance` reads `results/e2/reign_gte-large_s<stride>.json`, so keep a row whose `short` is `gte-large` if you intend to run that stage afterwards.

### Table 4 — DAPFAM rows outside the default set

`main-dapfam`'s default rows are the six systems Table 5 compares. The rest of Table 4:

- **IN-/cross-IPC columns:** `DAPFAM_SPLITS="test test_in test_out"`. The three splits share one corpus encode, so the extra columns cost little once `test` is done.
- **Remaining dense rows:** `DENSE_DAPFAM="bge-m3 nomic-v1.5 gte-small-chunked gte-base-chunked bge-large-chunked"`.
- **Remaining REIGN rows:** `DAPFAM_ZS_ROWS_OVERRIDE` with `reign-base-l3_gn-gte-small_s512_val-selected` (stride 512) and `reign-base-l3_gn-gte-base_val-selected` (stride 384), which are the strides those two rows report.

### The DAPFAM fine-tuned family (Appendix J)

`main-dapfam` runs the headline fine-tune by default: `reign-base-l3_gn-gte-base_dapfam-ft-c512s512`. The other ten released members, plus the cold-start control behind the paper's warm-vs-cold comparison, come from `scripts/dapfam_finetune.sh` directly. Every run is `base-l3`, chunk 512, InfoNCE at τ = 0.07, `--partial-policy ignore`, seed 42; what varies is below. Warm-start runs load the matching GoodWiki-Long backbone.

| Checkpoint | GN | Start | `LR` | `WD` | `EPOCHS` | `BS` | `EVAL_BS` | `STRIDE` | `TAG` |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `reign-base-l3_gn-gte-base_dapfam-ft-c512s512` | gte-base | cold | 1e-5 | 1e-4 | 15 | 4 | 4 | 512 | `dapfam-ft` *(stage default)* |
| `reign-base-l3_gn-gte-base_dapfam-ft-c512s384` | gte-base | cold | 1e-5 | 1e-4 | 15 | 2 | 4 | 384 | `dapfam-ft-s384` |
| `reign-base-l3_gn-gte-base_dapfam-ftwarm-c512s512` | gte-base | warm | 1e-5 | 1e-4 | 15 | 2 | 2 | 512 | `dapfam-ftwarm-base-s512` |
| `reign-base-l3_gn-gte-base_dapfam-ftwarm-c512s384` | gte-base | warm | 1e-5 | 1e-4 | 15 | 2 | 2 | 384 | `dapfam-ftwarm-base-s384` |
| `reign-base-l3_gn-gte-base_dapfam-ftreg-r1-c512s512` | gte-base | warm | 5e-6 | 1e-2 | 6 | 2 | 2 | 512 | `dapfam-ftreg-r1` |
| `reign-base-l3_gn-gte-base_dapfam-ftreg-r2-c512s512` | gte-base | warm | 2e-6 | 1e-2 | 6 | 2 | 2 | 512 | `dapfam-ftreg-r2` |
| `reign-base-l3_gn-gte-base_dapfam-ftreg-r3-c512s512` | gte-base | warm | 1e-6 | 1e-2 | 8 | 2 | 2 | 512 | `dapfam-ftreg-r3` |
| `reign-base-l3_gn-gte-base_dapfam-ftreg-r4-c512s512` | gte-base | warm | 2e-6 | 1e-1 | 6 | 2 | 2 | 512 | `dapfam-ftreg-r4` |
| `reign-base-l3_gn-gte-base_dapfam-ftcold-long-c512s512` | gte-base | cold | 1e-5 | 1e-4 | 60 | 4 | 4 | 512 | `dapfam-ftcold-long-s512` |
| `reign-base-l3_gn-gte-large_dapfam-ftwarm-c512s512` | gte-large | warm | 1e-5 | 1e-4 | 15 | 2 | 2 | 512 | `dapfam-ftwarm-large-s512` |
| `reign-base-l3_gn-gte-large_dapfam-ftwarm-c512s384` | gte-large | warm | 1e-5 | 1e-4 | 15 | 2 | 2 | 384 | `dapfam-ftwarm-large-s384` |
| `reign-base-l3_gn-gte-base_dapfam-coldreg-r3-c512s512` | gte-base | cold | 1e-6 | 1e-2 | 8 | 2 | 2 | 512 | `dapfam-coldreg-r3` |

The warm-start backbones are `reign-base-l3_gn-gte-base_val-selected/best` and `reign-base-l3_gn-gte-large_val-selected/best` respectively — note the `/best` is part of the spec's fifth field. The last row is the cold-start control at the `r3` hyperparameters; it is what the paper's "cold-start loses an additional 0.43 to warm-start at matched hyperparameters (32.30 vs. 32.73)" compares. **Its weights are not part of the released model zoo** — it is reproducible, not downloadable.

One invocation per row, e.g. the `ftreg-r3` cell (the best fine-tune in the sweep, 32.73 nDCG@100):

```bash
MODELS_DIR=./models DATA=data/dapfam_ir_fulltext TAG=dapfam-ftreg-r3 \
  CHUNK=512 STRIDE=512 BS=2 EVAL_BS=2 EPOCHS=8 LR=1e-6 WD=1e-2 \
  FT_RUNS_OVERRIDE="base-l3;thenlper/gte-base;gte-base;reign-base-l3_gn-gte-base_dapfam-ftreg-r3-c512s512;reign-base-l3_gn-gte-base_val-selected/best" \
  bash scripts/dapfam_finetune.sh
```

and the cold-start form, which is the same with an empty fifth field and no warm backbone:

```bash
MODELS_DIR=./models DATA=data/dapfam_ir_fulltext TAG=dapfam-ft-s384 \
  CHUNK=512 STRIDE=384 BS=2 EVAL_BS=4 EPOCHS=15 LR=1e-5 WD=1e-4 \
  FT_RUNS_OVERRIDE="base-l3;thenlper/gte-base;gte-base;reign-base-l3_gn-gte-base_dapfam-ft-c512s384;" \
  bash scripts/dapfam_finetune.sh
```

Result files are `results/reign-<short>-ft_<split>_<TAG>.json`, so the `TAG` column is what keeps the runs from overwriting each other — give each run the tag in the table, or a distinct `short`. Each fine-tune is roughly 6–10 h on a 24 GB card, and the 60-epoch cold-start run is longer.

## Expected variance

fp16 inference is not bit-reproducible across runs: cuBLAS can select different algorithms depending on free-memory state, and each batch is padded to its longest member, so **changing the batch size changes the padding pattern and perturbs embeddings slightly**. In practice this shifts nDCG by roughly 0.01 at a matched batch size and up to about 0.1 when the batch size differs, which is enough to churn near-ties deep in a 45,336-document ranking while leaving the top of the ranking stable.

Treat a deviation of a few hundredths of a point as reproduction, and a deviation of whole points as a genuinely different system — a wrong checkpoint, split, stride, or protocol.

The significance conclusions in Table 5 are robust to this variance: the fp16-scale shifts are far smaller than the reported confidence intervals, and the reported check of shifting Stella-1.5B to its published mean leaves the verdict unchanged.

Training is seeded (seed 42) but runs under 16-mixed precision, so retrained checkpoints will not be bit-identical to the released ones either. Compare against the reference metrics, not against weights.
