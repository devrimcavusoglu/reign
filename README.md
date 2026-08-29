# REIGN

**Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling** — a long-document bi-encoder that reads sequences of cached chunk embeddings instead of tokens.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![EMNLP 2026 Findings](https://img.shields.io/badge/EMNLP%202026-Findings-b31b1b.svg)](https://devrimcavusoglu.github.io/reign)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-goodwiki__long__synthetic__ir-yellow.svg)](https://huggingface.co/datasets/devrim/goodwiki_long_synthetic_ir)
[![Project page](https://img.shields.io/badge/Project-page-informational.svg)](https://devrimcavusoglu.github.io/reign)

## What is REIGN

REIGN is a contrastively trained bi-encoder that operates on a sequence of contextualised chunk embeddings produced by a frozen pretrained text embedder — the Guidance Network (GN) — rather than on raw sub-word tokens. A document is split by a sliding window of size *K* (typically 512, matching the GN's context window) advanced with stride *S ≤ K*; each window is encoded by the GN, and a small Transformer aggregates the resulting embedding sequence as a permutation-equivariant set function with no positional encoding, average-pooled into one document vector. Because the GN is frozen and its outputs are deterministic, they are content-hashed and cached to disk, which moves the token-level cost out of the training loop entirely.

REIGN targets multi-chunk inputs and primarily document-to-document retrieval; single-chunk inputs are served by the guidance network (GN) alone.

On the DAPFAM patent task, a 357M REIGN+GTE-large is statistically indistinguishable from dense baselines 1.6–4.3× larger under paired tests (paper Table 5); on LoCo, REIGN comes within 0.65 nDCG@10 of a 20×-larger model, though the bare chunked GTE-large mean-pool is the Pareto-best LoCo configuration the paper observes (71.38 macro against REIGN's 70.77) and the REIGN encoder marginally detracts there. With cached GN chunk embeddings REIGN answers queries 49–229× faster than re-running the GN per query, and is at parity — not faster — when uncached. Peak GPU memory spans 0.24–1.73 GB across REIGN configurations, against 4.8–18.9 GB for the native long-context dense baselines (Jina-v3 measured at batch 4).

## Install

Python ≥ 3.11.

```bash
conda env create -f environment.yml
conda activate reign
```

or, into an existing environment:

```bash
pip install -e .
```

## Quickstart

Encode two long documents and score them by cosine similarity. `ReignBaselineEncoder` returns L2-normalised vectors, so the cosine is a dot product.

```python
import numpy as np
from reign.encoders.reign import ReignBaselineEncoder

encoder = ReignBaselineEncoder(
    checkpoint_path="models/reign-base-l3_gn-gte-small_s512_val-selected/best",
    gn_model="thenlper/gte-small",
    chunk_size=512,
    stride=512,
)
docs = [open("doc_a.txt").read(), open("doc_b.txt").read()]
emb = encoder.encode(docs, batch_size=8)   # (2, hidden_size), L2-normalised
print(float(np.dot(emb[0], emb[1])))       # cosine similarity
```

`checkpoint_path` points at a checkpoint directory containing `config.json` and `model.safetensors`. Loading by local path always works; loading by Hugging Face Hub identifier works for any checkpoint published under [huggingface.co/devrim](https://huggingface.co/devrim), where the Hub identifiers mirror the checkpoint names in the model zoo below. `chunk_size` is the GN's sliding-window size — 512 for every released checkpoint, matching the GN's context window — and `stride` controls the overlap, with `stride == chunk_size` giving non-overlapping chunking. Both should match the values the checkpoint was trained and evaluated at.

For the lower-level surface, `ReignModel` (a `PreTrainedModel` consuming `inputs_embeds`) and `ReignFeatureExtractor` (the GN wrapper, with the on-disk embedding cache) are importable directly from `reign` and `reign.feature_extractor`.

## Model zoo

The zoo is the released-weights inventory, not the list of runs any one stage performs: `scripts/reproduce.sh` evaluates the headline row set by default and reaches the rest through row-set overrides, which [docs/REPRODUCING.md](docs/REPRODUCING.md) maps table by table.

Encoder sizes are the paper's configuration sweep: `tiny-l1` 0.56M, `small-l2` 3.85M, `base-l3` 22.45M, `large-l4` 52.49M parameters. `base-l3` is the paper-default encoder used in the headline tables. Where a checkpoint name carries a stride tag (`_s<N>_`, `_st-<N>_`), it identifies the chunking stride that run was trained at; the evaluation-time stride is a command-line argument (`--gn-stride`), and Tables 2 and 4 report the best-performing stride per GN as stated in their captions, while Table 3 reports both strides.

| Checkpoint | REIGN encoder | GN backbone | Stride tag | Appears in |
| --- | --- | --- | --- | --- |
| `reign-base-l3_gn-gte-small_s384_val-selected` | base-l3 | `thenlper/gte-small` | 384 | Tables 2, 3, 6, 7, 8, 12 |
| `reign-base-l3_gn-gte-small_s512_val-selected` | base-l3 | `thenlper/gte-small` | 512 | Tables 3, 4, 8, 11, 12 |
| `reign-base-l3_gn-gte-base_val-selected` | base-l3 | `thenlper/gte-base` | — | Tables 2, 3, 4, 7, 8, 11 |
| `reign-base-l3_gn-gte-large_val-selected` | base-l3 | `thenlper/gte-large` | — | Tables 2, 3, 4, 5, 7, 8, 11 |
| `reign-base-l3_gn-bge-base_val-selected` | base-l3 | `BAAI/bge-base-en-v1.5` | — | Table 2 |
| `reign-base-l3_gn-bge-large_val-selected` | base-l3 | `BAAI/bge-large-en-v1.5` | — | Table 2 |
| `reign-small-l2_gn-gte-small_s384_val-selected` | small-l2 | `thenlper/gte-small` | 384 | Tables 7, 8 |
| `reign-small-l2_gn-gte-base_val-selected` | small-l2 | `thenlper/gte-base` | — | Tables 7, 8 |
| `reign-small-l2_gn-gte-large_val-selected` | small-l2 | `thenlper/gte-large` | — | Table 8 |
| `reign-small-l2_gn-bge-base_val-selected` | small-l2 | `BAAI/bge-base-en-v1.5` | — | — |
| `reign-small-l2_gn-bge-large_val-selected` | small-l2 | `BAAI/bge-large-en-v1.5` | — | — |
| `reign-small-l2_gn-gte-large_st-128_val-selected` | small-l2 | `thenlper/gte-large` | 128 | — |
| `reign-small-l2_gn-gte-large_st-256_val-selected` | small-l2 | `thenlper/gte-large` | 256 | — |
| `reign-small-l2_gn-gte-large_st-384_val-selected` | small-l2 | `thenlper/gte-large` | 384 | — |
| `reign-small-l2_gn-gte-large_st-512_val-selected` | small-l2 | `thenlper/gte-large` | 512 | Table 7 |
| `reign-tiny-l1_gn-gte-small_s384_val-selected` | tiny-l1 | `thenlper/gte-small` | 384 | Table 7 |
| `reign-tiny-l1_gn-gte-base_val-selected` | tiny-l1 | `thenlper/gte-base` | — | Table 7 |
| `reign-tiny-l1_gn-gte-large_val-selected` | tiny-l1 | `thenlper/gte-large` | — | Table 7 |
| `reign-tiny-l1_gn-bge-base_val-selected` | tiny-l1 | `BAAI/bge-base-en-v1.5` | — | — |
| `reign-tiny-l1_gn-bge-large_val-selected` | tiny-l1 | `BAAI/bge-large-en-v1.5` | — | — |
| `reign-large-l4_gn-gte-small_s384_val-selected` | large-l4 | `thenlper/gte-small` | 384 | Table 7 |
| `reign-large-l4_gn-gte-base_s384_val-selected` | large-l4 | `thenlper/gte-base` | 384 | Table 7 |
| `reign-large-l4_gn-gte-large_s384_val-selected` | large-l4 | `thenlper/gte-large` | 384 | Table 7 |

A dash in *Appears in* marks a checkpoint that is released but not individually reported in the paper. The four `st-<N>` rows are a train-time stride sweep on `small-l2` + GTE-large; only `st-512` is reported (the Table 7 `small-l2` × GTE-large cell), and Table 8's `small-l2` + GTE-large rows at both strides come from the untagged `reign-small-l2_gn-gte-large_val-selected`. Table 11 reports measured latency and memory per Guidance Network rather than per checkpoint, so the paper names no checkpoint there; the three rows credited with it are the ones `reproduce.sh e1-efficiency` measures.

The `_val-selected` suffix marks best-validation checkpoint selection on nDCG@10 over the `val` qrels split. Each checkpoint directory holds a `best/` and a `last/` snapshot; use `best/`.

### DAPFAM fine-tuned family

The patent fine-tuning study (paper §5.3, protocol in Appendix J) produces a second family, all `base-l3` over a GTE backbone, named `reign-base-l3_gn-<gn>_dapfam-<variant>-c<chunk>s<stride>`. `ft` is the plain cold-start fine-tune, `ftwarm` warm-starts from the GoodWiki-Long-trained checkpoint, `ftreg-r<N>` are the regularised warm-start cells, and `ftcold-long` is the same cold start on a 60-epoch schedule. `reproduce.sh main-dapfam` trains the headline run, `reign-base-l3_gn-gte-base_dapfam-ft-c512s512`, by default; the other ten come from `scripts/dapfam_finetune.sh` under the per-checkpoint overrides tabulated in [docs/REPRODUCING.md](docs/REPRODUCING.md).

| Checkpoint | GN backbone | Chunk / stride |
| --- | --- | --- |
| `reign-base-l3_gn-gte-base_dapfam-ft-c512s384` | `thenlper/gte-base` | 512 / 384 |
| `reign-base-l3_gn-gte-base_dapfam-ft-c512s512` | `thenlper/gte-base` | 512 / 512 |
| `reign-base-l3_gn-gte-base_dapfam-ftwarm-c512s384` | `thenlper/gte-base` | 512 / 384 |
| `reign-base-l3_gn-gte-base_dapfam-ftwarm-c512s512` | `thenlper/gte-base` | 512 / 512 |
| `reign-base-l3_gn-gte-base_dapfam-ftreg-r1-c512s512` | `thenlper/gte-base` | 512 / 512 |
| `reign-base-l3_gn-gte-base_dapfam-ftreg-r2-c512s512` | `thenlper/gte-base` | 512 / 512 |
| `reign-base-l3_gn-gte-base_dapfam-ftreg-r3-c512s512` | `thenlper/gte-base` | 512 / 512 |
| `reign-base-l3_gn-gte-base_dapfam-ftreg-r4-c512s512` | `thenlper/gte-base` | 512 / 512 |
| `reign-base-l3_gn-gte-base_dapfam-ftcold-long-c512s512` | `thenlper/gte-base` | 512 / 512 |
| `reign-base-l3_gn-gte-large_dapfam-ftwarm-c512s384` | `thenlper/gte-large` | 512 / 384 |
| `reign-base-l3_gn-gte-large_dapfam-ftwarm-c512s512` | `thenlper/gte-large` | 512 / 512 |

The DAPFAM fine-tunes use InfoNCE at temperature 0.07 by design, not the released cosine recipe; see [docs/TRAINING.md](docs/TRAINING.md).

## Evaluation and reproducing

`scripts/reproduce.sh` is the entry point. It exposes one stage per paper artifact: `main-goodwiki`, `main-loco`, `main-dapfam` (Tables 2–4), `e1-efficiency` (Appendix G, Table 11), `e2-significance` (Table 5), `e4-pe-ablation` (Appendix E), `e5-objective-ablation` (Appendix I), and `mteb` (Appendix B). Each stage takes no positional arguments, is configured through environment variables, writes under `results/`, and is re-runnable — a completed row is skipped rather than recomputed. Run a stage as `bash scripts/reproduce.sh main-goodwiki`. Per-stage commands, expected outputs, runtime and hardware budgets, and the row-set overrides that reproduce the appendix sweeps are in [docs/REPRODUCING.md](docs/REPRODUCING.md).

`results/reference/` ships fourteen curated outputs to diff against, one directory per stage: aggregate metrics for GoodWiki-Long (Tables 2, 7), LoCo (Tables 3, 8), DAPFAM (Tables 4, 8, and the fine-tuned family) and MTEB (Table 6), plus the rendered Table 11 and Table 12, the Table 5 significance output, and the three Table 9 ablation arms. Each file, and what it lets you diff, is enumerated in [docs/REPRODUCING.md](docs/REPRODUCING.md#what-ships-in-resultsreference) — including what deliberately does not ship.

## Training

The released checkpoints come from `scripts/train_reign_emnlp26.sh`: a three-way cosine embedding loss with partial weight λ = 0.5, batch 18, AdamW at lr 1e-5 and weight decay 1e-4 with cosine annealing, 50 epochs, validation every 4 epochs with best-validation selection on nDCG@10, cached GN embeddings, 16-mixed precision, seed 42. A single run is:

```bash
python -m reign.train \
  --dataset devrim/goodwiki_long_synthetic_ir --train-split train --eval-split val \
  --model-config base-l3 --gn-model thenlper/gte-small \
  --gn-chunk-size 512 --gn-stride 512 \
  --batch-size 18 --eval-batch-size 18 --negative-batch-size-multiplier 17 \
  --weight-partial 0.5 --max-epochs 50 --lr 1e-5 --weight-decay 1e-4 \
  --enable-cache --precision 16-mixed \
  --metric-to-monitor ndcg@10 --check-val-every-n-epoch 4 \
  --seed 42 --device cuda --output-dir reign-base-l3_gn-gte-small_s512_val-selected
```

**Two training protocols exist and their numbers are not comparable.** The released checkpoints use the cosine recipe above; the Appendix E and Appendix I ablation arms use a separate controlled protocol (InfoNCE at τ = 0.07, batch 48, 20 epochs). Read [docs/TRAINING.md](docs/TRAINING.md) before training or comparing anything.

## Dataset

[`devrim/goodwiki_long_synthetic_ir`](https://huggingface.co/datasets/devrim/goodwiki_long_synthetic_ir) is a long-document retrieval benchmark derived from GoodWiki, a cleaned English Wikipedia release, filtered to articles over 16,000 characters — documents averaging 5,065 words. Queries are the original articles (17,854 of them); the corpus (53,562 documents) is one LLM-rephrased positive per query plus roughly two topical distractors, giving graded relevance where score 2 marks the rephrasal and score 1 a distractor. It ships in the canonical BEIR/MTEB tri-config layout (`corpus`, `queries`, `default`) with query-disjoint train/val/test qrels splits.

The rephrasals are machine-generated with GPT-4o-mini and are marked as synthetic in the dataset card.

## Paper and project page

- Project page: <https://devrimcavusoglu.github.io/reign>
- Code: <https://github.com/devrimcavusoglu/reign>
- Dataset: <https://huggingface.co/datasets/devrim/goodwiki_long_synthetic_ir>
- Models: <https://huggingface.co/collections/devrim/reign-emnlp-2026-findings-6a9202c43943622462e6ed9c>
- Paper: *REIGN: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling*, Findings of the ACL: EMNLP 2026 (to appear). arXiv link coming soon.

## Citation

```bibtex
@inproceedings{cavusoglu2026reign,
  title     = {{REIGN}: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling},
  author    = {{\c{C}}avu{\c{s}}o{\u{g}}lu, Devrim and Akba{\c{s}}, Emre},
  booktitle = {Findings of the Association for Computational Linguistics: {EMNLP} 2026},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  note      = {To appear}
}
```

Authors: Devrim Çavuşoğlu (Middle East Technical University; OBSS AI), Emre Akbaş (Middle East Technical University). Correspondence: devrim.cavusoglu@metu.edu.tr

## License

Code is released under the Apache License 2.0. The `devrim/goodwiki_long_synthetic_ir` dataset is released under CC BY-SA 4.0, preserving the share-alike licensing and attribution of GoodWiki and the underlying Wikipedia text.

## Acknowledgements

This work was conducted and supported by OBSS under the project code ARGEM-024.
