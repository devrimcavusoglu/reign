# Training REIGN

> **⚠️ Two training protocols exist in this repository, and their numbers are not comparable.**
>
> **Protocol A — the released-checkpoint recipe.** A three-way cosine embedding loss at batch 18 for 50 epochs. This is what produced every released checkpoint and every headline number in Tables 2, 3, and 4. It is what `scripts/train_reign_emnlp26.sh` runs and what `reign.train` does by default.
>
> **Protocol B — the ablation protocol.** InfoNCE at τ = 0.07, batch 48, 20 epochs. This is used **only** for the Appendix E positional-encoding ablation and the Appendix I training-objective ablation, where every arm shares one controlled protocol so the arms are comparable to each other.
>
> Protocol B arms sit **below** the released operating point by construction — the paper states this explicitly. Do not attempt to reproduce a Protocol A number with Protocol B settings, do not compare a Protocol B arm against a headline table, and do not read the gap between them as a result. The only valid comparisons are A against A and B against B.

---

## Protocol A — the released-checkpoint recipe

### Objective

A three-way cosine embedding loss over document pairs *(x, y)* carrying graded targets *s ∈ {1, 0, −1}* — positive, partial, negative:

- *s = 1*: `1 − c`
- *s = 0*: `λ(1 − c) + (1 − λ)·c₊`
- *s = −1*: `c₊`

where *c* = cos(*x*, *y*), *c₊* = max(0, *c*), and the partial weight is **λ = 0.5**.

### Batch construction

Batch size 18 anchors. Each anchor is paired with:

- its rephrased positive (target 1),
- its two topical distractors as partials (target 0),
- 17 in-batch negatives formed by shifting the positives (target −1),

giving **360 pairs per optimisation step**.

### Optimisation

| Setting | Value |
| --- | --- |
| Optimiser | AdamW |
| Learning rate | 1e-5 |
| Weight decay | 1e-4 |
| Schedule | Cosine annealing |
| Epochs | 50 |
| Validation | Every 4 epochs |
| Checkpoint selection | Best validation nDCG@10 on the `val` split |
| GN embeddings | Precomputed and cached |
| Precision | 16-mixed |
| Seed | 42 |
| Chunking | Sliding window K = 512, stride S = K unless otherwise specified |
| Hardware | Single 24 GB consumer GPU |

The `_val-selected` suffix on the released checkpoint names records the best-validation selection; each checkpoint directory holds a `best/` and a `last/` snapshot.

### Running it

`scripts/train_reign_emnlp26.sh` runs the queue across GN backbones. A single run is:

```bash
python -m reign.train \
  --dataset devrim/goodwiki_long_synthetic_ir --train-split train --eval-split val \
  --model-config base-l3 --gn-model thenlper/gte-small \
  --gn-chunk-size 512 --gn-stride 512 \
  --batch-size 18 --eval-batch-size 18 --negative-batch-size-multiplier 17 \
  --weight-partial 0.5 \
  --max-epochs 50 --lr 1e-5 --weight-decay 1e-4 \
  --enable-cache --precision 16-mixed \
  --metric-to-monitor ndcg@10 --check-val-every-n-epoch 4 \
  --seed 42 --device cuda \
  --output-dir reign-base-l3_gn-gte-small_s512_val-selected
```

`--model-config` selects the encoder size (`tiny-l1`, `small-l2`, `base-l3`, `large-l4` are the four the paper sweeps; `base-l3` is the paper default). `--gn-model` selects the frozen Guidance Network. `--output-dir` is interpreted relative to the models directory — `./models` by default, overridable with `REIGN_MODEL_DIR`.

### Defaults that already match Protocol A

`reign.train` ships with Protocol A as its default where a default exists, so most of the recipe is opt-out rather than opt-in:

| Argument | Default | Matches recipe A |
| --- | --- | --- |
| `--loss-function` | `cosine` (`ThreeWayCosineEmbeddingLoss`) | ✅ |
| `--weight-partial` | `0.5` | ✅ |
| `--lr` | `1e-5` | ✅ |
| `--weight-decay` | `1e-4` | ✅ |
| `--max-epochs` | `50` | ✅ |
| `--check-val-every-n-epoch` | `4` | ✅ |
| `--seed` | `42` | ✅ |
| `--gn-chunk-size` | `512` | ✅ |
| `--position-embedding-type` | `none` | ✅ (the published design) |
| `--batch-size` | `12` | ❌ pass `18` |
| `--negative-batch-size-multiplier` | `1` | ❌ pass `17` |
| `--precision` | `32` | ❌ pass `16-mixed` |
| `--metric-to-monitor` | `map@10` | ❌ pass `ndcg@10` |
| `--gn-stride` | `384` | ❌ pass `512` for the non-overlapping setting |
| `--enable-cache` | off | ❌ pass the flag |

`--loss-function cosine` is the default precisely because it is the released recipe. If you invoke the trainer without specifying a loss, you are training under Protocol A.

---

## Protocol B — the ablation protocol

Used **only** by the Appendix E (positional-encoding) and Appendix I (training-objective) ablations.

| Setting | Value |
| --- | --- |
| Objective | InfoNCE |
| Temperature | 0.07 |
| Batch size | 48 (47 in-batch negatives) |
| Negative pool | 2,304 pairs per step |
| Epochs | 20 |
| Encoder | `base-l3` on GTE-small |
| Chunk / stride | 512 / 512 |
| Learning rate | 1e-5, weight decay 1e-4 |
| Precision | 16-mixed |
| Seed | 42 |
| Checkpoint selection | Best validation nDCG@10 |

```bash
python -m reign.train \
  --dataset devrim/goodwiki_long_synthetic_ir --train-split train --eval-split val \
  --model-config base-l3 --gn-model thenlper/gte-small \
  --gn-chunk-size 512 --gn-stride 512 \
  --batch-size 48 --eval-batch-size 48 --negative-batch-size-multiplier 47 \
  --loss-function infonce --temperature 0.07 --weight-partial 0.5 \
  --max-epochs 20 --lr 1e-5 --weight-decay 1e-4 \
  --enable-cache --precision 16-mixed \
  --metric-to-monitor ndcg@10 --check-val-every-n-epoch 2 \
  --seed 42 --device cuda \
  --output-dir <arm-name>
```

Within Protocol B, arms differ in exactly one thing — the chunk-position signal for Appendix E (`--position-embedding-type none|absolute|sinusoidal`), or the objective, distractor weight, batch size, and start point for Appendix I.

### Stability: InfoNCE needs a large negative pool

This is the practical reason Protocol B uses batch 48 and not the released batch 18.

The InfoNCE path contrasts each anchor against the shifted in-batch positives only. At **batch 18 the per-step pool is 324 pairs**, against 360 pairs including partials on the cosine path — and at τ = 0.07 that pool is too small to train. Validation nDCG@10 peaks at the first validation epoch (0.73) and decays monotonically to 0.20 by epoch 19, so a batch-18 InfoNCE row is effectively an epoch-1 snapshot rather than a converged model.

**Batch 48 (2,304 pairs) removes the collapse.** The α = 0 arm improves steadily (0.76 → 0.78), while the α = 0.5 arm still decays after an early peak (0.76 → 0.61).

The cosine objective shows no such sensitivity: its validation improves monotonically at both batch sizes. Anyone reproducing REIGN with InfoNCE should budget the larger batch.

### The DAPFAM fine-tune is InfoNCE by design

The DAPFAM patent fine-tuning experiments use InfoNCE at temperature 0.07 over the dataset's provided negatives. This is a deliberate choice, not a Protocol B artefact: DAPFAM's relevance labels are binary, so it runs a standard query/positive/negative contrastive path with false-negative masking and `partial-policy = ignore`, rather than the graded three-way objective the GoodWiki-Long recipe uses.

Fine-tuning details (Appendix J): AdamW with lr ∈ {1e-5, 5e-6, 2e-6, 1e-6} and weight decay ∈ {1e-4, 1e-2, 1e-1}, cosine schedule, temperature 0.07, 16-mixed precision, seed 42, validation every 3 epochs. Negatives are DAPFAM's provided random score = 0 families at 4 per sample plus in-batch negatives; the dataset authors' original 20 negatives exceeds 24 GB at FullText sequence length.

Warm-start runs initialise from the GoodWiki-Long-trained checkpoint; cold-start runs initialise the REIGN encoder from scratch with the same frozen GN. Fine-tuning does not exceed zero-shot: the best run — a regularised warm-start at lr 1e-6, wd 1e-2 — reaches 32.73 nDCG@100 against the matched zero-shot backbone's 32.68 (+0.05, statistical parity), and naive fine-tuning at lr 1e-5 degrades by 0.4–1.5 points.

---

## Practical notes

**Cached GN embeddings.** `--enable-cache` precomputes and caches the frozen GN's chunk embeddings under `--cache-root` (default `~/.reign_cache`), keyed by GN model, chunk size, and stride. This is what makes training cheap: only the small REIGN encoder runs per step, over a fixed-length embedding sequence. It also makes multi-worker data loading safe on CUDA — see [CUDA_MULTIPROCESSING_GUIDE.md](CUDA_MULTIPROCESSING_GUIDE.md).

**Memory.** REIGN training uses roughly **2–3 GB of VRAM**, so several runs fit concurrently on one 24 GB card. Memory scales with chunk count rather than parameter count: a `small-l2` model paired with GTE-large at effective batch size 312 consumes approximately 1.8, 3.6, 4.5, and 8.4 GiB for chunk sizes 512, 256, 128, and 64 respectively.

**Never retrain into an existing checkpoint directory.** A new run gets a new `--output-dir`. The trainer refuses to create an output directory that already exists, which enforces this; the reason it matters for cached artefacts is in [PROVENANCE.md](PROVENANCE.md).

**Precision.** The released recipe uses `16-mixed`. See [MIXED_PRECISION_TRAINING.md](MIXED_PRECISION_TRAINING.md) for the precision modes, hardware requirements, and gradient clipping. Note that 16-mixed training is not bit-reproducible even at a fixed seed, so a retrained checkpoint will not match the released weights bit-for-bit; compare metrics, not weights.

**Metric logs.** `--logger` defaults to `csv`, writing `logs/<run-name>/<version>/metrics.csv` relative to the working directory; `tensorboard` and `none` are the other choices. Logging is entirely local — no external tracking service is contacted. `logs/` is gitignored.

**Encoder capacity.** The in-distribution optimum and the out-of-distribution optimum differ. `small-l2` wins the in-distribution sweep on two of three GNs, but `base-l3` wins 11 of 12 out-of-distribution cells and is the paper default for that reason. If you adapt REIGN to a new domain, plan an out-of-distribution verification step rather than trusting an in-distribution capacity sweep alone.
