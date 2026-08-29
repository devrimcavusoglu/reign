# Mixed-precision training

How to select a precision mode in the REIGN trainer, what hardware each mode needs, and what mixed precision costs you in reproducibility.

## Supported precision modes

`--precision` accepts three values.

### Full precision (`32`)

```bash
python -m reign.train --precision 32
```

The default. Highest numerical precision, most memory, compatible with all hardware.

### FP16 mixed precision (`16-mixed`)

```bash
python -m reign.train --precision 16-mixed
```

16-bit forward pass and gradient computation, 32-bit parameter updates. **This is what the released checkpoints were trained with.** Supported on NVIDIA Pascal and newer.

### BF16 mixed precision (`bf16-mixed`)

```bash
python -m reign.train --precision bf16-mixed
```

bfloat16 has better numerical range than fp16 and is generally more stable, at the cost of requiring Ampere or newer.

## Hardware requirements

| Mode | Minimum architecture | Compute capability | Examples |
| --- | --- | --- | --- |
| `16-mixed` | Pascal | ≥ 6.0 | GTX 10xx, RTX 20xx/30xx/40xx, V100, A100 |
| `bf16-mixed` | Ampere | ≥ 8.0 | RTX 30xx, RTX 40xx, A100, H100 |

The trainer warns if the selected mode is not supported by the detected hardware.

## Gradient clipping

Mixed precision can produce gradient instability. Clip to keep training well-behaved:

```bash
python -m reign.train \
    --precision 16-mixed \
    --gradient-clip-val 1.0 \
    --gradient-clip-algorithm norm
```

- `--gradient-clip-val` — maximum gradient norm or value (default: none)
- `--gradient-clip-algorithm` — `norm` or `value` (default: `norm`)

Reasonable ranges: 1.0–5.0 for norm clipping, 0.1–1.0 for value clipping. Loss scaling is handled automatically; no manual intervention is required.

## A complete run

The released recipe with mixed precision, gradient clipping added:

```bash
python -m reign.train \
    --dataset devrim/goodwiki_long_synthetic_ir --train-split train --eval-split val \
    --model-config base-l3 --gn-model thenlper/gte-small \
    --gn-chunk-size 512 --gn-stride 512 \
    --batch-size 18 --eval-batch-size 18 --negative-batch-size-multiplier 17 \
    --weight-partial 0.5 \
    --max-epochs 50 --lr 1e-5 --weight-decay 1e-4 \
    --precision 16-mixed \
    --gradient-clip-val 1.0 --gradient-clip-algorithm norm \
    --enable-cache \
    --metric-to-monitor ndcg@10 --check-val-every-n-epoch 4 \
    --seed 42 --device cuda \
    --output-dir my-reign-run
```

Swap `--precision 16-mixed` for `bf16-mixed` on Ampere or newer if you prefer the wider range.

## Reproducibility

**Mixed-precision training is not bit-reproducible, even at a fixed seed.** Two runs with identical arguments will produce slightly different weights, and evaluating under fp16 adds its own non-determinism: cuBLAS can select different algorithms depending on free-memory state, and changing the batch size changes the padding pattern within a batch. In practice this shifts nDCG by roughly 0.01 at a matched batch size and up to about 0.1 when the batch size differs.

The consequence for reproduction is that you should compare **metrics against the reference results**, not weights against the released checkpoints. See [REPRODUCING.md](REPRODUCING.md) for the tolerance to expect per stage.

## Output directory naming

When `--output-dir` is not given, the trainer derives a directory name that records the run's configuration, including the precision mode:

```
models/reign-base-l3_lr-1e-05_bs-18_..._cached_16-mixed/    # fp16 mixed
models/reign-base-l3_lr-1e-05_bs-18_..._cached_bf16-mixed/  # bf16 mixed
models/reign-base-l3_lr-1e-05_bs-18_..._cached/             # fp32, no suffix
```

Passing `--output-dir` explicitly overrides this; the released checkpoints use explicit names.

## Troubleshooting

**NaN or Inf during training.** Add `--gradient-clip-val 1.0`, reduce the learning rate slightly, or switch to `bf16-mixed` if the hardware supports it.

**Mixed precision is no faster than fp32.** REIGN's trainable encoder is small and, with `--enable-cache`, each step runs over a short precomputed embedding sequence — so the step is often not compute-bound and mixed precision has little to speed up. Check GPU utilisation before assuming something is wrong.

**CUDA out of memory.** Reduce `--batch-size`, or enable `--enable-cache` so the GN forward pass is not re-run inside the training loop. REIGN training itself uses roughly 2–3 GB of VRAM; if you are far above that, the GN is probably still in the loop.

**Warnings about an unsupported precision mode.** Fall back one step: `bf16-mixed` → `16-mixed` → `32`.

## Practical guidance

1. Use `16-mixed` unless you have Ampere or newer and want bf16's range.
2. Add `--gradient-clip-val 1.0` if you see instability.
3. Watch the loss curve over the first few validation epochs; the cosine objective should improve monotonically.
4. Use `--enable-cache` for any serious run — it dominates the memory and speed picture far more than the precision mode does.
