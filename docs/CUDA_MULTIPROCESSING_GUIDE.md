# CUDA and DataLoader workers

## The problem

Using a PyTorch `DataLoader` with `num_workers > 0` on a CUDA device can fail with:

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

CUDA contexts cannot be safely shared between processes after a fork. `ReignFeatureExtractor` initialises the Guidance Network on the CUDA device in the main process, so when the DataLoader forks workers they inherit a CUDA context they cannot use.

## Solutions

### 1. Single-process data loading (applied automatically)

The trainer detects the unsafe combination and sets `num_workers = 0` when all of the following hold: the device is CUDA, `--data-loader-num-workers > 0`, and caching is not enabled. It logs a warning naming the two ways out.

```bash
# num_workers is forced to 0, with a warning
python -m reign.train --device cuda --data-loader-num-workers 4
```

`--data-loader-num-workers` defaults to 0, so this path is only reached if you ask for workers.

### 2. Cached embeddings (recommended)

Precomputing the GN's chunk embeddings takes CUDA out of the worker processes entirely, which makes multi-worker loading safe — and is what the released recipe does anyway:

```bash
python -m reign.train \
    --device cuda \
    --data-loader-num-workers 4 \
    --enable-cache \
    --cache-root ~/.reign_cache
```

Benefits: multiprocessing becomes safe, training is substantially faster after the first pass, and the cache persists across runs.

### 3. Force multiprocessing (advanced)

To override the safety check:

```bash
export REIGN_FORCE_MULTIPROCESSING=true
python -m reign.train --device cuda --data-loader-num-workers 4
```

⚠️ The trainer will warn that this may cause CUDA context errors, and it means it. Use `--enable-cache` instead unless you have a specific reason not to.

## Choosing between them

**`num_workers = 0`** — no CUDA context issues, simplest setup, may be slower on I/O-bound work. Good for small runs and debugging.

**`--enable-cache`** — fastest, supports multiprocessing, persists across runs, at the cost of an initial computation pass and disk space. This is the right default for any serious training.

## Troubleshooting

**"Cannot re-initialize CUDA in forked subprocess."** Use `--enable-cache`, or let the trainer force `num_workers = 0`.

**"CUDA out of memory."** Reduce `--batch-size`, or use `--enable-cache` so the GN forward pass is not repeated inside the training loop.

**Corrupted cache file.** The cache detects a corrupted file, warns, and regenerates it. To force a rebuild, use `--force-cache-refresh`.

## Cache management

```bash
# show cache information and exit
python -m reign.train --cache-info

# rebuild cached embeddings even if they exist
python -m reign.train --force-cache-refresh --enable-cache

# drop one model's cache by hand
rm -rf ~/.reign_cache/<model_hash>/
```

Cache entries are keyed by GN model, chunk size, and stride, so different configurations do not collide. See [PROVENANCE.md](PROVENANCE.md) for the integrity guarantees on the cache and on the separate corpus-embedding cache used at evaluation time.

## How the detection works

The logic in `reign/train.py` is:

```python
effective_num_workers = args.data_loader_num_workers
force_multiprocessing = os.environ.get("REIGN_FORCE_MULTIPROCESSING", "false").lower() == "true"

if (
    args.device == "cuda"
    and args.data_loader_num_workers > 0
    and not args.enable_cache
    and not force_multiprocessing
):
    # warn, then:
    effective_num_workers = 0
```

The effective worker count is logged at startup, so you can confirm which path was taken.
