#!/usr/bin/env bash
# Train + evaluate REIGN-small-l2 on top of 4 short-context Guidance Network
# (GN) backbones, with checkpoint selection on the `val` qrels split.
# Produces the "REIGN-on-X" rows of paper Tables 3-4 alongside the X-alone rows
# emitted by `bash scripts/reproduce.sh main-goodwiki`.
#
# The hyperparameters below are the published recipe. Do not change them if you
# are trying to reproduce the paper's numbers.
#
# Modes:
#   bash scripts/train_reign_emnlp26.sh --smoke   # 1-epoch sanity check, ~15 min
#   bash scripts/train_reign_emnlp26.sh           # full queue, ~14-20 hours
#
# Requires a CUDA GPU. The full queue trains the 4 backbones in parallel and
# needs roughly 8 GB of GPU memory in total; it was developed on a 24 GB card.
#
# Re-runnable: skips any (GN, run) that already wrote both a best/config.json
# and a results/reign_<gn>_test.json, so a crashed mid-queue run can be resumed
# by re-invoking the same command.
#
# Outputs:
#   models/reign-small-l2_gn-<gn>_val-selected/{best,last}/{config.json,model.safetensors}
#   results/reign_<gn>_test.json
#   logs/reign_<gn>_<timestamp>.log

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs results models

MODE="${1:-full}"

DATASET="devrim/goodwiki_long_synthetic_ir"

# (GN model, short-name-for-paths) — order matters for the queue.
GN_LIST=(
  "thenlper/gte-large           gte-large"
  "BAAI/bge-large-en-v1.5       bge-large"
  "thenlper/gte-base            gte-base"
  "BAAI/bge-base-en-v1.5        bge-base"
)

# Hyperparameters held constant across all 4 runs, so the backbones are
# directly comparable. This is the published recipe.
COMMON_TRAIN_ARGS=(
  --dataset "$DATASET"
  --train-split train
  --eval-split val
  --model-config small-l2
  --batch-size 18
  --eval-batch-size 18
  --max-epochs 50
  --lr 1e-5
  --negative-batch-size-multiplier 17
  --weight-partial 0.5
  --chunk-size 512
  --enable-cache
  --precision 16-mixed
  --metric-to-monitor ndcg@10
  --check-val-every-n-epoch 4
  --seed 42
  --device cuda
)

run_smoke() {
  # 1-epoch / 200-sample run on gte-base. Goal: catch dataloader, cache, OOM and
  # arg-parse problems before committing 14+ hours.
  # train.py prepends MODEL_DIR to --output-dir, so we pass just a directory
  # name and rely on rm to clear stale state between smoke attempts.
  local out="_smoke_reign_gte-base"
  rm -rf "models/$out"
  local log="logs/reign_smoke_$(date +%Y%m%d_%H%M%S).log"
  echo "[smoke] $(date +%T) — 1 epoch on gte-base, max_samples=200 → $log"
  python -m reign.train \
    "${COMMON_TRAIN_ARGS[@]}" \
    --gn-model thenlper/gte-base \
    --max-epochs 1 \
    --max-samples 200 \
    --check-val-every-n-epoch 1 \
    --output-dir "$out" \
    > "$log" 2>&1
  echo "[smoke] $(date +%T) — done. Tail of log:"
  tail -10 "$log"
  echo "[smoke] If the above shows a saved best checkpoint and no Traceback, the main queue is safe to start."
}

run_one() {
  # Train + eval one (GN, REIGN-small-l2) pair.
  local gn="$1"
  local short="$2"
  # train.py prepends MODEL_DIR to --output-dir, so we pass the bare name and
  # reconstruct the full path locally for existence checks.
  local out_name="reign-small-l2_gn-${short}_val-selected"
  local out="models/${out_name}"
  local result="results/reign_${short}_test.json"
  local log="logs/reign_${short}_$(date +%Y%m%d_%H%M%S).log"

  if [ -f "$out/best/config.json" ] && [ -f "$result" ]; then
    echo "[$short] $(date +%T) — already complete (have $out/best + $result), skipping."
    return 0
  fi

  if [ ! -f "$out/best/config.json" ]; then
    echo "[$short] $(date +%T) — TRAIN start ($gn) → $log"
    # Each run writes to its own fresh output directory. Never point a new run
    # at an existing checkpoint directory: evaluate_reign.py's corpus-embedding
    # cache is keyed on the checkpoint path, and reusing a path for different
    # weights is exactly what its fingerprint guard exists to catch.
    python -m reign.train \
      "${COMMON_TRAIN_ARGS[@]}" \
      --gn-model "$gn" \
      --output-dir "$out_name" \
      >> "$log" 2>&1
    echo "[$short] $(date +%T) — TRAIN done."
  else
    echo "[$short] $(date +%T) — train output already exists at $out/best, skipping training."
  fi

  echo "[$short] $(date +%T) — EVAL on test → $result"
  python scripts/evaluate_reign.py \
    --checkpoint "$out/best" \
    --gn-model "$gn" \
    --chunk-size 512 \
    --dataset "$DATASET" \
    --split test \
    --top_k 10 \
    --batch_size 8 \
    --output_path "$result" \
    >> "$log" 2>&1
  echo "[$short] $(date +%T) — EVAL done."
}

run_compute_queue() {
  # After all REIGN training + eval finishes, fill the runtime/memory columns
  # of the paper table with measure_compute.py runs (sub-sampled for repeatable
  # timing on a fixed hardware target). Timing runs need an otherwise-idle GPU.
  echo "=== COMPUTE PHASE @ $(date +%T) ==="
  local ncorpus="${COMPUTE_N_CORPUS:-200}"
  local nq="${COMPUTE_N_QUERIES:-50}"

  for r in bm25 tfidf; do
    echo "  - sparse: $r"
    python scripts/measure_compute.py \
      --kind sparse --retriever "$r" \
      --dataset "$DATASET" \
      --n_corpus "$ncorpus" --n_queries "$nq" \
      --output_path "results/compute_sparse_${r}.json" || true
  done

  for b in bge-m3 nomic-v1.5 bge-large-chunked bge-base-chunked gte-large-chunked gte-base-chunked; do
    echo "  - dense: $b"
    python scripts/measure_compute.py \
      --kind dense --baseline "$b" \
      --dataset "$DATASET" \
      --n_corpus "$ncorpus" --n_queries "$nq" \
      --output_path "results/compute_dense_${b}.json" || true
  done

  for entry in "${GN_LIST[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    local gn="$1" short="$2"
    local ckpt="models/reign-small-l2_gn-${short}_val-selected/best"
    if [ ! -f "$ckpt/config.json" ]; then
      echo "  - reign-on-${short}: SKIPPED (no $ckpt/config.json)"
      continue
    fi
    echo "  - reign-on-${short} (gn=$gn)"
    python scripts/measure_compute.py \
      --kind reign --checkpoint "$ckpt" --gn-model "$gn" \
      --chunk-size 512 \
      --dataset "$DATASET" \
      --n_corpus "$ncorpus" --n_queries "$nq" \
      --name "reign-on-${short}" \
      --output_path "results/compute_reign_${short}.json" || true
  done

  echo "=== COMPUTE PHASE DONE @ $(date +%T) ==="
}

run_full_queue() {
  # REIGN-small-l2 training uses ~2 GB of GPU memory and ~7-12% utilisation on a
  # 24 GB card, so all 4 backbones run in parallel (~8 GB total) while keeping
  # the GPU busy. Each run writes to its own output directory and its own cache
  # files (keyed by GN model), so there is no on-disk contention.
  local pids=()
  for entry in "${GN_LIST[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry  # split on whitespace
    run_one "$1" "$2" &
    pids+=($!)
  done

  # Wait for every parallel run, recording per-PID exit codes. Any failure is
  # surfaced but doesn't abort the rest.
  local rc=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "=== parallel REIGN run (pid $pid) failed @ $(date +%T) ==="
      rc=1
    fi
  done

  echo "=== ALL REIGN RUNS DONE @ $(date +%T) (parallel rc=$rc) ==="
  echo "Result JSONs:"
  ls -la results/reign_*_test.json 2>&1

  # Compute measurements for every baseline + REIGN, so the final table
  # carries both accuracy and runtime columns out-of-the-box.
  run_compute_queue
  echo "Accuracy rows are in results/reign_*_test.json; timing rows in results/compute_*.json."
}

case "$MODE" in
  --smoke|smoke)  run_smoke ;;
  --full|full|"") run_full_queue ;;
  *) echo "usage: $0 [--smoke | --full]"; exit 2 ;;
esac
