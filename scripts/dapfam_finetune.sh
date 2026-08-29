#!/usr/bin/env bash
# DAPFAM REIGN domain-adaptation fine-tune (paper Section 5.3 / Table 4).
#
# DAPFAM relevance is BINARY, so this uses plain InfoNCE with no graded soft
# positives: --partial-policy ignore (explicit score=0 rows are never read as
# positives). Protocol: train on `train`, select on `val`, report the fine-tuned
# model on the held-out query-disjoint `test` (+ IN/OUT via test_in/test_out) —
# the SAME held-out set the zero-shot baselines are scored on. Assumes the
# query-disjoint split has already been written into default/ (run
# `python -m reign.dapfam.split_qrels --data-dir <DATA>` once, before
# any baseline).
#
# Requires a CUDA GPU. Roughly 6-10 h per run on a 24 GB card at these settings.
#
# The hyperparameters below are the published recipe. Environment variables
# exist so the same script can drive the regularization grid, but the defaults
# are the reported configuration.
#
#   PYTHON=... DATA=data/dapfam_ir_fulltext TAG=dapfam-ft bash scripts/dapfam_finetune.sh
#
# The default run is the headline fine-tune of the paper (§5.3; protocol in Appendix J):
# reign-base-l3_gn-gte-base_dapfam-ft-c512s512 (cold start, lr 1e-5, wd 1e-4,
# 15 epochs, chunk/stride 512). The other ten members of the released
# DAPFAM fine-tuned family come from FT_RUNS_OVERRIDE plus the matching
# LR / WD / EPOCHS / BS / STRIDE values; docs/REPRODUCING.md lists all of them
# verbatim, one line per checkpoint.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Reduce CUDA fragmentation (helps the per-step REIGN forwards over full-text
# chunk sequences fit alongside the negative pool).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON="${PYTHON:-python}"
DATA="${DATA:-data/dapfam_ir_fulltext}"
TAG="${TAG:-dapfam-ft}"
# Checkpoint root. `reign.train_ir` resolves --output-dir against REIGN_MODEL_DIR,
# so keep the two in step or training writes where this script cannot find it.
MODELS_DIR="${MODELS_DIR:-./models}"
export REIGN_MODEL_DIR="${REIGN_MODEL_DIR:-$MODELS_DIR}"
CHUNK="${CHUNK:-512}"
STRIDE="${STRIDE:-512}"
EPOCHS="${EPOCHS:-15}"
# LR / weight-decay are overridable for the regularization grid: warm-start
# fine-tuning is prone to catastrophic forgetting here (validation peaks around
# epoch 2 then degrades) and to overfitting (train loss goes to 0). The defaults
# are the published values.
LR="${LR:-1e-5}"
WD="${WD:-1e-4}"
# Post-REIGN corpus doc-embedding cache: test/test_in/test_out share one full
# corpus pool, so encode it once per (checkpoint, GN, chunk, stride) and reuse
# it across the 3 splits (skips ~2x45K redundant REIGN forwards per stride). The
# cache-miss split self-checks a bit-identical round-trip plus identical metrics
# before any later split reuses the entry, and every entry records a fingerprint
# of the checkpoint weights so a retrained checkpoint can never be served stale
# embeddings. Repo-relative by default; override to put it elsewhere.
CORPUS_EMB_CACHE="${CORPUS_EMB_CACHE:-.cache/reign/corpus_emb}"
mkdir -p logs results

# spec: model-config ; gn-model ; short ; out-subdir(under $MODELS_DIR) ; [warm-start-subdir(under $MODELS_DIR)]
# A non-empty 5th field warm-starts the fine-tune from that GoodWiki-trained
# backbone (load weights, fresh optimizer/LR/epoch) instead of random init — the
# domain-adaptation experiment. Empty 5th field means cold start.
# `short` names the result files (results/reign-<short>-ft_<split>_<TAG>.json) and
# the logs, so give every override run a distinct short or TAG.
declare -a FT_RUNS=(
  "base-l3;thenlper/gte-base;gte-base;reign-base-l3_gn-gte-base_dapfam-ft-c${CHUNK}s${STRIDE};"
)
[ -n "${FT_RUNS_OVERRIDE:-}" ] && IFS=$'\n' read -rd '' -a FT_RUNS <<<"$FT_RUNS_OVERRIDE" || true

echo "=== [1/3] verify train/val/test split present @ $(date +%T) ==="
"$PYTHON" - "$DATA" <<'PY' || { echo "  split missing — run: python -m reign.dapfam.split_qrels --data-dir $DATA"; exit 1; }
import sys, datasets
d = datasets.load_from_disk(f"{sys.argv[1]}/default")
need = {"train", "val", "test"}
missing = need - set(d)
assert not missing, f"default/ missing splits {missing}; have {list(d)}"
print(f"  splits OK: { {k: len(d[k]) for k in d} }")
PY

for spec in "${FT_RUNS[@]}"; do
  IFS=';' read -r mcfg gn short outdir warm <<<"$spec"
  warm_args=()
  if [ -n "${warm:-}" ]; then
    if [ ! -f "${MODELS_DIR}/${warm}/config.json" ]; then
      echo "  WARM-START backbone ${MODELS_DIR}/${warm}/config.json missing — skipping $short"; continue
    fi
    warm_args=(--warm-start-from "${MODELS_DIR}/${warm}")
    echo "  warm-start from ${MODELS_DIR}/${warm}"
  fi
  echo "=== [2/3] fine-tune $mcfg / $short → ${MODELS_DIR}/$outdir @ $(date +%T) ==="
  if [ -f "${MODELS_DIR}/${outdir}/best/config.json" ]; then
    echo "  checkpoint present — skipping training"
  else
    # Standard contrastive fine-tune (DPR/RocketQA recipe). DAPFAM is binary, so
    # --partial-policy ignore. Negatives = a few of DAPFAM's OWN provided hard
    # negatives per step (N_NEG) PLUS in-batch other-query positives, with
    # false-negative masking. Re-encoding 20 dedicated negatives per step runs
    # out of memory on 24 GB (160 full-text REIGN forwards), so N_NEG~4 plus
    # in-batch at batch 4 is the memory-feasible setting. The in-training val
    # monitor (ndcg@10) is an in-batch proxy used for checkpoint selection only;
    # the AUTHORITATIVE metric is evaluate_reign.py against the full 45K corpus
    # at top_k=100, run in step [3/3] below.
    #
    # Each run gets a fresh $MODELS_DIR/<outdir>. Never retrain into an existing
    # checkpoint directory — the corpus-embedding cache below is keyed on the
    # checkpoint path.
    "$PYTHON" -m reign.train_ir \
      --dataset "$DATA" --train-split train --eval-split val \
      --gn-model "$gn" --model-config "$mcfg" --output-dir "$outdir" \
      ${warm_args[@]+"${warm_args[@]}"} \
      --partial-policy ignore --n-negatives-per-sample "${N_NEG:-4}" --in-batch-negatives \
      --batch-size "${BS:-4}" --eval-batch-size "${EVAL_BS:-4}" --max-epochs "$EPOCHS" \
      --lr "$LR" --weight-decay "$WD" \
      --temperature 0.07 \
      --chunk-size "$CHUNK" --gn-stride "$STRIDE" --enable-cache \
      --precision 16-mixed --device cuda --seed 42 \
      --data-loader-num-workers 4 --check-val-every-n-epoch 3 \
      --top-k 10 --metric-to-monitor "ndcg@10" \
      > "logs/${TAG}_finetune_${short}.log" 2>&1 \
      || { echo "  TRAIN FAILED (see logs/${TAG}_finetune_${short}.log)"; continue; }
  fi

  ckpt="${MODELS_DIR}/${outdir}/best"
  echo "=== [3/3] eval fine-tuned $short on test / test_in / test_out @ $(date +%T) ==="
  for sp in test test_in test_out; do
    out="results/reign-${short}-ft_${sp}_${TAG}.json"
    [ -f "$out" ] && { echo "  [$sp] present"; continue; }
    "$PYTHON" scripts/evaluate_reign.py \
      --checkpoint "$ckpt" --gn-model "$gn" \
      --gn-chunk-size "$CHUNK" --gn-stride "$STRIDE" \
      --dataset "$DATA" --split "$sp" --top_k 100 \
      --batch_size 4 --gn-batch-size 8 \
      --corpus-embed-cache "$CORPUS_EMB_CACHE" \
      --name "reign-${short}-dapfam-ft-on-${sp}" --output_path "$out" \
      > "logs/${TAG}_ft_eval_${short}_${sp}.log" 2>&1 \
      && echo "  [$sp] OK → $out" \
      || echo "  [$sp] FAILED (see logs/${TAG}_ft_eval_${short}_${sp}.log)"
  done
done

echo "=== fine-tune done @ $(date +%T) ==="
echo "Results: results/reign-*-ft_{test,test_in,test_out}_${TAG}.json"
