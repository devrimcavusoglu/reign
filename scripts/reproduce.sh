#!/usr/bin/env bash
# REIGN paper reproduction entry point.
#
#   bash scripts/reproduce.sh <stage>
#
# ---------------------------------------------------------------------------
# CHECKPOINTS
# ---------------------------------------------------------------------------
# The trained REIGN checkpoints are distributed through the Hugging Face Hub.
# The weights release is still pending; until it lands, place checkpoints
# yourself under $MODELS_DIR (default ./models), using the directory names the
# stages below expect. The exact expected layout — one directory per run, each
# containing best/{config.json,model.safetensors} — is documented in
# docs/REPRODUCING.md, together with which released checkpoint corresponds to
# which paper row.
#
# Any stage whose checkpoints are missing skips the affected rows with a message
# rather than failing, so the baseline-only rows of a table can be reproduced
# before the weights are available.
#
# ---------------------------------------------------------------------------
# CONFIGURATION (environment variables)
# ---------------------------------------------------------------------------
#   MODELS_DIR    where checkpoints live            (default: ./models)
#   RESULTS_DIR   where result JSONs are written    (default: ./results)
#   PYTHON        interpreter to use                (default: python)
#   DAPFAM_DATA   local DAPFAM dataset directory    (default: data/dapfam_ir_fulltext)
#   GN_STRIDE     Guidance Network stride for REIGN evals (default: 512)
#
# Row-set overrides (one spec per line; see docs/REPRODUCING.md for the exact
# values that reproduce each paper table):
#   REIGN_ROWS_OVERRIDE      REIGN rows for main-goodwiki / main-loco / e1,
#                            as "short|checkpoint-dir|gn-model"
#   DAPFAM_ZS_ROWS_OVERRIDE  REIGN zero-shot rows for main-dapfam, same format
#   DAPFAM_SPLITS            DAPFAM splits to score (default: test; the paper's
#                            IN-/cross-IPC columns are "test test_in test_out")
#   DENSE_NATIVE / DENSE_CHUNKED / DENSE_DAPFAM   baseline lists
#   MTEB_TASKS / MTEB_CKPT / MTEB_GN              the App. B row
#   FT_RUNS_OVERRIDE         passed through to scripts/dapfam_finetune.sh
#
# Every stage is re-runnable: an output file that already exists is left alone,
# so an interrupted stage resumes by re-invoking the same command.
#
# ---------------------------------------------------------------------------
# STAGES
# ---------------------------------------------------------------------------
#   main-goodwiki          paper Tables 2, 3, 4 — in-distribution GoodWiki-Long
#   main-loco              paper Tables 3, 4 — zero-shot LoCo (12 subtasks)
#   main-dapfam            paper Table 4 / Section 5.3 — DAPFAM patent prior art
#   e1-efficiency          paper App. G, Table 11 — measured latency and memory
#   e2-significance        paper Table 5 — paired significance tests on DAPFAM
#   e4-pe-ablation         paper App. E — positional-encoding ablation
#   e5-objective-ablation  paper App. I — training-objective ablation
#   mteb                   paper App. B — short-context MTEB sanity check
#   all                    every stage above, in order
#
# GPU: every stage except e2-significance needs a CUDA GPU. e2-significance is
# pure CPU post-processing over per-query dumps that main-dapfam produced.
# Runtime estimates below are for a single 24 GB card (RTX 4090 class).

set -uo pipefail

STAGE="${1:-help}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
MODELS_DIR="${MODELS_DIR:-./models}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
DAPFAM_DATA="${DAPFAM_DATA:-data/dapfam_ir_fulltext}"
GOODWIKI="devrim/goodwiki_long_synthetic_ir"
GN_STRIDE="${GN_STRIDE:-512}"
GN_CHUNK="${GN_CHUNK:-512}"

# `reign.train` resolves its checkpoint root from REIGN_MODEL_DIR and takes a
# bare run name in --output-dir. Point it at MODELS_DIR so training writes where
# the evaluation steps below look.
export REIGN_MODEL_DIR="${REIGN_MODEL_DIR:-$MODELS_DIR}"

mkdir -p "$RESULTS_DIR" logs

# --- REIGN rows: short-name | checkpoint dir under $MODELS_DIR | GN model id ---
# These are the base-l3 val-selected runs of the headline tables. Tables 3 and 4
# pair each REIGN row with its own Guidance Network run alone (the "-chunked"
# baselines).
#
# Any released checkpoint can be evaluated by replacing this row set with
# REIGN_ROWS_OVERRIDE — one "short|checkpoint-dir|gn-model" spec per line. That
# is how the encoder-capacity sweeps of Tables 7 and 8 are produced; the exact
# override values for each table are listed in docs/REPRODUCING.md. `short` is
# the key in every output filename, so give each row a distinct short (and set
# GN_STRIDE to the stride you want evaluated) or runs will overwrite each other.
REIGN_ROWS=(
  "gte-small|reign-base-l3_gn-gte-small_s512_val-selected|thenlper/gte-small"
  "gte-base|reign-base-l3_gn-gte-base_val-selected|thenlper/gte-base"
  "gte-large|reign-base-l3_gn-gte-large_val-selected|thenlper/gte-large"
  "bge-base|reign-base-l3_gn-bge-base_val-selected|BAAI/bge-base-en-v1.5"
  "bge-large|reign-base-l3_gn-bge-large_val-selected|BAAI/bge-large-en-v1.5"
)
[ -n "${REIGN_ROWS_OVERRIDE:-}" ] && IFS=$'\n' read -rd '' -a REIGN_ROWS <<<"$REIGN_ROWS_OVERRIDE" || true

# Native long-context dense encoders (read the whole document in one pass).
# The three baseline lists use ${VAR-default}, not ${VAR:-default}, so setting one
# to the empty string genuinely suppresses its block (the appendix sweeps compare
# REIGN rows only and do not re-run the baselines).
DENSE_NATIVE="${DENSE_NATIVE-bge-m3 jina-v3 stella-1.5b nomic-v1.5}"
# The bare Guidance Networks, chunked + mean-pooled: REIGN's like-for-like
# comparison. Add --protocol truncate to a call to get the truncated variant.
DENSE_CHUNKED="${DENSE_CHUNKED-gte-small-chunked gte-base-chunked gte-large-chunked bge-base-chunked bge-large-chunked}"
# Dense comparators on DAPFAM. The default is exactly the set the Table 5 paired
# tests consume; widen it to cover the rest of Table 4's dense block.
DENSE_DAPFAM="${DENSE_DAPFAM-jina-v3 stella-1.5b gte-large-chunked}"

# --- REIGN zero-shot rows for main-dapfam, same "short|checkpoint|gn" format ---
# The default is the single row Table 5 tests. Override with DAPFAM_ZS_ROWS to
# score any other released checkpoint; docs/REPRODUCING.md lists the values that
# reproduce the REIGN rows of Table 4 and the DAPFAM half of Table 8. Note that
# e2-significance reads results/e2/reign_gte-large_s<stride>.json, so keep a row
# whose short is `gte-large` if you intend to run that stage afterwards.
DAPFAM_ZS_ROWS=(
  "gte-large|reign-base-l3_gn-gte-large_val-selected|thenlper/gte-large"
)
[ -n "${DAPFAM_ZS_ROWS_OVERRIDE:-}" ] && IFS=$'\n' read -rd '' -a DAPFAM_ZS_ROWS <<<"$DAPFAM_ZS_ROWS_OVERRIDE" || true

have_ckpt() {  # have_ckpt <checkpoint-dir-name> -> 0 if present
  [ -f "${MODELS_DIR}/$1/best/config.json" ]
}

skip_ckpt() {  # skip_ckpt <short> <checkpoint-dir-name>
  echo "  [skip] $1 — no checkpoint at ${MODELS_DIR}/$2/best (see docs/REPRODUCING.md)"
}

# LoCo writes one JSON per subtask rather than a single output file, so its
# "already done" check counts the 12 subtask files instead of stat-ing one path.
LOCO_N_SUBTASKS=12
have_loco() {  # have_loco <tag> -> 0 if all 12 subtask JSONs are present
  local n
  n=$(find "${RESULTS_DIR}/loco" -maxdepth 1 -name "loco_$1_*.json" 2>/dev/null | wc -l)
  [ "$n" -ge "$LOCO_N_SUBTASKS" ]
}

# ===========================================================================
# main-goodwiki — paper Tables 2, 3, 4 (in-distribution GoodWiki-Long test)
# ===========================================================================
# GPU required. ~4-6 h for the full set (the 1.5B Stella row dominates).
stage_main_goodwiki() {
  echo "=== [main-goodwiki] GoodWiki-Long test — paper Tables 2, 3, 4 ==="
  local out

  echo "--- classical sparse baselines (CPU) ---"
  for r in bm25 tfidf; do
    out="${RESULTS_DIR}/sparse_${r}_goodwiki_test.json"
    [ -f "$out" ] && { echo "  [have] $out"; continue; }
    echo "  - $r"
    "$PYTHON" scripts/evaluate_sparse_baselines.py \
      --retriever "$r" --dataset "$GOODWIKI" --split test \
      --top_k 10 --output_path "$out"
  done

  echo "--- native long-context dense baselines (fp16, the reported dtype) ---"
  for b in $DENSE_NATIVE; do
    out="${RESULTS_DIR}/dense_${b}_goodwiki_test.json"
    [ -f "$out" ] && { echo "  [have] $out"; continue; }
    echo "  - $b"
    "$PYTHON" scripts/evaluate_dense_baselines.py \
      --baseline "$b" --dataset "$GOODWIKI" --split test \
      --top_k 10 --batch_size 8 --torch-dtype float16 --output_path "$out"
  done

  echo "--- bare Guidance Networks, chunked mean-pool ---"
  for b in $DENSE_CHUNKED; do
    out="${RESULTS_DIR}/dense_${b}_goodwiki_test.json"
    [ -f "$out" ] && { echo "  [have] $out"; continue; }
    echo "  - $b"
    "$PYTHON" scripts/evaluate_dense_baselines.py \
      --baseline "$b" --dataset "$GOODWIKI" --split test \
      --top_k 10 --batch_size 8 --output_path "$out"
  done

  echo "--- REIGN on each Guidance Network ---"
  for row in "${REIGN_ROWS[@]}"; do
    IFS='|' read -r short ckpt gn <<< "$row"
    out="${RESULTS_DIR}/reign_${short}_goodwiki_test.json"
    [ -f "$out" ] && { echo "  [have] $out"; continue; }
    have_ckpt "$ckpt" || { skip_ckpt "$short" "$ckpt"; continue; }
    echo "  - reign-on-${short}"
    "$PYTHON" scripts/evaluate_reign.py \
      --checkpoint "${MODELS_DIR}/${ckpt}/best" --gn-model "$gn" \
      --gn-chunk-size "$GN_CHUNK" --gn-stride "$GN_STRIDE" \
      --dataset "$GOODWIKI" --split test \
      --top_k 10 --batch_size 8 \
      --name "reign-${short}-s${GN_STRIDE}" --output_path "$out"
  done
}

# ===========================================================================
# main-loco — paper Tables 3, 4 (zero-shot LoCo, 12 subtasks)
# ===========================================================================
# GPU required. ~1-2 h per encoder for all 12 subtasks.
stage_main_loco() {
  echo "=== [main-loco] LoCo, 12 subtasks, zero-shot — paper Tables 3, 4 ==="

  echo "--- dense baselines ---"
  for b in $DENSE_NATIVE $DENSE_CHUNKED; do
    have_loco "dense-${b}" && { echo "  [have] loco_dense-${b}_*.json"; continue; }
    echo "  - $b"
    "$PYTHON" scripts/evaluate_loco.py \
      --baseline "$b" --subtask all \
      --top_k 10 --batch_size 8 --torch-dtype float16 \
      --output-dir "${RESULTS_DIR}/loco" --tag "dense-${b}"
  done

  echo "--- REIGN ---"
  for row in "${REIGN_ROWS[@]}"; do
    IFS='|' read -r short ckpt gn <<< "$row"
    local tag="reign-${short}-s${GN_STRIDE}"
    have_loco "$tag" && { echo "  [have] loco_${tag}_*.json"; continue; }
    have_ckpt "$ckpt" || { skip_ckpt "$short" "$ckpt"; continue; }
    echo "  - reign-on-${short}"
    "$PYTHON" scripts/evaluate_loco.py \
      --reign-checkpoint "${MODELS_DIR}/${ckpt}/best" --gn-model "$gn" \
      --gn-chunk-size "$GN_CHUNK" --gn-stride "$GN_STRIDE" --subtask all \
      --top_k 10 --batch_size 8 \
      --output-dir "${RESULTS_DIR}/loco" --tag "$tag"
  done
}

# ===========================================================================
# main-dapfam — paper Table 4 / Section 5.3 (DAPFAM patent prior art)
# ===========================================================================
# GPU required. ~8-12 h. Also writes the per-query dumps that e2-significance
# consumes, so run this stage before e2-significance.
stage_main_dapfam() {
  echo "=== [main-dapfam] DAPFAM patent prior art — paper Table 4 / Section 5.3 ==="
  local e2="${RESULTS_DIR}/e2"
  mkdir -p "$e2"

  if [ ! -f "${DAPFAM_DATA}/default/dataset_dict.json" ]; then
    echo "--- building DAPFAM (fulltext view) → ${DAPFAM_DATA} ---"
    "$PYTHON" -m reign.dapfam.build_dataset \
      --text-view fulltext --out-dir "$DAPFAM_DATA" --download
  fi
  # Query-disjoint 70/15/15 split, written into default/. Idempotent for a fixed
  # seed; only (re)split when the train split is absent.
  if ! "$PYTHON" -c "import datasets,sys; sys.exit(0 if 'train' in datasets.load_from_disk('${DAPFAM_DATA}/default') else 1)" 2>/dev/null; then
    echo "--- splitting DAPFAM (query-disjoint 70/15/15) ---"
    "$PYTHON" -m reign.dapfam.split_qrels \
      --data-dir "$DAPFAM_DATA" --train-ratio 0.70 --val-ratio 0.15 --seed 42
  fi

  # DAPFAM is scored at top_k=100 (nDCG@100), unlike GoodWiki-Long's top_k=10.
  # Results land in results/e2/ under the exact names e2-significance expects.
  # `test` is the split every paper row reports; DAPFAM_SPLITS="test test_in
  # test_out" additionally produces Table 4's IN-/cross-IPC columns for the
  # chunked-GN and REIGN rows (the three splits share one corpus encode).
  local splits="${DAPFAM_SPLITS:-test}"
  local suffix

  echo "--- classical sparse baselines (CPU) ---"
  for r in bm25 tfidf; do
    local out="${e2}/sparse_${r}.json"
    [ -f "$out" ] && { echo "  [have] $out"; continue; }
    echo "  - $r"
    "$PYTHON" scripts/evaluate_sparse_baselines.py \
      --retriever "$r" --dataset "$DAPFAM_DATA" --split test \
      --top_k 100 --output_path "$out"
  done

  echo "--- dense comparators (fp16, the reported dtype) ---"
  for b in $DENSE_DAPFAM; do
    for sp in $splits; do
      suffix=""; [ "$sp" = "test" ] || suffix="_${sp}"
      local out="${e2}/dense_${b}${suffix}.json"
      [ -f "$out" ] && { echo "  [have] $out"; continue; }
      echo "  - ${b} (${sp})"
      "$PYTHON" scripts/evaluate_dense_baselines.py \
        --baseline "$b" --dataset "$DAPFAM_DATA" --split "$sp" \
        --top_k 100 --batch_size 4 --torch-dtype float16 --output_path "$out"
    done
  done

  echo "--- REIGN zero-shot (the gte-large row is the system under test in Table 5) ---"
  for row in "${DAPFAM_ZS_ROWS[@]}"; do
    IFS='|' read -r short ckpt gn <<< "$row"
    have_ckpt "$ckpt" || { skip_ckpt "$short" "$ckpt"; continue; }
    for sp in $splits; do
      suffix=""; [ "$sp" = "test" ] || suffix="_${sp}"
      local out="${e2}/reign_${short}_s${GN_STRIDE}${suffix}.json"
      [ -f "$out" ] && { echo "  [have] $out"; continue; }
      echo "  - reign-on-${short} (${sp})"
      "$PYTHON" scripts/evaluate_reign.py \
        --checkpoint "${MODELS_DIR}/${ckpt}/best" --gn-model "$gn" \
        --gn-chunk-size "$GN_CHUNK" --gn-stride "$GN_STRIDE" \
        --dataset "$DAPFAM_DATA" --split "$sp" \
        --top_k 100 --batch_size 4 --gn-batch-size 8 \
        --corpus-embed-cache "${RESULTS_DIR}/.corpus_emb_cache" \
        --name "reign-${short}-zs-s${GN_STRIDE}" --output_path "$out"
    done
  done

  echo "--- REIGN in-domain fine-tune (Section 5.3 / App. J) ---"
  # Default: the headline fine-tune, $MODELS_DIR/reign-base-l3_gn-gte-base_dapfam-ft-c512s512.
  # FT_RUNS_OVERRIDE (with the matching LR/WD/EPOCHS/BS/STRIDE) reproduces the
  # other released family members; docs/REPRODUCING.md lists the exact values.
  MODELS_DIR="$MODELS_DIR" DATA="$DAPFAM_DATA" TAG="${TAG:-dapfam-ft}" \
    CHUNK="$GN_CHUNK" STRIDE="$GN_STRIDE" bash scripts/dapfam_finetune.sh
}

# ===========================================================================
# e1-efficiency — paper App. G, Table 11 (measured latency and peak memory)
# ===========================================================================
# GPU required, and it MUST be otherwise idle: these are timing measurements and
# a co-scheduled job invalidates every row. Strictly sequential. ~1-2 h.
stage_e1_efficiency() {
  echo "=== [e1-efficiency] measured latency + memory — paper App. G, Table 11 ==="
  local d="${RESULTS_DIR}/e1"
  mkdir -p "$d"
  local nc="${E1_N_CORPUS:-500}" nq="${E1_N_QUERIES:-100}"
  local common=(--dataset "$GOODWIKI" --split test --n_corpus "$nc" --n_queries "$nq"
                --batch_size 8 --n_warmup 1 --n_repeat 3 --top_k 10)

  run_e1() {  # run_e1 <row-name> <args...>
    local name="$1"; shift
    local out="${d}/compute_${name}.json"
    [ -f "$out" ] && { echo "  [have] compute_${name}"; return 0; }
    echo "  - $name"
    "$PYTHON" scripts/measure_compute.py "$@" --output_path "$out" || true
  }

  for r in bm25 tfidf; do
    run_e1 "sparse_${r}" --kind sparse --retriever "$r" "${common[@]}"
  done
  # dtypes match the configuration that produced the reported accuracy numbers.
  run_e1 "dense_bge-m3"      --kind dense --baseline bge-m3      "${common[@]}"
  run_e1 "dense_jina-v3"     --kind dense --baseline jina-v3     --torch-dtype float16 "${common[@]}"
  run_e1 "dense_stella-1.5b" --kind dense --baseline stella-1.5b --torch-dtype float16 "${common[@]}"
  run_e1 "dense_nomic-v1.5"  --kind dense --baseline nomic-v1.5  "${common[@]}"
  for b in $DENSE_CHUNKED; do
    run_e1 "dense_${b}" --kind dense --baseline "$b" "${common[@]}"
  done

  # REIGN is measured three ways: uncached (cold cache, GN runs per query),
  # build (the one-time GN cache write being amortised), cached (steady state).
  for row in "${REIGN_ROWS[@]}"; do
    IFS='|' read -r short ckpt gn <<< "$row"
    case "$short" in gte-small|gte-base|gte-large) ;; *) continue ;; esac
    have_ckpt "$ckpt" || { skip_ckpt "$short" "$ckpt"; continue; }
    local base=(--kind reign --checkpoint "${MODELS_DIR}/${ckpt}/best" --gn-model "$gn"
                --chunk-size "$GN_CHUNK" --gn-stride "$GN_STRIDE"
                --name "reign-on-${short}" --cache-tag "e1_${short}" "${common[@]}")
    run_e1 "reign_${short}_uncached" "${base[@]}" --gn-cache-mode uncached
    run_e1 "reign_${short}_build"    "${base[@]}" --gn-cache-mode build
    run_e1 "reign_${short}_cached"   "${base[@]}" --gn-cache-mode cached
  done

  "$PYTHON" scripts/_print_e1_table.py --results-dir "$d" \
    --markdown "${d}/e1_table.md" --latex "${d}/e1_table.tex"
}

# ===========================================================================
# e2-significance — paper Table 5 (paired significance tests on DAPFAM)
# ===========================================================================
# CPU only, ~1 min. Consumes the per-query dumps written by main-dapfam, so run
# that stage first. All five baselines go in ONE invocation so the
# Holm-Bonferroni correction spans the whole comparison family; splitting this
# into five separate runs would under-correct and inflate significance.
stage_e2_significance() {
  echo "=== [e2-significance] paired bootstrap + randomization — paper Table 5 ==="
  local e2="${RESULTS_DIR}/e2"
  local sys_json="${e2}/reign_gte-large_s${GN_STRIDE}.json"

  local missing=0
  for f in "$sys_json" "${e2}/dense_jina-v3.json" "${e2}/dense_stella-1.5b.json" \
           "${e2}/dense_gte-large-chunked.json" "${e2}/sparse_bm25.json" \
           "${e2}/sparse_tfidf.json"; do
    [ -f "$f" ] || { echo "  missing per-query dump: $f"; missing=1; }
  done
  if [ "$missing" -ne 0 ]; then
    echo "  → run 'bash scripts/reproduce.sh main-dapfam' first; it writes these files."
    return 1
  fi

  "$PYTHON" scripts/paired_bootstrap.py \
    --system "$sys_json" \
    --baseline "${e2}/dense_jina-v3.json" \
    --baseline "${e2}/dense_stella-1.5b.json" \
    --baseline "${e2}/dense_gte-large-chunked.json" \
    --baseline "${e2}/sparse_bm25.json" \
    --baseline "${e2}/sparse_tfidf.json" \
    --metric nDCG@100 --n-boot 10000 --n-perm 10000 --seed 42 --alpha 0.05 \
    --output "${e2}/significance.json" \
    --markdown "${e2}/significance.md"
  echo "  Reference output to diff against: results/reference/e2/significance_dapfam_5way.md"
}

# ===========================================================================
# e4-pe-ablation — paper App. E (positional-encoding ablation)
# ===========================================================================
# GPU required. Trains 3 arms then evaluates each on 3 benchmarks: ~20-30 h
# sequential (the arms are independent and can be trained in parallel on a card
# with room for ~2.5 GB each).
stage_e4_pe_ablation() {
  echo "=== [e4-pe-ablation] chunk-position signal — paper App. E ==="
  local d="${RESULTS_DIR}/e4"
  mkdir -p "$d"
  local gn="thenlper/gte-small"
  # One identical protocol for all three arms; the ONLY thing that varies is
  # --position-embedding-type, which is what makes the ablation interpretable.
  local common=(--dataset "$GOODWIKI" --train-split train --eval-split val
                --model-config base-l3
                --gn-model "$gn" --gn-chunk-size 512 --gn-stride 512
                --batch-size 48 --eval-batch-size 48
                --max-epochs 20 --lr 1e-5 --weight-decay 1e-4
                --negative-batch-size-multiplier 47 --weight-partial 0.5
                --loss-function infonce --temperature 0.07
                --enable-cache --precision 16-mixed
                --metric-to-monitor ndcg@10 --check-val-every-n-epoch 2
                --seed 42 --device cuda)

  for arm in none absolute sinusoidal; do
    local out="reign-base-l3_gn-gte-small_pe-${arm}"
    if have_ckpt "$out"; then
      echo "  [have] trained arm $arm"
    else
      echo "  - training arm: $arm"
      "$PYTHON" -m reign.train "${common[@]}" \
        --position-embedding-type "$arm" --output-dir "$out" || continue
    fi

    local gw="${d}/goodwiki_pe-${arm}.json"
    if [ ! -f "$gw" ]; then
      echo "  - eval $arm: GoodWiki-Long"
      "$PYTHON" scripts/evaluate_reign.py \
        --checkpoint "${MODELS_DIR}/${out}/best" --gn-model "$gn" \
        --gn-chunk-size 512 --gn-stride 512 --dataset "$GOODWIKI" --split test \
        --top_k 10 --batch_size 8 --name "reign-pe-${arm}" --output_path "$gw" || true
    fi

    local dp="${d}/dapfam_pe-${arm}.json"
    if [ ! -f "$dp" ] && [ -d "$DAPFAM_DATA" ]; then
      echo "  - eval $arm: DAPFAM (zero-shot)"
      "$PYTHON" scripts/evaluate_reign.py \
        --checkpoint "${MODELS_DIR}/${out}/best" --gn-model "$gn" \
        --gn-chunk-size 512 --gn-stride 512 --dataset "$DAPFAM_DATA" --split test \
        --top_k 100 --batch_size 4 --gn-batch-size 8 \
        --corpus-embed-cache "${RESULTS_DIR}/.corpus_emb_cache" \
        --name "reign-pe-${arm}-dapfam" --output_path "$dp" || true
    fi

    if [ ! -d "${d}/loco_pe-${arm}" ]; then
      echo "  - eval $arm: LoCo (12 subtasks)"
      "$PYTHON" scripts/evaluate_loco.py \
        --reign-checkpoint "${MODELS_DIR}/${out}/best" --gn-model "$gn" \
        --gn-chunk-size 512 --gn-stride 512 --subtask all \
        --output-dir "${d}/loco_pe-${arm}" --tag "pe-${arm}" || true
    fi
  done

  "$PYTHON" scripts/_print_e4_table.py --results-dir "$d" --markdown "${d}/e4_table.md"
}

# ===========================================================================
# e5-objective-ablation — paper App. I (training-objective ablation)
# ===========================================================================
# GPU required. Nine arms trained under one protocol, each evaluated on four
# benchmarks: ~60-90 h sequential. Arms are independent and parallelise well.
stage_e5_objective_ablation() {
  echo "=== [e5-objective-ablation] loss / temperature / distractors — paper App. I ==="
  local d="${RESULTS_DIR}/e5"
  mkdir -p "$d"
  local gn="thenlper/gte-small"
  local published="${MODELS_DIR}/reign-base-l3_gn-gte-small_s512_val-selected/best"
  # Shared protocol. What varies per arm is in the table below: the loss, the
  # InfoNCE temperature, the distractor weight, the batch size (which sets the
  # in-batch negative count), warm vs cold start, and one epoch-budget extension.
  local common=(--dataset "$GOODWIKI" --train-split train --eval-split val
                --model-config base-l3 --gn-model "$gn"
                --gn-chunk-size 512 --gn-stride 512
                --max-epochs 20 --lr 1e-5 --weight-decay 1e-4
                --enable-cache --precision 16-mixed
                --metric-to-monitor ndcg@10 --check-val-every-n-epoch 2
                --seed 42 --device cuda)

  # arm-name | extra training args   (arm names must match _print_e5_table.ARMS)
  local arms=(
    "cosine-bs18|--loss-function cosine --weight-partial 0.5 --batch-size 18 --eval-batch-size 18 --negative-batch-size-multiplier 17"
    "cosine-bs48|--loss-function cosine --weight-partial 0.5 --batch-size 48 --eval-batch-size 48 --negative-batch-size-multiplier 47"
    "infonce-pw05-bs18|--loss-function infonce --temperature 0.07 --weight-partial 0.5 --batch-size 18 --eval-batch-size 18 --negative-batch-size-multiplier 17"
    "infonce-pw05-bs48|--loss-function infonce --temperature 0.07 --weight-partial 0.5 --batch-size 48 --eval-batch-size 48 --negative-batch-size-multiplier 47"
    "infonce-pw00-bs48|--loss-function infonce --temperature 0.07 --weight-partial 0.0 --batch-size 48 --eval-batch-size 48 --negative-batch-size-multiplier 47"
    "infonce-pw05-warm|--loss-function infonce --temperature 0.07 --weight-partial 0.5 --batch-size 48 --eval-batch-size 48 --negative-batch-size-multiplier 47 --from-checkpoint ${published}"
    "infonce-pw05-t01-bs48|--loss-function infonce --temperature 0.1 --weight-partial 0.5 --batch-size 48 --eval-batch-size 48 --negative-batch-size-multiplier 47"
    "infonce-pw00-t01-bs48|--loss-function infonce --temperature 0.1 --weight-partial 0.0 --batch-size 48 --eval-batch-size 48 --negative-batch-size-multiplier 47"
    "infonce-pw00-bs48-e50|--loss-function infonce --temperature 0.07 --weight-partial 0.0 --batch-size 48 --eval-batch-size 48 --negative-batch-size-multiplier 47 --max-epochs 50"
  )

  for spec in "${arms[@]}"; do
    IFS='|' read -r name extra <<< "$spec"
    local out="reign-base-l3_gte-small_e5-${name}"
    if have_ckpt "$out"; then
      echo "  [have] trained arm $name"
    else
      echo "  - training arm: $name"
      # shellcheck disable=SC2086
      "$PYTHON" -m reign.train "${common[@]}" $extra --output-dir "$out" || continue
    fi
    local ckpt="${MODELS_DIR}/${out}/best"

    [ -f "${d}/goodwiki_${name}.json" ] || {
      echo "  - eval $name: GoodWiki-Long"
      "$PYTHON" scripts/evaluate_reign.py --checkpoint "$ckpt" --gn-model "$gn" \
        --gn-chunk-size 512 --gn-stride 512 --dataset "$GOODWIKI" --split test \
        --top_k 10 --batch_size 8 --name "e5-${name}" \
        --output_path "${d}/goodwiki_${name}.json" || true; }

    if [ ! -f "${d}/dapfam_${name}.json" ] && [ -d "$DAPFAM_DATA" ]; then
      echo "  - eval $name: DAPFAM (zero-shot)"
      "$PYTHON" scripts/evaluate_reign.py --checkpoint "$ckpt" --gn-model "$gn" \
        --gn-chunk-size 512 --gn-stride 512 --dataset "$DAPFAM_DATA" --split test \
        --top_k 100 --batch_size 4 --gn-batch-size 8 \
        --corpus-embed-cache "${RESULTS_DIR}/.corpus_emb_cache" \
        --name "e5-${name}-dapfam" --output_path "${d}/dapfam_${name}.json" || true
    fi

    [ -d "${d}/loco_${name}" ] || {
      echo "  - eval $name: LoCo (12 subtasks)"
      "$PYTHON" scripts/evaluate_loco.py --reign-checkpoint "$ckpt" --gn-model "$gn" \
        --gn-chunk-size 512 --gn-stride 512 --subtask all \
        --output-dir "${d}/loco_${name}" --tag "e5-${name}" || true; }

    [ -d "${d}/mteb_${name}" ] || {
      echo "  - eval $name: MTEB (ArguAna, FiQA2018)"
      "$PYTHON" scripts/evaluate_mteb.py --model-path "$ckpt" --gn-model "$gn" \
        --task-names ArguAna FiQA2018 --eval-splits test --batch-size 8 \
        --output-dir "${d}/mteb_${name}" || true; }
  done

  "$PYTHON" scripts/_print_e5_table.py --results-dir "$d" --markdown "${d}/e5_table.md"
}

# ===========================================================================
# mteb — paper App. B (short-context MTEB sanity check)
# ===========================================================================
# GPU required. ~2-4 h for the retrieval subset per checkpoint. REIGN is a
# long-document model, so this stage checks that it has not regressed
# catastrophically on short-context tasks rather than chasing leaderboard rank.
stage_mteb() {
  echo "=== [mteb] short-context sanity check — paper App. B, Table 6 ==="
  # Table 6 is exactly two retrieval tasks and two systems: the paper-default
  # base-l3 encoder on GTE-small, and the same GN alone truncated at its native
  # 512-token window. Both rows are produced here.
  local tasks="${MTEB_TASKS:-ArguAna FiQA2018}"
  local ckpt="${MTEB_CKPT:-reign-base-l3_gn-gte-small_s384_val-selected}"
  local gn="${MTEB_GN:-thenlper/gte-small}"

  local out="${RESULTS_DIR}/mteb/reign-base-l3_gte-small"
  if [ -d "$out" ]; then
    echo "  [have] $out"
  elif have_ckpt "$ckpt"; then
    echo "  - REIGN row: $tasks"
    # shellcheck disable=SC2086
    "$PYTHON" scripts/evaluate_mteb.py \
      --model-path "${MODELS_DIR}/${ckpt}/best" --gn-model "$gn" \
      --task-names $tasks --eval-splits test --batch-size 8 \
      --max-seq-length 512 \
      --output-dir "$out" || true
  else
    skip_ckpt "base-l3 on gte-small" "$ckpt"
  fi

  # Baseline row: the bare Guidance Network, no REIGN encoder. This is the stock
  # `mteb` runner over the Hub model (installed with `pip install -e ".[eval]"`),
  # not scripts/evaluate_mteb.py, which always builds a REIGN encoder.
  local bout="${RESULTS_DIR}/mteb/baseline_gte-small"
  if [ -d "$bout" ]; then
    echo "  [have] $bout"
  elif command -v mteb >/dev/null 2>&1; then
    echo "  - truncated-GN baseline row: $tasks"
    # shellcheck disable=SC2086
    mteb run -m "$gn" -t $tasks --eval_splits test --batch_size 8 \
      --output_folder "$bout" || true
  else
    echo "  [skip] baseline row — the 'mteb' CLI is not on PATH"
  fi
}

stage_all() {
  stage_main_goodwiki
  stage_main_loco
  stage_main_dapfam
  stage_e1_efficiency
  stage_e2_significance
  stage_e4_pe_ablation
  stage_e5_objective_ablation
  stage_mteb
}

stage_help() {
  cat <<'USAGE'
usage: bash scripts/reproduce.sh <stage>

Stages (paper artifact each one produces):
  main-goodwiki          Tables 2, 3, 4   in-distribution GoodWiki-Long        [GPU, ~4-6 h]
  main-loco              Tables 3, 4      zero-shot LoCo, 12 subtasks          [GPU, ~1-2 h/encoder]
  main-dapfam            Table 4 / Sec 5.3  DAPFAM patent prior art            [GPU, ~8-12 h]
  e1-efficiency          App. G, Table 11  measured latency + peak memory      [GPU, idle card, ~1-2 h]
  e2-significance        Table 5          paired bootstrap + randomization     [CPU, ~1 min]
  e4-pe-ablation         App. E           positional-encoding ablation         [GPU, ~20-30 h]
  e5-objective-ablation  App. I           training-objective ablation          [GPU, ~60-90 h]
  mteb                   App. B, Table 6  short-context MTEB sanity check      [GPU, ~1 h]
  all                    every stage above, in order

Environment: MODELS_DIR (default ./models), RESULTS_DIR (default ./results),
PYTHON, DAPFAM_DATA, GN_STRIDE.

Row-set overrides — REIGN_ROWS_OVERRIDE, DAPFAM_ZS_ROWS_OVERRIDE, DAPFAM_SPLITS,
DENSE_NATIVE, DENSE_CHUNKED, DENSE_DAPFAM, MTEB_TASKS, MTEB_CKPT, MTEB_GN,
FT_RUNS_OVERRIDE — reach the checkpoints and columns the default rows leave out
(paper Tables 4, 7, 8 and the DAPFAM fine-tuned family). docs/REPRODUCING.md
lists the exact override values table by table.

Checkpoints come from the Hugging Face Hub; the weights release is pending.
Until then place them under MODELS_DIR — see docs/REPRODUCING.md for the layout.
Curated reference outputs to diff against live in results/reference/, one
subdirectory per stage; docs/REPRODUCING.md enumerates them file by file.
USAGE
}

case "$STAGE" in
  main-goodwiki)          stage_main_goodwiki ;;
  main-loco)              stage_main_loco ;;
  main-dapfam)            stage_main_dapfam ;;
  e1-efficiency)          stage_e1_efficiency ;;
  e2-significance)        stage_e2_significance ;;
  e4-pe-ablation)         stage_e4_pe_ablation ;;
  e5-objective-ablation)  stage_e5_objective_ablation ;;
  mteb)                   stage_mteb ;;
  all)                    stage_all ;;
  help|--help|-h)         stage_help ;;
  *) echo "unknown stage: $STAGE"; echo; stage_help; exit 2 ;;
esac
