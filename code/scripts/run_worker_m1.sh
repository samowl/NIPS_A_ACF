#!/bin/bash
# M1 worker: PASS-A optional + PASS-B multiclass head training.
#
# Usage (inside a tmux Singularity session):
#   bash scripts/run_worker_m1.sh WORKER_ID N_WORKERS [GPU_ID]
#
# Slices jobs_m1.txt into N_WORKERS slices by line index modulo and runs
# the slice assigned to WORKER_ID. Two verbs are supported per line:
#
#   EXTRACT {cache_task} {fm}       -> scripts/extract_features.py (PASS-A)
#   HEAD    brats_multiclass {fm} {seed}
#                                    -> scripts/train_head_multiclass.py
#
# By default jobs_m1.txt contains only HEAD lines (the brats_wt feature
# cache is reused from jobs_v2.txt). To force EXTRACT, regenerate the
# manifest with ``python scripts/make_jobs_m1.py --with-extract``.
#
# HEAD lines block (poll-and-wait, capped at 30 minutes total) until the
# matching ``results/feature_cache/brats_wt/{fm}/manifest.json`` exists.
# EXTRACT and HEAD are both skip-if-output-exists so the worker is fully
# resume-safe and idempotent.
#
# GPU_ID (optional) selects which physical GPU to bind via
# CUDA_VISIBLE_DEVICES. Default uses the Singularity-visible device 0.
#
# Logs append to results/logs/worker_m1_${WORKER_ID}.log; final status
# line lands in results/logs/worker_m1_${WORKER_ID}.status.

set -uo pipefail

WORKER_ID="${1:?'WORKER_ID required (integer 0..N_WORKERS-1)'}"
N_WORKERS="${2:?'N_WORKERS required'}"
GPU_ID="${3:-}"

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

JOBS_FILE="${REPO_ROOT}/jobs_m1.txt"
LOG_DIR="${REPO_ROOT}/results/logs"
STATUS_FILE="${LOG_DIR}/worker_m1_${WORKER_ID}.status"
LOG_FILE="${LOG_DIR}/worker_m1_${WORKER_ID}.log"
mkdir -p "${LOG_DIR}"

PYTHON="${FMPOOL_PYTHON:-python}"
DATA_ROOT="${FMPOOL_DATA_ROOT:-data/raw}"
CACHE_ROOT="${FMPOOL_CACHE_ROOT:-${REPO_ROOT}/results/feature_cache}"

# M1 PASS-B output and checkpoint roots are kept distinct from the
# binary-WT pipeline so both matrices can coexist in the same repo.
M1_OUT_ROOT="${REPO_ROOT}/results/per_case_dice_m1"
M1_CKPT_ROOT="${REPO_ROOT}/results/checkpoints_m1"

export PYTHONPATH="${REPO_ROOT}/src"
export FMPOOL_DATA_ROOT="${DATA_ROOT}"
if [[ -n "${GPU_ID}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

log() {
    printf '[%s] worker=%s %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${WORKER_ID}" "$*" \
        | tee -a "${LOG_FILE}"
}

log "starting M1 WORKER_ID=${WORKER_ID} N_WORKERS=${N_WORKERS} GPU=${CUDA_VISIBLE_DEVICES:-default} PYTHON=${PYTHON}"
log "DATA_ROOT=${DATA_ROOT} CACHE_ROOT=${CACHE_ROOT}"

if [[ ! -f "${JOBS_FILE}" ]]; then
    log "FATAL jobs file missing: ${JOBS_FILE} (run 'python scripts/make_jobs_m1.py' first)"
    echo "FAIL missing_jobs_file" > "${STATUS_FILE}"
    exit 2
fi

ran=0
skipped=0
failed=0

# --- Phase 1: EXTRACT jobs assigned to this worker --------------------------
log "=== Phase 1: EXTRACT slice ==="
extract_idx=-1
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    set -- $line
    verb="$1"
    if [[ "${verb}" != "EXTRACT" ]]; then
        continue
    fi
    extract_idx=$((extract_idx + 1))
    slice_mod=$((extract_idx % N_WORKERS))
    if [[ "${slice_mod}" != "${WORKER_ID}" ]]; then
        continue
    fi
    cache_task="$2"
    fm="$3"
    cache_dir="${CACHE_ROOT}/${cache_task}/${fm}"
    manifest_path="${cache_dir}/manifest.json"
    if [[ -f "${manifest_path}" ]]; then
        log "SKIP EXTRACT idx=${extract_idx} cache_task=${cache_task} fm=${fm} (cache present)"
        skipped=$((skipped + 1))
        continue
    fi
    log "RUN EXTRACT idx=${extract_idx} cache_task=${cache_task} fm=${fm}"
    start_ts=$(date +%s)
    "${PYTHON}" "${REPO_ROOT}/scripts/extract_features.py" \
        --task "${cache_task}" --fm "${fm}" \
        --data-root "${DATA_ROOT}" \
        --cache-root "${CACHE_ROOT}" \
        --batch-size 32 --num-workers 4 \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 && -f "${manifest_path}" ]]; then
        log "OK EXTRACT idx=${extract_idx} elapsed=${elapsed}s"
        ran=$((ran + 1))
    else
        log "FAIL EXTRACT idx=${extract_idx} rc=${rc} elapsed=${elapsed}s"
        failed=$((failed + 1))
    fi
done < "${JOBS_FILE}"

# --- Phase 2: HEAD jobs assigned to this worker -----------------------------
# HEAD lines have the form: HEAD brats_multiclass {fm} {seed}
# The feature cache used is brats_wt/{fm} (image features are
# label-independent); the trainer verifies the manifest task matches one
# of {brats_wt, brats_multiclass}.
log "=== Phase 2: HEAD slice ==="
head_idx=-1
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    set -- $line
    verb="$1"
    if [[ "${verb}" != "HEAD" ]]; then
        continue
    fi
    head_idx=$((head_idx + 1))
    slice_mod=$((head_idx % N_WORKERS))
    if [[ "${slice_mod}" != "${WORKER_ID}" ]]; then
        continue
    fi
    head_task="$2"
    fm="$3"
    seed="$4"
    if [[ "${head_task}" != "brats_multiclass" ]]; then
        log "SKIP HEAD idx=${head_idx} task=${head_task} (not brats_multiclass)"
        skipped=$((skipped + 1))
        continue
    fi
    out_dir="${M1_OUT_ROOT}/${head_task}/${fm}"
    out_json="${out_dir}/seed_${seed}.json"
    ckpt_dir="${M1_CKPT_ROOT}/${head_task}/${fm}"
    # Cache lookup is keyed by brats_wt (the image-only feature cache).
    cache_task="brats_wt"
    cache_dir="${CACHE_ROOT}/${cache_task}/${fm}"
    manifest_path="${cache_dir}/manifest.json"
    if [[ -f "${out_json}" ]]; then
        log "SKIP HEAD idx=${head_idx} task=${head_task} fm=${fm} seed=${seed} (json present)"
        skipped=$((skipped + 1))
        continue
    fi
    # Wait up to 30 minutes for the brats_wt extract job to finish.
    waited=0
    while [[ ! -f "${manifest_path}" && ${waited} -lt 180 ]]; do
        sleep 10
        waited=$((waited + 1))
    done
    if [[ ! -f "${manifest_path}" ]]; then
        log "FAIL HEAD idx=${head_idx} task=${head_task} fm=${fm} seed=${seed} cache_missing (${cache_dir})"
        failed=$((failed + 1))
        continue
    fi
    log "RUN HEAD idx=${head_idx} task=${head_task} fm=${fm} seed=${seed}"
    start_ts=$(date +%s)
    "${PYTHON}" "${REPO_ROOT}/scripts/train_head_multiclass.py" \
        --task "${head_task}" --fm "${fm}" --seed "${seed}" \
        --cache-task "${cache_task}" \
        --cache-root "${CACHE_ROOT}" \
        --out "${out_dir}" \
        --checkpoint-dir "${ckpt_dir}" \
        --epochs 30 --batch-size 16 \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 && -f "${out_json}" ]]; then
        md=$(python3 -c "import json; print(json.load(open('${out_json}')).get('mean_dice', 'NA'))" 2>/dev/null || echo NA)
        log "OK HEAD idx=${head_idx} elapsed=${elapsed}s mean_dice=${md}"
        ran=$((ran + 1))
    else
        log "FAIL HEAD idx=${head_idx} rc=${rc} elapsed=${elapsed}s"
        failed=$((failed + 1))
    fi
done < "${JOBS_FILE}"

log "M1 worker ${WORKER_ID} complete: ran=${ran} skipped=${skipped} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} failed=${failed}" > "${STATUS_FILE}"
