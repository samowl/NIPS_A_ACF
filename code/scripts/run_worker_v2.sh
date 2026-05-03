#!/bin/bash
# PASS-A + PASS-B worker for the two-pass job queue (jobs_v2.txt).
#
# Usage (inside a tmux Singularity session):
#   bash scripts/run_worker_v2.sh WORKER_ID N_WORKERS [GPU_ID]
#
# Slices jobs_v2.txt into N_WORKERS slices by line index modulo and runs
# the slice assigned to WORKER_ID. Two verbs are supported per line:
#
#   EXTRACT {task} {fm}            -> scripts/extract_features.py (PASS-A)
#   HEAD    {task} {fm} {seed}     -> scripts/train_head_cached.py (PASS-B)
#
# HEAD lines block (poll-and-wait, capped) until the matching EXTRACT has
# written ``results/feature_cache/{task}/{fm}/manifest.json``. EXTRACT and
# HEAD are both skipped if the expected output already exists, so the
# worker is fully resume-safe and idempotent.
#
# GPU_ID (optional) selects which physical GPU to bind via
# CUDA_VISIBLE_DEVICES. Default uses the Singularity-visible device 0.
#
# Logs append to results/logs/worker_${WORKER_ID}.log; final status line
# lands in results/logs/worker_${WORKER_ID}.status.

set -uo pipefail

WORKER_ID="${1:?'WORKER_ID required (integer 0..N_WORKERS-1)'}"
N_WORKERS="${2:?'N_WORKERS required'}"
GPU_ID="${3:-}"

if ! [[ "${WORKER_ID}" =~ ^[0-9]+$ && "${N_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL WORKER_ID and N_WORKERS must be non-negative / positive integers" >&2
    exit 2
fi
if [[ "${WORKER_ID}" -ge "${N_WORKERS}" ]]; then
    echo "FATAL WORKER_ID=${WORKER_ID} must be < N_WORKERS=${N_WORKERS}" >&2
    exit 2
fi

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

JOBS_FILE="${FMPOOL_JOBS_FILE:-${REPO_ROOT}/jobs_v2.txt}"
LOG_DIR="${REPO_ROOT}/results/logs"
WORKER_TAG="${FMPOOL_WORKER_TAG:-}"
if [[ -n "${WORKER_TAG}" ]]; then
    STATUS_FILE="${LOG_DIR}/worker_${WORKER_TAG}_${WORKER_ID}.status"
    LOG_FILE="${LOG_DIR}/worker_${WORKER_TAG}_${WORKER_ID}.log"
else
    STATUS_FILE="${LOG_DIR}/worker_${WORKER_ID}.status"
    LOG_FILE="${LOG_DIR}/worker_${WORKER_ID}.log"
fi
mkdir -p "${LOG_DIR}"

PYTHON="${FMPOOL_PYTHON:-python}"
DATA_ROOT="${FMPOOL_DATA_ROOT:-data/raw}"
CACHE_ROOT="${FMPOOL_CACHE_ROOT:-${REPO_ROOT}/results/feature_cache}"
PER_CASE_ROOT="${FMPOOL_PER_CASE_ROOT:-${REPO_ROOT}/results/per_case_dice}"
CHECKPOINT_ROOT="${FMPOOL_CHECKPOINT_ROOT:-${REPO_ROOT}/results/checkpoints}"

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

log "starting WORKER_ID=${WORKER_ID} N_WORKERS=${N_WORKERS} GPU=${CUDA_VISIBLE_DEVICES:-default} PYTHON=${PYTHON}"
log "JOBS_FILE=${JOBS_FILE}"
log "DATA_ROOT=${DATA_ROOT} CACHE_ROOT=${CACHE_ROOT} PER_CASE_ROOT=${PER_CASE_ROOT} CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"

if [[ ! -f "${JOBS_FILE}" ]]; then
    log "FATAL jobs file missing: ${JOBS_FILE}"
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
    task="$2"
    fm="$3"
    cache_dir="${CACHE_ROOT}/${task}/${fm}"
    manifest_path="${cache_dir}/manifest.json"
    train_path="${cache_dir}/train.pt"
    test_path="${cache_dir}/test.pt"
    if [[ -f "${manifest_path}" && -f "${train_path}" && -f "${test_path}" ]]; then
        log "SKIP EXTRACT idx=${extract_idx} task=${task} fm=${fm} (complete cache present)"
        skipped=$((skipped + 1))
        continue
    fi
    log "RUN EXTRACT idx=${extract_idx} task=${task} fm=${fm}"
    start_ts=$(date +%s)
    "${PYTHON}" "${REPO_ROOT}/scripts/extract_features.py" \
        --task "${task}" --fm "${fm}" \
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
    task="$2"
    fm="$3"
    seed="$4"
    out_dir="${PER_CASE_ROOT}/${task}/${fm}"
    out_json="${out_dir}/seed_${seed}.json"
    cache_dir="${CACHE_ROOT}/${task}/${fm}"
    manifest_path="${cache_dir}/manifest.json"
    train_path="${cache_dir}/train.pt"
    test_path="${cache_dir}/test.pt"
    if [[ -f "${out_json}" ]]; then
        log "SKIP HEAD idx=${head_idx} task=${task} fm=${fm} seed=${seed} (json present)"
        skipped=$((skipped + 1))
        continue
    fi
    waited=0
    while [[ (! -f "${manifest_path}" || ! -f "${train_path}" || ! -f "${test_path}") && ${waited} -lt 180 ]]; do
        sleep 10
        waited=$((waited + 1))
    done
    if [[ ! -f "${manifest_path}" || ! -f "${train_path}" || ! -f "${test_path}" ]]; then
        log "FAIL HEAD idx=${head_idx} task=${task} fm=${fm} seed=${seed} cache_missing_or_incomplete"
        failed=$((failed + 1))
        continue
    fi
    log "RUN HEAD idx=${head_idx} task=${task} fm=${fm} seed=${seed}"
    start_ts=$(date +%s)
    "${PYTHON}" "${REPO_ROOT}/scripts/train_head_cached.py" \
        --task "${task}" --fm "${fm}" --seed "${seed}" \
        --cache-root "${CACHE_ROOT}" \
        --out "${out_dir}" \
        --checkpoint-dir "${CHECKPOINT_ROOT}/${task}/${fm}/seed_${seed}" \
        --epochs 30 --batch-size 16 \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 && -f "${out_json}" ]]; then
        md=$("${PYTHON}" -c "import json; print(json.load(open('${out_json}')).get('mean_dice', 'NA'))" 2>/dev/null || echo NA)
        log "OK HEAD idx=${head_idx} elapsed=${elapsed}s mean_dice=${md}"
        ran=$((ran + 1))
    else
        log "FAIL HEAD idx=${head_idx} rc=${rc} elapsed=${elapsed}s"
        failed=$((failed + 1))
    fi
done < "${JOBS_FILE}"

log "worker ${WORKER_ID} complete: ran=${ran} skipped=${skipped} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} failed=${failed}" > "${STATUS_FILE}"
if [[ "${failed}" -gt 0 ]]; then
    exit 3
fi
