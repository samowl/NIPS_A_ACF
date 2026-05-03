#!/bin/bash
# M3 worker (head-design sweep). Reads jobs_m3.txt, runs slice modulo
# WORKER_ID. Each line is:
#
#   M3 {head_design} {task} {fm} {seed}
#
# Requires the PASS-A feature cache to be present at
#   results/feature_cache/{task}/{fm}/manifest.json
# This script never invokes extract_features.py.
#
# Usage (inside a tmux Singularity session):
#   bash scripts/run_worker_m3.sh WORKER_ID N_WORKERS [GPU_ID]
#
# Resume safety: skips when the per-case JSON already exists. Logs are
# appended to results/logs/worker_m3_${WORKER_ID}.log; final status to
# results/logs/worker_m3_${WORKER_ID}.status.

set -uo pipefail

WORKER_ID="${1:?'WORKER_ID required (integer 0..N_WORKERS-1)'}"
N_WORKERS="${2:?'N_WORKERS required'}"
GPU_ID="${3:-}"

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

JOBS_FILE="${REPO_ROOT}/jobs_m3.txt"
LOG_DIR="${REPO_ROOT}/results/logs"
STATUS_FILE="${LOG_DIR}/worker_m3_${WORKER_ID}.status"
LOG_FILE="${LOG_DIR}/worker_m3_${WORKER_ID}.log"
mkdir -p "${LOG_DIR}"

PYTHON="${FMPOOL_PYTHON:-python}"
DATA_ROOT="${FMPOOL_DATA_ROOT:-data/raw}"
CACHE_ROOT="${FMPOOL_CACHE_ROOT:-${REPO_ROOT}/results/feature_cache}"

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

log "starting M3 WORKER_ID=${WORKER_ID} N_WORKERS=${N_WORKERS} GPU=${CUDA_VISIBLE_DEVICES:-default} PYTHON=${PYTHON}"
log "DATA_ROOT=${DATA_ROOT} CACHE_ROOT=${CACHE_ROOT}"

if [[ ! -f "${JOBS_FILE}" ]]; then
    log "FATAL jobs file missing: ${JOBS_FILE}"
    echo "FAIL missing_jobs_file" > "${STATUS_FILE}"
    exit 2
fi

ran=0
skipped=0
failed=0

job_idx=-1
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    set -- $line
    verb="$1"
    if [[ "${verb}" != "M3" ]]; then
        continue
    fi
    job_idx=$((job_idx + 1))
    slice_mod=$((job_idx % N_WORKERS))
    if [[ "${slice_mod}" != "${WORKER_ID}" ]]; then
        continue
    fi

    head_design="$2"
    task="$3"
    fm="$4"
    seed="$5"

    out_dir="${REPO_ROOT}/results/per_case_dice_m3/${head_design}/${task}/${fm}"
    out_json="${out_dir}/seed_${seed}.json"
    cache_dir="${CACHE_ROOT}/${task}/${fm}"
    manifest_path="${cache_dir}/manifest.json"

    if [[ -f "${out_json}" ]]; then
        log "SKIP M3 idx=${job_idx} head=${head_design} task=${task} fm=${fm} seed=${seed} (json present)"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ ! -f "${manifest_path}" ]]; then
        log "FAIL M3 idx=${job_idx} head=${head_design} task=${task} fm=${fm} seed=${seed} cache_missing"
        failed=$((failed + 1))
        continue
    fi

    log "RUN M3 idx=${job_idx} head=${head_design} task=${task} fm=${fm} seed=${seed}"
    start_ts=$(date +%s)
    "${PYTHON}" "${REPO_ROOT}/scripts/train_head_m3.py" \
        --task "${task}" --fm "${fm}" --seed "${seed}" \
        --head-design "${head_design}" \
        --cache-root "${CACHE_ROOT}" \
        --out "${out_dir}" \
        --epochs 30 --batch-size 16 \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 && -f "${out_json}" ]]; then
        md=$(python3 -c "import json; print(json.load(open('${out_json}')).get('mean_dice', 'NA'))" 2>/dev/null || echo NA)
        log "OK M3 idx=${job_idx} elapsed=${elapsed}s mean_dice=${md}"
        ran=$((ran + 1))
    else
        log "FAIL M3 idx=${job_idx} rc=${rc} elapsed=${elapsed}s"
        failed=$((failed + 1))
    fi
done < "${JOBS_FILE}"

log "M3 worker ${WORKER_ID} complete: ran=${ran} skipped=${skipped} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} failed=${failed}" > "${STATUS_FILE}"
