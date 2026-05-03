#!/bin/bash
# M19 worker: train-set fraction PASS-B head trainer over jobs_m19.txt.
#
# Usage (inside a tmux Singularity session):
#   bash scripts/run_worker_m19.sh WORKER_ID N_WORKERS [GPU_ID]
#
# Slices jobs_m19.txt by line index modulo N_WORKERS. The single supported
# verb is:
#
#   HEAD_FRAC {task} {fm} {seed} {train_frac}
#
# Each line consumes the existing PASS-A feature cache at
# ``results/feature_cache/{task}/{fm}/manifest.json``. Lines whose JSON
# output already exists are skipped (resume-safe).
#
# Logs append to ``results/logs/worker_m19_${WORKER_ID}.log``.

set -uo pipefail

WORKER_ID="${1:?'WORKER_ID required (integer 0..N_WORKERS-1)'}"
N_WORKERS="${2:?'N_WORKERS required'}"
GPU_ID="${3:-}"

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

JOBS_FILE="${REPO_ROOT}/jobs_m19.txt"
LOG_DIR="${REPO_ROOT}/results/logs"
STATUS_FILE="${LOG_DIR}/worker_m19_${WORKER_ID}.status"
LOG_FILE="${LOG_DIR}/worker_m19_${WORKER_ID}.log"
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
    printf '[%s] worker_m19=%s %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${WORKER_ID}" "$*" \
        | tee -a "${LOG_FILE}"
}

log "starting WORKER_ID=${WORKER_ID} N_WORKERS=${N_WORKERS} GPU=${CUDA_VISIBLE_DEVICES:-default}"
log "PYTHON=${PYTHON} DATA_ROOT=${DATA_ROOT} CACHE_ROOT=${CACHE_ROOT}"

if [[ ! -f "${JOBS_FILE}" ]]; then
    log "FATAL jobs file missing: ${JOBS_FILE}"
    echo "FAIL missing_jobs_file" > "${STATUS_FILE}"
    exit 2
fi

ran=0
skipped=0
failed=0
idx=-1
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    set -- $line
    verb="$1"
    if [[ "${verb}" != "HEAD_FRAC" ]]; then
        continue
    fi
    idx=$((idx + 1))
    slice_mod=$((idx % N_WORKERS))
    if [[ "${slice_mod}" != "${WORKER_ID}" ]]; then
        continue
    fi
    task="$2"
    fm="$3"
    seed="$4"
    frac="$5"

    out_dir="${REPO_ROOT}/results/per_case_dice_m19/frac${frac}/${task}/${fm}"
    out_json="${out_dir}/seed_${seed}.json"
    cache_dir="${CACHE_ROOT}/${task}/${fm}"
    manifest_path="${cache_dir}/manifest.json"

    if [[ -f "${out_json}" ]]; then
        log "SKIP idx=${idx} task=${task} fm=${fm} seed=${seed} frac=${frac} (json present)"
        skipped=$((skipped + 1))
        continue
    fi
    if [[ ! -f "${manifest_path}" ]]; then
        log "FAIL idx=${idx} task=${task} fm=${fm} seed=${seed} frac=${frac} cache_missing"
        failed=$((failed + 1))
        continue
    fi

    log "RUN idx=${idx} task=${task} fm=${fm} seed=${seed} frac=${frac}"
    start_ts=$(date +%s)
    "${PYTHON}" "${REPO_ROOT}/scripts/train_head_subset.py" \
        --task "${task}" --fm "${fm}" --seed "${seed}" \
        --train-frac "${frac}" \
        --cache-root "${CACHE_ROOT}" \
        --out "${out_dir}" \
        --epochs 30 --batch-size 16 \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 && -f "${out_json}" ]]; then
        md=$(python3 -c "import json;d=json.load(open('${out_json}'));print('mean_dice=%.4f n_used=%d/%d'%(d.get('mean_dice',float('nan')),d.get('n_train_used',-1),d.get('n_train_total',-1)))" 2>/dev/null || echo NA)
        log "OK idx=${idx} elapsed=${elapsed}s ${md}"
        ran=$((ran + 1))
    else
        log "FAIL idx=${idx} rc=${rc} elapsed=${elapsed}s"
        failed=$((failed + 1))
    fi
done < "${JOBS_FILE}"

log "worker_m19 ${WORKER_ID} complete: ran=${ran} skipped=${skipped} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} failed=${failed}" > "${STATUS_FILE}"
