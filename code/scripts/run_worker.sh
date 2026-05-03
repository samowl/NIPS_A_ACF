#!/bin/bash
# Worker for the fmpool_clean job queue. SPEC §7.
#
# Usage (inside a tmux Singularity session):
#   bash scripts/run_worker.sh WORKER_ID N_WORKERS
#
# Splits jobs.txt into N_WORKERS slices by line index (0-based modulo) and runs
# the slice assigned to WORKER_ID sequentially. Skips lines where the expected
# output JSON already exists (resume-safe). Appends all stdout/stderr to
# results/logs/worker_${WORKER_ID}.log.

set -uo pipefail

WORKER_ID="${1:?'WORKER_ID required (integer 0..N_WORKERS-1)'}"
N_WORKERS="${2:?'N_WORKERS required'}"

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

JOBS_FILE="${REPO_ROOT}/jobs.txt"
LOG_DIR="${REPO_ROOT}/results/logs"
STATUS_FILE="${LOG_DIR}/worker_${WORKER_ID}.status"
LOG_FILE="${LOG_DIR}/worker_${WORKER_ID}.log"
mkdir -p "${LOG_DIR}"

# Resolve Python + data root. Use the configured Python interpreter and raw-data root,
# override via env FMPOOL_PYTHON / FMPOOL_DATA_ROOT.
PYTHON="${FMPOOL_PYTHON:-python}"
DATA_ROOT="${FMPOOL_DATA_ROOT:-data/raw}"

export PYTHONPATH="${REPO_ROOT}/src"
export FMPOOL_DATA_ROOT="${DATA_ROOT}"

log() { printf '[%s] worker=%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${WORKER_ID}" "$*" | tee -a "${LOG_FILE}"; }

log "starting WORKER_ID=${WORKER_ID} N_WORKERS=${N_WORKERS}"
log "PYTHON=${PYTHON}"
log "DATA_ROOT=${DATA_ROOT}"
log "JOBS_FILE=${JOBS_FILE}"

if [[ ! -f "${JOBS_FILE}" ]]; then
    log "FATAL jobs file missing: ${JOBS_FILE}"
    echo "FAIL missing_jobs_file" > "${STATUS_FILE}"
    exit 2
fi

TOTAL=$(wc -l < "${JOBS_FILE}")
log "total jobs in queue: ${TOTAL}"

ran=0
skipped=0
failed=0
idx=-1
while IFS= read -r line; do
    # Skip blank lines before incrementing idx, so they don't shift assignments.
    [[ -z "$line" ]] && continue
    idx=$((idx + 1))
    slice_mod=$((idx % N_WORKERS))
    if [[ "$slice_mod" != "$WORKER_ID" ]]; then
        continue
    fi
    read -r task fm seed <<< "$line"
    out_dir="${REPO_ROOT}/results/per_case_dice/${task}/${fm}"
    out_json="${out_dir}/seed_${seed}.json"
    if [[ -f "${out_json}" ]]; then
        log "SKIP idx=${idx} task=${task} fm=${fm} seed=${seed} (exists ${out_json})"
        skipped=$((skipped + 1))
        continue
    fi
    log "RUN idx=${idx} task=${task} fm=${fm} seed=${seed}"
    start_ts=$(date +%s)
    "${PYTHON}" "${REPO_ROOT}/scripts/train_frozen_fm.py" \
        --task "${task}" --fm "${fm}" --seed "${seed}" \
        --data-root "${DATA_ROOT}" \
        --epochs 30 --batch-size 16 \
        --out "${out_dir}" \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 && -f "${out_json}" ]]; then
        md=$(python3 -c "import json; print(json.load(open('${out_json}')).get('mean_dice', 'NA'))" 2>/dev/null || echo NA)
        log "OK idx=${idx} elapsed=${elapsed}s mean_dice=${md}"
        ran=$((ran + 1))
    else
        log "FAIL idx=${idx} rc=${rc} elapsed=${elapsed}s  (see ${LOG_FILE})"
        failed=$((failed + 1))
    fi
done < "${JOBS_FILE}"

log "worker ${WORKER_ID} complete: ran=${ran} skipped=${skipped} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} failed=${failed}" > "${STATUS_FILE}"
