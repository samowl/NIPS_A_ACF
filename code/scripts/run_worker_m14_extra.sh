#!/bin/bash
# Worker for optional M14 extension queues.
#
# Usage (inside a tmux/PBS GPU allocation):
#   FMPOOL_JOBS_FILE=jobs_m14_extra_acdc_kvasir.txt \
#   bash scripts/run_worker_m14_extra.sh WORKER_ID N_WORKERS [GPU_ID]
#
# Each line in the jobs file has the form:
#   M14EXTRA {task} {fm} {seed}
# Output JSON path: results/per_case_dice_m14/{task}/{fm}/seed_{seed}.json

set -uo pipefail

WORKER_ID="${1:?'WORKER_ID required (integer 0..N_WORKERS-1)'}"
N_WORKERS="${2:?'N_WORKERS required'}"
GPU_ID="${3:-}"

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

JOBS_FILE="${FMPOOL_JOBS_FILE:-${REPO_ROOT}/jobs_m14_extra_acdc_kvasir.txt}"
LOG_DIR="${REPO_ROOT}/results/logs"
STATUS_FILE="${LOG_DIR}/worker_m14_extra_${WORKER_ID}.status"
LOG_FILE="${LOG_DIR}/worker_m14_extra_${WORKER_ID}.log"
mkdir -p "${LOG_DIR}"

PYTHON="${FMPOOL_PYTHON:-python}"
DATA_ROOT="${FMPOOL_DATA_ROOT:-data/raw}"
JOB_VERB="${FMPOOL_JOB_VERB:-M14EXTRA}"

export PYTHONPATH="${REPO_ROOT}/src"
export FMPOOL_DATA_ROOT="${DATA_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${GPU_ID}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

log() {
    printf '[%s] worker_m14_extra=%s %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${WORKER_ID}" "$*" \
        | tee -a "${LOG_FILE}"
}

log "starting WORKER_ID=${WORKER_ID} N_WORKERS=${N_WORKERS} GPU=${CUDA_VISIBLE_DEVICES:-default}"
log "PYTHON=${PYTHON}"
log "DATA_ROOT=${DATA_ROOT}"
log "JOBS_FILE=${JOBS_FILE}"
log "JOB_VERB=${JOB_VERB}"

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
    idx=$((idx + 1))
    slice_mod=$((idx % N_WORKERS))
    if [[ "${slice_mod}" != "${WORKER_ID}" ]]; then
        continue
    fi
    set -- $line
    verb="$1"
    if [[ "${verb}" != "${JOB_VERB}" ]]; then
        log "SKIP idx=${idx} unknown verb=${verb}"
        skipped=$((skipped + 1))
        continue
    fi
    task="$2"
    fm="$3"
    seed="$4"
    out_dir="${REPO_ROOT}/results/per_case_dice_m14/${task}/${fm}"
    out_json="${out_dir}/seed_${seed}.json"
    if [[ -f "${out_json}" ]]; then
        valid=$("${PYTHON}" -c "
import json
try:
    d = json.load(open('${out_json}'))
    ok = (d.get('schema_version') == 'm14_clinical_v1'
          and d.get('task') == '${task}' and d.get('fm') == '${fm}'
          and int(d.get('seed', -1)) == ${seed})
    print('ok' if ok else 'stale')
except Exception:
    print('stale')
" 2>/dev/null || echo "stale")
        if [[ "${valid}" == "ok" ]]; then
            log "SKIP idx=${idx} task=${task} fm=${fm} seed=${seed} (valid ${out_json})"
            skipped=$((skipped + 1))
            continue
        else
            log "REGEN idx=${idx} task=${task} fm=${fm} seed=${seed} (stale ${out_json}; will overwrite)"
        fi
    fi
    log "RUN idx=${idx} task=${task} fm=${fm} seed=${seed}"
    start_ts=$(date +%s)
    "${PYTHON}" "${REPO_ROOT}/scripts/train_head_clinical.py" \
        --task "${task}" --fm "${fm}" --seed "${seed}" \
        --data-root "${DATA_ROOT}" \
        --epochs 30 --batch-size 16 \
        --out "${out_dir}" \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 && -f "${out_json}" ]]; then
        md=$("${PYTHON}" -c "import json; print(json.load(open('${out_json}')).get('mean_dice','NA'))" 2>/dev/null || echo NA)
        log "OK idx=${idx} elapsed=${elapsed}s mean_dice=${md}"
        ran=$((ran + 1))
    else
        log "FAIL idx=${idx} rc=${rc} elapsed=${elapsed}s (see ${LOG_FILE})"
        failed=$((failed + 1))
    fi
done < "${JOBS_FILE}"

log "worker_m14_extra ${WORKER_ID} complete: ran=${ran} skipped=${skipped} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} failed=${failed}" > "${STATUS_FILE}"
if [[ ${failed} -gt 0 ]]; then
    exit 3
fi
