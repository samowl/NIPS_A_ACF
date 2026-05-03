#!/bin/bash
# Worker for the M15 5-fold-CV head trainer queue (jobs_m15.txt).
#
# Usage (inside a tmux Singularity session):
#   bash scripts/run_worker_m15.sh WORKER_ID N_WORKERS [GPU_ID]
#
# Each line in jobs_m15.txt has the form:
#   M15 {task} {fm} {fold} {seed}
# The worker dispatches each assigned line to scripts/train_head_5fold.py.
# Output JSON path: results/per_case_dice_m15/{task}/{fm}/fold_{k}_seed_{seed}.json
# (presence of that file -> SKIP, so the worker is resume-safe.)
#
# Cache dependency: the trainer reads results/feature_cache/{task}/{fm}/{train,test}.pt
# previously produced by scripts/extract_features.py (PASS-A). The worker
# polls (capped at ~30 minutes) for the manifest before dispatching, so M15
# can be launched alongside an in-progress PASS-A run.

set -uo pipefail

WORKER_ID="${1:?'WORKER_ID required (integer 0..N_WORKERS-1)'}"
N_WORKERS="${2:?'N_WORKERS required'}"
GPU_ID="${3:-}"

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

JOBS_FILE="${REPO_ROOT}/jobs_m15.txt"
LOG_DIR="${REPO_ROOT}/results/logs"
STATUS_FILE="${LOG_DIR}/worker_m15_${WORKER_ID}.status"
LOG_FILE="${LOG_DIR}/worker_m15_${WORKER_ID}.log"
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
    printf '[%s] worker_m15=%s %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${WORKER_ID}" "$*" \
        | tee -a "${LOG_FILE}"
}

log "starting WORKER_ID=${WORKER_ID} N_WORKERS=${N_WORKERS} GPU=${CUDA_VISIBLE_DEVICES:-default}"
log "PYTHON=${PYTHON}"
log "DATA_ROOT=${DATA_ROOT}"
log "CACHE_ROOT=${CACHE_ROOT}"
log "JOBS_FILE=${JOBS_FILE}"

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
    if [[ "${verb}" != "M15" ]]; then
        log "SKIP idx=${idx} unknown verb=${verb}"
        skipped=$((skipped + 1))
        continue
    fi
    task="$2"
    fm="$3"
    fold="$4"
    seed="$5"
    out_dir="${REPO_ROOT}/results/per_case_dice_m15/${task}/${fm}"
    out_json="${out_dir}/fold_${fold}_seed_${seed}.json"
    cache_dir="${CACHE_ROOT}/${task}/${fm}"
    manifest_path="${cache_dir}/manifest.json"
    if [[ -f "${out_json}" ]]; then
        # Validate that the existing JSON matches the requested cell to
        # prevent stale incompatible outputs from being silently kept.
        valid=$(python3 -c "
import json,sys
try:
    d = json.load(open('${out_json}'))
    ok = (d.get('schema_version') == 'm15_5fold_v1'
          and d.get('task') == '${task}' and d.get('fm') == '${fm}'
          and int(d.get('fold', -1)) == ${fold}
          and int(d.get('seed', -1)) == ${seed})
    print('ok' if ok else 'stale')
except Exception:
    print('stale')
" 2>/dev/null || echo "stale")
        if [[ "${valid}" == "ok" ]]; then
            log "SKIP idx=${idx} task=${task} fm=${fm} fold=${fold} seed=${seed} (valid ${out_json})"
            skipped=$((skipped + 1))
            continue
        else
            log "REGEN idx=${idx} task=${task} fm=${fm} fold=${fold} seed=${seed} (stale ${out_json}; will overwrite)"
        fi
    fi
    waited=0
    while [[ ! -f "${manifest_path}" && ${waited} -lt 180 ]]; do
        sleep 10
        waited=$((waited + 1))
    done
    if [[ ! -f "${manifest_path}" ]]; then
        log "FAIL idx=${idx} task=${task} fm=${fm} fold=${fold} seed=${seed} cache_missing"
        failed=$((failed + 1))
        continue
    fi
    log "RUN idx=${idx} task=${task} fm=${fm} fold=${fold} seed=${seed}"
    start_ts=$(date +%s)
    "${PYTHON}" "${REPO_ROOT}/scripts/train_head_5fold.py" \
        --task "${task}" --fm "${fm}" --fold "${fold}" --seed "${seed}" \
        --cache-root "${CACHE_ROOT}" \
        --out "${out_dir}" \
        --epochs 30 --batch-size 16 \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 && -f "${out_json}" ]]; then
        md=$(python3 -c "import json,sys; print(json.load(open('${out_json}')).get('mean_dice','NA'))" 2>/dev/null || echo NA)
        log "OK idx=${idx} elapsed=${elapsed}s mean_dice=${md}"
        ran=$((ran + 1))
    else
        log "FAIL idx=${idx} rc=${rc} elapsed=${elapsed}s (see ${LOG_FILE})"
        failed=$((failed + 1))
    fi
done < "${JOBS_FILE}"

log "worker_m15 ${WORKER_ID} complete: ran=${ran} skipped=${skipped} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} failed=${failed}" > "${STATUS_FILE}"
# Propagate failures so schedulers do not treat partial completion as success.
if [[ ${failed} -gt 0 ]]; then
    exit 3
fi
