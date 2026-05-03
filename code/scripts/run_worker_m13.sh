#!/bin/bash
# M13 worker: PASS-A + PASS-B for the leave-MESSIDOR-out OOD matrix.
#
# Usage (inside a raw-data environment where RIGA images are available):
#   bash scripts/run_worker_m13.sh WORKER_ID N_WORKERS [GPU_ID]
#
# Slices jobs_m13.txt by line index modulo N_WORKERS. Two verbs:
#
#   EXTRACT riga_cup_ood_messidor {fm}            -> extract_features.py
#   HEAD    riga_cup_ood_messidor {fm} {seed}     -> train_head_cached.py
#
# HEAD lines block (poll-and-wait, capped at ~30 min) until the matching
# EXTRACT has written
# results/feature_cache/riga_cup_ood_messidor/{fm}/manifest.json. EXTRACT
# and HEAD are both skipped if their expected output already exists, so the
# worker is fully resume-safe and idempotent.
#
# IMPORTANT: M13 requires a raw-data environment because the RIGA TIFF
# corpus lives at data/raw/RIGA/RIGAPlus/; environments without the image files
# can run only the CSV-based checks.
#
# GPU_ID (optional) selects which physical GPU to bind via
# CUDA_VISIBLE_DEVICES. Default uses the Singularity-visible device 0.
#
# Logs append to results/logs/worker_m13_${WORKER_ID}.log; final status
# line lands in results/logs/worker_m13_${WORKER_ID}.status.

set -uo pipefail

WORKER_ID="${1:?'WORKER_ID required (integer 0..N_WORKERS-1)'}"
N_WORKERS="${2:?'N_WORKERS required'}"
GPU_ID="${3:-}"

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

JOBS_FILE="${REPO_ROOT}/jobs_m13.txt"
LOG_DIR="${REPO_ROOT}/results/logs"
STATUS_FILE="${LOG_DIR}/worker_m13_${WORKER_ID}.status"
LOG_FILE="${LOG_DIR}/worker_m13_${WORKER_ID}.log"
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
    printf '[%s] worker_m13=%s %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${WORKER_ID}" "$*" \
        | tee -a "${LOG_FILE}"
}

log "starting WORKER_ID=${WORKER_ID} N_WORKERS=${N_WORKERS} GPU=${CUDA_VISIBLE_DEVICES:-default} PYTHON=${PYTHON}"
log "DATA_ROOT=${DATA_ROOT} CACHE_ROOT=${CACHE_ROOT}"

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
    if [[ -f "${manifest_path}" ]]; then
        log "SKIP EXTRACT idx=${extract_idx} task=${task} fm=${fm} (cache present)"
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
    out_dir="${REPO_ROOT}/results/per_case_dice/${task}/${fm}"
    out_json="${out_dir}/seed_${seed}.json"
    cache_dir="${CACHE_ROOT}/${task}/${fm}"
    manifest_path="${cache_dir}/manifest.json"
    if [[ -f "${out_json}" ]]; then
        log "SKIP HEAD idx=${head_idx} task=${task} fm=${fm} seed=${seed} (json present)"
        skipped=$((skipped + 1))
        continue
    fi
    waited=0
    while [[ ! -f "${manifest_path}" && ${waited} -lt 180 ]]; do
        sleep 10
        waited=$((waited + 1))
    done
    if [[ ! -f "${manifest_path}" ]]; then
        log "FAIL HEAD idx=${head_idx} task=${task} fm=${fm} seed=${seed} cache_missing"
        failed=$((failed + 1))
        continue
    fi
    log "RUN HEAD idx=${head_idx} task=${task} fm=${fm} seed=${seed}"
    start_ts=$(date +%s)
    "${PYTHON}" "${REPO_ROOT}/scripts/train_head_cached.py" \
        --task "${task}" --fm "${fm}" --seed "${seed}" \
        --cache-root "${CACHE_ROOT}" \
        --out "${out_dir}" \
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

log "worker_m13 ${WORKER_ID} complete: ran=${ran} skipped=${skipped} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} failed=${failed}" > "${STATUS_FILE}"
