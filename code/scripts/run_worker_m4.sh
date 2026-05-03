#!/bin/bash
# M4 worker (UNet-skip head sweep over 5 FMs x 4 tasks x 2 seeds = 40 cells).
# Reads jobs_m4.txt, runs slice modulo WORKER_ID. Each line is:
#
#   M4 {task} {fm} {seed}
#
# DEVIATION FROM SPEC §4: M4 swaps the SPEC §4 1x1-conv decoder for a
# multi-scale UNet-skip decoder.
#
# Unlike M3, this worker does NOT require a PASS-A feature cache: the
# multi-scale UNet-skip trainer extracts FM features online (encoder
# frozen, bf16 autocast, no_grad).
#
# Usage (inside a tmux Singularity session):
#   bash scripts/run_worker_m4.sh WORKER_ID N_WORKERS [GPU_ID]
#
# Optional env:
#   FMPOOL_TASKS_ALLOW="acdc_lv,riga_cup"   # comma-list of allowed tasks
#                                            # (others skipped); useful on
#                                            # nodes that lack Kvasir/BraTS.
#   FMPOOL_PYTHON                            # override interpreter
#   FMPOOL_DATA_ROOT                         # override dataset root
#
# Resume safety: skips when the per-case JSON already exists. Logs are
# appended to results/logs/worker_m4_${WORKER_ID}.log; final status to
# results/logs/worker_m4_${WORKER_ID}.status.

set -uo pipefail

WORKER_ID="${1:?'WORKER_ID required (integer 0..N_WORKERS-1)'}"
N_WORKERS="${2:?'N_WORKERS required'}"
GPU_ID="${3:-}"

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

JOBS_FILE="${REPO_ROOT}/jobs_m4.txt"
LOG_DIR="${REPO_ROOT}/results/logs"
STATUS_FILE="${LOG_DIR}/worker_m4_${WORKER_ID}.status"
LOG_FILE="${LOG_DIR}/worker_m4_${WORKER_ID}.log"
mkdir -p "${LOG_DIR}"

PYTHON="${FMPOOL_PYTHON:-python}"
DATA_ROOT="${FMPOOL_DATA_ROOT:-data/raw}"
TASKS_ALLOW="${FMPOOL_TASKS_ALLOW:-}"

export PYTHONPATH="${REPO_ROOT}/src"
export FMPOOL_DATA_ROOT="${DATA_ROOT}"
if [[ -n "${GPU_ID}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

log() {
    printf '[%s] worker_m4=%s %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${WORKER_ID}" "$*" \
        | tee -a "${LOG_FILE}"
}

is_task_allowed() {
    local task="$1"
    if [[ -z "${TASKS_ALLOW}" ]]; then
        return 0
    fi
    IFS=',' read -ra ALLOW_ARR <<< "${TASKS_ALLOW}"
    for t in "${ALLOW_ARR[@]}"; do
        if [[ "${t}" == "${task}" ]]; then
            return 0
        fi
    done
    return 1
}

log "starting M4 WORKER_ID=${WORKER_ID} N_WORKERS=${N_WORKERS} GPU=${CUDA_VISIBLE_DEVICES:-default} PYTHON=${PYTHON}"
log "DATA_ROOT=${DATA_ROOT} TASKS_ALLOW=${TASKS_ALLOW:-<all>}"

if [[ ! -f "${JOBS_FILE}" ]]; then
    log "FATAL jobs file missing: ${JOBS_FILE}"
    echo "FAIL missing_jobs_file" > "${STATUS_FILE}"
    exit 2
fi

ran=0
skipped=0
filtered=0
failed=0
job_idx=-1

OUT_ROOT="${REPO_ROOT}/results/per_case_dice_m4"

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    set -- $line
    verb="$1"
    if [[ "${verb}" != "M4" ]]; then
        continue
    fi
    job_idx=$((job_idx + 1))
    slice_mod=$((job_idx % N_WORKERS))
    if [[ "${slice_mod}" != "${WORKER_ID}" ]]; then
        continue
    fi

    task="$2"
    fm="$3"
    seed="$4"

    if ! is_task_allowed "${task}"; then
        log "FILTER idx=${job_idx} task=${task} (not in allow-list)"
        filtered=$((filtered + 1))
        continue
    fi

    out_dir="${OUT_ROOT}/${task}/${fm}"
    out_json="${out_dir}/seed_${seed}.json"

    if [[ -f "${out_json}" ]]; then
        log "SKIP M4 idx=${job_idx} task=${task} fm=${fm} seed=${seed} (json present)"
        skipped=$((skipped + 1))
        continue
    fi

    log "RUN M4 idx=${job_idx} task=${task} fm=${fm} seed=${seed}"
    start_ts=$(date +%s)
    "${PYTHON}" "${REPO_ROOT}/scripts/train_head_unet_skip.py" \
        --task "${task}" --fm "${fm}" --seed "${seed}" \
        --data-root "${DATA_ROOT}" \
        --out "${OUT_ROOT}" \
        --epochs 30 --batch-size 16 \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 && -f "${out_json}" ]]; then
        md=$(python3 -c "import json; print(json.load(open('${out_json}')).get('mean_dice', 'NA'))" 2>/dev/null || echo NA)
        log "OK M4 idx=${job_idx} elapsed=${elapsed}s mean_dice=${md}"
        ran=$((ran + 1))
    else
        log "FAIL M4 idx=${job_idx} rc=${rc} elapsed=${elapsed}s"
        failed=$((failed + 1))
    fi
done < "${JOBS_FILE}"

log "M4 worker ${WORKER_ID} complete: ran=${ran} skipped=${skipped} filtered=${filtered} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} filtered=${filtered} failed=${failed}" > "${STATUS_FILE}"
