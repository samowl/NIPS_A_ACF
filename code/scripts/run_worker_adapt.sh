#!/bin/bash
# Adaptation factorial worker (M6-M11 cells). Reads jobs_adaptation.txt and
# dispatches each line to scripts/train_adapted.py. Resume-safe.
#
# Usage (inside a tmux Singularity session):
#   bash scripts/run_worker_adapt.sh WORKER_ID N_WORKERS [GPU_ID]
#
# Optional env:
#   FMPOOL_TASKS_ALLOW="acdc_lv,riga_cup"   # comma-list of allowed tasks
#                                            # (others skipped when those raw datasets are unavailable).
#   FMPOOL_PYTHON                            # override interpreter
#   FMPOOL_DATA_ROOT                         # override dataset root
#
# Job line format (8 fields):
#   ADAPT {task} {fm} {seed} {regime} {lora_rank|-} {lora_blocks|-} {epochs}
#
# regime in {lora, full_ft, rand_init}; lora_rank/blocks are "-" for non-LoRA.

set -uo pipefail

WORKER_ID="${1:?'WORKER_ID required'}"
N_WORKERS="${2:?'N_WORKERS required'}"
GPU_ID="${3:-}"

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

JOBS_FILE="${REPO_ROOT}/jobs_adaptation.txt"
LOG_DIR="${REPO_ROOT}/results/logs"
STATUS_FILE="${LOG_DIR}/worker_adapt_${WORKER_ID}.status"
LOG_FILE="${LOG_DIR}/worker_adapt_${WORKER_ID}.log"
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
    printf '[%s] adapt_worker=%s %s\n' \
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

log "starting WORKER_ID=${WORKER_ID} N_WORKERS=${N_WORKERS} GPU=${CUDA_VISIBLE_DEVICES:-default}"
log "PYTHON=${PYTHON} DATA_ROOT=${DATA_ROOT} TASKS_ALLOW=${TASKS_ALLOW:-<all>}"

if [[ ! -f "${JOBS_FILE}" ]]; then
    log "FATAL jobs file missing: ${JOBS_FILE}"
    echo "FAIL missing_jobs_file" > "${STATUS_FILE}"
    exit 2
fi

ran=0
skipped=0
failed=0
filtered=0
idx=-1

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    set -- $line
    verb="$1"
    if [[ "${verb}" != "ADAPT" ]]; then
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
    regime="$5"
    lora_rank="$6"
    lora_blocks="$7"
    epochs="$8"

    if ! is_task_allowed "${task}"; then
        log "FILTER idx=${idx} task=${task} (not in allow-list)"
        filtered=$((filtered + 1))
        continue
    fi

    if [[ "${regime}" == "lora" ]]; then
        config="lora_r${lora_rank}_b${lora_blocks}"
    elif [[ "${regime}" == "full_ft" ]]; then
        config="ft_e${epochs}"
    elif [[ "${regime}" == "rand_init" ]]; then
        config="rand_init"
    else
        log "FAIL idx=${idx} unknown regime=${regime}"
        failed=$((failed + 1))
        continue
    fi

    out_root="${REPO_ROOT}/results/per_case_dice_adapted"
    out_dir="${out_root}/${regime}/${config}/${task}/${fm}"
    out_json="${out_dir}/seed_${seed}.json"
    if [[ -f "${out_json}" ]]; then
        log "SKIP idx=${idx} ${regime}/${config} ${task}/${fm} seed=${seed} (json present)"
        skipped=$((skipped + 1))
        continue
    fi

    log "RUN idx=${idx} ${regime}/${config} task=${task} fm=${fm} seed=${seed} epochs=${epochs}"
    start_ts=$(date +%s)

    cmd=(
        "${PYTHON}" "${REPO_ROOT}/scripts/train_adapted.py"
        --task "${task}" --fm "${fm}" --seed "${seed}"
        --regime "${regime}" --epochs "${epochs}"
        --data-root "${DATA_ROOT}"
        --out "${out_root}"
    )
    if [[ "${regime}" == "lora" ]]; then
        cmd+=( --lora-rank "${lora_rank}" --lora-blocks "${lora_blocks}" )
    fi

    "${cmd[@]}" >> "${LOG_FILE}" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 && -f "${out_json}" ]]; then
        md=$(python3 -c "import json; print(json.load(open('${out_json}')).get('mean_dice','NA'))" 2>/dev/null || echo NA)
        log "OK idx=${idx} elapsed=${elapsed}s mean_dice=${md}"
        ran=$((ran + 1))
    else
        log "FAIL idx=${idx} rc=${rc} elapsed=${elapsed}s"
        failed=$((failed + 1))
    fi
done < "${JOBS_FILE}"

log "worker_adapt ${WORKER_ID} complete: ran=${ran} skipped=${skipped} filtered=${filtered} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} filtered=${filtered} failed=${failed}" > "${STATUS_FILE}"
