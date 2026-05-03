#!/bin/bash
# Run one task/FM primary-completion pair with cached features, then four heads.
#
# Usage:
#   bash scripts/run_strengthen_pair_cached.sh TASK FM GPU_ID [SEEDS_CSV]
#
# This is intentionally small and resume-safe: extraction is skipped when the
# cache manifest exists, and each head is skipped when its JSON exists.

set -uo pipefail

TASK="${1:?'TASK required'}"
FM="${2:?'FM required'}"
GPU_ID="${3:?'GPU_ID required'}"
SEEDS_CSV="${4:-42,43,44,45}"

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

PYTHON="${FMPOOL_PYTHON:-python}"
DATA_ROOT="${FMPOOL_DATA_ROOT:-data/raw}"
CACHE_ROOT="${FMPOOL_CACHE_ROOT:-${REPO_ROOT}/results/feature_cache}"
LOG_DIR="${REPO_ROOT}/results/logs"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${REPO_ROOT}/src"
export FMPOOL_DATA_ROOT="${DATA_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

LOG_FILE="${LOG_DIR}/strengthen_pair_${TASK}_${FM}.log"
STATUS_FILE="${LOG_DIR}/strengthen_pair_${TASK}_${FM}.status"

log() {
    printf '[%s] strengthen_pair task=%s fm=%s gpu=%s %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${TASK}" "${FM}" "${GPU_ID}" "$*" \
        | tee -a "${LOG_FILE}"
}

log "start PYTHON=${PYTHON} DATA_ROOT=${DATA_ROOT} CACHE_ROOT=${CACHE_ROOT}"

manifest="${CACHE_ROOT}/${TASK}/${FM}/manifest.json"
if [[ -f "${manifest}" ]]; then
    log "SKIP EXTRACT cache_present=${manifest}"
else
    log "RUN EXTRACT"
    "${PYTHON}" "${REPO_ROOT}/scripts/extract_features.py" \
        --task "${TASK}" --fm "${FM}" \
        --data-root "${DATA_ROOT}" \
        --cache-root "${CACHE_ROOT}" \
        --batch-size 32 --num-workers 4 \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    if [[ ${rc} -ne 0 || ! -f "${manifest}" ]]; then
        log "FAIL EXTRACT rc=${rc} manifest_present=$([[ -f "${manifest}" ]] && echo yes || echo no)"
        echo "FAIL extract rc=${rc}" > "${STATUS_FILE}"
        exit 2
    fi
    log "OK EXTRACT"
fi

failed=0
ran=0
skipped=0
IFS=',' read -r -a seeds <<< "${SEEDS_CSV}"
for seed in "${seeds[@]}"; do
    out_dir="${REPO_ROOT}/results/per_case_dice/${TASK}/${FM}"
    out_json="${out_dir}/seed_${seed}.json"
    mkdir -p "${out_dir}"
    if [[ -f "${out_json}" ]]; then
        log "SKIP HEAD seed=${seed} json_present=${out_json}"
        skipped=$((skipped + 1))
        continue
    fi
    log "RUN HEAD seed=${seed}"
    "${PYTHON}" "${REPO_ROOT}/scripts/train_head_cached.py" \
        --task "${TASK}" --fm "${FM}" --seed "${seed}" \
        --cache-root "${CACHE_ROOT}" \
        --out "${out_dir}" \
        --epochs 30 --batch-size 16 \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    if [[ ${rc} -eq 0 && -f "${out_json}" ]]; then
        md=$("${PYTHON}" - <<PY 2>/dev/null || echo NA
import json
print(json.load(open("${out_json}")).get("mean_dice", "NA"))
PY
)
        log "OK HEAD seed=${seed} mean_dice=${md}"
        ran=$((ran + 1))
    else
        log "FAIL HEAD seed=${seed} rc=${rc}"
        failed=$((failed + 1))
    fi
done

log "complete ran=${ran} skipped=${skipped} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} failed=${failed}" > "${STATUS_FILE}"

