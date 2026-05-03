#!/bin/bash
# Run missing M19 BraTS train-size cells for one FM.
#
# Usage:
#   bash scripts/run_strengthen_m19_fm.sh FM GPU_ID [FRACS_CSV]
#
# Extraction is run once if the feature cache is missing; then seed-42 heads
# are trained for the requested fractions.

set -uo pipefail

FM="${1:?'FM required'}"
GPU_ID="${2:?'GPU_ID required'}"
FRACS_CSV="${3:-0.1,0.25,0.5,1.0}"
TASK="brats_wt"
SEED="42"

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

LOG_FILE="${LOG_DIR}/strengthen_m19_${FM}.log"
STATUS_FILE="${LOG_DIR}/strengthen_m19_${FM}.status"

log() {
    printf '[%s] strengthen_m19 fm=%s gpu=%s %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "${FM}" "${GPU_ID}" "$*" \
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
IFS=',' read -r -a fracs <<< "${FRACS_CSV}"
for frac in "${fracs[@]}"; do
    out_dir="${REPO_ROOT}/results/per_case_dice_m19/frac${frac}/${TASK}/${FM}"
    out_json="${out_dir}/seed_${SEED}.json"
    mkdir -p "${out_dir}"
    if [[ -f "${out_json}" ]]; then
        log "SKIP HEAD_FRAC frac=${frac} json_present=${out_json}"
        skipped=$((skipped + 1))
        continue
    fi
    log "RUN HEAD_FRAC frac=${frac}"
    "${PYTHON}" "${REPO_ROOT}/scripts/train_head_subset.py" \
        --task "${TASK}" --fm "${FM}" --seed "${SEED}" \
        --train-frac "${frac}" \
        --cache-root "${CACHE_ROOT}" \
        --out "${out_dir}" \
        --epochs 30 --batch-size 16 \
        >> "${LOG_FILE}" 2>&1
    rc=$?
    if [[ ${rc} -eq 0 && -f "${out_json}" ]]; then
        md=$("${PYTHON}" - <<PY 2>/dev/null || echo NA
import json
d=json.load(open("${out_json}"))
print("mean_dice=%s n_used=%s/%s" % (d.get("mean_dice", "NA"), d.get("n_train_used", "NA"), d.get("n_train_total", "NA")))
PY
)
        log "OK HEAD_FRAC frac=${frac} ${md}"
        ran=$((ran + 1))
    else
        log "FAIL HEAD_FRAC frac=${frac} rc=${rc}"
        failed=$((failed + 1))
    fi
done

log "complete ran=${ran} skipped=${skipped} failed=${failed}"
echo "DONE ran=${ran} skipped=${skipped} failed=${failed}" > "${STATUS_FILE}"

