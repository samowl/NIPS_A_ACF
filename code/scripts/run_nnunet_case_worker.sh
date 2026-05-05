#!/usr/bin/env bash
# Distributed worker for the case-identical RIGA Cup nnU-Net scope check.
#
# Prepare once per host:
#   PYTHON=/path/to/python bash scripts/run_nnunet_case_worker.sh prepare
#
# Then launch one worker per GPU:
#   CUDA_VISIBLE_DEVICES=0 WORKER_ID=0 N_WORKERS=4 PYTHON=/path/to/python \
#     bash scripts/run_nnunet_case_worker.sh worker
#
# The worker grid is 5 folds x 2 seeds (13, 37) using the nnU-Net v2 2D
# seeded trainer. By default this runs the 100-epoch scope check; set
# TRAINER_EPOCHS=1000 and EXP_NAME=nnunet_case_riga_2d_1000ep for the
# default-length companion. Training folds use BinRushed+MESSIDOR; held-out
# scoring uses the same Magrabia cases as the primary RIGA frozen-encoder row.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
PYBIN="$( dirname "${PYTHON}" )"
export PATH="${PYBIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export FMPOOL_DATA_ROOT="${FMPOOL_DATA_ROOT:-/home/${USER}/datasets}"
export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnunet_case_workspace/raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnunet_case_workspace/preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_ROOT}/nnunet_case_workspace/results}"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

RIGA_CASE_DATASET_ID="${RIGA_CASE_DATASET_ID:-551}"
TRAINER_EPOCHS="${TRAINER_EPOCHS:-100}"
if [ "${TRAINER_EPOCHS}" != "100" ] && [ "${TRAINER_EPOCHS}" != "1000" ]; then
  echo "TRAINER_EPOCHS must be 100 or 1000, got ${TRAINER_EPOCHS}" >&2
  exit 2
fi
BASE_TRAINER="nnUNetTrainer_${TRAINER_EPOCHS}epochs"
EXP_NAME="${EXP_NAME:-nnunet_case_riga_2d_${TRAINER_EPOCHS}ep}"
OUT_DIR="${REPO_ROOT}/results/nnunet/${EXP_NAME}"
LOG_DIR="${REPO_ROOT}/results/logs/${EXP_NAME}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

log() {
  printf '[nnunet_case %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

verify_env() {
  log "python=${PYTHON}"
  "${PYTHON}" - <<'PY'
import nnunetv2
import torch
print("nnunetv2", nnunetv2.__file__)
print("torch_cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
  for cli in nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict; do
    command -v "${cli}" >/dev/null
    log "${cli}=$(command -v "${cli}")"
  done
}

install_manual_split() {
  local pad raw_split pre_dir
  pad="$(printf '%03d' "${RIGA_CASE_DATASET_ID}")"
  raw_split="$(find "${nnUNet_raw}" -maxdepth 2 -path "*/Dataset${pad}_*/splits_final.json" -type f -print -quit)"
  pre_dir="$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "Dataset${pad}_*" -print -quit)"
  if [ -n "${raw_split}" ] && [ -n "${pre_dir}" ]; then
    cp "${raw_split}" "${pre_dir}/splits_final.json"
    log "installed manual splits: ${raw_split} -> ${pre_dir}/splits_final.json"
  fi
}

prepare_case_dataset() {
  verify_env
  log "install seeded nnU-Net trainer discovery shim"
  "${PYTHON}" "${REPO_ROOT}/scripts/install_nnunet_seeded_trainers.py"
  log "export RIGA primary split dataset_id=${RIGA_CASE_DATASET_ID}"
  "${PYTHON}" "${REPO_ROOT}/scripts/prepare_nnunet_data.py" \
    --task riga_cup \
    --dataset-id "${RIGA_CASE_DATASET_ID}" \
    ${FORCE_REPREP:+--force}
  local pad
  pad="$(printf '%03d' "${RIGA_CASE_DATASET_ID}")"
  if [ "${FORCE_REPLAN:-0}" != "1" ] && \
     compgen -G "${nnUNet_preprocessed}/Dataset${pad}_*/nnUNetPlans.json" >/dev/null; then
    log "plan already exists for Dataset${pad}; skip"
    install_manual_split
  else
    log "plan_and_preprocess Dataset${pad}"
    nnUNetv2_plan_and_preprocess -d "${RIGA_CASE_DATASET_ID}" --verify_dataset_integrity
    install_manual_split
  fi
}

run_cell() {
  local fold="$1"
  local seed="$2"
  local trainer="nnUNetTrainerSeed${seed}_${TRAINER_EPOCHS}epochs"
  local out_json="${OUT_DIR}/riga_cup_fold${fold}_seed${seed}.json"
  if [ -f "${out_json}" ]; then
    log "skip existing ${out_json}"
    return 0
  fi
  log "train fold=${fold} seed=${seed} gpu=${CUDA_VISIBLE_DEVICES:-all}"
  "${PYTHON}" "${REPO_ROOT}/scripts/install_nnunet_seeded_trainers.py"
  local t0 t1 elapsed
  t0="$(date +%s)"
  nnUNetv2_train "${RIGA_CASE_DATASET_ID}" 2d "${fold}" -tr "${trainer}" --c
  t1="$(date +%s)"
  elapsed="$((t1 - t0))"
  log "predict fold=${fold} seed=${seed} elapsed=${elapsed}s"
  "${PYTHON}" "${REPO_ROOT}/scripts/predict_nnunet_holdout.py" \
    --task riga_cup \
    --dataset-id "${RIGA_CASE_DATASET_ID}" \
    --config 2d \
    --fold "${fold}" \
    --seed "${seed}" \
    --base-trainer "${BASE_TRAINER}" \
    --out "${out_json}" \
    --training-elapsed-s "${elapsed}"
}

run_worker() {
  verify_env
  local worker_id="${WORKER_ID:?WORKER_ID required}"
  local n_workers="${N_WORKERS:?N_WORKERS required}"
  log "worker ${worker_id}/${n_workers} starting"
  local idx=0
  local fold seed
  for fold in 0 1 2 3 4; do
    for seed in 13 37; do
      if [ $(( idx % n_workers )) -eq "${worker_id}" ]; then
        run_cell "${fold}" "${seed}" 2>&1 | tee -a "${LOG_DIR}/worker_${worker_id}.log"
      fi
      idx=$((idx + 1))
    done
  done
  log "worker ${worker_id}/${n_workers} done"
}

cmd="${1:-worker}"
case "${cmd}" in
  prepare)
    prepare_case_dataset
    ;;
  worker)
    run_worker
    ;;
  *)
    echo "Usage: $0 {prepare|worker}" >&2
    exit 2
    ;;
esac
