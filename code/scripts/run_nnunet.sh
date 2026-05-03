#!/usr/bin/env bash
# nnU-Net v2 launch driver for SPEC §5 matrices M16 + M17.
#
# Usage:
#   bash scripts/run_nnunet.sh verify            # check install + env
#   bash scripts/run_nnunet.sh prepare           # convert datasets + plan_and_preprocess
#   bash scripts/run_nnunet.sh m16               # 2D 100ep   RIGA Cup   (5x2=10 cells)
#   bash scripts/run_nnunet.sh m17_riga          # 2D 1000ep  RIGA Cup   (5x2=10 cells)
#   bash scripts/run_nnunet.sh m17_acdc          # 3D fullres 1000ep ACDC LV (5x2=10 cells)
#   bash scripts/run_nnunet.sh m17               # m17_riga then m17_acdc
#
# Environment overrides (defaults use local workspace paths):
#   nnUNet_raw           nnunet_workspace/raw
#   nnUNet_preprocessed  nnunet_workspace/preprocessed
#   nnUNet_results       nnunet_workspace/results
#   FOLDS                "0 1 2 3 4"
#   SEEDS                "13 37"        (5 folds x 2 seeds = 10 cells)
#   PYTHON               python
#   RIGA_DATASET_ID      501
#   ACDC_DATASET_ID      502
#   FORCE_REPREP=1       force re-export of raw data
#   FORCE_REPLAN=1       force re-run plan_and_preprocess
#
set -euo pipefail

# --- Locate repo ------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-python}"

# --- nnU-Net workspace ------------------------------------------------------
export nnUNet_raw="${nnUNet_raw:-nnunet_workspace/raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-nnunet_workspace/preprocessed}"
export nnUNet_results="${nnUNet_results:-nnunet_workspace/results}"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

FOLDS="${FOLDS:-0 1 2 3 4}"
SEEDS="${SEEDS:-13 37}"
RIGA_DATASET_ID="${RIGA_DATASET_ID:-501}"
ACDC_DATASET_ID="${ACDC_DATASET_ID:-502}"

log() { printf '[run_nnunet %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

verify_install() {
  log "verifying nnunetv2 install + env vars"
  if ! "${PYTHON}" -c "import nnunetv2; print('nnunetv2', nnunetv2.__file__)"; then
    echo "ERROR: nnunetv2 not importable. Install with: pip install nnunetv2" >&2
    return 2
  fi
  for var in nnUNet_raw nnUNet_preprocessed nnUNet_results; do
    log "  ${var}=${!var}"
    [ -d "${!var}" ] || { echo "ERROR: ${var} dir missing" >&2; return 3; }
  done
  for cli in nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict; do
    if ! command -v "${cli}" >/dev/null; then
      echo "ERROR: ${cli} not on PATH" >&2
      return 4
    fi
    log "  ${cli} -> $(command -v "${cli}")"
  done
  log "install/check FMPool seeded trainer discovery shim"
  "${PYTHON}" "${REPO_ROOT}/scripts/install_nnunet_seeded_trainers.py"
  log "verify OK"
}

prepare_dataset() {
  local task="$1" did="$2"
  local force_flag=()
  [ "${FORCE_REPREP:-0}" = "1" ] && force_flag+=(--force)
  log "prepare_nnunet_data --task ${task} --dataset-id ${did}"
  "${PYTHON}" "${REPO_ROOT}/scripts/prepare_nnunet_data.py" \
      --task "${task}" --dataset-id "${did}" "${force_flag[@]}"
}

plan_and_preprocess() {
  local did="$1"
  local pad
  pad=$(printf '%03d' "${did}")
  if [ "${FORCE_REPLAN:-0}" != "1" ] && \
     compgen -G "${nnUNet_preprocessed}/Dataset${pad}_*/nnUNetPlans.json" > /dev/null; then
    log "plan_and_preprocess for ${did} already complete; skip"
    return 0
  fi
  log "plan_and_preprocess -d ${did}"
  nnUNetv2_plan_and_preprocess -d "${did}" --verify_dataset_integrity
}

# Single training cell.
# Args: task dataset_id config base_trainer fold seed exp_name
train_cell() {
  local task="$1" did="$2" cfg="$3" tr="$4" fold="$5" seed="$6" exp="$7"
  local epoch_suffix="${tr#nnUNetTrainer_}"
  local trainer_cls="nnUNetTrainerSeed${seed}_${epoch_suffix}"
  local out_json="${REPO_ROOT}/results/nnunet/${exp}/${task}_fold${fold}_seed${seed}.json"

  if [ -f "${out_json}" ]; then
    log "SKIP (output exists): ${out_json}"
    return 0
  fi

  log "TRAIN  task=${task} cfg=${cfg} fold=${fold} seed=${seed} trainer=${trainer_cls}"
  # Verify the seeded trainer subclass is registered through nnU-Net's
  # official recursive trainer-discovery path before invoking the upstream CLI.
  "${PYTHON}" "${REPO_ROOT}/scripts/install_nnunet_seeded_trainers.py"

  local t0 t1 elapsed
  t0=$(date +%s)
  nnUNetv2_train "${did}" "${cfg}" "${fold}" -tr "${trainer_cls}" --c
  t1=$(date +%s)
  elapsed=$(( t1 - t0 ))
  log "  training elapsed ${elapsed}s"

  log "PREDICT task=${task} fold=${fold} seed=${seed} -> ${out_json}"
  "${PYTHON}" "${REPO_ROOT}/scripts/predict_nnunet_holdout.py" \
      --task "${task}" \
      --dataset-id "${did}" \
      --config "${cfg}" \
      --fold "${fold}" \
      --seed "${seed}" \
      --base-trainer "${tr}" \
      --out "${out_json}" \
      --training-elapsed-s "${elapsed}"
}

run_matrix() {
  local task="$1" did="$2" cfg="$3" tr="$4" exp="$5"
  log "==== matrix ${exp}: task=${task} cfg=${cfg} trainer=${tr} ===="
  for fold in ${FOLDS}; do
    for seed in ${SEEDS}; do
      train_cell "${task}" "${did}" "${cfg}" "${tr}" "${fold}" "${seed}" "${exp}"
    done
  done
}

cmd="${1:-verify}"
case "${cmd}" in
  verify)
    verify_install
    ;;
  prepare)
    verify_install
    prepare_dataset riga_cup "${RIGA_DATASET_ID}"
    prepare_dataset acdc_lv  "${ACDC_DATASET_ID}"
    plan_and_preprocess "${RIGA_DATASET_ID}"
    plan_and_preprocess "${ACDC_DATASET_ID}"
    ;;
  m16)
    run_matrix riga_cup "${RIGA_DATASET_ID}" 2d \
        nnUNetTrainer_100epochs nnunet_2d_100ep
    ;;
  m17_riga)
    run_matrix riga_cup "${RIGA_DATASET_ID}" 2d \
        nnUNetTrainer_1000epochs nnunet_2d_1000ep
    ;;
  m17_acdc)
    run_matrix acdc_lv "${ACDC_DATASET_ID}" 3d_fullres \
        nnUNetTrainer_1000epochs nnunet_3d_fullres
    ;;
  m17)
    run_matrix riga_cup "${RIGA_DATASET_ID}" 2d \
        nnUNetTrainer_1000epochs nnunet_2d_1000ep
    run_matrix acdc_lv "${ACDC_DATASET_ID}" 3d_fullres \
        nnUNetTrainer_1000epochs nnunet_3d_fullres
    ;;
  *)
    echo "Usage: $0 {verify|prepare|m16|m17_riga|m17_acdc|m17}" >&2
    exit 1
    ;;
esac
