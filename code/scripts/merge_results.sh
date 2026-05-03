#!/bin/bash
# Merge result directories from user-provided rsync sources into one local
# destination. Used post-experiment to feed compute_all.py.
#
# Usage:
#   FMPOOL_REMOTE_RESULTS_SOURCES="hostA:/path/to/results hostB:/path/to/results" \
#     bash scripts/merge_results.sh [DEST_DIR]
#
# Default DEST_DIR: results/_merged

set -uo pipefail

DEST="${1:-results/_merged}"
mkdir -p "$DEST"

SSH_OPTS="-o ProxyJump=none -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=30"
REMOTE_RESULTS_SOURCES="${FMPOOL_REMOTE_RESULTS_SOURCES:-}"

if [[ -z "${REMOTE_RESULTS_SOURCES}" ]]; then
    echo "Set FMPOOL_REMOTE_RESULTS_SOURCES to one or more host:/path/to/results roots." >&2
    exit 2
fi

merge_source() {
    local src_root="$1"
    echo "=== ${src_root} ==="
    # Per-case Dice (CORE + M2 + M5 + M16/17/18 etc.)
    rsync -avz --rsync-path='rsync' -e "ssh $SSH_OPTS" \
        "${src_root}/per_case_dice/" \
        "${DEST}/per_case_dice/" 2>&1 | tail -5

    # Adaptation factorial (M6-M11)
    rsync -avz --rsync-path='rsync' -e "ssh $SSH_OPTS" \
        "${src_root}/per_case_dice_adapted/" \
        "${DEST}/per_case_dice_adapted/" 2>&1 | tail -5 || true

    # 3D segmentation-architecture pool (M21)
    rsync -avz --rsync-path='rsync' -e "ssh $SSH_OPTS" \
        "${src_root}/per_case_dice_3d/" \
        "${DEST}/per_case_dice_3d/" 2>&1 | tail -5 || true

    # nnU-Net outputs (M16/M17)
    rsync -avz --rsync-path='rsync' -e "ssh $SSH_OPTS" \
        "${src_root}/nnunet/" \
        "${DEST}/nnunet/" 2>&1 | tail -5 || true

    # Tier-3 matrix outputs (best-effort; skip on missing)
    for sub in per_case_dice_m1 per_case_dice_m3 per_case_dice_m14 \
               per_case_dice_m15 per_case_dice_m19 per_case_dice_m22 \
               m12_bootstrap; do
        rsync -avz --rsync-path='rsync' -e "ssh $SSH_OPTS" \
            "${src_root}/${sub}/" \
            "${DEST}/${sub}/" 2>&1 | tail -3 || true
    done

    echo
}

for src in ${REMOTE_RESULTS_SOURCES}; do
    merge_source "${src}"
done

echo "=== merge complete: $DEST ==="
echo "per_case_dice JSON count:"
find "${DEST}/per_case_dice" -name "seed_*.json" 2>/dev/null | wc -l
echo "per_case_dice_adapted JSON count:"
find "${DEST}/per_case_dice_adapted" -name "seed_*.json" 2>/dev/null | wc -l
