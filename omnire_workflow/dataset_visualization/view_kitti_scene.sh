#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VIZ_ENV_NAME="${VIZ_ENV_NAME:-waymo-kitti-viz}"
KITTI_ROOT="${KITTI_ROOT:-OutPut/waymo_training_10scenes/scene_114/dataset_visualization/kitti}"
CAMERA_ID="${CAMERA_ID:-0}"
CLASSES="${CLASSES:-Car}"
START_FRAME="${START_FRAME:-0}"
END_FRAME="${END_FRAME:--1}"

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda was not found in PATH." >&2
  exit 1
fi

conda run -n "$VIZ_ENV_NAME" python -m omnire_workflow.dataset_visualization.cli view \
  --kitti_root "$KITTI_ROOT" \
  --camera_id "$CAMERA_ID" \
  --classes "$CLASSES" \
  --start_frame "$START_FRAME" \
  --end_frame "$END_FRAME"
