#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VIZ_ENV_NAME="${VIZ_ENV_NAME:-waymo-kitti-viz}"
KITTI_OUT="${KITTI_OUT:-OutPut/waymo_training_10scenes/scene_114/dataset_visualization/kitti}"
NUM_PROC="${NUM_PROC:-1}"
PREFIX="${PREFIX:-}"
CAMERA_ID="${CAMERA_ID:-0}"

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

extra_args=()
if [[ "${CONVERT_ALL:-0}" == "1" ]]; then
  extra_args+=(--convert_all)
fi
if [[ -n "${TFRECORD_FILE:-}" ]]; then
  extra_args+=(--tfrecord_file "$TFRECORD_FILE")
fi
if [[ -n "${RAW_DIR:-}" ]]; then
  extra_args+=(--raw_dir "$RAW_DIR")
fi

conda run -n "$VIZ_ENV_NAME" python -m omnire_workflow.dataset_visualization.cli convert-one \
  --kitti_out "$KITTI_OUT" \
  --prefix "$PREFIX" \
  --num_proc "$NUM_PROC" \
  --camera_id "$CAMERA_ID" \
  "${extra_args[@]}"
