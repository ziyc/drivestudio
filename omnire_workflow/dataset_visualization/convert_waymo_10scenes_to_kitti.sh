#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VIZ_ENV_NAME="${VIZ_ENV_NAME:-waymo-kitti-viz}"
RAW_DIR="${RAW_DIR:-data/waymo/raw}"
PROJECT_OUTPUT="${PROJECT_OUTPUT:-OutPut/waymo_training_10scenes}"
MAP_FILE="${MAP_FILE:-omnire_workflow/dataset_visualization/waymo_10scene_map.txt}"
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
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  extra_args+=(--overwrite)
fi

conda run -n "$VIZ_ENV_NAME" python -m omnire_workflow.dataset_visualization.cli convert-ten \
  --raw_dir "$RAW_DIR" \
  --project_output "$PROJECT_OUTPUT" \
  --map_file "$MAP_FILE" \
  --prefix "$PREFIX" \
  --num_proc "$NUM_PROC" \
  --camera_id "$CAMERA_ID" \
  "${extra_args[@]}"
