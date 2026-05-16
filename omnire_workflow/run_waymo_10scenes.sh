#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-drivestudio}"
SCENE_IDS="${SCENE_IDS:-0 1 4 8 32 102 109 114 149 156}"
WORKERS="${WORKERS:-4}"
START_TIMESTEP="${START_TIMESTEP:-0}"
END_TIMESTEP="${END_TIMESTEP:--1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./OutPut}"
PROJECT="${PROJECT:-waymo_training_10scenes}"
CONFIG_FILE="${CONFIG_FILE:-configs/omnire.yaml}"
DATASET="${DATASET:-waymo/3cams}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

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

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  conda run -n "$ENV_NAME" python datasets/preprocess.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --scene_ids $SCENE_IDS \
    --workers "$WORKERS" \
    --process_keys images lidar calib pose dynamic_masks objects
fi

if [[ -n "${SEGFORMER_PATH:-}" ]]; then
  SEGFORMER_CHECKPOINT="${SEGFORMER_CHECKPOINT:-$SEGFORMER_PATH/pretrained/segformer.b5.1024x1024.city.160k.pth}"
  conda run -n "$ENV_NAME" python datasets/tools/extract_masks.py \
    --data_root data/waymo/processed/training \
    --segformer_path "$SEGFORMER_PATH" \
    --checkpoint "$SEGFORMER_CHECKPOINT" \
    --scene_ids $SCENE_IDS \
    --process_dynamic_mask
else
  echo "[INFO] Skipping SegFormer masks. Set SEGFORMER_PATH to enable mask extraction."
fi

for scene_idx in $SCENE_IDS; do
  run_name="scene_${scene_idx}"
  echo
  echo "===== Training scene ${scene_idx} as ${run_name} ====="
  conda run -n "$ENV_NAME" python tools/train.py \
    --config_file "$CONFIG_FILE" \
    --output_root "$OUTPUT_ROOT" \
    --project "$PROJECT" \
    --run_name "$run_name" \
    dataset="$DATASET" \
    data.scene_idx="$scene_idx" \
    data.start_timestep="$START_TIMESTEP" \
    data.end_timestep="$END_TIMESTEP"
done

echo "All requested scenes finished."
