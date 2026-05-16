#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PROJECT_OUTPUT="${PROJECT_OUTPUT:-OutPut/waymo_training_10scenes}"
MAP_FILE="${MAP_FILE:-omnire_workflow/dataset_visualization/waymo_10scene_map.txt}"
CAMERA_ID="${CAMERA_ID:-0}"

python -m omnire_workflow.dataset_visualization.cli check-ten \
  --project_output "$PROJECT_OUTPUT" \
  --map_file "$MAP_FILE" \
  --camera_id "$CAMERA_ID"
