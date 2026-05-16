#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

KITTI_ROOT="${KITTI_ROOT:-OutPut/waymo_training_10scenes/scene_114/dataset_visualization/kitti}"
CAMERA_ID="${CAMERA_ID:-0}"

python -m omnire_workflow.dataset_visualization.cli check \
  --kitti_root "$KITTI_ROOT" \
  --camera_id "$CAMERA_ID"
