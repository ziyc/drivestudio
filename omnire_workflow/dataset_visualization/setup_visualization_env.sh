#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VIZ_ENV_NAME="${VIZ_ENV_NAME:-waymo-kitti-viz}"

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

if ! conda env list | awk '{print $1}' | grep -qx "$VIZ_ENV_NAME"; then
  conda create -n "$VIZ_ENV_NAME" python=3.8 -y
else
  echo "Conda env already exists: $VIZ_ENV_NAME"
fi

conda run -n "$VIZ_ENV_NAME" python -m pip install --upgrade pip
conda run -n "$VIZ_ENV_NAME" pip install \
  tensorflow==2.11.* \
  waymo-open-dataset-tf-2-11-0==1.6.0 \
  opencv-python \
  tqdm \
  matplotlib

conda run -n "$VIZ_ENV_NAME" pip install \
  numpy==1.21.3 \
  vedo==2021.0.6 \
  vtk==9.0.3 \
  opencv-python==4.5.4.58 \
  matplotlib==3.4.3

echo "Visualization environment is ready: $VIZ_ENV_NAME"
