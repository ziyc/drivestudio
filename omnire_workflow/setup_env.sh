#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-drivestudio}"
INSTALL_WAYMO="${INSTALL_WAYMO:-1}"

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

git submodule update --init --recursive

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.9 -y
else
  echo "Conda env already exists: $ENV_NAME"
fi

conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" pip install -r requirements.txt
conda run -n "$ENV_NAME" pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
conda run -n "$ENV_NAME" pip install git+https://github.com/facebookresearch/pytorch3d.git
conda run -n "$ENV_NAME" pip install git+https://github.com/NVlabs/nvdiffrast
conda run -n "$ENV_NAME" pip install -e third_party/smplx

if [[ "$INSTALL_WAYMO" == "1" ]]; then
  conda run -n "$ENV_NAME" pip install waymo-open-dataset-tf-2-11-0==1.6.0
fi

echo "Environment setup completed: $ENV_NAME"
