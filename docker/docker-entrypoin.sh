#!/bin/bash
set -e

# GPU info (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "    ✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1 | sed 's/^/    GPU: /'
else
    echo "    ⚠️  NVIDIA GPU not detected"
fi

# Optional EGL/Xvfb bootstrap
if [ "${EGL_MODE}" = "1" ]; then
    echo "  - Starting virtual display (Xvfb) on :99 with EGL support"
    Xvfb ${DISPLAY:-:99} -screen 0 1024x768x24 -ac +extension GLX +render -noreset > /dev/null 2>&1 &
    export EGL_DEVICE_ID=${EGL_DEVICE_ID:-0}
    export __GLX_VENDOR_LIBRARY_NAME=${__GLX_VENDOR_LIBRARY_NAME:-nvidia}
    export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-0}
fi

# Always try to link SMPL assets if available (no nosmplx path)
PROJECT_ROOT="/workspace/drivestudio"
SMPL_NFS_PATH="${NFS_ASSETS_PATH:-/nfs/assets}/smpl/SMPL_NEUTRAL.pkl"

if [ -f "$SMPL_NFS_PATH" ]; then
    echo "  - Linking SMPL model from $SMPL_NFS_PATH..."
    CACHE_SMPL_DIR="$HOME/.cache/4DHumans/data/smpl"
    mkdir -p "$CACHE_SMPL_DIR" "$PROJECT_ROOT/data" "$PROJECT_ROOT/third_party/Humans4D/data/smpl"
    ln -sfn "$SMPL_NFS_PATH" "$CACHE_SMPL_DIR/SMPL_NEUTRAL.pkl"
    ln -sfn "$SMPL_NFS_PATH" "$CACHE_SMPL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"

    ln -sfn "$SMPL_NFS_PATH" "$PROJECT_ROOT/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    ln -sfn "$SMPL_NFS_PATH" "$PROJECT_ROOT/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"

    ln -sfn "$SMPL_NFS_PATH" "$PROJECT_ROOT/third_party/Humans4D/data/smpl/SMPL_NEUTRAL.pkl"
else
    echo "  - ⚠️  WARNING: SMPL model not found at '$SMPL_NFS_PATH'."
    echo "    Please check NFS_ASSETS_PATH env var if you expect SMPL assets."
fi

exec "$@"
