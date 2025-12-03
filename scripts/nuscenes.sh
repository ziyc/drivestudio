#!/usr/bin/env bash
# Run NuScenes preprocessing inside the drivestudio:nuscenes Docker image.
# Usage: scripts/nuscenes.sh /abs/path/to/nuscenes/raw [--split v1.0-mini] [--num-scenes 10] [--start-idx 0] [--interpolate 4] [--workers 32] [--image drivestudio:nuscenes] [--cpu] [--no-checkpoint-download] [--no-humanpose] [--no-dynamic-mask]
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/nuscenes.sh /abs/path/to/nuscenes/raw [options]

Required:
  /abs/path/to/nuscenes/raw   Directory containing NuScenes raw data (v1.0-mini / v1.0-trainval / v1.0-test)

Options:
  --split <name>              NuScenes split (default: v1.0-mini)
  --num-scenes <n>            Number of scenes to process (default: 10)
  --start-idx <n>             Start index for scenes (default: 0)
  --interpolate <n>           Interpolation factor N (0-4) for 2Hz->(N+1)*2Hz (default: 4 -> 10Hz)
  --workers <n>               Workers for preprocessing (default: 32)
  --image <tag>               Docker image to use (default: drivestudio:nuscenes)
  --cpu                       Run container without --gpus all
  --no-checkpoint-download    Skip SegFormer checkpoint download (expects file at /opt/segformer/pretrained/segformer.b5.1024x1024.city.160k.pth)
  --no-humanpose              Skip downloading preprocessed human pose zip
  --no-dynamic-mask           Skip fine dynamic mask extraction with SegFormer
  -h, --help                  Show this help
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

RAW_PATH=""
SPLIT="v1.0-mini"
NUM_SCENES=10
START_IDX=0
INTERPOLATE=4
WORKERS=32
IMAGE="drivestudio:nuscenes"
GPU_FLAG="--gpus all"
DOWNLOAD_CHECKPOINT=1
DOWNLOAD_HUMANPOSE=1
PROCESS_DYNAMIC_MASK=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2;;
    --num-scenes) NUM_SCENES="$2"; shift 2;;
    --start-idx) START_IDX="$2"; shift 2;;
    --interpolate) INTERPOLATE="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --cpu) GPU_FLAG=""; shift;;
    --no-checkpoint-download) DOWNLOAD_CHECKPOINT=0; shift;;
    --no-humanpose) DOWNLOAD_HUMANPOSE=0; shift;;
    --no-dynamic-mask) PROCESS_DYNAMIC_MASK=0; shift;;
    -h|--help) usage; exit 0;;
    *)
      if [[ -z "${RAW_PATH}" ]]; then
        RAW_PATH="$1"; shift
      else
        echo "Unexpected argument: $1" >&2
        usage; exit 1
      fi
      ;;
  esac
done

if [[ -z "${RAW_PATH}" ]]; then
  echo "Missing NuScenes raw path." >&2
  usage
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not found in PATH." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_REALPATH="$(python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "${RAW_PATH}")"

if [[ ! -d "${RAW_REALPATH}" ]]; then
  echo "NuScenes raw path does not exist: ${RAW_REALPATH}" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/data/nuscenes"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Building Docker image ${IMAGE} ..."
  (cd "${REPO_ROOT}" && docker buildx bake nuscenes)
fi

echo "Running NuScenes preprocessing inside container..."

docker run ${GPU_FLAG} --rm -it \
  -v "${REPO_ROOT}:/workspace/drivestudio" \
  -v "${RAW_REALPATH}:/workspace/drivestudio/data/nuscenes/raw" \
  -e NUSC_SPLIT="${SPLIT}" \
  -e NUSC_NUM_SCENES="${NUM_SCENES}" \
  -e NUSC_START_IDX="${START_IDX}" \
  -e NUSC_INTERPOLATE="${INTERPOLATE}" \
  -e NUSC_WORKERS="${WORKERS}" \
  -e NUSC_DOWNLOAD_CHECKPOINT="${DOWNLOAD_CHECKPOINT}" \
  -e NUSC_DOWNLOAD_HUMANPOSE="${DOWNLOAD_HUMANPOSE}" \
  -e NUSC_PROCESS_DYNAMIC_MASK="${PROCESS_DYNAMIC_MASK}" \
  "${IMAGE}" bash -c '
    set -euo pipefail
    RAW=/workspace/drivestudio/data/nuscenes/raw
    TARGET=/workspace/drivestudio/data/nuscenes/processed
    SPLIT="${NUSC_SPLIT}"
    START_IDX="${NUSC_START_IDX}"
    NUM_SCENES="${NUSC_NUM_SCENES}"
    INTERPOLATE="${NUSC_INTERPOLATE}"
    WORKERS="${NUSC_WORKERS}"
    DOWNLOAD_CHECKPOINT="${NUSC_DOWNLOAD_CHECKPOINT}"
    DOWNLOAD_HUMANPOSE="${NUSC_DOWNLOAD_HUMANPOSE}"
    PROCESS_DYNAMIC_MASK="${NUSC_PROCESS_DYNAMIC_MASK}"

    SEGFORMER_ROOT=/opt/segformer
    CKPT_PATH=${SEGFORMER_ROOT}/pretrained/segformer.b5.1024x1024.city.160k.pth
    CKPT_ID=1e7DECAH0TRtPZM6hTqRGoboq1XPqSmuj
    HUMANPOSE_ID=1Z0gJVRtPnjvusQVaW7ghZnwfycZStCZx

    echo "[1/3] Preprocessing raw NuScenes..."
    python3 datasets/preprocess.py \
      --data_root "${RAW}" \
      --target_dir "${TARGET}" \
      --dataset nuscenes \
      --split "${SPLIT}" \
      --start_idx "${START_IDX}" \
      --num_scenes "${NUM_SCENES}" \
      --interpolate_N "${INTERPOLATE}" \
      --workers "${WORKERS}" \
      --process_keys images lidar calib dynamic_masks objects

    FREQ=$(( (INTERPOLATE + 1) * 2 ))
    PROCESSED_BASE="${TARGET/processed/processed_${FREQ}Hz}"
    SPLIT_LEAF="${SPLIT##*-}"
    MASK_DATA_ROOT="${PROCESSED_BASE}/${SPLIT_LEAF}"

    if [[ "${DOWNLOAD_CHECKPOINT}" == "1" && ! -f "${CKPT_PATH}" ]]; then
      echo "[2/3] Downloading SegFormer checkpoint..."
      mkdir -p "$(dirname "${CKPT_PATH}")"
      conda run -n segformer gdown "${CKPT_ID}" -O "${CKPT_PATH}"
    fi

    if [[ ! -f "${CKPT_PATH}" ]]; then
      echo "SegFormer checkpoint missing at ${CKPT_PATH}. Download or specify --no-checkpoint-download only if already present." >&2
      exit 1
    fi

    echo "[2/3] Extracting sky/fine dynamic masks with SegFormer..."
    MASK_ARGS=(
      --data_root "${MASK_DATA_ROOT}"
      --segformer_path "${SEGFORMER_ROOT}"
      --checkpoint "${CKPT_PATH}"
      --start_idx "${START_IDX}"
      --num_scenes "${NUM_SCENES}"
    )
    if [[ "${PROCESS_DYNAMIC_MASK}" == "1" ]]; then
      MASK_ARGS+=(--process_dynamic_mask)
    fi
    conda run -n segformer python datasets/tools/extract_masks.py "${MASK_ARGS[@]}"

    HUMANPOSE_MARKER=""
    if [[ -d "${PROCESSED_BASE}/${SPLIT_LEAF}" ]]; then
      HUMANPOSE_MARKER="$(find "${PROCESSED_BASE}/${SPLIT_LEAF}" -maxdepth 2 -type d -name humanpose | head -n 1 || true)"
    fi
    if [[ "${DOWNLOAD_HUMANPOSE}" == "1" ]]; then
      if [[ -d "${HUMANPOSE_MARKER}" ]]; then
        echo "[3/3] Human pose data already present at ${HUMANPOSE_MARKER}, skipping download."
      else
        echo "[3/3] Downloading preprocessed human pose data..."
        HP_ZIP=/workspace/drivestudio/data/nuscenes_preprocess_humanpose.zip
        conda run -n segformer gdown "${HUMANPOSE_ID}" -O "${HP_ZIP}"
        python3 - <<PY
import os, zipfile
zip_path = r'"'"'${HP_ZIP}'"'"'
extract_dir = r'"'"'/workspace/drivestudio/data'"'"'
with zipfile.ZipFile(zip_path, '"'"'r'"'"') as zf:
    zf.extractall(extract_dir)
os.remove(zip_path)
print("Human pose data extracted to", extract_dir)
PY
      fi
    else
      echo "[3/3] Skipping human pose download."
    fi

    echo "NuScenes preprocessing finished. Processed data at ${MASK_DATA_ROOT}"
  '
