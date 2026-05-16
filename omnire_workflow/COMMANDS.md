# OmniRe Workflow Commands

This file collects the commands used for OmniRe reproduction, Waymo preprocessing/training, dataset visualization, and offset-trajectory rendering.

For conda environment / image planning, see `omnire_workflow/ENVIRONMENTS.md`.

## Linux Cloud Server

### 1. DriveStudio / OmniRe Environment

```bash
bash omnire_workflow/setup_env.sh
```

Optional:

```bash
ENV_NAME=drivestudio INSTALL_WAYMO=1 bash omnire_workflow/setup_env.sh
```

### 2. Waymo Preprocess And Train Ten Scenes

Default ten scenes:

```text
0 1 4 8 32 102 109 114 149 156
```

Run the full preprocessing and training flow:

```bash
bash omnire_workflow/run_waymo_10scenes.sh
```

Common overrides:

```bash
ENV_NAME=drivestudio \
SCENE_IDS="0 1 4 8 32 102 109 114 149 156" \
WORKERS=4 \
CONFIG_FILE=configs/omnire.yaml \
DATASET=waymo/3cams \
PROJECT=waymo_training_10scenes \
bash omnire_workflow/run_waymo_10scenes.sh
```

Skip preprocessing and only train:

```bash
SKIP_PREPROCESS=1 bash omnire_workflow/run_waymo_10scenes.sh
```

### 3. Original Manual Preprocess Commands

These were previously recorded under `data/waymo/raw/command.txt`.

```bash
export PYTHONPATH=/path/to/project
python datasets/preprocess.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --scene_ids 0 1 4 8 32 \
    --workers 5 \
    --process_keys images lidar calib pose dynamic_masks objects
```

```bash
export PYTHONPATH=/path/to/project
python datasets/preprocess.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --scene_ids 0 1 4 8 \
    --workers 4 \
    --process_keys images lidar calib pose dynamic_masks objects
```

## Dataset Visualization

### 1. Visualization Environment

```bash
bash omnire_workflow/dataset_visualization/setup_visualization_env.sh
```

If proxy settings interfere with TensorFlow/Waymo downloads:

```bash
unset http_proxy
unset https_proxy
```

### 2. Convert Ten Waymo Scenes To KITTI-Like Viewer Data

Output is colocated with reconstruction runs:

```text
OutPut/waymo_training_10scenes/scene_<idx>/dataset_visualization/kitti
```

Run:

```bash
bash omnire_workflow/dataset_visualization/convert_waymo_10scenes_to_kitti.sh
```

Force regeneration:

```bash
OVERWRITE=1 bash omnire_workflow/dataset_visualization/convert_waymo_10scenes_to_kitti.sh
```

Equivalent Python CLI:

```bash
conda run -n waymo-kitti-viz python -m omnire_workflow.dataset_visualization.cli convert-ten
```

### 3. Check Converted Visualization Data

Check all ten scenes:

```bash
bash omnire_workflow/dataset_visualization/check_waymo_10scenes_io.sh
```

Check one scene:

```bash
KITTI_ROOT=OutPut/waymo_training_10scenes/scene_114/dataset_visualization/kitti \
bash omnire_workflow/dataset_visualization/check_kitti_io.sh
```

Equivalent Python CLI:

```bash
python -m omnire_workflow.dataset_visualization.cli check-ten
python -m omnire_workflow.dataset_visualization.cli check \
  --kitti_root OutPut/waymo_training_10scenes/scene_114/dataset_visualization/kitti
```

### 4. Open Viewer

Open problem scene `114`:

```bash
SCENE_IDX=114 END_FRAME=80 \
bash omnire_workflow/dataset_visualization/view_waymo_scene.sh
```

Open all object classes:

```bash
SCENE_IDX=114 CLASSES=ALL END_FRAME=80 \
bash omnire_workflow/dataset_visualization/view_waymo_scene.sh
```

Equivalent Python CLI:

```bash
conda run -n waymo-kitti-viz python -m omnire_workflow.dataset_visualization.cli view-scene \
  --scene_idx 114 \
  --classes Car,Pedestrian,Cyclist \
  --end_frame 80
```

## Offset-Trajectory Rendering

After a scene is reconstructed and has a checkpoint:

```bash
conda run -n drivestudio python omnire_workflow/render_offset_views.py \
  --resume_from OutPut/waymo_training_10scenes/scene_114/checkpoint_final.pth \
  --right_m 1.5 \
  --up_m 0.5 \
  --yaw_deg -5 \
  --frames 150 \
  --fps 24
```

Outputs:

```text
OutPut/waymo_training_10scenes/scene_114/videos_offset/*.mp4
OutPut/waymo_training_10scenes/scene_114/videos_offset/*.npy
```

## Windows Fallback

The `.bat` files mirror the Linux shell wrappers for Windows `cmd`. On the cloud server, prefer the `.sh` scripts or direct Python CLI commands above.

Examples:

```bat
omnire_workflow\setup_env.bat
omnire_workflow\run_waymo_10scenes.bat
omnire_workflow\dataset_visualization\convert_waymo_10scenes_to_kitti.bat
```
