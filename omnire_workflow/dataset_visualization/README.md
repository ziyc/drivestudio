# Waymo dataset visualization flow

This folder reproduces the dataset-inspection workflow from `OmniRe复现.docx`: convert a Waymo tfrecord into KITTI-like files, then inspect camera images, labels, and LiDAR points with the modified 3D Detection & Tracking Viewer.

For a consolidated command index, see `omnire_workflow/COMMANDS.md`.

Core workflow logic lives in the Python package:

```bat
python -m omnire_workflow.dataset_visualization.cli --help
```

The `.bat` files in this folder are thin Windows launchers for convenience. `setup_visualization_env.bat` is the only substantial batch file because conda environment creation is most reliable from the shell.

The external tools are packaged under:

- `external_tools/waymo_kitti_converter`
- `external_tools/3D-Detection-Tracking-Viewer`

The viewer copy keeps the fixes already made during debugging:

- uses `image_0` and `label_0` for Waymo front camera output
- passes real image shape into LiDAR FOV filtering instead of KITTI's hard-coded `374 x 1241`
- uses `np.asmatrix` in places that previously broke under NumPy 2.x
- guards empty scatter colors and 3/4-channel color assignment in `viewer.py`

## 1. Create the visualization environment

On Linux cloud servers:

```bash
bash omnire_workflow/dataset_visualization/setup_visualization_env.sh
```

Windows fallback:

```bat
omnire_workflow\dataset_visualization\setup_visualization_env.bat
```

If proxy settings break TensorFlow or Waymo package downloads, close the proxy for this shell first:

```bat
set http_proxy=
set https_proxy=
```

## 2. Convert Waymo tfrecords to KITTI-like layout

Default single-scene input/output target the problematic `scene_idx=114` tfrecord discussed in the reproduction notes, colocated with the reconstruction run:

```text
data\waymo\raw\train_segment-12505030131868863688_1740_000_1760_000_with_camera_labels.tfrecord
OutPut\waymo_training_10scenes\scene_114\dataset_visualization\kitti
```

Run:

```bat
omnire_workflow\dataset_visualization\convert_waymo_to_kitti.bat
```

Common override for a single-scene comparison folder:

```bat
set TFRECORD_FILE=data\waymo\raw\train_segment-12505030131868863688_1740_000_1760_000_with_camera_labels.tfrecord
set KITTI_OUT=OutPut\waymo_training_10scenes\scene_114\dataset_visualization\kitti
set NUM_PROC=1
omnire_workflow\dataset_visualization\convert_waymo_to_kitti.bat
```

To convert all ten local scenes into reconstruction-adjacent folders, run:

```bat
omnire_workflow\dataset_visualization\convert_waymo_10scenes_to_kitti.bat
```

Equivalent Python package command:

```bat
conda run -n waymo-kitti-viz python -m omnire_workflow.dataset_visualization.cli convert-ten
```

Linux shell wrapper:

```bash
bash omnire_workflow/dataset_visualization/convert_waymo_10scenes_to_kitti.sh
```

To force regeneration of existing converted folders:

```bat
set OVERWRITE=1
omnire_workflow\dataset_visualization\convert_waymo_10scenes_to_kitti.bat
```

The output layout is:

```text
OutPut\waymo_training_10scenes\scene_0\dataset_visualization\kitti
OutPut\waymo_training_10scenes\scene_1\dataset_visualization\kitti
...
OutPut\waymo_training_10scenes\scene_156\dataset_visualization\kitti
```

Expected converted structure:

```text
calib\
image_0\ ... image_4\
label_0\ ... label_4\
label_all\
pose\
velodyne\
```

## 3. View images, labels, and LiDAR

Before opening the GUI, check file alignment:

```bat
set KITTI_ROOT=OutPut\waymo_training_10scenes\scene_114\dataset_visualization\kitti
set CAMERA_ID=0
omnire_workflow\dataset_visualization\check_kitti_io.bat
```

To check all ten converted scene folders:

```bat
omnire_workflow\dataset_visualization\check_waymo_10scenes_io.bat
```

Equivalent Python package command:

```bat
python -m omnire_workflow.dataset_visualization.cli check-ten
```

Linux shell wrapper:

```bash
bash omnire_workflow/dataset_visualization/check_waymo_10scenes_io.sh
```

```bat
set KITTI_ROOT=OutPut\waymo_training_10scenes\scene_114\dataset_visualization\kitti
set CAMERA_ID=0
set CLASSES=Car,Pedestrian,Cyclist
set START_FRAME=0
set END_FRAME=80
omnire_workflow\dataset_visualization\view_kitti_scene.bat
```

Or use the scene-index wrapper:

```bat
set SCENE_IDX=114
set END_FRAME=80
omnire_workflow\dataset_visualization\view_waymo_scene.bat
```

Equivalent Python package command:

```bat
conda run -n waymo-kitti-viz python -m omnire_workflow.dataset_visualization.cli view-scene --scene_idx 114 --end_frame 80
```

Linux shell wrapper:

```bash
SCENE_IDX=114 END_FRAME=80 bash omnire_workflow/dataset_visualization/view_waymo_scene.sh
```

Controls: focus the VTK/Vedo window, then press `Q`, `Enter`, or `Esc` to advance frames.

## Comparison use

For the green-belt/tree reconstruction issue, inspect the same frame range in:

1. raw converted LiDAR/image/labels with this viewer
2. DriveStudio preprocessed outputs under `data/waymo/processed/training/<scene_id>`
3. OmniRe render videos under `OutPut/.../videos`

If the converted LiDAR already lacks returns or has strong FOV filtering artifacts around the greenery, the issue is likely upstream data/pose/point-cloud related. If converted LiDAR and labels look normal while OmniRe output fails, focus on mask quality, sky/dynamic masks, AABB coverage, and Gaussian initialization/optimization for that scene.
