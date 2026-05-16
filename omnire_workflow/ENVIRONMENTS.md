# OmniRe Environment Plan

This project is easier to run with several task-specific conda environments. The dependencies for OmniRe training, SegFormer masks, and Waymo/KITTI visualization are not cleanly compatible in one environment.

## Recommended Environments

### 1. `drivestudio`

Use for:

- DriveStudio / OmniRe training
- Waymo preprocessing with `datasets/preprocess.py`
- checkpoint evaluation with `tools/eval.py`
- offset-trajectory rendering with `omnire_workflow/render_offset_views.py`

Main dependencies:

```text
python=3.9
torch==2.0.0+cu117
torchvision==0.15.0+cu117
xformers==0.0.18
gsplat==v1.3.0
pytorch3d
nvdiffrast
waymo-open-dataset-tf-2-11-0==1.6.0
third_party/smplx editable install
```

Create:

```bash
bash omnire_workflow/setup_env.sh
```

Suggested image name:

```text
drivestudio-omnire-cu117
```

## 2. `segformer`

Use for:

- sky mask extraction
- optional fine dynamic mask extraction
- `datasets/tools/extract_masks.py`

Why separate: SegFormer uses older PyTorch / MMCV dependencies that conflict with the OmniRe training environment.

Main dependencies:

```text
python=3.8
torch==1.8.1+cu111
torchvision==0.9.1+cu111
torchaudio==0.8.1
mmcv-full==1.2.7
timm==0.3.2
opencv-python-headless
imageio
scikit-image
omegaconf
SegFormer local install
```

Typical command:

```bash
conda activate segformer

python datasets/tools/extract_masks.py \
  --data_root data/waymo/processed/training \
  --segformer_path /path/to/SegFormer \
  --checkpoint /path/to/SegFormer/pretrained/segformer.b5.1024x1024.city.160k.pth \
  --scene_ids 0 1 4 8 32 102 109 114 149 156 \
  --process_dynamic_mask
```

Suggested image name:

```text
segformer-mask-cu111
```

## 3. `waymo-kitti-viz`

Use for:

- Waymo tfrecord to KITTI-like conversion
- converted dataset I/O checks
- modified 3D Detection & Tracking Viewer

Main dependencies:

```text
python=3.8
tensorflow==2.11.*
waymo-open-dataset-tf-2-11-0==1.6.0
numpy==1.21.3
vedo==2021.0.6
vtk==9.0.3
opencv-python==4.5.4.58
matplotlib==3.4.3
tqdm
```

Create:

```bash
bash omnire_workflow/dataset_visualization/setup_visualization_env.sh
```

Common commands:

```bash
bash omnire_workflow/dataset_visualization/convert_waymo_10scenes_to_kitti.sh
bash omnire_workflow/dataset_visualization/check_waymo_10scenes_io.sh
SCENE_IDX=114 END_FRAME=80 bash omnire_workflow/dataset_visualization/view_waymo_scene.sh
```

Suggested image name:

```text
waymo-kitti-viz
```

Note: conversion and I/O checks can run headlessly. The interactive viewer needs GUI support, such as X11 forwarding, VNC, NoMachine, a remote desktop, or a future offscreen render/export path.

## Optional: `kitti-viewer-lite`

Use for:

- only opening already-converted KITTI-like data
- avoiding TensorFlow and Waymo toolkit in the viewer-only environment

Main dependencies:

```text
python=3.8
numpy==1.21.3
vedo==2021.0.6
vtk==9.0.3
opencv-python==4.5.4.58
matplotlib==3.4.3
```

Suggested image name:

```text
kitti-viewer-lite
```

## Task Matrix

| Task | Environment |
|---|---|
| OmniRe training | `drivestudio` |
| Waymo raw preprocessing | `drivestudio` |
| SegFormer sky/fine dynamic masks | `segformer` |
| checkpoint evaluation | `drivestudio` |
| original trajectory videos | `drivestudio` |
| offset trajectory videos | `drivestudio` |
| Waymo tfrecord to KITTI-like conversion | `waymo-kitti-viz` |
| KITTI-like I/O checks | any Python env, recommended `waymo-kitti-viz` |
| 3D point cloud / label viewer | `waymo-kitti-viz` or `kitti-viewer-lite` |

## Minimal Image Set

For cloud usage, start with these three:

```text
drivestudio-omnire-cu117
segformer-mask-cu111
waymo-kitti-viz
```

Create `kitti-viewer-lite` only if you want to split GUI viewing away from tfrecord conversion.
