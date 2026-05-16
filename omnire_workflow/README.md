# OmniRe local workflow

This directory keeps the local reproduction helpers separate from DriveStudio core code.

For a consolidated command index, see `omnire_workflow/COMMANDS.md`.
For conda environment / image planning, see `omnire_workflow/ENVIRONMENTS.md`.

## 1. Configure the environment

On Linux cloud servers:

```bash
bash omnire_workflow/setup_env.sh
```

Windows `cmd` fallback:

```bat
omnire_workflow\setup_env.bat
```

Defaults:

- conda env: `drivestudio`
- installs `requirements.txt`, `gsplat`, `pytorch3d`, `nvdiffrast`, local `third_party\smplx`
- installs `waymo-open-dataset-tf-2-11-0==1.6.0`

Override example:

```bat
set ENV_NAME=drivestudio
set INSTALL_WAYMO=0
omnire_workflow\setup_env.bat
```

## 2. Preprocess and train the ten raw Waymo scenes

The current raw tfrecords map to:

```text
0 1 4 8 32 102 109 114 149 156
```

On Linux:

```bash
bash omnire_workflow/run_waymo_10scenes.sh
```

Windows fallback:

```bat
omnire_workflow\run_waymo_10scenes.bat
```

Useful overrides:

```bat
set ENV_NAME=drivestudio
set SCENE_IDS=0 1 4 8 32 102 109 114 149 156
set END_TIMESTEP=150
set CONFIG_FILE=configs\omnire.yaml
set DATASET=waymo/3cams
set PROJECT=waymo_training_10scenes
omnire_workflow\run_waymo_10scenes.bat
```

Mask extraction is optional in the batch file because SegFormer normally uses a separate environment. To enable it:

```bat
set SEGFORMER_PATH=D:\path\to\SegFormer
set SEGFORMER_CHECKPOINT=D:\path\to\SegFormer\pretrained\segformer.b5.1024x1024.city.160k.pth
omnire_workflow\run_waymo_10scenes.bat
```

## 3. Render a deviated camera trajectory after reconstruction

After a scene has a checkpoint:

```bat
conda activate drivestudio
set PYTHONPATH=%CD%
python omnire_workflow\render_offset_views.py ^
  --resume_from OutPut\waymo_training_10scenes\scene_114\checkpoint_final.pth ^
  --right_m 1.5 ^
  --up_m 0.5 ^
  --yaw_deg -5 ^
  --frames 150 ^
  --fps 24
```

Outputs go to:

```text
<experiment>\videos_offset\*.mp4
<experiment>\videos_offset\*.npy
```

Offset convention:

- `right_m`: camera-local right, meters
- `up_m`: physical up, meters
- `forward_m`: camera-local forward, meters
- `yaw_deg`, `pitch_deg`, `roll_deg`: camera-local rotation offsets
