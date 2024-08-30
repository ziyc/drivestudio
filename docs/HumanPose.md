# Human Body Pose Processing Guide
This guide details the process of extracting body poses of pedestrians in various datasets.

## Prerequisites
:warning: To utilize the SMPL-Gaussians to model pedestrians, please first download the SMPL models.

1. Download SMPL v1.1 (`SMPL_python_v.1.1.0.zip`) from the [SMPL official website](https://smpl.is.tue.mpg.de/download.php)
2. Move `SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `PROJECT_ROOT/smpl_models/SMPL_NEUTRAL.pkl`

## Obtaining Human Body Pose Data

### Option 1: Using Pre-processed Data

We provide pre-processed human body pose data for a subset of scenes from various datasets.

| Dataset    | File ID                         | Note                                                           |
|------------|----------------------------------|----------------------------------------------------------------|
| Waymo      | 1QrtMrPAQhfSABpfgQWJZA2o_DDamL_7_ | *Processed SMPL data for 60+ scenes, including all listed in `data/waymo_example_scenes.txt `              |
| PandaSet   | 1ODzoH7SxNzjOThhKUc_n2LLXcOAeGMCM | Scenes listed in `data/pandaset_example_scenes.txt`                                            |
| ArgoVerse2 | 1XbYannJpQ9SRAL1-49XDLL833wqQolDd | Scenes listed in `data/argoverse_example_scenes.txt`           |
| NuScenes   | 1Z0gJVRtPnjvusQVaW7ghZnwfycZStCZx | v1.0-mini split (10 scenes)                                    |
| KITTI      | 1eAMNi5NFMU8T7tjQBT_jzxeX-yJRwVKM | Scenes listed in `data/kitti_example_scenes.txt`               |
| NuPlan     | 1EohZnZCUPDmqsaC1p5WBCDYJi1u9cymt | Scenes listed in `data/nuplan_example_scenes.txt`              |

*We are currently working on processing SMPL data for all Waymo scenes. Stay tuned for updates.

To download:
```shell
pip install gdown  # if not installed
cd data
gdown <gdown_id>
unzip <dataset>_preprocess_humanpose.zip
rm <dataset>_preprocess_humanpose.zip
```

Replace `<gdown_id>` and `<dataset>` with the appropriate values from the table.

For scenes not included in pre-processed files, please use Option 2 to process them.

### Option 2: Run Processing Pipeline

Supported datasets for human body pose extraction:
- [x] Waymo
- [x] PandaSet
- [x] Argoverse
- [x] Nuscene
- [x] KITTI
- [x] NuPlan

**1. Update submodules and set up the environment:**
   ```bash
   # Update submodules
   git submodule update --init --recursive

   # Create and activate the environment
   conda create --name 4D-humans python=3.10 -y
   conda activate 4D-humans

   # Install PyTorch
   pip install torch

   # Install 4D-Humans
   cd third_party/Humans4D
   pip install -e .[all]

   # Install additional dependencies
   pip install git+https://github.com/brjathu/PHALP.git
   pip install git+https://github.com/facebookresearch/pytorch3d.git

   # Return to the project root
   cd ../..
   ```

**2. Ensure dataset preprocessing is complete.** If not, please refer to the preprocessing instructions for each dataset:
   - [Waymo Preprocessing Instructions](./Waymo.md)
   - [PandaSet Preprocessing Instructions](./Pandaset.md)
   - [ArgoVerse2 Preprocessing Instructions](./ArgoVerse.md)
   - [NuScenes Preprocessing Instructions](./NuScenes.md)
   - [KITTI Preprocessing Instructions](./KITTI.md)
   - [NuPlan Preprocessing Instructions](./Nuplan.md)

**3. Run the extraction script:**

   **Waymo**
   ```bash
   conda activate 4D-humans

   python datasets/tools/humanpose_process.py \
   --dataset waymo \
   --data_root data/waymo/processed/training \
   --split_file data/waymo_example_scenes.txt \
   [--save_temp] [--verbose]
   ```

   **PandaSet**
   ```bash
   conda activate 4D-humans

   python datasets/tools/humanpose_process.py \
   --dataset pandaset \
   --data_root data/pandaset/processed \
   --split_file data/pandaset_example_scenes.txt \
   [--save_temp] [--verbose]
   ```

   **ArgoVerse2**
   ```bash
   conda activate 4D-humans

   python datasets/tools/humanpose_process.py \
   --dataset argoverse \
   --data_root data/argoverse/processed/training \
   --split_file data/argoverse_example_scenes.txt \
   [--save_temp] [--verbose]
   ```

   **NuScenes**
   ```bash
   conda activate 4D-humans

   python datasets/tools/humanpose_process.py \
   --dataset nuscenes \
   --data_root data/nuscenes/processed_10Hz/mini \
   --scene_ids 0 1 2 3 4 5 6 7 8 9 \
   [--save_temp] [--verbose]
   ```

   **KITTI**
   ```bash
   conda activate 4D-humans

   python datasets/tools/humanpose_process.py \
   --dataset kitti \
   --data_root data/kitti/processed \
   --split_file data/kitti_example_scenes.txt \
   [--save_temp] [--verbose]
   ```

   **NuPlan**
   ```bash
   conda activate 4D-humans

   python datasets/tools/humanpose_process.py \
   --dataset nuplan \
   --data_root data/nuplan/processed/mini \
   --split_file data/nuplan_example_scenes.txt \
   [--save_temp] [--verbose]
   ```

   - `--save_temp`: Save intermediate results (requires additional storage)
   - `--verbose`: Visualize some results during processing

   Notes:
- Ensure sufficient storage space, especially with `--save_temp` and `--verbose` options.
- Processing time example: ~30 minutes per scene (5 cameras, 200 frames each) on a single RTX 4090 GPU.
- Processing time varies with hardware, number of scenes, cameras, and frames.

**4. Output:** Processed human poses will be saved in each processed scene's `humanpose/` directory.