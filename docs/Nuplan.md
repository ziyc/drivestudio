# Preparing the NuPlan Dataset

Before downloading or using the NuPlan dataset, it is crucial to:

1. Register for an account on the [NuPlan website](https://www.nuscenes.org/nuplan).
2. Carefully read and agree to the NuPlan dataset's terms of use.
3. Ensure you comply with all licensing requirements and usage restrictions.

Ensure you have completed these steps before proceeding with the dataset download and use.

## 1. Download Raw Data

To save disk space, we'll use part of the mini split as an example. If you want to process other splits, please download the corresponding files.

To download the first several scenes of the NuPlan mini split, use the following commands:

```shell
# Download Maps
wget https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1/nuplan-maps-v1.0.zip

# Download log databases for the mini split
wget https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1/nuplan-v1.1_mini.zip

# Download mini sensors for NuPlan
wget https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan-v1.1_mini_camera_0.zip
wget https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan-v1.1_mini_lidar_0.zip
```

After downloading, unzip the files and organize them according to the following structure:

```
ProjectPath/data/nuplan/raw/
  ├── maps
  │   ├── nuplan-maps-v1.0.json
  │   ├── sg-one-north
  │   │   └── 9.17.1964
  │   │       └── map.gpkg
  │   ├── us-ma-boston
  │   │   └── 9.12.1817
  │   │       └── map.gpkg
  │   ├── us-nv-las-vegas-strip
  │   │   └── 9.15.1915
  │   │       └── map.gpkg
  │   └── us-pa-pittsburgh-hazelwood
  │       └── 9.17.1937
  │           └── map.gpkg
  └── nuplan-v1.1
      ├── splits
      │     ├── mini
      │     │    ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
      │     │    ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
      │     │    └── ...
      │     └── trainval
      │          └── ...
      └── sensor_blobs
            ├── 2021.05.12.22.00.38_veh-35_01008_01518
            │    ├── CAM_F0
            │    │     ├── c082c104b7ac5a71.jpg
            │    │     ├── af380db4b4ca5d63.jpg
            │    │     └── ...
            │    ├── CAM_B0
            │    ├── CAM_L0
            │    ├── CAM_L1
            │    ├── CAM_L2
            │    ├── CAM_R0
            │    ├── CAM_R1
            │    ├── CAM_R2
            │    └──MergedPointCloud
            │         ├── 03fafcf2c0865668.pcd
            │         ├── 5aee37ce29665f1b.pcd
            │         └── ...
            ├── 2021.06.09.17.23.18_veh-38_00773_01140
            └──  ...
```

**Note**: You may want to create a symbolic link if you prefer to store the data elsewhere.

## 2. Install the Development Toolkit

We recommend creating a new environment for the NuPlan devkit:

```shell
conda create -n nuplan python=3.9
conda activate nuplan
```

Install the [official NuPlan toolkit](https://github.com/motional/nuplan-devkit):

```shell
git clone https://github.com/motional/nuplan-devkit
cd nuplan-devkit
pip install -e .
```

## 3. Process Raw Data
You can provide a split file (e.g. `data/nuplan_example_scenes.txt`) to process a batch of scenes at once:
```shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/nuplan/raw \
    --target_dir data/nuplan/processed \
    --dataset nuplan \
    --split mini \
    --split_file data/nuplan_example_scenes.txt \
    --workers 6 \
    --start_frame_idx 1000 \
    --max_frame_limit 300 \
    --process_keys images lidar pose calib dynamic_masks objects
```
The extracted data will be stored in the `data/nuplan/processed/mini` directory.

**NOTE:** We skip the first `start_frame_idx` (default=1000) frames to avoid ego static frames; the duration of each scene is around 8 mins, resulting in ~5000 frames. We only process the first `max_frame_limit` (default=300) frames.

## 4. Extract Masks

To generate:

- **sky masks (required)** 
- fine dynamic masks (optional)

Follow these steps:

#### Install Data Prep Dependencies

`extract_masks.py` now uses the Hugging Face SegFormer backend directly from the main project environment.

<details>
<summary>Troubleshooting: Hugging Face Model Download</summary>

If the default Hugging Face model download is slow or blocked, pre-download the model with the Hugging Face CLI and keep the same `--hf_model_id` in the command below.
</details>

#### Run Mask Extraction Script

```shell
uv sync --group data

python datasets/tools/extract_masks.py \
    --data_root data/nuplan/processed/mini \
    --split_file data/nuplan_example_scenes.txt \
    --process_dynamic_mask
```

Note: The `--process_dynamic_mask` flag is included to process fine dynamic masks along with sky masks.

This process will extract the required masks from your processed data.

## 5. Human Body Pose Processing

#### Prerequisites
To utilize the SMPL-Gaussian to model pedestrians, please first download the SMPL models.

1. Download SMPL v1.1 (`SMPL_python_v.1.1.0.zip`) from the [SMPL official website](https://smpl.is.tue.mpg.de/download.php)
2. Move `SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `PROJECT_ROOT/smpl_models/SMPL_NEUTRAL.pkl`

SMPL-Nodes (SMPL-Gaussian Representation) requires Human Body Pose Sequences of pedestrians. We've developed a human body pose processing pipeline for in-the-wild driving data to generate this information. There are two ways to obtain these data:

#### Option 1: Download Preprocessed Human Pose Data

We have uploaded preprocessed human pose data for a subset of NuPlan scenes to [Google Drive](https://drive.google.com/drive/folders/187w1rwEZ5i9tb4y-dOJVTnIZAtKPR7_j). You can download and unzip these files without installing any additional environment.

```shell
# https://drive.google.com/file/d/1EohZnZCUPDmqsaC1p5WBCDYJi1u9cymt/view?usp=drive_link
# filename: nuplan_preprocess_humanpose.zip
cd data
gdown 1EohZnZCUPDmqsaC1p5WBCDYJi1u9cymt

unzip nuplan_preprocess_humanpose.zip
rm nuplan_preprocess_humanpose.zip
```

#### Option 2: Run the Extraction Pipeline

To process human body poses for other NuPlan scenes or to run the processing pipeline yourself, follow the instructions in our [Human Pose Processing Guide](./HumanPose.md).

## 6. Data Structure

After completing all preprocessing steps, the project files should be organized according to the following structure:

```
ProjectPath/data/
  └── nuplan/
    ├── raw/
    │    ├── maps/
    │    │   └── ...
    │    └── nuplan-v1.1/
    │        ├── splits/
    │        │   └── ...
    │        └── sensor_blobs/
    │            └── ...
    └── processed/
         └── mini/
              ├── 2021.05.12.22.00.38_veh-35_01008_01518/
              │  ├──images/             # Images: {timestep:03d}_{cam_id}.jpg
              │  ├──lidar/              # LiDAR data: {timestep:03d}.pcd
              │  ├──ego_pose/           # Ego vehicle poses: {timestep:03d}.txt
              │  ├──extrinsics/         # Camera extrinsics: {cam_id}.txt
              │  ├──intrinsics/         # Camera intrinsics: {cam_id}.txt
              │  ├──sky_masks/          # Sky masks: {timestep:03d}_{cam_id}.png
              │  ├──dynamic_masks/      # Dynamic masks: {timestep:03d}_{cam_id}.png
              │  ├──fine_dynamic_masks/ # (Optional) Fine dynamic masks: {timestep:03d}_{cam_id}.png
              │  ├──objects/            # Object information
              │  └──humanpose/          # Preprocessed human body pose: smpl.pkl
              ├── 2021.05.12.22.28.35_veh-35_00620_01164/
              └── ...
```
