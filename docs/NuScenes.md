# Preparing NuScenes Dataset

Before downloading or using the NuScenes dataset, please follow these important steps:

1. Visit the [official NuScenes website](https://www.nuscenes.org/).
2. Register for an account if you haven't already done so.
3. Carefully read and agree to the [NuScenes terms of use](https://www.nuscenes.org/terms-of-use).

Ensure you have completed these steps before proceeding with the dataset download and use.

## 1. Download the Raw Data

Download the raw data from the [official NuScenes website](https://www.nuscenes.org/download). Then, create directories for NuScenes data and optionally create a symbolic link if you have the data elsewhere.

```shell
mkdir -p ./data/nuscenes
ln -s $PATH_TO_NUSCENES ./data/nuscenes/raw # ['v1.0-mini', 'v1.0-trainval', 'v1.0-test'] lies in it
```

We'll use the **v1.0-mini split** in our examples. The process is similar for other splits.

## 2. Install the Development Toolkit
```shell
pip install nuscenes-devkit
```

## 3. Process Raw Data

To process the 10 scenes in NuScenes **v1.0-mini split**, you can run:

```shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/nuscenes/raw \
    --target_dir data/nuscenes/processed \
    --dataset nuscenes \
    --split v1.0-mini \
    --start_idx 0 \
    --num_scenes 10 \
    --interpolate_N 4 \
    --workers 32 \
    --process_keys images lidar calib dynamic_masks objects
```

The extracted data will be stored in the `data/nuscenes/processed_10Hz` directory.

`interpolate_N`: Increases frame rate by interpolating between keyframes.

* NuScenes provides synchronized keyframes at `2Hz`. Our script allows interpolation to increase up to `10Hz`.
* `interpolate_N = 4`: Interpolates 4 frames between original synchronized keyframes.
* Result: `10Hz` frame rate `((4 + 1) * 2 Hz)`
* Note: We recommend using `interpolate_N = 4`. While `interpolate_N = 5 (12 Hz)` is possible, it may lead to frame drop issues. Although the camera captures at `12 Hz`, occasional frame misses during recording can cause data gaps at higher interpolation rates.

## 4. Extract Masks

To generate:

- **sky masks (required)** 
- fine dynamic masks (optional)

Follow these steps:

#### Install `SegFormer` (Skip if already installed)

:warning: SegFormer relies on `mmcv-full=1.2.7`, which relies on `pytorch=1.8` (pytorch<1.9). Hence, a separate conda env is required.

```shell
#-- Set conda env
conda create -n segformer python=3.8
conda activate segformer
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#-- Install mmcv-full
pip install timm==0.3.2 pylint debugpy opencv-python-headless attrs ipython tqdm imageio scikit-image omegaconf
pip install mmcv-full==1.2.7 --no-cache-dir

#-- Clone and install segformer
git clone https://github.com/NVlabs/SegFormer
cd SegFormer
pip install .
```

Download the pretrained model `segformer.b5.1024x1024.city.160k.pth` from the google_drive / one_drive links in https://github.com/NVlabs/SegFormer#evaluation .

Remember the location where you download into, and pass it to the script in the next step with `--checkpoint`.

#### Run Mask Extraction Script

```shell
conda activate segformer
segformer_path=/path/to/segformer
split=mini

python datasets/tools/extract_masks.py \
    --data_root data/nuscenes/processed_10Hz/$split \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --start_idx 0 \
    --num_scenes 10 \
    --process_dynamic_mask
```

Replace `/path/to/segformer` with the actual path to your Segformer installation.

Note: The `--process_dynamic_mask` flag is included to process fine dynamic masks along with sky masks.

This process will extract the required masks from your processed data.

## 5. Human Body Pose Processing

SMPL-Nodes (SMPL Gaussian Representation) requires Human Body Pose Sequences of pedestrians. We've developed a human body pose processing pipeline for in-the-wild driving data to generate this information. There are two ways to obtain these data:

#### Option 1: Download Preprocessed Human Pose Data

We have uploaded preprocessed human pose data for **v1.0-mini split** of NuScenes scenes to [Google Drive](https://drive.google.com/drive/folders/187w1rwEZ5i9tb4y-dOJVTnIZAtKPR7_j). You can download and unzip these files without installing any additional environment.

```shell
# https://drive.google.com/file/d/1Z0gJVRtPnjvusQVaW7ghZnwfycZStCZx/view?usp=drive_link
# filename: nuscenes_preprocess_humanpose.zip
cd data
gdown 1Z0gJVRtPnjvusQVaW7ghZnwfycZStCZx

unzip nuscenes_preprocess_humanpose.zip
rm nuscenes_preprocess_humanpose.zip
```

#### Option 2: Run the Extraction Pipeline

To process human body poses for other NuScenes scenes or to run the processing pipeline yourself, follow the instructions in our [Human Pose Processing Guide](./HumanPose.md).

## 6. Data Structure

After completing all preprocessing steps, the project files should be organized according to the following structure:

```shell
ProjectPath/data/
  └── nuscenes/
    ├── raw/
    │    └── [original NuScenes structure]
    └── processed_10Hz/
         └── mini/
              ├── 001/
              │  ├──images/             # Images: {timestep:03d}_{cam_id}.jpg
              │  ├──lidar/              # LiDAR data: {timestep:03d}.bin
              │  ├──lidar_pose/         # Lidar poses: {timestep:03d}.txt
              │  ├──extrinsics/         # Camera extrinsics: {cam_id}.txt
              │  ├──intrinsics/         # Camera intrinsics: {cam_id}.txt
              │  ├──sky_masks/          # Sky masks: {timestep:03d}_{cam_id}.png
              │  ├──dynamic_masks/      # Dynamic masks: {timestep:03d}_{cam_id}.png
              │  ├──fine_dynamic_masks/ # (Optional) Fine dynamic masks: {timestep:03d}_{cam_id}.png
              │  ├──objects/            # Object information
              │  └──humanpose/          # Preprocessed human body pose: smpl.pkl
              ├── 002/
              └── ...
```
