# Preparing Argoverse2 Dataset

ArgoVerse2 is a large-scale dataset for autonomous driving research. Before using the dataset, please carefully read and comply with the [ArgoVerse dataset Terms of Use](https://www.argoverse.org/about.html#terms-of-use).

For more information, visit the [ArgoVerse2 webpage](https://www.argoverse.org/av2.html).

## 1. Install the Environment
To install the development toolkit, we follow the [official setup instructions](https://argoverse.github.io/user-guide/getting_started.html#installing-via-pip). Note that the installation process requires manually installing Rust via rustup before proceeding with the PyPI installation.
1. Install Rust via rustup:
   ```shell
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   export PATH=$HOME/.cargo/bin:$PATH
   rustup default nightly
   ```

2. Install our modified av2 API:
   ```shell
   pip install git+https://github.com/ziyc/av2-api
   ```
   Note: This is a modified API forked from the [original av2-api](https://github.com/argoverse/av2-api). Our version allows loading tracking IDs of objects in the sensor dataset.

## 2. Download the raw data

1. Install s5cmd following the [instructions](https://argoverse.github.io/user-guide/getting_started.html#installing-s5cmd).

2. Set Up the Data Directory

   ```shell
   # Create the data directory or create a symbolic link to the data directory
   mkdir -p ./data/argoverse/raw  
   mkdir -p ./data/argoverse/processed
   ```

3. Download the dataset:

   ```sh
    # Set the name of the dataset subset you want to download
    export DATASET_NAME="sensor"
   
    # Set the target directory where you want to save the dataset
    export TARGET_DIR="data/argoverse/raw"
   
    # Create the target directory if it doesn't exist
    mkdir -p $TARGET_DIR
   
    # Download the dataset using s5cmd
    s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" $TARGET_DIR
   ```

## 3. Process the Data

After downloading the raw dataset, you'll need to preprocess these data to our desired format. For the ArgoVerse2 Dataset, we first organize the scene names alphabetically and store them in `data/argoverse_train_list.txt`. The scene index is then determined by the line number minus one.

#### Running the preprocessing script

You can provide a split file (e.g. `data/argoverse_example_scenes.txt`) to process a batch of scenes at once:

```shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/argoverse/raw \
    --target_dir data/argoverse/processed/training \
    --dataset argoverse \
    --split_file data/argoverse_example_scenes.txt \
    --workers 64 \
    --process_keys images lidar calib pose dynamic_masks objects
```

You can also process a specific range of scenes:

```shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/argoverse/raw \
    --target_dir data/argoverse/processed/training \
    --dataset argoverse \
    --start_idx 0 \
    --num_scenes 50 \
    --workers 64 \
    --process_keys images lidar calib pose dynamic_masks objects
```

The extracted data will be stored in the `data/argoverse/processed` directory.

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

 python datasets/tools/extract_masks.py \
    --data_root data/argoverse/processed/training \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --split_file data/argoverse_example_scenes.txt \
    --process_dynamic_mask
```

Replace `/path/to/segformer` with the actual path to your Segformer installation.

Note: The `--process_dynamic_mask` flag is included to process fine dynamic masks along with sky masks.

This process will extract the required masks from your processed data.

## 5. Human Body Pose Processing

SMPL-Nodes (SMPL Gaussian Representation) requires Human Body Pose Sequences of pedestrians. We've developed a human body pose processing pipeline for in-the-wild driving data to generate this information. There are two ways to obtain these data:

#### Option 1: Download Preprocessed Human Pose Data

We have uploaded preprocessed human pose data for a subset of Argoverse2 scenes to [Google Drive](https://drive.google.com/drive/folders/187w1rwEZ5i9tb4y-dOJVTnIZAtKPR7_j). You can download and unzip these files without installing any additional environment.

```shell
# https://drive.google.com/file/d/1XbYannJpQ9SRAL1-49XDLL833wqQolDd/view?usp=drive_link
# filename: argoverse_preprocess_humanpose.zip
cd data
gdown 1XbYannJpQ9SRAL1-49XDLL833wqQolDd

unzip argoverse_preprocess_humanpose.zip
rm argoverse_preprocess_humanpose.zip
```

#### Option 2: Run the Extraction Pipeline

To process human body poses for other ArgoVerse2 scenes or to run the processing pipeline yourself, follow the instructions in our [Human Pose Processing Guide](./HumanPose.md).

## 6. Data Structure

After completing all preprocessing steps, the project files should be organized according to the following structure:

```shell
ProjectPath/data/
  └── argoverse/
    ├── raw/
    │    └── train/
    │         ├── 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a/
    │         └──....
    └── processed/
         └──training/
              ├── 000/
              │  ├──images/             # Images: {timestep:03d}_{cam_id}.jpg
              │  ├──lidar/              # LiDAR data: {timestep:03d}.bin
              │  ├──ego_pose/           # Ego vehicle poses: {timestep:03d}.txt
              │  ├──extrinsics/         # Camera extrinsics: {cam_id}.txt
              │  ├──intrinsics/         # Camera intrinsics: {cam_id}.txt
              │  ├──sky_masks/          # Sky masks: {timestep:03d}_{cam_id}.png
              │  ├──dynamic_masks/      # Dynamic masks: {timestep:03d}_{cam_id}.png
              │  ├──fine_dynamic_masks/ # (Optional) Fine dynamic masks: {timestep:03d}_{cam_id}.png
              │  ├──objects/            # Object information
              │  └──humanpose/          # Preprocessed human body pose: smpl.pkl
              ├── 001/
              └── ...
```
