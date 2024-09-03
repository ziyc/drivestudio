# Preparing PandaSet

Before downloading or using the Pandaset dataset, carefully read and agree to [Pandaset's terms of use and licensing requirements](https://scale.com/legal/pandaset-terms-of-use).

For more information, visit the [PandaSet website](https://pandaset.org/)

## 1. Download and Organize Dataset
Due to reported [issues](https://github.com/scaleapi/pandaset-devkit/issues/151) with the [official PandaSet download link](https://pandaset.org/), we recommend an alternative download method via Kaggle. However, please note that the original data and all usage rights are provided by [PandaSet](https://pandaset.org/).

1. Install the Kaggle API:

   If you haven't already, install the Kaggle API. You can do this using pip:

   ```shell
   pip install kaggle
   ```
   NOTE: Ensure you've set up your **Kaggle API credentials** as per their [instructions](https://www.kaggle.com/docs/api).

2. Download the dataset:

   Run the following command in your terminal:

   ```shell
   kaggle datasets download pz19930809/pandaset
   ```

3. Organize the dataset:

   After downloading, organize the files with these commands:

   ```shell
   # Create the data directory or create a symbolic link to the data directory
   mkdir -p ./data/pandaset/raw
   mv pandaset-dataset.zip ./data/pandaset/raw
   cd ./data/pandaset/raw
   unzip pandaset-dataset.zip
   rm pandaset-dataset.zip
   ```

## 2. Install PandaSet Development Toolkit

Install the modified PandaSet development toolkit:

```shell
git clone https://github.com/ziyc/pandaset-devkit.git
cd pandaset-devkit/python
pip install -e .
```

NOTE: This fork of the [original devkit](https://github.com/scaleapi/pandaset-devkit) addresses file name mismatches for compatibility with our scripts.

## 3. Process Raw Data

You can provide a split file (e.g. `data/pandaset_example_scenes.txt`) to process a batch of scenes at once:

```shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/pandaset/raw \
    --target_dir data/pandaset/processed \
    --dataset pandaset \
    --split_file data/pandaset_example_scenes.txt \
    --workers 32 \
    --process_keys images lidar calib pose dynamic_masks objects
```

The extracted data will be stored in the `data/pandaset/processed` directory.

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
    --data_root data/pandaset/processed \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --split_file data/pandaset_example_scenes.txt \
    --process_dynamic_mask
```

Replace `/path/to/segformer` with the actual path to your Segformer installation.

Note: The `--process_dynamic_mask` flag is included to process fine dynamic masks along with sky masks.

This process will extract the required masks from your processed data.

## 5. Human Body Pose Processing

SMPL-Nodes (SMPL Gaussian Representation) requires Human Body Pose Sequences of #### Prerequisites
To utilize the SMPL-Gaussian to model pedestrians, please first download the SMPL models.

1. Download SMPL v1.1 (`SMPL_python_v.1.1.0.zip`) from the [SMPL official website](https://smpl.is.tue.mpg.de/download.php)
2. Move `SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `PROJECT_ROOT/smpl_models/SMPL_NEUTRAL.pkl`

SMPL-Nodes (SMPL-Gaussian Representation) requires Human Body Pose Sequences of pedestrians. We've developed a human body pose processing pipeline for in-the-wild driving data to generate this information. There are two ways to obtain these data:

#### Option 1: Download Preprocessed Human Pose Data

We have uploaded preprocessed human pose data for a subset of PandaSet scenes to [Google Drive](https://drive.google.com/drive/folders/187w1rwEZ5i9tb4y-dOJVTnIZAtKPR7_j). You can download and unzip these files without installing any additional environment.

```shell
# https://drive.google.com/file/d/1ODzoH7SxNzjOThhKUc_n2LLXcOAeGMCM/view?usp=drive_link
# filename: pandaset_preprocess_humanpose.zip
cd data
gdown 1ODzoH7SxNzjOThhKUc_n2LLXcOAeGMCM

unzip pandaset_preprocess_humanpose.zip
rm pandaset_preprocess_humanpose.zip
```

#### Option 2: Run the Extraction Pipeline

To process human body poses for other PandaSet scenes or to run the processing pipeline yourself, follow the instructions in our [Human Pose Processing Guide](./HumanPose.md).

## 6. Data Structure

After completing all preprocessing steps, the project files should be organized according to the following structure:

```
ProjectPath/data/
  └── pandaset/
    ├── raw/
    │    └── [original PandaSet structure]
    └── processed/
         ├── 001/
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
         ├── 002/
         └── ...
```
