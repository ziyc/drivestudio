# Preparing Waymo Dataset
## 1. Register on Waymo Open Dataset

#### Sign Up for a Waymo Open Dataset Account and Install gcloud SDK

To download the Waymo dataset, you need to register an account at [Waymo Open Dataset](https://waymo.com/open/). You also need to install gcloud SDK and authenticate your account. Please refer to [this page](https://cloud.google.com/sdk/docs/install) for more details.

#### Set Up the Data Directory

Once you've registered and installed the gcloud SDK, create a directory to house the raw data:

```shell
# Create the data directory or create a symbolic link to the data directory
mkdir -p ./data/waymo/raw   
mkdir -p ./data/waymo/processed 
```

## 2. Download the raw data
For the Waymo Open Dataset, we first organize the scene names alphabetically and store them in `data/waymo_train_list.txt`. The scene index is then determined by the line number minus one.

For example, to obtain the 23th, 114th, and 788th scenes from the Waymo Open Dataset, execute:

```shell
python datasets/waymo/waymo_download.py \
    --target_dir ./data/waymo/raw \
    --scene_ids 23 114 327 621 703 172 552 788
```

You can also provide a split file (e.g. `data/waymo_example_scenes.txt`) to download a batch of scenes at once:

```shell
python datasets/waymo/waymo_download.py \
    --target_dir ./data/waymo/raw \
    --split_file data/waymo_example_scenes.txt
```

If you wish to run experiments on different scenes, please specify your own list of scenes.

## 3. Preprocess the data
After downloading the raw dataset, you'll need to preprocess this compressed data to extract and organize various components.

#### Install Waymo Development Toolkit
```shell
pip install waymo-open-dataset-tf-2-11-0==1.6.0
```

#### Running the preprocessing script
To preprocess specific scenes of the dataset, use the following command:
```shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --scene_ids 23 114 327 621 703 172 552 788 \
    --workers 8 \
    --process_keys images lidar calib pose dynamic_masks objects
```
Alternatively, preprocess a batch of scenes by providing the split file:
```shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --split_file data/waymo_example_scenes.txt \
    --workers 8 \
    --process_keys images lidar calib pose dynamic_masks objects
```
The extracted data will be stored in the `data/waymo/processed` directory.

## 4. Extract Masks

To generate:

- **sky masks (required)** 
- fine dynamic masks (optional)

Follow these steps:

#### Install `SegFormer` (Skip if already installed)

:warning: SegFormer relies on `mmcv-full=1.2.7`, which relies on `pytorch=1.8` (pytorch<1.9). Hence, a seperate conda env is required.

```shell
#-- Set conda env
conda create -n segformer python=3.8
conda activate segformer
# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
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

Remember the location where you download into, and pass it to the script in the next step with `--checkpoint` .


#### Run Mask Extraction Script

```shell
conda activate segformer
segformer_path=/pathtosegformer

python datasets/tools/extract_masks.py \
    --data_root data/waymo/processed/training \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --split_file data/waymo_example_scenes.txt \
    --process_dynamic_mask
```
Replace `/pathtosegformer` with the actual path to your Segformer installation.

Note: The `--process_dynamic_mask` flag is included to process fine dynamic masks along with sky masks.

This process will extract the required masks from your processed data.

## 5. Human Body Pose Processing

#### Prerequisites
To utilize the SMPL-Gaussian to model pedestrians, please first download the SMPL models.

1. Download SMPL v1.1 (`SMPL_python_v.1.1.0.zip`) from the [SMPL official website](https://smpl.is.tue.mpg.de/download.php)
2. Move `SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `PROJECT_ROOT/smpl_models/SMPL_NEUTRAL.pkl`

SMPL-Nodes (SMPL-Gaussian Representation) requires Human Body Pose Sequences of pedestrians. We've developed a human body pose processing pipeline for in-the-wild driving data to generate this information. There are two ways to obtain these data:

#### Option 1: Download Preprocessed Human Pose Data

We have uploaded preprocessed human pose data for a subset of Waymo scenes to [Google Drive](https://drive.google.com/drive/folders/187w1rwEZ5i9tb4y-dOJVTnIZAtKPR7_j). You can download and unzip these files without installing any additional environment.

```shell
# https://drive.google.com/file/d/1QrtMrPAQhfSABpfgQWJZA2o_DDamL_7_/view?usp=drive_link
# filename: waymo_preprocess_humanpose.zip
cd data
gdown 1QrtMrPAQhfSABpfgQWJZA2o_DDamL_7_ 

unzip waymo_preprocess_humanpose.zip
rm waymo_preprocess_humanpose.zip
```

#### Option 2: Run the Extraction Pipeline

To process human body poses for other Waymo scenes or to run the processing pipeline yourself, follow the instructions in our [Human Pose Processing Guide](./HumanPose.md).

## 6. Data Structure
After completing all preprocessing steps, the project files should be organized according to the following structure:
```shell
ProjectPath/data/
  └── waymo/
    ├── raw/
    │    ├── segment-454855130179746819_4580_000_4600_000_with_camera_labels.tfrecord
    │    └── ...
    └── processed/
         └──training/
              ├── 001/
              │  ├──images/             # Images: {timestep:03d}_{cam_id}.jpg
              │  ├──lidar/              # LiDAR data: {timestep:03d}.bin
              │  ├──ego_pose/           # Ego vehicle poses: {timestep:03d}.txt
              │  ├──extrinsics/         # Camera extrinsics: {cam_id}.txt
              │  ├──intrinsics/         # Camera intrinsics: {cam_id}.txt
              │  ├──sky_masks/          # Sky masks: {timestep:03d}_{cam_id}.png
              │  ├──dynamic_masks/      # Coarse dynamic masks: category/{timestep:03d}_{cam_id}.png
              │  ├──fine_dynamic_masks/ # (Optional) Fine dynamic masks: category/{timestep:03d}_{cam_id}.png 
              │  ├──instances/          # Instances' bounding boxes information
              │  └──humanpose/          # Preprocessed human body pose: smpl.pkl
              ├── 002/
              ├── ...
```