# Preparing PhysicalAI Dataset

PhysicalAI is loaded by the regular `DrivingDataset`, using the same raw /
processed directory structure and entry points as the other DriveStudio datasets.
This integration accepts **PhysicalAI Autonomous Vehicles in NCore V4 format**.
The larger original PhysicalAI release must first be converted to NCore with
[NVIDIA's converter](https://github.com/NVIDIA/ncore); it is not the same input format.

The processed scene always contains native FTheta images. In
`configs/datasets/physicalai/7cams.yaml`, `data.pixel_source.undistort: False` selects native
training; set it to `True` to rectify images and masks to pinhole geometry at
load time. Processing never requires a NuRec model, pretrained scene, or
intermediate camera bundle.

Run the commands below from the repository root.

## 1. Register on Hugging Face

#### Sign Up for a Hugging Face Account and Install the CLI

Sign in to Hugging Face and accept the
[PhysicalAI AV NCore dataset agreement](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore).
Authenticate locally using your own read token. The dataset license is separate
from this code's license; do not redistribute downloaded/processed data with the
repository. Permission to publish dataset-derived media is not implied.

```shell
pip install "huggingface_hub>=0.34,<1.0"
hf auth login
```

#### Set Up the Data Directory

```shell
# Create the data directory or create a symbolic link to the data directory
mkdir -p ./data/physicalai/raw
mkdir -p ./data/physicalai/processed
```

## 2. Download the raw data

To download the full seven-camera demo clip, execute:

```shell
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NCore \
    --repo-type dataset \
    --include 'clips/000da9de-0ee5-465a-9a2d-e7e91d3016bb/*' \
    --local-dir data/physicalai/raw
```

For another clip, replace the UUID in `--include`. Leave the `.itar` archives
intact; the NCore SDK reads them directly.

## 3. Preprocess the data

#### Install the NCore Development Toolkit

Install the NCore SDK in the existing DriveStudio environment:

```shell
pip install nvidia-ncore==19.8.0 zarr==2.17.1 numcodecs==0.12.1
```

Zarr and Numcodecs are pinned to versions compatible with the existing NumPy
installation. NCore is needed only for preprocessing, not training or rendering.
The preprocessor handles NCore's FTheta enum annotation issue on Python 3.9.

#### Running the preprocessing script

To preprocess the demo clip, use the following command. Scene `0` selects
the demo when it is the only clip in the raw folder:

```shell
# export PYTHONPATH=\path\to\project
python datasets/preprocess.py \
    --data_root data/physicalai/raw \
    --target_dir data/physicalai/processed \
    --dataset physicalai \
    --scene_ids 0 \
    --workers 1 \
    --process_keys images lidar calib pose dynamic_masks objects
```

The extracted data will be stored in `data/physicalai/processed`.

Scene indices refer to the lexicographically sorted `pai_*.json` files under the
raw root. `metadata.json` records the source clip, selected camera frames and
exact timestamps. Keep that mapping when adding more downloaded clips. To use
stable UUID scene names instead, pass a `--split_file` CSV with a header and one
clip UUID per row; those scenes are written under their UUID rather than `000`.
The downstream mask/pose tools also support `--split_file` scene lists. Include
`physicalai` in the split-file path (for example, `physicalai_demo.csv`) so the
mask tool preserves UUID scene IDs as strings.

Processing covers the full clip, sampling every third front-wide camera frame
(approximately 10 Hz from a 30 Hz source), and preserves the original image
resolution and lens projection. No PhysicalAI-specific preprocessing flags are
needed. Select training frame ranges with `data.start_timestep` /
`data.end_timestep` and loading resolution with
`data.pixel_source.downscale_when_loading` in the existing dataset configuration.
Other cameras are matched to the front-wide reference, retaining their actual
exposure timestamps. Missing cameras are recorded; adjust the camera list in
the dataset config accordingly.

## 4. Extract Masks

To generate:

- **sky masks (required)**
- fine dynamic masks (optional)

#### Install `SegFormer` (Skip if already installed)

Follow the [SegFormer installation instructions](./Waymo.md#install-segformer-skip-if-already-installed)
to create the separate `segformer` Conda environment and download the pretrained
`segformer.b5.1024x1024.city.160k.pth` checkpoint.

#### Run Mask Extraction Script

```shell
conda activate segformer
segformer_path=/pathtosegformer

# export PYTHONPATH=\path\to\project
python datasets/tools/extract_masks.py \
    --data_root data/physicalai/processed \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --scene_ids 0 \
    --process_dynamic_mask
```

Replace `/pathtosegformer` with the actual path to your SegFormer installation.
The `--process_dynamic_mask` flag extracts fine dynamic masks along with sky masks.
This writes `sky_masks/` and `fine_dynamic_masks/`. The shared mask tool supports
the processed PNG images. The example dataset configuration expects sky and dynamic masks.

## 5. Human Body Pose Processing

#### Prerequisites

For optional SMPL annotations, follow the [Human Pose Processing Guide](./HumanPose.md)
to install the `4D-humans` environment and download the SMPL and estimator model assets.

#### Run the Extraction Pipeline

```shell
conda activate 4D-humans

# export PYTHONPATH=\path\to\project
python datasets/tools/humanpose_process.py \
    --dataset physicalai \
    --data_root data/physicalai/processed \
    --scene_ids 0 \
    --fps 10 \
    --save_temp
```

Add `--verbose` to save per-person box overlays under
`humanpose/temp/Pedes_GTTracks/vis/images/<camera_id>/` and a video for each camera
at `humanpose/temp/Pedes_GTTracks/vis/cam_<camera_id>.mp4`. Video playback uses
`--fps`; frames without accepted pedestrian boxes are included without overlays.

The PhysicalAI adapter projects pedestrian boxes using native calibration and
the same midpoint pose used for training. The existing image-based human estimator
is not retrained for fisheye images, so its predictions remain approximate. After the step creates
`humanpose/smpl.pkl`, enable `data.pixel_source.load_smpl=true`. The default is
false; pedestrians may instead be represented by the existing deformable nodes.

## 6. Data Structure

After completing preprocessing, the data directory should be organized as follows:

```text
ProjectPath/data/
└── physicalai/
    ├── raw/
    │   └── clips/<clip_uuid>/
    │       ├── pai_<clip_uuid>.json
    │       ├── pai_<clip_uuid>.ncore4.zarr.itar
    │       ├── pai_<clip_uuid>.ncore4-camera_*.zarr.itar
    │       └── pai_<clip_uuid>.ncore4-lidar_*.zarr.itar
    └── processed/
        └── 000/
            ├── metadata.json        # Source clip, camera calibration and frame mapping
            ├── timestamps.txt       # Reference frame timestamps
            ├── images/              # Native images: {timestep:03d}_{cam_id}.png
            ├── lidar/               # LiDAR data: {timestep:03d}.bin
            ├── lidar_pose/          # LiDAR sensor poses: {timestep:03d}.txt
            ├── ego_pose/            # Ego vehicle poses: {timestep:03d}.txt
            ├── extrinsics/          # Camera extrinsics: {cam_id}.txt
            ├── intrinsics/          # FTheta calibration: {cam_id}.json
            ├── camera_pose/         # Exposure-end poses: {timestep:03d}_{cam_id}.txt
            ├── camera_pose_start/   # Exposure-start poses: {timestep:03d}_{cam_id}.txt
            ├── invalid_masks/       # Invalid/ego masks: {timestep:03d}_{cam_id}.png
            ├── sky_masks/           # Sky masks: {timestep:03d}_{cam_id}.png
            ├── dynamic_masks/       # Coarse masks: category/{timestep:03d}_{cam_id}.png
            ├── fine_dynamic_masks/  # (Optional) Fine masks: category/{timestep:03d}_{cam_id}.png
            ├── instances/           # Instances' bounding boxes information
            └── humanpose/           # (Optional) Human body poses: smpl.pkl
```
