# Preparing UrbanV2X Dataset

UrbanV2X is a vehicle-infrastructure dataset collected in dense urban areas of Hong Kong. The ego vehicle carries
seven surround cameras (2200 × 1740, 10 Hz), a Velodyne HDL-32E LiDAR, an IMU and a tactical-grade GNSS/INS whose
post-processed solution serves as pose ground truth. Recordings are ROS1 bags. This page covers the ego-vehicle data;
3D object annotations are not provided, so object-based models (OmniRe, StreetGS) need boxes from another source.

## 1. Download Dataset

Request access and download from the UrbanV2X project page: [arXiv:2512.20224](https://arxiv.org/abs/2512.20224).
You need, per recording:
- the ego-vehicle ROS1 bag(s),
- the sensor calibration package (camera intrinsics, camera/LiDAR → IMU extrinsics, Kalibr convention),
- the GNSS/INS ground-truth export (Inertial Explorer text file, 100 Hz).

##### File Structure of Raw Data
```
ProjectPath/data/urbanv2x/raw/
├── bags/
│   └── <recording>.bag
├── calibration/
│   ├── intrinsics/intrinsics_cam{0..6}.yaml
│   └── extrinsics/
│       ├── Camera-IMU/imu_to_cam{0..6}.yaml
│       └── LiDAR-IMU/hdl32_to_imu.yaml
└── gt/
    └── <gnss_ins>.txt
```

## 2. Install the Extra Dependency

Bags are read with [rosbags](https://gitlab.com/ternaris/rosbags) (pure Python, no ROS installation needed):
```shell
pip install rosbags
```

## 3. Process Raw Data

Scenes are listed in a text file, one per line: `<scene_name> <bag> <gnss_ins_file>`
(see `data/urbanv2x_example_scenes.txt`). `--scene_ids` are line indices into this list.

```shell
python datasets/preprocess.py \
    --dataset urbanv2x \
    --data_root data/urbanv2x/raw \
    --target_dir data/urbanv2x/processed \
    --split_file data/urbanv2x_example_scenes.txt \
    --scene_ids 0 \
    --workers 1 \
    --process_keys images lidar calib pose
```

What the preprocessing does:
- **Images**: every camera is undistorted with its calibrated intrinsics and rescaled (0.5× → 1100 × 870). The
  stored intrinsics are those of the undistorted image, so `undistort: False` in the configs.
- **Timesteps**: one timestep per frame of the front camera (CAM_F, hikcamera 2); the other cameras are matched by
  nearest header time (typically within ~2 ms).
- **Ego poses**: GNSS/INS poses (local ENU anchored at the first epoch of the clip) chained with the GNSS/INS body →
  IMU extrinsic. This extrinsic and the sensor clock offset are refined by multi-scan LiDAR consistency
  (point-to-plane between deskewed scans ≥ 0.5 s apart). On the example scene this reduces the median multi-scan
  residual from 6.4 cm to 1.7 cm and estimates a −15 ms clock offset; the values are written to `meta.json`.
  Disable with `--no_refine` when running `datasets/urbanv2x/urbanv2x_preprocess.py` directly.
- **LiDAR**: each HDL-32E point is deskewed with its own timestamp and expressed in the ego frame at the camera
  time of its timestep.

## 4. Extract Masks

Sky masks (required), dynamic masks (optional, for metrics) and ego-hood masks use a HuggingFace SegFormer model, so
no separate mmcv environment is needed:
```shell
python datasets/urbanv2x/urbanv2x_masks.py --scene_dir data/urbanv2x/processed/hkstp_2a_20s
```
The ego hood is visible in CAM_F1, CAM_F and CAM_R. Its mask is estimated per camera from the clip (pixels touching
the bottom edge that are static, saturated, or almost always classified as car) and written to
`data/ego_masks/urbanv2x/<cam>.png`. Dynamic masks are semantic, so they also include parked vehicles.

## 5. Data Structure

```
ProjectPath/data/
  └── urbanv2x/
    ├── raw/
    └── processed/
         └── <scene_name>/
              ├── images/          # {timestep:03d}_{cam_id}.jpg, undistorted
              ├── intrinsics/      # {cam_id}.txt: fx fy cx cy 0 0 0 0 0
              ├── extrinsics/      # {cam_id}.txt: camera -> ego (OpenCV camera axes); ego = IMU frame
              ├── ego_pose/        # {timestep:03d}.txt: ego -> world (local ENU)
              ├── lidar/           # {timestep:03d}.bin: float32 x y z intensity, ego frame
              ├── sky_masks/       # {timestep:03d}_{cam_id}.png
              ├── dynamic_masks/   # all / vehicle / human
              └── meta.json        # timing, anchor, refined extrinsic and residuals
ProjectPath/data/ego_masks/urbanv2x/{cam_id}.png
```

## 6. Training

```shell
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
python tools/train.py \
    --config_file configs/pvg_urbanv2x.yaml \
    --output_root outputs --project urbanv2x --run_name pvg_cam2 \
    dataset=urbanv2x/1cams \
    data.scene_idx=hkstp_2a_20s
```
- Dataset configs: `urbanv2x/1cams` (CAM_F), `urbanv2x/3cams` (CAM_FL, CAM_F, CAM_FR), `urbanv2x/7cams`.
- `configs/pvg_urbanv2x.yaml` is `configs/pvg.yaml` without camera-pose refinement. The GNSS/INS poses are already
  accurate to a few centimetres, and the learned per-image correction is only applied to training images, so with it
  enabled held-out images are rendered from uncorrected poses.
- The dataset configs set `trainer.render.exclude_egocar_in_metrics: True`, so the ego hood (never supervised) is
  excluded from PSNR / SSIM / LPIPS. The default for other datasets is unchanged.
- On a 16 GB GPU, the default PVG densification limit (3M points) runs out of memory at full resolution; add
  `model.Background.ctrl.densify_until_num_points=1500000 model.Background.init.from_lidar.num_samples=600000`.

Reference (example scene `hkstp_2a_20s`, CAM_F only, frames 0–79, every 10th frame held out, 30k iterations):
PVG held-out PSNR 23.37 / SSIM 0.807 / LPIPS 0.209.

## Notes and Limitations
- The calibration package used for development is a reviewed candidate (`calibration_accepted: false`); check for
  the final release.
- No 3D object annotations are provided (`load_objects: False`).
- The front camera timestep defines the scene; LiDAR scans are paired with the nearest camera time (median ~14 ms)
  and deskewed to it.
