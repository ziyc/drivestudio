# Waymo-KITTI Converter

This repository provides tools for:
- [x] Converting Waymo Open Dataset(WOD)-format data to KITTI-format data
- [x] Converting KITTI-format prediction results to WOD-format results
- [x] Visualization for both formats

The tools convert the following data types:
- [x] Point clouds
- [x] Images
- [x] Bounding box labels
- [x] Calibration
- [x] Self driving car's poses

The tools have some additional features:
- [x] Multiprocessing
- [x] Progress bar

## Technical Report

More implementation details are included in our [technical report](https://arxiv.org/abs/2006.16796)

## Setup

#### Step 1. Create and use a new environment (taking anaconda for example)

Note: Python versions 3.6/3.7 are supported.
```
conda create -n waymo-kitti python=3.7
conda activate waymo-kitti
```

#### Step 2. Install TensorFlow

Note: TensorFlow version 1.1.5/2.0.0/2.1.0 are supported.

The following command install the newest version 2.1.0. 
For this repository, the cpu version is sufficient.
```
pip3 install tensorflow==2.1.0 
```

#### Step 3. Install Waymo Open Dataset precompiled packages.

Note: the following command assumes TensorFlow version 2.1.0. 
You may modify the command according to your TensorFlow version. 
See [official guide](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md) 
for additional details.

```
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-1-0==1.2.0 --user
```

#### Step 4. Install other packages

```
pip3 install opencv-python matplotlib tqdm
```
The following optional package is needed for the 3D visualization tools in tools/:
```
pip3 install open3d
```
 
## Convert WOD-format data to KITTI-format data

#### Step 1. Download and decompress data

Data can be downloaded from the [official website](https://waymo.com/open/download/).
Note that domain adaptation data is not needed. 

Decompress the zip files into different directories.
Each directory should contain tfrecords.
Example:
```
waymo_open_dataset
├── training
├── validation
├── testing
```


#### Step 2. Run the conversion tool
```
python converter.py <load_dir> <save_dir> [--prefix prefix] [--num_proc num_proc]
```
- load_dir: directory to load Waymo Open Dataset tfrecords
- save_dir: directory to save converted KITTI-format data
- (optional) prefix: prefix to be added to converted file names
- (optional) num_proc: number of processes to spawn

**Important Notes:**
- **tensorflow warnings that appear at the start can be ignored**
- **the progress bar is shown only after the first file is processed**
- **time for conversion (single process) ~40 mins / tfrecord (Intel Xeon Gold 6126)**

The reason for having a prefix is that KITTI format does not have a separate val set.
Hence, training set and validation set are merged into one directory.
To differentiate data from these two sets, the files have different prefixes.
In addition, WOD has some labelled data for domain adaptation task.
They can be added to the training directory with different prefixes as well.

Example:
```
python converter.py waymo_open_dataset/training waymo_open_dataset_kitti/training --prefix 0 --num_proc 8
python converter.py waymo_open_dataset/validation waymo_open_dataset_kitti/training --prefix 1 --num_proc 8
python converter.py waymo_open_dataset/testing waymo_open_dataset_kitti/testing --prefix 2 --num_proc 8
```
Note: 
- if both training and validation sets are saved in the same directory (waymo_open_dataset_kitti/training), it is necessary to give them different prefix to avoid overwriting.
- as the WOD is huge, the process can take very long; it is thus recommended to use more processes as long as there are enough CPU cores.

#### More on the converted data

The converted data should have the following file structure:
```
save_dir
├── calib
├── image_0
├── image_1
├── image_2
├── image_3
├── image_4
├── label_0
├── label_1
├── label_2
├── label_3
├── label_4
├── label_all
├── pose
├── velodyne
```
Important Notes: 
- KITTI only annotates 3D bounding boxes visible in the front camera,
whereas WOD annotates 3D bounding boxes both within and out of the field of views of all 5 cameras
- KITTI's front camera has a index 2, whereas the WOD's front camera has a index 0
- The calibration files are for the front camera
- label_x only contains 3D bounding boxes visible in camera idx x.
- label_all contains all 3D bounding boxes, even those out of the field of views of all cameras.
- calib contains calibration files for the front camera only at the moment
- pose file is not included in the regular KITTI set. It contains self driving car's pose, 
in the form of a transformation matrix (4x4) from the vehicle frame to the global frame

To read a pose file:
```python3
import numpy as np
pose = np.loadtxt(<pathname>, dtype=np.float32)
```

The label files in label_x (x in {0,1,2,3}) are similar to the regular KITTI label:

```
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Pedestrian', 'Cyclist'
   1    truncated    Always 0
   1    occluded     Always 0
   1    alpha        Always -10
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
```

The label files in label_all have the following format:
```
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Pedestrian', 'Cyclist'
   1    truncated    Always 0
   1    occluded     Always 0
   1    alpha        Always -10
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    cam_idx	     The index of the camera in which the 3D bounding box is visible. 
                     Set to 0 if not visible in any cameras   
```

## Convert KITTI-format results to WOD-format results

#### Generation of the prediction results
It is assumed that the user's model generates prediction results .txt files in the KITTI format.

#### Run the conversion tool

```
python prediction_kitti_to_waymo.py <kitti_results_load_dir> <waymo_tfrecords_load_dir> \
                                    <waymo_results_save_dir> <waymo_results_comb_save_pathname> \
                                    [--prefix prefix] [--num_proc num_proc]
```
- kitti_results_load_dir: directory to load KITTI-format results
- waymo_tfrecords_load_dir: directory to load corresponding Waymo Open Dataset tfrecords
- waymo_results_save_dir: directory to save temporary output files
- waymo_results_comb_save_pathname: pathname to save the single output file
- (optional) prefix: prefix to be added to converted file names
- (optional) num_proc: number of processes to spawn

Example:
```
python prediction_kitti_to_waymo.py prediction/  waymo_open_dataset/testing \
                                    temp/  output.bin \
                                    --prefix 2 --num_proc 8
```
Note: 
- this example shows how to convert the prediction results on the test set.
For the validation set, waymo_tfrecords_load_dir must be modified accordingly.
- the prefix must match the one used for Waymo-to-KITTI data conversion.
- it is important that the tfrecords in the directory are not modified (added, deleted etc)

#### Evaluate the results

The output file of the conversion tool is a single .bin.
Waymo Open Dataset provides methods for creating submission (create_submission) or
local evaluation (compute_detection_metrics_main). See the [official guide](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md) for details.


## References

- [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset)
- [Waymo_Kitti_Adapter](https://github.com/Yao-Shao/Waymo_Kitti_Adapter)

## Citation

If you find this toolkit useful, please consider citing:

```
@article{yucai2020leveraging,
  title={Leveraging Temporal Information for 3D Detection and Domain Adaptation},
  author={Yu, Cunjun and Cai, Zhongang and Ren, Daxuan and Zhao, Haiyu},
  journal={arXiv preprint arXiv:2006.16796},
  year={2020}
}
```
