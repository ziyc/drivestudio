import os
import cv2
import re
import numpy as np

"""
input: calib txt path
return: P2: (4,4) 3D camera coordinates to 2D image pixels
        vtc_mat: (4,4) 3D velodyne Lidar coordinates to 3D camera coordinates
"""
def read_calib(calib_path):
    P2 = None
    vtc_mat = None
    R0 = np.eye(4, dtype=np.float32)
    with open(calib_path) as f:
        for line in f.readlines():
            if line[:2] == "P2":
                P2 = re.split(" ", line.strip())
                P2 = np.array(P2[-12:], np.float32)
                P2 = P2.reshape((3, 4))
            if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                vtc_mat = re.split(" ", line.strip())
                vtc_mat = np.array(vtc_mat[-12:], np.float32)
                vtc_mat = vtc_mat.reshape((3, 4))
                vtc_mat = np.concatenate([vtc_mat, [[0, 0, 0, 1]]])
            if line[:7] == "R0_rect" or line[:6] == "R_rect":
                R0 = re.split(" ", line.strip())
                R0 = np.array(R0[-9:], np.float32)
                R0 = R0.reshape((3, 3))
                R0 = np.concatenate([R0, [[0], [0], [0]]], -1)
                R0 = np.concatenate([R0, [[0, 0, 0, 1]]])
    if P2 is None:
        raise ValueError(f"Missing P2 in calibration file: {calib_path}")
    if vtc_mat is None:
        raise ValueError(f"Missing Tr_velo_to_cam/Tr_velo_cam in calibration file: {calib_path}")
    vtc_mat = np.matmul(R0, vtc_mat)
    return (P2, vtc_mat)


"""
description: read lidar data given 
input: lidar bin path "path", cam 3D to cam 2D image matrix (4,4), lidar 3D to cam 3D matrix (4,4)
output: valid points in lidar coordinates (PointsNum,4)
"""
def read_velodyne(path, P, vtc_mat, IfReduce=True, image_shape=None):
    if image_shape is not None:
        max_row = image_shape[0]
        max_col = image_shape[1]
    else:
        max_row = 374  # y
        max_col = 1241  # x
    lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))

    if not IfReduce:
        return lidar

    mask = lidar[:, 0] > 0
    lidar = lidar[mask]
    lidar_copy = np.zeros(shape=lidar.shape)
    lidar_copy[:, :] = lidar[:, :]

    velo_tocam = vtc_mat
    lidar[:, 3] = 1
    lidar = np.matmul(lidar, velo_tocam.T)
    img_pts = np.matmul(lidar, P.T)
    velo_tocam = np.asmatrix(velo_tocam).I
    velo_tocam = np.array(velo_tocam)
    normal = velo_tocam
    normal = normal[0:3, 0:4]
    lidar = np.matmul(lidar, normal.T)
    lidar_copy[:, 0:3] = lidar
    x, y = img_pts[:, 0] / img_pts[:, 2], img_pts[:, 1] / img_pts[:, 2]
    mask = np.logical_and(np.logical_and(x >= 0, x < max_col), np.logical_and(y >= 0, y < max_row))

    return lidar_copy[mask]


"""
description: convert 3D camera coordinates to Lidar 3D coordinates.
input: (PointsNum,3)
output: (PointsNum,3)
"""
def cam_to_velo(cloud,vtc_mat):
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    mat=np.asmatrix(mat)
    normal=np.asmatrix(vtc_mat).I
    normal=normal[0:3,0:4]
    transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T

"""
description: convert 3D camera coordinates to Lidar 3D coordinates.
input: (PointsNum,3)
output: (PointsNum,3)
"""
def velo_to_cam(cloud,vtc_mat):
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    mat=np.asmatrix(mat)
    normal=np.asmatrix(vtc_mat).I
    normal=normal[0:3,0:4]
    transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T

def read_image(path):
    im=cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return im

def read_detection_label(path):

    boxes = []
    names = []

    if not os.path.exists(path):
        return np.empty((0, 7), dtype=np.float32), np.array([])

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[0]
            if this_name != "DontCare":
                line = np.array(line[-7:],np.float32)
                boxes.append(line)
                names.append(this_name)

    if len(boxes) == 0:
        return np.empty((0, 7), dtype=np.float32), np.array([])

    return np.array(boxes, dtype=np.float32).reshape(-1, 7),np.array(names)

def read_tracking_label(path):

    frame_dict={}

    names_dict={}

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[2]
            frame_id = line[0]
            ob_id = line[1]

            if this_name != "DontCare":
                line = np.array(line[10:17],np.float32).tolist()
                line.append(ob_id)


                if frame_id in frame_dict.keys():
                    frame_dict[frame_id].append(line)
                    names_dict[frame_id].append(this_name)
                else:
                    frame_dict[frame_id] = [line]
                    names_dict[frame_id] = [this_name]

    return frame_dict,names_dict

if __name__ == '__main__':
    path = 'H:/dataset/traking/training/label_0/0000.txt'
    labels,a = read_tracking_label(path)
    print(a)

