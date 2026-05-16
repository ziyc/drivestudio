"""
Load both ground truth tfrecord
and processed prediction in combined.bin (before running create_submission)
Visualize and compare
"""

import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from multiprocessing import Pool
import open3d as o3d
from collections import defaultdict

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset


all_dt_pathname = '/home/alex/github/waymo_to_kitti_converter/submission/20200513/PartA2_waymo_nosign_40_mini_all_m1    40.bin'
gt_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/waymo/validation/segment-1024360143612057520_3580_000_3600_000_with_camera_labels.tfrecord'
# gt_pathname = '/media/alex/Seagate Expansion Drive/waymo_open_dataset/validation(partial)/segment-10837554759555844344_6525_000_6545_000_with_camera_labels.tfrecord'

# all_dt_pathname = '/home/alex/github/waymo_to_kitti_converter/submission/20200430_pointpillar_waymo_8_16_z4_test/20200430_pointpillar_waymo_8_16_z4_test.bin'
# gt_pathname = '/media/alex/Seagate Expansion Drive/waymo_open_dataset/testing(partial)/segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord'


def parse_range_image_and_camera_projection(frame):
    ranged_images = {}
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_compressed, 'ZLIB')
            ri = open_dataset.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            ranged_images[laser.name] = [ri]

            if laser.name == open_dataset.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                range_image_top_pose = open_dataset.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    bytearray(range_image_top_pose_str_tensor.numpy()))

        if len(laser.ri_return2.range_image_compressed) > 0:
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_compressed, 'ZLIB')
            ri = open_dataset.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            ranged_images[laser.name].append(ri)

    return ranged_images, range_image_top_pose


def convert_range_image_to_point_cloud(frame, range_images, range_image_top_pose, ri_index=0):
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    # lasers = sorted(frame.lasers, key=lambda laser: laser.name)
    points = []
    # cp_points = []
    intensity = []

    frame_pose = tf.convert_to_tensor(
        np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == open_dataset.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.where(range_image_mask))
        intensity_tensor = tf.gather_nd(range_image_tensor,
                                        tf.where(range_image_mask))
        # cp = camera_projections[c.name][0]
        # cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
        # cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
        points.append(points_tensor.numpy())
        # cp_points.append(cp_points_tensor.numpy())
        intensity.append(intensity_tensor.numpy()[:, 1])

    return points, intensity


def get_lidar(frame):

    range_images, range_image_top_pose = parse_range_image_and_camera_projection(frame)
    points, intensity = convert_range_image_to_point_cloud(frame, range_images, range_image_top_pose)

    points = np.concatenate(points, axis=0)
    intensity = np.concatenate(intensity, axis=0)

    point_cloud = np.column_stack((points, intensity))
    return point_cloud


def corners_to_lines(qs, color):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    idx = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7)]
    cl = [color for i in range(12)]

    # print('draw bbox')
    # print(qs)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(qs),
        lines=o3d.utility.Vector2iVector(idx),
    )
    line_set.colors = o3d.utility.Vector3dVector(cl)

    return line_set


def boxes3d_to_corners3d_lidar(x, y, z, l, w, h, rz):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, rz] in LiDAR/Vehicle coords
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
         z x
         |/
      y--O
    """
    x_corners = np.array([l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2.])
    y_corners = np.array([-w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2., w / 2.])
    z_corners = np.array([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.])

    corners = np.concatenate((x_corners.reshape(1, 8), y_corners.reshape(1, 8), z_corners.reshape(1, 8)), axis=0)  # (3, 8)

    Rz = np.array([[ np.cos(rz), -np.sin(rz), 0],
                   [ np.sin(rz),  np.cos(rz), 0],
                   [          0,           0, 1]])

    rotated_corners = np.matmul(Rz, corners)  # (3, 8)
    rotated_corners += np.array([x, y, z]).reshape(3, 1)

    return rotated_corners.astype(np.float32)


def parse_all_dt_objects(pathname):
    objects = metrics_pb2.Objects()
    with open(pathname, 'rb') as f:
        objects.ParseFromString(f.read())

    all_dt_objects = defaultdict(list)
    for o in objects.objects:
        context_name = o.context_name
        frame_timestamp_micros = o.frame_timestamp_micros
        all_dt_objects[str(context_name) + '-' + str(frame_timestamp_micros)].append(o)

    return all_dt_objects


def get_dt_bbox(o):

    x = o.object.box.center_x
    y = o.object.box.center_y
    z = o.object.box.center_z
    print('get_dt_bbox', x, y, z)

    l = o.object.box.length
    w = o.object.box.width
    h = o.object.box.height
    rz = o.object.box.heading

    bbox = boxes3d_to_corners3d_lidar(x, y, z, l, w, h, rz)
    return bbox


def get_gt_bboxes(frame):
    gt_bboxes = []

    for o in frame.laser_labels:

        x = o.box.center_x
        y = o.box.center_y
        z = o.box.center_z
        l = o.box.length
        w = o.box.width
        h = o.box.height
        rz = o.box.heading

        bbox = boxes3d_to_corners3d_lidar(x, y, z, l, w, h, rz)
        gt_bboxes.append(bbox)

    return gt_bboxes


def main():

    all_dt = parse_all_dt_objects(all_dt_pathname)

    dataset = tf.data.TFRecordDataset(gt_pathname, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # print(frame)
        key = str(frame.context.name) + '-' + str(frame.timestamp_micros)  # unique identifier

        point_cloud = get_lidar(frame)
        gt_bboxes = get_gt_bboxes(frame)

        dt_objects = all_dt[key]
        dt_bboxes = [get_dt_bbox(o) for o in dt_objects]

        # draw
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])

        visual = [pcd, axis]

        for bbox3d in gt_bboxes:
            bbox3d = bbox3d.T.tolist()  # (8, 3)
            # print(bbox3d)
            visual.append(corners_to_lines(bbox3d, color=[0,0,1]))

        for bbox3d in dt_bboxes:
            bbox3d = bbox3d.T.tolist()
            print('dt_bbox', bbox3d)
            visual.append(corners_to_lines(bbox3d, color=[1,0,0]))

        o3d.visualization.draw_geometries(visual)


if __name__ == '__main__':
    main()
