import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from multiprocessing import Pool
from os.path import join, isdir
import argparse
from glob import glob

from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils

# Abbreviations:
# WOD: Waymo Open Dataset
# FOV: field of view
# SDC: self-driving car
# 3dbox: 3D bounding box

# Some 3D bounding boxes do not contain any points
# This switch, when set True, filters these boxes
# It is safe to filter these boxes because they are not counted towards evaluation anyway
filter_empty_3dboxes = False


# There is no bounding box annotations in the No Label Zone (NLZ)
# if set True, points in the NLZ are filtered
filter_no_label_zone_points = True


# Only bounding boxes of certain classes are converted
# Note: Waymo Open Dataset evaluates for ALL_NS, including only 'VEHICLE', 'PEDESTRIAN', 'CYCLIST'
selected_waymo_classes = [
    # 'UNKNOWN',
    'VEHICLE',
    'PEDESTRIAN',
    # 'SIGN',
    'CYCLIST'
]


# Only data collected in specific locations will be converted
# If set None, this filter is disabled (all data will thus be converted)
# Available options: location_sf (main dataset)
selected_waymo_locations = None

# Save track id
save_track_id = True

# DATA_PATH = '/media/alex/Seagate Expansion Drive/waymo_open_dataset/domain_adaptation_training_labelled(partial)'
# KITTI_PATH = '/home/alex/github/waymo_to_kitti_converter/tools/pose'


class WaymoToKITTI(object):

    def __init__(self, load_dir, save_dir, prefix, num_proc):
        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']
        self.type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
        self.waymo_to_kitti_class_map = {
            'UNKNOWN': 'DontCare',
            'PEDESTRIAN': 'Pedestrian',
            'VEHICLE': 'Car',
            'CYCLIST': 'Cyclist',
            'SIGN': 'Sign'  # not in kitti
        }

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix   = prefix
        self.num_proc = int(num_proc)

        self.tfrecord_pathnames = sorted(glob(join(self.load_dir, '*.tfrecord')))

        self.label_save_dir       = self.save_dir + '/label_'
        self.label_all_save_dir   = self.save_dir + '/label_all'
        self.image_save_dir       = self.save_dir + '/image_'
        self.calib_save_dir       = self.save_dir + '/calib'
        self.point_cloud_save_dir = self.save_dir + '/velodyne'
        self.pose_save_dir        = self.save_dir + '/pose'

        self.create_folder()

    def convert(self):
        print("start converting ...")
        with Pool(self.num_proc) as p:
            r = list(tqdm.tqdm(p.imap(self.convert_one, range(len(self))), total=len(self)))
        print("\nfinished ...")

    def convert_one(self, file_idx):
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')
        
        for frame_idx, data in enumerate(dataset):

            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if selected_waymo_locations is not None and frame.context.stats.location not in selected_waymo_locations:
                continue

            # save images
            self.save_image(frame, file_idx, frame_idx)

            # parse calibration files
            self.save_calib(frame, file_idx, frame_idx)

            # parse point clouds
            self.save_lidar(frame, file_idx, frame_idx)

            # parse label files
            self.save_label(frame, file_idx, frame_idx)

            # parse pose files
            self.save_pose(frame, file_idx, frame_idx)

    def __len__(self):
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx):
        """ parse and save the images in png format
                :param frame: open dataset frame proto
                :param file_idx: the current file number
                :param frame_idx: the current frame number
                :return:
        """
        for img in frame.images:
            img_path = self.image_save_dir + str(img.name - 1) + '/' + self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.png'
            img = cv2.imdecode(np.frombuffer(img.image, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.imsave(img_path, rgb_img, format='png')

    def save_calib(self, frame, file_idx, frame_idx):
        """ parse and save the calibration data
                :param frame: open dataset frame proto
                :param file_idx: the current file number
                :param frame_idx: the current frame number
                :return:
        """
        # kitti:
        #   bbox in reference camera frame (right-down-front)
        #       image_x_coord = Px * R0_rect * R0_rot * bbox_coord
        #   lidar points in lidar frame (front-right-up)
        #       image_x_coord = Px * R0_rect * Tr_velo_to_cam * lidar_coord
        #   note:   R0_rot is caused by bbox rotation
        #           Tr_velo_to_cam projects lidar points to cam_0 frame
        # waymo:
        #   bbox in vehicle frame, hence, use a virtual reference frame
        #   since waymo camera uses frame front-left-up, the virtual reference frame (right-down-front) is
        #   built on a transformed front camera frame, name this transform T_front_cam_to_ref
        #   and there is no rectified camera frame
        #       image_x_coord = intrinsics_x * Tr_front_cam_to_cam_x * inv(T_front_cam_to_ref) * R0_rot * bbox_coord(now in ref frame)
        #   lidar points in vehicle frame
        #       image_x_coord = intrinsics_x * Tr_front_cam_to_cam_x * inv(T_front_cam_to_ref) * T_front_cam_to_ref * Tr_velo_to_front_cam * lidar_coord
        # hence, waymo -> kitti:
        #   set Tr_velo_to_cam = T_front_cam_to_ref * Tr_vehicle_to_front_cam = T_front_cam_to_ref * inv(Tr_front_cam_to_vehicle)
        #       as vehicle and lidar use the same frame after fusion
        #   set R0_rect = identity
        #   set P2 = front_cam_intrinsics * Tr_waymo_to_conv * Tr_front_cam_to_front_cam * inv(T_front_cam_to_ref)
        #   note: front cam is cam_0 in kitti, whereas has name = 1 in waymo
        #   note: waymo camera has a front-left-up frame,
        #       instead of the conventional right-down-front frame
        #       Tr_waymo_to_conv is used to offset this difference. However, Tr_waymo_to_conv is the same as
        #       T_front_cam_to_ref, hence,
        #   set P2 = front_cam_intrinsics

        calib_context = ''

        # front-left-up -> right-down-front
        # T_front_cam_to_ref = np.array([
        #     [0.0, -1.0, 0.0],
        #     [-1.0, 0.0, 0.0],
        #     [0.0, 0.0, 1.0]
        # ])
        T_front_cam_to_ref = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0]
        ])
        # T_ref_to_front_cam = np.array([
        #     [0.0, 0.0, 1.0],
        #     [-1.0, 0.0, 0.0],
        #     [0.0, -1.0, 0.0]
        # ])

        # print('context\n',frame.context)

        for camera in frame.context.camera_calibrations:
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                T_front_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(4, 4)
                # print('T_front_cam_to_vehicle\n', T_front_cam_to_vehicle)
                T_vehicle_to_front_cam = np.linalg.inv(T_front_cam_to_vehicle)

                front_cam_intrinsic = np.zeros((3, 4))
                front_cam_intrinsic[0, 0] = camera.intrinsic[0]
                front_cam_intrinsic[1, 1] = camera.intrinsic[1]
                front_cam_intrinsic[0, 2] = camera.intrinsic[2]
                front_cam_intrinsic[1, 2] = camera.intrinsic[3]
                front_cam_intrinsic[2, 2] = 1

                break

        # print('front_cam_intrinsic\n', front_cam_intrinsic)

        self.T_front_cam_to_ref = T_front_cam_to_ref.copy()
        self.T_vehicle_to_front_cam = T_vehicle_to_front_cam.copy()

        identity_3x4 = np.eye(4)[:3, :]

        # although waymo has 5 cameras, for compatibility, we produces 4 P
        for i in range(4):
            if i == 2:
                # note: front camera is labeled camera 2 (kitti) or camera 0 (waymo)
                #   other Px are given dummy values. this is to ensure compatibility. They are seldom used anyway.
                # tmp = cart_to_homo(np.linalg.inv(T_front_cam_to_ref))
                # print(front_cam_intrinsic.shape, tmp.shape)
                # P2 = np.matmul(front_cam_intrinsic, tmp).reshape(12)
                P2 = front_cam_intrinsic.reshape(12)
                calib_context += "P2: " + " ".join(['{}'.format(i) for i in P2]) + '\n'
            else:
                calib_context += "P" + str(i) + ": " + " ".join(['{}'.format(i) for i in identity_3x4.reshape(12)]) + '\n'

        calib_context += "R0_rect" + ": " + " ".join(['{}'.format(i) for i in np.eye(3).astype(np.float32).flatten()]) + '\n'

        Tr_velo_to_cam = self.cart_to_homo(T_front_cam_to_ref) @ np.linalg.inv(T_front_cam_to_vehicle)
        # print('T_front_cam_to_vehicle\n', T_front_cam_to_vehicle)
        # print('np.linalg.inv(T_front_cam_to_vehicle)\n', np.linalg.inv(T_front_cam_to_vehicle))
        # print('cart_to_homo(T_front_cam_to_ref)\n', cart_to_homo(T_front_cam_to_ref))
        # print('Tr_velo_to_cam\n',Tr_velo_to_cam)
        calib_context += "Tr_velo_to_cam" + ": " + " ".join(['{}'.format(i) for i in Tr_velo_to_cam[:3, :].reshape(12)]) + '\n'

        with open(self.calib_save_dir + '/' + self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt', 'w+') as fp_calib:
            fp_calib.write(calib_context)

    def save_lidar(self, frame, file_idx, frame_idx):
        """ parse and save the lidar data in psd format
                :param frame: open dataset frame proto
                :param file_idx: the current file number
                :param frame_idx: the current frame number
                :return:
                """
        range_images, camera_projections, range_image_top_pose = parse_range_image_and_camera_projection(frame)
        points_0, cp_points_0, intensity_0 = self.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0
        )
        points_0 = np.concatenate(points_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)

        points_1, cp_points_1, intensity_1 = self.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1
        )
        points_1 = np.concatenate(points_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)

        points = np.concatenate([points_0, points_1], axis=0)
        # print('points_0', points_0.shape, 'points_1', points_1.shape, 'points', points.shape)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        # points = points_1
        # intensity = intensity_1

        # reference frame:
        # front-left-up (waymo) -> right-down-front(kitti)
        # lidar frame:
        # ?-?-up (waymo) -> front-right-up (kitti)

        # print('bef\n', points)
        # print('bef\n', points.dtype)
        # points = np.transpose(points)  # (n, 3) -> (3, n)
        # tf = np.array([
        #     [0.0, -1.0,  0.0],
        #     [0.0,  0.0, -1.0],
        #     [1.0,  0.0,  0.0]
        # ])
        # points = np.matmul(tf, points)
        # points = np.transpose(points)  # (3, n) -> (n, 3)
        # print('aft\n', points)
        # print('aft\n', points.dtype)

        # concatenate x,y,z and intensity
        point_cloud = np.column_stack((points, intensity))


        # print(point_cloud.shape)

        # save
        pc_path = self.point_cloud_save_dir + '/' + self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.bin'
        point_cloud.astype(np.float32).tofile(pc_path)  # note: must save as float32, otherwise loading errors

    def save_label(self, frame, file_idx, frame_idx):
        """ parse and save the label data in .txt format
                :param frame: open dataset frame proto
                :param file_idx: the current file number
                :param frame_idx: the current frame number
                :return:
                """
        fp_label_all = open(self.label_all_save_dir + '/' + self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt', 'w+')
        # preprocess bounding box data
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # waymo: bounding box origin is at the center
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [label.box.center_x - label.box.length / 2, label.box.center_y - label.box.width / 2,
                        label.box.center_x + label.box.length / 2, label.box.center_y + label.box.width / 2]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        # print([i.type for i in frame.laser_labels])
        for obj in frame.laser_labels:

            # calculate bounding box
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break

            # TODO: temp fix
            if bounding_box == None or name == None:
                name = '0'
                bounding_box = (0, 0, 0, 0)

            my_type = self.type_list[obj.type]

            if my_type not in selected_waymo_classes:
                continue

            if filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            # from waymo_open_dataset.utils.box_utils import compute_num_points_in_box_3d
            # print('annot:', obj.num_lidar_points_in_box)
            # num_points_in_gt_waymo = compute_num_points_in_box_3d(
            #     tf.convert_to_tensor(self.pc.astype(np.float32), dtype=tf.float32),
            #     tf.convert_to_tensor(np.array([[obj.box.center_x, obj.box.center_y, obj.box.center_z,  obj.box.length,obj.box.width,  obj.box.height,obj.box.heading]]).astype(np.float32), dtype=tf.float32))
            # print('actual:', num_points_in_gt_waymo.numpy())

            # visualizer
            # [261   56   24   15   46  254   24  824  146   26    5   13   30   45
            #  60  184  347  222 1774    2   46]

            # converter
            # 264, 59, 24, 16, 51, 268, 24, 847, 149, 28, 6, 13, 30, 45, \
            # 64, 192, 353, 229, 1848, 2, 48

            my_type = self.waymo_to_kitti_class_map[my_type]

            # length: along the longer axis that is perpendicular to gravity direction
            # width: along the shorter axis  that is perpendicular to gravity direction
            # height: along the gravity direction
            # the same for waymo and kitti
            height = obj.box.height  # up/down
            width = obj.box.width  # left/right
            length = obj.box.length  # front/back

            # waymo: bbox label in lidar/vehicle frame. kitti: bbox label in reference image frame
            # however, kitti uses bottom center as the box origin, whereas waymo uses the true center
            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2

            # print('bef', x,y,z)

            # project bounding box to the virtual reference frame
            pt_ref = self.cart_to_homo(self.T_front_cam_to_ref) @ self.T_vehicle_to_front_cam @ np.array([x,y,z,1]).reshape((4,1))
            x, y, z, _ = pt_ref.flatten().tolist()

            # print('aft', x,y,z)

            # x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
            # length, width, height = length, height, width

            # front-left-up (waymo) -> right-down-front(kitti)
            # bbox origin at volumetric center (waymo) -> bottom center (kitti)
            # x, y, z = -waymo_y, -waymo_z + height / 2, waymo_x

            # rotation: +x around y-axis (kitti) -> +x around y-axis (waymo)
            #           right-down-front            front-left-up
            # note: the "rotation_y" is kept as the name of the rotation variable for compatibility
            # it is, in fact, rotation around positive z
            rotation_y = -obj.box.heading - np.pi / 2

            # track id
            track_id = obj.id

            # not available
            truncated = 0
            occluded = 0

            # alpha:
            # we set alpha to the default -10, the same as nuscenes to kitti tool
            # contribution is welcome
            alpha = -10

            # save the labels
            line = my_type + ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(round(truncated, 2),
                                                                                   occluded,
                                                                                   round(alpha, 2),
                                                                                   round(bounding_box[0], 2),
                                                                                   round(bounding_box[1], 2),
                                                                                   round(bounding_box[2], 2),
                                                                                   round(bounding_box[3], 2),
                                                                                   round(height, 2),
                                                                                   round(width, 2),
                                                                                   round(length, 2),
                                                                                   round(x, 2),
                                                                                   round(y, 2),
                                                                                   round(z, 2),
                                                                                   round(rotation_y, 2))
            if save_track_id:
                line_all = line[:-1] + ' ' + name + ' ' + track_id + '\n'
            else:
                line_all = line[:-1] + ' ' + name + '\n'

            # store the label
            fp_label = open(self.label_save_dir + name + '/' + self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt', 'a')
            fp_label.write(line)
            fp_label.close()

            fp_label_all.write(line_all)

        fp_label_all.close()

        # print(file_idx, frame_idx)

    def save_pose(self, frame, file_idx, frame_idx):
        """ Save self driving car (SDC)'s own pose

        Note that SDC's own pose is not included in the regular training of KITTI dataset
        KITTI raw dataset contains ego motion files but are not often used
        Pose is important for algorithms that takes advantage of the temporal information

        """

        pose = np.array(frame.pose.transform).reshape(4,4)
        np.savetxt(join(self.pose_save_dir, self.prefix + str(file_idx).zfill(3) + str(frame_idx).zfill(3) + '.txt'), pose)


    def create_folder(self):
        for d in [self.label_all_save_dir, self.calib_save_dir, self.point_cloud_save_dir, self.pose_save_dir]:
            if not isdir(d):
                os.makedirs(d)
        for d in [self.label_save_dir, self.image_save_dir]:
            for i in range(5):
                if not isdir(d + str(i)):
                    os.makedirs(d + str(i))

    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        """Convert range images to point cloud.
        Args:
          frame: open dataset frame
           range_images: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
             camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
          ri_index: 0 for the first return, 1 for the second return.
        Returns:
          points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
          cp_points: {[N, 6]} list of camera projections of length 5
            (number of lidars).
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        intensity = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
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
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            # No Label Zone
            if filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                # print(range_image_tensor[range_image_tensor[..., 3] == 1.0])
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.compat.v1.where(range_image_mask))

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor,
                                            tf.compat.v1.where(range_image_mask))
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor,
                                            tf.where(range_image_mask))
            intensity.append(intensity_tensor.numpy()[:, 1])

        return points, cp_points, intensity


    # def get_intensity(self, frame, range_images, ri_index=0):
    #     """Convert range images to point cloud.
    #     Args:
    #       frame: open dataset frame
    #        range_images: A dict of {laser_name,
    #          [range_image_first_return, range_image_second_return]}.
    #        camera_projections: A dict of {laser_name,
    #          [camera_projection_from_first_return,
    #           camera_projection_from_second_return]}.
    #       range_image_top_pose: range image pixel pose for top lidar.
    #       ri_index: 0 for the first return, 1 for the second return.
    #     Returns:
    #       intensity: {[N, 1]} list of intensity of length 5 (number of lidars).
    #     """
    #     calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    #     intensity = []
    #     for c in calibrations:
    #         range_image = range_images[c.name][ri_index]
    #         range_image_tensor = tf.reshape(
    #             tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    #         range_image_mask = range_image_tensor[..., 0] > 0
    #         intensity_tensor = tf.gather_nd(range_image_tensor,
    #                                         tf.where(range_image_mask))
    #         intensity.append(intensity_tensor.numpy()[:, 1])
    #
    #     return intensity

    def cart_to_homo(self, mat):
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', help='Directory to load Waymo Open Dataset tfrecords')
    parser.add_argument('save_dir', help='Directory to save converted KITTI-format data')
    parser.add_argument('--prefix', default='', help='Prefix to be added to converted file names')
    parser.add_argument('--num_proc', default=1, help='Number of processes to spawn')
    args = parser.parse_args()

    converter = WaymoToKITTI(args.load_dir, args.save_dir, args.prefix, args.num_proc)
    converter.convert()
