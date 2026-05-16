"""Adopted from waymo open dataset repository"""
import os
from os.path import join, isdir
import tensorflow as tf
from glob import glob
import tqdm
from multiprocessing import Pool
import numpy as np
import argparse

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset

# kitti_results_load_dir = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_kitti_results/data'
# waymo_tfrecords_load_dir = '/home/alex/github/waymo_to_kitti_converter/tools/waymo/testing'
#
# waymo_results_save_dir = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_submission/20200417-bin'
# waymo_results_comb_save_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_submission/20200417.bin'
#
# prefix = '1'
#
# NUM_PROC = 1

# def _create_pd_file_example():
#     """Creates a prediction objects file."""
#     objects = metrics_pb2.Objects()
#
#     o = metrics_pb2.Object()
#     # The following 3 fields are used to uniquely identify a frame a prediction
#     # is predicted at. Make sure you set them to values exactly the same as what
#     # we provided in the raw data. Otherwise your prediction is considered as a
#     # false negative.
#     o.context_name = ('context_name for the prediction. See Frame::context::name '
#                       'in    dataset.proto.')
#     # The frame timestamp for the prediction. See Frame::timestamp_micros in
#     # dataset.proto.
#     invalid_ts = -1
#     o.frame_timestamp_micros = invalid_ts
#     # This is only needed for 2D detection or tracking tasks.
#     # Set it to the camera name the prediction is for.
#     o.camera_name = dataset_pb2.CameraName.FRONT
#
#     # Populating box and score.
#     box = label_pb2.Label.Box()
#     box.center_x = 0
#     box.center_y = 0
#     box.center_z = 0
#     box.length = 0
#     box.width = 0
#     box.height = 0
#     box.heading = 0
#     o.object.box.CopyFrom(box)
#     # This must be within [0.0, 1.0]. It is better to filter those boxes with
#     # small scores to speed up metrics computation.
#     o.score = 0.5
#     # For tracking, this must be set and it must be unique for each tracked
#     # sequence.
#     o.object.id = 'unique object tracking ID'
#     # Use correct type.
#     o.object.type = label_pb2.Label.TYPE_PEDESTRIAN
#
#     objects.objects.append(o)
#
#     # Add more objects. Note that a reasonable detector should limit its maximum
#     # number of boxes predicted per frame. A reasonable value is around 400. A
#     # huge number of boxes can slow down metrics computation.
#
#     # Write objects to a file.
#     f = open('/tmp/your_preds.bin', 'wb')
#     f.write(objects.SerializeToString())
#     f.close()


class KITTI2Waymo(object):
    def __init__(self, kitti_results_load_dir, waymo_tfrecords_load_dir,
                 waymo_results_save_dir, waymo_results_comb_save_pathname,
                 prefix, num_proc):

        self.kitti_results_load_dir           = kitti_results_load_dir
        self.waymo_tfrecords_load_dir         = waymo_tfrecords_load_dir
        self.waymo_results_save_dir           = waymo_results_save_dir
        self.waymo_results_comb_save_pathname = waymo_results_comb_save_pathname
        self.prefix                           = prefix
        self.num_proc                         = num_proc

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

        self.T_ref_to_front_cam = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.get_file_names()
        self.create_folder()

    def get_file_names(self):
        self.waymo_tfrecord_pathnames = sorted(glob(join(self.waymo_tfrecords_load_dir, '*.tfrecord')))
        print(len(self.waymo_tfrecord_pathnames), 'tfrecords found.')

    def create_folder(self):
        if not isdir(self.waymo_results_save_dir):
            os.makedirs(self.waymo_results_save_dir)

    def parse_objects(self, kitti_result_pathname, T_k2w, context_name, frame_timestamp_micros):

        def parse_one_object(line):
            attrs = line.split()

            cls = attrs[0]
            height = float(attrs[8])
            width = float(attrs[9])
            length = float(attrs[10])
            x = float(attrs[11])
            y = float(attrs[12])
            z = float(attrs[13])
            rotation_y = float(attrs[14])
            score = float(attrs[15])

            # y: downwards; move box origin from bottom center (kitti) to true center (waymo)
            y = float(attrs[12]) - height / 2
            x, y, z = self.transform(T_k2w, x, y, z)  # frame transformation: kitti -> waymo

            # different conventions
            heading = - (rotation_y + np.pi / 2)
            while heading < -np.pi:
                heading += 2*np.pi
            while heading > np.pi:
                heading -= 2*np.pi

            # populate box
            box = label_pb2.Label.Box()
            box.center_x = x
            box.center_y = y
            box.center_z = z
            box.length = length
            box.width = width
            box.height = height
            box.heading = heading

            o = metrics_pb2.Object()
            o.object.box.CopyFrom(box)
            o.object.type = self.k2w_cls_map[cls]
            o.score = score

            # for identification of the frame
            o.context_name = context_name
            o.frame_timestamp_micros = frame_timestamp_micros

            return o

        objects = metrics_pb2.Objects()
        with open(kitti_result_pathname, 'r') as f:
            lines = f.readlines()

        for line in lines:
            o = parse_one_object(line)
            objects.objects.append(o)

        return objects

    def process_one(self, file_num):
        file_pathname = self.waymo_tfrecord_pathnames[file_num]
        file_data = tf.data.TFRecordDataset(file_pathname, compression_type='')

        # process each frame
        for frame_num, frame_data in enumerate(file_data):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(frame_data.numpy()))

            kitti_result_pathname = join(self.kitti_results_load_dir, '{}{:03d}{:03d}.txt'.format(self.prefix, file_num, frame_num))

            # prepare transformation matrix from kitti to waymo
            # here, the kitti frame is a virtual reference frame
            # the bounding boxes are in the vehicle frame
            for camera in frame.context.camera_calibrations:
                if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                    T_front_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(4, 4)

            T_k2w = T_front_cam_to_vehicle @ self.T_ref_to_front_cam

            # prepare context_name and frame_timestamp_micros
            context_name = frame.context.name
            frame_timestamp_micros = frame.timestamp_micros

            try:
                objects = self.parse_objects(kitti_result_pathname, T_k2w, context_name, frame_timestamp_micros)
            except:
                print(kitti_result_pathname, 'not found.')
                objects = metrics_pb2.Objects()

            # print(file_num, frame_num, '\n', objects)

            # Write objects to a file.
            with open(join(self.waymo_results_save_dir, '{}{:03d}{:03d}.bin'.format(self.prefix, file_num, frame_num)), 'wb') as f:
                f.write(objects.SerializeToString())

    def convert(self):

        print("start converting ...")
        with Pool(self.num_proc) as p:
            r = list(tqdm.tqdm(p.imap(self.process_one, range(len(self))), total=len(self)))
        print("\nfinished ...")

        # combine all files into one .bin
        pathnames = sorted(glob(join(self.waymo_results_save_dir, '*.bin')))
        combined = self.combine(pathnames)

        with open(self.waymo_results_comb_save_pathname, 'wb') as f:
            f.write(combined.SerializeToString())

    def __len__(self):
        return len(self.waymo_tfrecord_pathnames)

    def transform(self, T, x, y, z):
        pt_bef = np.array([x, y, z, 1.0]).reshape(4,1)
        pt_aft = np.matmul(T, pt_bef)
        # print(pt_aft)
        return pt_aft[:3].flatten().tolist()

    def combine(self, pathnames):
        combined = metrics_pb2.Objects()

        for pathname in pathnames:
            objects = metrics_pb2.Objects()
            with open(pathname, 'rb') as f:
                objects.ParseFromString(f.read())
            for o in objects.objects:
                combined.objects.append(o)

        return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kitti_results_load_dir', help='Directory to load KITTI-format results')
    parser.add_argument('waymo_tfrecords_load_dir', help='Directory to load corresponding Waymo Open Dataset tfrecords')
    parser.add_argument('waymo_results_save_dir', help='Directory to save temporary output files')
    parser.add_argument('waymo_results_comb_save_pathname', help='Pathname to save the single output file')
    parser.add_argument('--prefix', default='', help='Prefix to be added to converted file names')
    parser.add_argument('--num_proc', default=1, help='Number of processes to spawn')
    args = parser.parse_args()

    converter = KITTI2Waymo(args.kitti_results_load_dir,
                            args.waymo_tfrecords_load_dir,
                            args.waymo_results_save_dir,
                            args.waymo_results_comb_save_pathname,
                            args.prefix,
                            args.num_proc)
    converter.convert()


if __name__ == '__main__':
    main()
