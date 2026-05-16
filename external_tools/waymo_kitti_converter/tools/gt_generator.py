"""This script generates gt.bin
"""

from glob import glob
from os.path import join
from tqdm import tqdm

import tensorflow as tf
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset

tfrecord_load_dir = '/media/alex/Seagate Expansion Drive/waymo_open_dataset/val_temp'
gt_bin_save_pathname = '/home/alex/gt.bin'


# convert from waymo.open_dataset.Label to waymo.open_dataset.Object
def convert(obj, context_name, frame_timestamp_micros):

    o = metrics_pb2.Object()
    o.object.box.CopyFrom(obj.box)
    o.object.type = obj.type
    o.score = 1.0
    o.object.num_lidar_points_in_box = obj.num_lidar_points_in_box  # needed for gt generation

    # for identification of the frame
    o.context_name = context_name
    o.frame_timestamp_micros = frame_timestamp_micros

    return o


def main():

    gt = metrics_pb2.Objects()
    for pathname in tqdm(glob(join(tfrecord_load_dir, '*.tfrecord'))):

        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        for frame_idx, data in enumerate(dataset):

            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            for obj in frame.laser_labels:
                if obj.num_lidar_points_in_box < 1:
                    continue

                o = convert(obj, frame.context.name, frame.timestamp_micros)
                gt.objects.append(o)

    with open(gt_bin_save_pathname, 'wb') as f:
        f.write(gt.SerializeToString())
    print(gt_bin_save_pathname, 'saved.')


if __name__ == '__main__':
    main()
