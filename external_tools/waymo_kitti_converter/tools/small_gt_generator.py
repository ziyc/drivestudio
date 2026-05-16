"""
    Generates a smaller gt.bin
    Scale is fixed to be 1/10 of the data
"""

from glob import glob
from os.path import join
from collections import defaultdict
import tensorflow as tf

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset


tfrecords_load_dir = ''
val_list_load_pathname = ''
gt_load_pathname = ''

gt_small_save_pathname = ''


def main():

    # get selected num list
    with open(val_list_load_pathname, 'r') as f:
        lines = f.readlines()
    selected = lines[:int(len(lines) / 10)]

    # get file idx and frame idx
    file_frame = defaultdict(list)
    for num in selected:
        prefix, file_idx, frame_idx = num[0], int(num[1:4]), int(num[4:7])
        file_frame[file_idx].append(frame_idx)

    # get ids
    selected_ids = set()
    tfrecord_pathnames = sorted(glob(join(tfrecords_load_dir, '*.tfrecord')))
    for file_idx, frame_idxs in file_frame.items():

        file_pathname = tfrecord_pathnames[file_idx]
        file_data = tf.data.TFRecordDataset(file_pathname, compression_type='')

        # process each frame
        for frame_idx in frame_idxs:
            frame = file_data[frame_idx]

        # prepare context_name and frame_timestamp_micros
        context_name = frame.context.name
        frame_timestamp_micros = frame.timestamp_micros

        selected_ids.add('{}-{}'.format(str(context_name), str(frame_timestamp_micros)))

    # get the gt with selected ids
    gt_objects = metrics_pb2.Objects()
    gt_small_objects = metrics_pb2.Objects()
    in_cnt = 0
    out_cnt = 0
    with open(gt_load_pathname, 'rb') as f:
        gt_objects.ParseFromString(f.read())
    for obj in gt_objects.objects:
        context_name = obj.context_name
        frame_timestamp_micros = obj.frame_timestamp_micros
        id = '{}-{}'.format(str(context_name), str(frame_timestamp_micros))
        if id in selected_ids:
            in_cnt += 1
            gt_small_objects.objects.append(obj)
        else:
            out_cnt += 1

    print(in_cnt, out_cnt, 'should be around 1:9')

    with open(gt_small_save_pathname, 'wb') as f:
        f.write(gt_small_objects.SerializeToString())


if __name__ == '__main__':
    main()
