import tensorflow as tf
# from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset
# import numpy as np

tfrecord_file_name = '/media/alex/Seagate Expansion Drive/waymo_open_dataset/domain_adaptation_training_labelled(partial)/segment-10495858009395654700_197_000_217_000.tfrecord'
# tfrecord_file_name = '/media/alex/Seagate Expansion Drive/waymo_open_dataset/testing(partial)/segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord'
# gt_bin_pathname = '/home/alex/github/waymo_to_kitti_converter/submission/validation_ground_truth_objects_gt.bin'

def read_tfrecord():
    dataset = tf.data.TFRecordDataset(tfrecord_file_name, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # print(frame)
        # print('\n\n\n\n')
        # print(frame.context)

        # print(np.array(frame.pose.transform).reshape(4,4))

        for obj in frame.laser_labels:
            print(obj)
            # print(obj.num_lidar_points_in_box)


        # print(frame.context.name)
        # print(type(frame))
        # print(type(frame.laser_labels[0]))
        # # print([i.num_lidar_points_in_box for i in frame.laser_labels])
        # print([i.type for i in frame.laser_labels])

        # for obj in frame.laser_labels:
        #     print(obj.num_lidar_points_in_box)

        assert 0


# def read_gt_bin():
#     gt_objects = metrics_pb2.Objects()
#     with open(gt_bin_pathname, 'rb') as f:
#         gt_objects.ParseFromString(f.read())
#     print(type(gt_objects))
#     print(gt_objects.objects[0])


if __name__ == '__main__':
    read_tfrecord()
    # read_gt_bin()
