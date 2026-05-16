"""
Ensemble individual .bin files into one .bin file
"""

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


# (pathname, [class])
load_pathnames_classes = [
    ('/home/alex/github/waymo_to_kitti_converter/submission/20200519/PartA2_waymo_nosign_40_all12_test.bin',
     ['VEHICLE']),

    ('/home/alex/github/waymo_to_kitti_converter/submission/20200519/PartA2_waymo_nosign_40_cyconly_50_aug38_test.bin',
     ['CYCLIST']),

    ('/home/alex/github/waymo_to_kitti_converter/submission/20200519/PartA2_waymo_nosign_40_ped_1034_test.bin',
     ['PEDESTRIAN'])
]

save_pathname = '/home/alex/github/waymo_to_kitti_converter/submission/20200519/ensemble.bin'


def main():

    cls_map = {
        'VEHICLE': label_pb2.Label.TYPE_VEHICLE,
        'CYCLIST': label_pb2.Label.TYPE_CYCLIST,
        'PEDESTRIAN': label_pb2.Label.TYPE_PEDESTRIAN
    }

    ensemble = metrics_pb2.Objects()
    for file in load_pathnames_classes:
        pathname, cls = file
        print('Processing', pathname, 'for class', cls)

        cls = [cls_map[c] for c in cls]
        # load bin
        objects = metrics_pb2.Objects()
        with open(pathname, 'rb') as f:
            objects.ParseFromString(f.read())
        for o in objects.objects:
            if o.object.type in cls:
                ensemble.objects.append(o)

    # save ensemble
    with open(save_pathname, 'wb') as f:
        f.write(ensemble.SerializeToString())
    print(save_pathname, 'saved.')


if __name__ == '__main__':
    main()
