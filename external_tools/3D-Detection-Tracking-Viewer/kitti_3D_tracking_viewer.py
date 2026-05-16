from viewer.viewer import Viewer
import numpy as np
from dataset.kitti_dataset import KittiTrackingDataset

def kitti_viewer():
    root = r"D:\GithubDocument\3D-Detection-Tracking-Viewer\dataset\KITTI"
    label_path = r"D:\GithubDocument\3D-Detection-Tracking-Viewer\dataset\KITTI"
    dataset = KittiTrackingDataset(root,seq_id=1,label_path=label_path)

    vi = Viewer(box_type="Kitti")

    for i in range(len(dataset)):
        P2, V2C, points, image, labels, label_names = dataset[i]


        if labels is not None:
            mask = (label_names=="Car")
            labels = labels[mask]
            label_names = label_names[mask]
            vi.add_3D_boxes(labels, ids=labels[:, -1].astype(int), box_info=label_names,caption_size=(0.09,0.09))
            vi.add_3D_cars(labels, ids=labels[:, -1].astype(int), mesh_alpha=1)
        vi.add_points(points[:,:3])

        vi.add_image(image)
        vi.set_extrinsic_mat(V2C)
        vi.set_intrinsic_mat(P2)

        vi.show_2D()

        vi.show_3D()


if __name__ == '__main__':
    kitti_viewer()
