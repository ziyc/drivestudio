from viewer.viewer import Viewer
import numpy as np
from dataset.waymo_base import WaymoDataset

def waymo_viewer():
    root = "D:\GithubDocument\3D-Detection-Tracking-Viewer\dataset\waymo"
    gt_info = "D:\GithubDocument\3D-Detection-Tracking-Viewer\dataset\waymo"
    pred_info = "D:\GithubDocument\3D-Detection-Tracking-Viewer\dataset\waymo"
    data = WaymoDataset(root, gt_info, pred_info)

    vi = Viewer()

    for i in range(len(data)):

        infos = data[i]

        lidar_points=infos['points']
        gt_boxes=infos['gt_boxes']
        gt_names=infos['gt_names']
        pred_boxes=infos['pred_boxes']
        pred_scores=infos['pred_scores']
        pred_names=infos['pred_names']

        vi.add_3D_boxes(gt_boxes,color='red',box_info=gt_names,show_corner_spheres=False)
        vi.add_3D_boxes(pred_boxes, color='blue',box_info=pred_names,show_corner_spheres=False)
        vi.add_points(lidar_points[:,0:3],color='gray')
        vi.show_3D()

if __name__ == '__main__':
    waymo_viewer()
