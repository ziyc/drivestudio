import numpy as np
from glob import glob
from os.path import join, basename, splitext
import open3d as o3d

load_dir = '/home/alex/github/waymo_to_kitti_converter/tools/dataloader/one_example'
pc_range = [-51.2, -51.2, -3, 51.2, 51.2, 9]
test_range = [0, -40, -3.0, 70.4, 40, 3.0]

# load_dir = '/home/alex/github/waymo_to_kitti_converter/tools/dataloader/kitti_one_example'
# pc_range = [0, -39.68, -3, 69.12, 39.68, 1]
# test_range = [0, -40, -3.0, 70.4, 40, 3.0]


def corners_to_lines(qs, color=[0,0,0]):
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
    # print('qs', qs)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(qs),
        lines=o3d.utility.Vector2iVector(idx),
    )
    line_set.colors = o3d.utility.Vector3dVector(cl)

    return line_set


def boxes3d_to_corners3d_lidar(x, y, z, w, l, h, ry):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, rz] in LiDAR coords
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

    # due to convention difference, h is for y and w is for x and l is for z
    # also, box origin at bottom

    x_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.])
    y_corners = np.array([-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.])
    z_corners = np.array([0, 0, 0, 0, h, h, h, h])

    # x_corners = np.array([l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2.])
    # y_corners = np.array([-w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2., w / 2.])
    # z_corners = np.array([0,0,0,0,h,h,h,h])

    corners = np.concatenate((x_corners.reshape(1, 8), y_corners.reshape(1, 8), z_corners.reshape(1, 8)), axis=0)  # (3, 8)

    print('ry\n', ry)

    # actually Rz, as ry is in ref frame
    Ry = np.array([[ np.cos(ry), -np.sin(ry), 0],
                   [ np.sin(ry),  np.cos(ry), 0],
                   [          0,           0, 1]])

    print('Rot\n', Ry)
    print('corners', corners.T)

    rotated_corners = np.matmul(Ry, corners)  # (3, 8)

    print('rotated_corners', rotated_corners.T)

    rotated_corners += np.array([x, y, z]).reshape(3, 1)

    print('shifted_corners', rotated_corners.T)

    return rotated_corners.astype(np.float32)


def get_range_box(xmin, ymin, zmin, xmax, ymax, zmax):
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

    x_corners = np.array([xmax, xmin, xmin, xmax, xmax, xmin, xmin, xmax])
    y_corners = np.array([ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax])
    z_corners = np.array([zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax])

    corners = np.concatenate((x_corners.reshape(1, 8), y_corners.reshape(1, 8), z_corners.reshape(1, 8)), axis=0)  # (3, 8)

    return corners.astype(np.float32)


def main():
    all_content = {}
    all_bin_pathnames = glob(join(load_dir, '*.bin'))
    for bin_pathname in all_bin_pathnames:
        stem, _ = splitext(basename(bin_pathname))
        splits = stem.split('-')
        k, d = splits[0], splits[1]
        if d == 'object':
            continue
        shape = splits[2:]
        shape = tuple([int(s) for s in shape])
        all_content[k] = np.fromfile(bin_pathname, dtype=d).reshape(shape)
        print(k, all_content[k].shape)

    print(all_content['points'])
    point_cloud = all_content['points'][:, 1:4]  # (N, 3) TODO: what??? idx 1:4
    bboxes = all_content['gt_boxes'][0]  # (N, 8), 7 param + type

    print(all_content['gt_boxes'])
    print('==============')
    print(all_content['sample_idx'])

    # test point cloud
    full_pc_pathname = glob(join(load_dir, 'full_pc/*.bin'))[0]
    full_pc = np.fromfile(full_pc_pathname, dtype=np.float32).reshape(-1, 4)
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(full_pc[:, :3])
    full_pcd.paint_uniform_color([0, 0, 0])

    bboxes_corners = []
    for i in range(bboxes.shape[0]):
        x, y, z, w, l, h, ry, t = bboxes[i].tolist()  # lidar coord: wlh
        print(x,y,z,w,l,h,ry,t)
        bbox_corners = boxes3d_to_corners3d_lidar(x, y, z, w, l, h, ry)  # lidar coord
        bboxes_corners.append(bbox_corners)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    voxels = o3d.geometry.PointCloud()
    voxels.points = o3d.utility.Vector3dVector(all_content['voxel_centers'])
    voxels.paint_uniform_color([1, 0, 0])

    pc_range_box = get_range_box(*pc_range)
    pc_range_box_lines = corners_to_lines(pc_range_box.T.tolist(), color=[1,0,0])

    test_range_box = get_range_box(*test_range)
    test_range_box_lines = corners_to_lines(test_range_box.T.tolist())

    voxel_box = get_range_box(0,0,-3,0.16,0.16,1)
    voxel_box_lines = corners_to_lines(voxel_box.T.tolist())

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])

    # visual = [full_pcd, pcd, voxels, axis, pc_range_box_lines, test_range_box_lines, voxel_box_lines]
    visual = [pcd, voxels, axis, pc_range_box_lines, test_range_box_lines, voxel_box_lines]
    # visual = [pcd, axis, pc_range_box_lines, test_range_box_lines, voxel_box_lines]

    for bbox3d in bboxes_corners:
        bbox3d = bbox3d.T.tolist()  # (8, 3)
        # print(bbox3d)
        visual.append(corners_to_lines(bbox3d, color=[0, 0, 1]))

    o3d.visualization.draw_geometries(visual)

    # print(all_content)


if __name__ == '__main__':
    main()
