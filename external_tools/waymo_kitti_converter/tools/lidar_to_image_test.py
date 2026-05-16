import cv2
import numpy as np
from calibration import get_calib_from_file

# kitti
# name = '000000'
# pc_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/kitti/velodyne/'+name+'.bin'
# img_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/kitti/image_2/'+name+'.png'
# calib_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/kitti/calib/'+name+'.txt'

# waymo-kitti
name = '00000-00001'
pc_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_kitti/velodyne/'+name+'.bin'
img_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_kitti/image_0/'+name+'.png'
calib_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_kitti/calib/'+name+'.txt'


def cart_to_homo(mat):
    mat = np.vstack([mat, np.ones((1, mat.shape[1]))])
    return mat


def pc_to_pt(pc, V2C, R0, P):

    def cart2hom(pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_velo_to_ref(pts_3d_velo):
        pts_3d_velo = cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(V2C))

    def project_ref_to_rect(pts_3d_ref):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(R0, np.transpose(pts_3d_ref)))

    def project_rect_to_image(pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    # filter behind
    ind = pc[:, 0] > 0  # lidar: x is front
    pc = pc[ind, :]

    print('pc', pc)
    ref = project_velo_to_ref(pc)
    print('ref',ref)

    rect = project_ref_to_rect(ref)
    print('rect', rect)

    depth = rect[:, 2]

    print(rect.shape, depth.shape)
    image = project_rect_to_image(rect)

    return image, depth

def main():
    calib = get_calib_from_file(calib_pathname)

    v2c = calib['Tr_velo2cam']
    r0 = calib['R0']
    px = calib['P2']

    # v2c = np.array([
    #     [7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03],
    #     [1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02],
    #     [9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01]])
    # r0 = np.array([
    #     [9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03],
    #     [-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03],
    #     [7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01]])
    # px = np.array([
    #     [7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
    #     [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
    #     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]])

    pc = np.fromfile(pc_pathname, dtype=np.float32).reshape((-1, 4))[:, :3]

    # filter all behind image plane
    keep = []
    for i in range(pc.shape[0]):
        p = pc[i, :]
        if p[0] > 0:
            keep.append(p)
    # pc = np.vstack(keep)

    #
    # tmp = np.eye(4)
    # tmp[:3, :3] = r0
    # r0 = tmp

    # pc = np.transpose(pc) # (n,3) -> (3,n)
    # pc = cart_to_homo(pc) # (3,n) -> (4,n)
    #
    # v2c = cart_to_homo(v2c) # (3,4) -> (4,4)
    #
    # print(px.shape, r0.shape, v2c.shape, pc.shape)

    pt, depth = pc_to_pt(pc, v2c, r0, px)
    print(pt.shape, depth.shape)

    # pt = px @ r0 @ v2c @ pc
    # print(pt.shape)
    # pt = pt[:2] / pt[2]

    print(pt)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    # draw
    img = cv2.imread(img_pathname)

    for i in range(pt.shape[0]):
        x = pt[i, 0]
        y = pt[i, 1]
        color = cmap[np.clip(640/depth[i], 0, 255).astype(np.int), :]
        # if 0 < x < 1920 and 0 < y < 1080:
        #     print('yah')
        # print(int(x), int(y))
        cv2.circle(img, (int(x), int(y)), 1, tuple(color), -1)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == 27:  # exit
            break
        elif key != -1:
            print('Undefined key:', key)


if __name__ == '__main__':
    main()


