import numpy as np
from multiprocessing import Pool
from os.path import join, basename, splitext
from glob import glob
from tqdm import tqdm

point_cloud_load_dir = 'test_point_cloud'
label_load_dir = 'test_label'
calib_load_dir = 'test_calib'
cropped_point_cloud_save_dir = 'test_crop_car'
num_proc = 8
min_points = 100
crop_class = 'Car'

def visual(pc):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc) 
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0,0,0]) 
    visual = [pcd, axis] 
    o3d.visualization.draw_geometries(visual)


def transform_point_cloud(pc, T):
    assert pc.shape[1] == 3
    pc = np.transpose(pc)
    
    if T.shape == (4, 4):
        pc = np.vstack([pc, np.ones((1, pc.shape[1]))])
    
    pc = T @ pc
    pc = pc[:3, :]
    pc = np.transpose(pc)
    return pc

def crop_point_cloud(pc, x, y, z, h, w, l, ry):  
    # center point cloud
    pc[:, 0] -= x
    pc[:, 1] -= y
    pc[:, 2] -= z
    
    # compute bbox heading in velo frame
    # kitti camera(label): rotate around y (down), starting from right, counter-clockwise
    # kitti velo(point cloud): rotate around z (up), starting from front, counter-clockwise
    heading = - (ry + np.pi / 2)
    while heading < -np.pi:
        heading += 2 * np.pi
    while heading > np.pi:
        heading -= 2 * np.pi
    
    # compute rz
    # rotate point cloud instead of the bounding box
    rz = - heading
    T = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz),  np.cos(rz), 0.0],
        [0.0       ,         0.0, 1.0]
    ])
    pc = transform_point_cloud(pc, T)
    
    # compute limits
    x_min = - l / 2.0
    x_max =   l / 2.0
    y_min = - w / 2.0
    y_max =   w / 2.0
    z_min =   0
    z_max =   h

    # crop
    mask = (
        (pc[:, 0] >= x_min) *
        (pc[:, 0] <= x_max) *
        (pc[:, 1] >= y_min) *
        (pc[:, 1] <= y_max) *
        (pc[:, 2] >= z_min) *
        (pc[:, 2] <= z_max)
    )
    cropped = pc[mask]
    
    return cropped        


def one_process(idx):
    # load point cloud
    point_cloud_load_pathname = join(point_cloud_load_dir, idx + '.bin')
    point_cloud = np.fromfile(point_cloud_load_pathname, dtype=np.float32).reshape(-1, 4)
    point_cloud = point_cloud[:, :3]
    
    # load calib
    calib_load_pathname = join(calib_load_dir, idx + '.txt')
    with open(calib_load_pathname, 'r') as f:
        calibs = f.readlines()

    for calib in calibs:
        calib = calib.split()
        if len(calib) != 0 and calib[0] == 'Tr_velo_to_cam:':
            T_v2c = np.array([float(x) for x in calib[1:]]).reshape(3, 4)
            T_v2c = np.vstack([T_v2c, np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,4)])
            T_c2v = np.linalg.inv(T_v2c)
            
    # load label
    label_load_pathname = join(label_load_dir, idx + '.txt')
    with open(label_load_pathname, 'r') as f:
        labels = f.readlines()
    crop_labels = []
    for label in labels:
        cls, tru, occ, alp, x1, y1, x2, y2, h, w, l, x, y, z, ry = label.split()       
        if cls == crop_class:
            x, y, z, h, w, l, ry = float(x), float(y), float(z), float(h), float(w), float(l), float(ry)
            t_v = T_c2v @ np.array([x, y, z, 1.0]).reshape(4, 1)
            x_v, y_v, z_v, _ = t_v.flatten().tolist()
            crop_labels.append((x_v, y_v, z_v, h, w, l, ry))
      
    # crop and save
    for i, crop_label in enumerate(crop_labels):
        cropped_point_cloud = crop_point_cloud(point_cloud.copy(), *crop_label)
        if cropped_point_cloud.shape[0] < min_points:
            continue
        
        cropped_point_cloud_save_pathname = join(cropped_point_cloud_save_dir, idx + '-{:03d}.bin'.format(i))
        cropped_point_cloud.astype(np.float32).tofile(cropped_point_cloud_save_pathname)
        
    
def main():
    label_pathnames = sorted(glob(join(label_load_dir, '*.txt')))
    idxs = [splitext(basename(pathname))[0] for pathname in label_pathnames]
        
    with Pool(num_proc) as p:
        r = list(tqdm(p.imap(one_process, idxs), total=len(idxs)))
    #for idx in idxs:
    #    one_process(idx)


if __name__ == '__main__':
    main()


