"""
This program checks the accuracy of oxts files (IMU) by overlaying the transformed point clouds together, followed by visualization
"""

import pykitti.utils as utils
import open3d as o3d
from os.path import basename, splitext, join
from glob import glob
import numpy as np


#load_dir = '/home/caizhongang/Datasets/kitti/kitti_raw_data/2011_09_26/2011_09_26_drive_0017_sync'
load_dir = '/media/alex/Seagate Expansion Drive/kitti/kitti_raw_data/2011_10_03/2011_10_03_drive_0042_sync'
pc_load_dir = join(load_dir, 'velodyne_points/data')
oxts_load_dir = join(load_dir, 'oxts/data')

def transform(pc, tf):
	tf_pc = []
	tf = np.array(tf)
	for p in pc:
		p = np.array(p[0:3].tolist() + [1.0])
		#print(tf, p)
		tf_p = np.matmul(tf, p)
		tf_p = tf_p[0:3] / tf_p[3]
		tf_pc.append(tf_p.tolist())
	return tf_pc		


if __name__ == '__main__':
	oxts_pathnames = sorted(glob(join(oxts_load_dir, '*.txt')))
	oxts_all = utils.load_oxts_packets_and_poses(oxts_pathnames)

	pc_all = []
	oxts_pathnames = oxts_pathnames[:1]
	for i, oxts_pathname in enumerate(oxts_pathnames):
		stem, ext = splitext(basename(oxts_pathname))
		pc_pathname = join(pc_load_dir, stem+'.bin')
		pc = utils.load_velo_scan(pc_pathname)
		
		oxts = oxts_all[i]
		#print(type(oxts))	
		tf = oxts[1]
		#print(tf)
		
		tf_pc = transform(pc, tf)
		
		pc_all += tf_pc

	pcd = o3d.geometry.PointCloud()
	print(np.array(pc_all).shape, np.array(pc_all).dtype)
	pcd.points = o3d.utility.Vector3dVector(np.array(pc_all))
	o3d.visualization.draw_geometries([pcd])		
			
	




