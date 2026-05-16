"""
This program parses all oxts files in kitti raw data
save them in the same file structure for zipping, uploading and easy insertion
"""

from os import makedirs, listdir
from os.path import join, isdir, basename
from pykitti import utils
from glob import glob
import numpy as np

kitti_raw_dir = '/home/caizhongang/Datasets/kitti/kitti_raw_data'
save_dir_root = '/home/caizhongang/Datasets/kitti/tf_all'

def main():
	l1_dirs = listdir(kitti_raw_dir)
	for l1_dir in l1_dirs:  # eg. '2011_09_26/'
		full_l1_dir = join(kitti_raw_dir, l1_dir)
		if not isdir(full_l1_dir):
			continue
		print(full_l1_dir)

		l2_dirs = listdir(full_l1_dir)
		for l2_dir in l2_dirs: # eg. '2011_09_26_drive_0017_sync/'		
			full_l2_dir = join(kitti_raw_dir, l1_dir, l2_dir)
			if not isdir(full_l2_dir):
				continue
			print(full_l2_dir)
			
			oxts_paths = glob(join(full_l2_dir, 'oxts/data/*.txt'))

			# parse
			oxts = utils.load_oxts_packets_and_poses(oxts_paths)

			# save
			for i, oxts_path in enumerate(oxts_paths):
				# extract transformation				
				tf = oxts[i][1]
				assert type(tf) is np.ndarray and tf.shape == (4,4)

				save_dir = join(save_dir_root, l1_dir, l2_dir, 'oxts/tf')
				if not isdir(save_dir):
					makedirs(save_dir)				

				save_pathname = join(save_dir, basename(oxts_path))
				np.savetxt(save_pathname, tf)
				print(save_pathname, 'saved.')
	
if __name__ == '__main__':
	main()
