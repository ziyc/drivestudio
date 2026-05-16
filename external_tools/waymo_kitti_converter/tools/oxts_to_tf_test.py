"""
This test program parses oxts .txt files and print transformation
"""


import pykitti.utils as utils
from glob import glob

load_dir = '/home/caizhongang/Datasets/kitti/kitti_raw_data/2011_09_26/2011_09_26_drive_0017_sync/oxts/data'

if __name__ == '__main__':
	paths = glob(load_dir + '/*.txt')
	oxts = utils.load_oxts_packets_and_poses(paths)
	print(oxts[1])
