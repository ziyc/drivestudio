"""
    This script generates number list text files
    Number list contains stem of filenames
    Number list specifies the files used for a certain data split

    Prefix:
        - training: 0 (training/)
        - validation: 1 (training/)
        - testing: 2 (testing/)
        - domain adaptation training labelled: 4 (training/)
        - domain adaptation training unlabelled: 5
        - domain adaptation validation labelled: 6 (training/)
        - domain adaptation validation unlabelled: 7
        - domain adaptation testing: 8
"""

import glob

path = '/mnt/lustre/share/DSK/datasets/waymo_open_dataset_kitti/training/label_all'
save_pathname = 'val.txt'
prefixes = []
skip_empty = True  # set false for testing set
num_files = 100  # set None to take all


def main():
    filenames = []
    for prefix in prefixes:
        filenames.extend(sorted(glob.glob(path + '/' + prefix + '*.txt')))
    if num_files is not None:
        filenames = filenames[:num_files]

    num_list = []
    for filename in filenames:
        # skip empty files
        if skip_empty:
            with open(filename, 'r') as ff:
                lines = ff.readlines()
            if not lines:  # empty
                print('Skipping', filename)
                continue

        num = filename.split('/')[-1][:-4]
        num_list.append(num)

    with open(save_pathname, 'w') as ot:
        for num in num_list:
            ot.write(num + '\n')


if __name__ == '__main__':
    main()
