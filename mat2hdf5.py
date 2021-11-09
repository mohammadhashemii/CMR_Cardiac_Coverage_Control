'''
This script takes the directory name of the mat files
and create a HDF5 file of the mat files for later use.
'''

import argparse
import os
import scipy.io as sio
import skimage.transform as skt
import numpy as np
import h5py

parser = argparse.ArgumentParser()

parser.add_argument('--data_directory', type=str, default='data/APEX/', help='directory of the mat files')
parser.add_argument('--save_as', type=str, default='dataset_APEX.hdf5', help='hdf5 file name')
parser.add_argument('--image_size', type=tuple, default=(128, 128), help='converted image shape')

args = parser.parse_args()

X = []
Y = []
mat_file_names_list = os.listdir(args.data_directory)
idx = 0

print("Converting mat files to HDF5 ...")
for mf_name in mat_file_names_list:
    if not mf_name.endswith(".mat"):
        continue
    mat_file_path = args.data_directory + mf_name
    mat_file = sio.loadmat(mat_file_path)
    mylist = list(mat_file.values())
    mat_file_values = np.array([mylist[3]])
    volume_size = (args.image_size[0], args.image_size[1], 3)
    X.append(skt.resize(mylist[3], volume_size, order=0))

    if mf_name.find("miss") != -1:
        Y.append(0)
    else:
        Y.append(1)
    idx += 1
    if idx % 1000 == 0:
        print('{}/{} files have loaded yet'.format(idx, len(mat_file_names_list)))
X = np.array(X)
Y = np.array(Y)

with h5py.File(args.data_directory.split("/")[0] + "/" + args.save_as, 'w') as hf:
    dset_x = hf.create_dataset('X', data=X, shape=(len(mat_file_names_list), args.image_size[0], args.image_size[1], 3),
                               compression='gzip', chunks=True)
    dset_y = hf.create_dataset('Y', data=Y, shape=(len(mat_file_names_list), 1), compression='gzip', chunks=True)

print("mat files converted to a single HDF5 file completely!")
print("Input shape: {}".format(X.shape))
print("Target shape: {}".format(Y.shape))
print("=============================")