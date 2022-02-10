import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LoadDataset(Dataset):
    def __init__(self, images_hdf5_path, targets_hdf5_path):
        self.images_hdf5_path = images_hdf5_path
        self.targets_hdf5_path = targets_hdf5_path
        self.images_ndarray = self._read_hdf5(self.images_hdf5_path)
        self.targets_ndarray = self._read_hdf5(self.targets_hdf5_path)

    def _read_hdf5(self, hdf5_path):
        print(f"LOADING DATA FROM {hdf5_path}...")
        with h5py.File(hdf5_path, 'r') as hf:
            #idx = np.array(hf['idx'])
            if hdf5_path.contains("masks"):
                X = np.array(hf['mask'])
            else:
                X = np.array(hf['X'])
            X = np.expand_dims(X, axis=1)

            return X
    def __len__(self):
        return self.images_ndarray.shape[0]

    def __getitem__(self, idx):
        img = self.images_ndarray[idx]
        target = self.targets_ndarray[idx]

        return {
            'image': torch.as_tensor(img).float(),
            'target': torch.as_tensor(target).float(),
        }






