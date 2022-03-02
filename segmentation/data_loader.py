import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class LoadDataset(Dataset):
    def __init__(self, images_hdf5_path, targets_hdf5_path, contains_target=True, transform=False):
        self.contains_target = contains_target
        self.images_hdf5_path = images_hdf5_path
        self.images_ndarray = self._read_hdf5(self.images_hdf5_path)
        self.transform = transform
        if contains_target:
            self.targets_hdf5_path = targets_hdf5_path
            self.targets_ndarray = self._read_hdf5(self.targets_hdf5_path)

    def _read_hdf5(self, hdf5_path):
        print(f"LOADING DATA FROM {hdf5_path}...")
        with h5py.File(hdf5_path, 'r') as hf:
            #idx = np.array(hf['idx'])
            if "masks" in hdf5_path:
                X = np.array(hf['mask'])
            else:
                X = np.array(hf['X'])
            X = np.expand_dims(X, axis=1)

            return X

    def __len__(self):
        return self.images_ndarray.shape[0]

    def augment(self, image, target):
        """ Apply some random transformation"""
        # 1. Rotate by one of the given angles.
        angles = [-30, -20, -10, 10, 20, 30]
        random_angle = random.choice(angles)
        image = TF.rotate(image, random_angle)
        target = TF.rotate(target, random_angle)

        # 2. Flip vertically or horizontally
        if random.random() > 0.5:
            image = TF.vflip(image)
            target = TF.vflip(target)

        if random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)

        # 3. Adjust brightness of an image by a random brightness factor
        brightness_factor = random.uniform(0.4, 1.3)
        image = TF.adjust_brightness(image, brightness_factor)

        return image, target


    def __getitem__(self, idx):
        img = self.images_ndarray[idx]
        if self.contains_target:
            if self.transform:
                img = torch.tensor(img).permute(0, 3, 1, 2)  # (batch, c, h, w)

                target = self.targets_ndarray[idx]
                target = torch.tensor(target).permute(0, 3, 1, 2)  # (batch, c, h, w)

                img, target = self.augment(image=img, target=target)
                img = img.permute(0, 2, 3, 1)  # (batch, h, w, c)
                target = target.permute(0, 2, 3, 1)  # (batch, h, w, c)

                #plot_volume(img[0])
                #plot_volume(target[0])
            else: # no augmentation
                img = self.images_ndarray[idx]
                target = self.targets_ndarray[idx]

            return {
                'image': torch.as_tensor(img).float(),
                'target': torch.as_tensor(target).float()
            }
        else:

            return {
                'image': torch.as_tensor(img).float(),
            }

'''
dataset = LoadDataset(images_hdf5_path='data/apex_lime/correct_predictions_apex.hdf5',
                      targets_hdf5_path='data/apex_lime/masks_apex.hdf5',
                      contains_target=True)

train_loader = DataLoader(dataset, shuffle=False, batch_size=1)

for batch in train_loader:
    img = batch['image']
'''


