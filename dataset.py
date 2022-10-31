import torch
from torch.utils import data
import scipy.io as spio
import skimage.transform as skt
import glob
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class MRIDataset(data.Dataset):
    def __init__(self, root, input_size=128, fold_no=0, is_training=True):
        self.root = root
        self.input_size = input_size
        self.filenames = glob.glob(root + "*.mat")
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        train_idx, test_idx = list(kfold.split(self.filenames))[fold_no]
        if is_training:
            self.filenames = list(map(self.filenames.__getitem__, train_idx))
        else:
            self.filenames = list(map(self.filenames.__getitem__, test_idx))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_path = self.filenames[index]
        mat_file = spio.loadmat(image_path)
        mylist = list(mat_file.values())
        volume_size = (self.input_size, self.input_size, 3)
        image = skt.resize(mylist[3], volume_size, order=0)
        image = torch.Tensor(image)

        label = 0 if image_path.find("miss") != -1 else 1

        return image, label

# mri = MRIDataset(root="./data/APEX/", input_size=128, is_training=False)
# print(len(mri))
# print(mri[0][0].shape)
# plt.imshow(mri[0][0][:, :, 2], cmap="gray")
# plt.show()