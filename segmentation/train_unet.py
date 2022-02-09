import argparse
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchgeometry as tgm

from data_loader import LoadDataset
from attention_unet import UNet3D
from dice_score import dice_loss

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--data_root', type=str, default='data/', help='path to the root of data directory')
parser.add_argument('--images_path', type=str, default='apex_lime/correct_predictions_apex.hdf5', help='images path')
parser.add_argument('--target_path', type=str, default='apex_lime/perturbations_apex.hdf5', help='targets path')

# training
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--log_step', type=int, default=5, help='steps to print the loss and ...')
parser.add_argument('--test_size', type=int, default=0.2, help='percentage of test size')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay rate')
args = parser.parse_args()

# load the images and targets
loader_args = dict(batch_size=args.batch_size,
                   num_workers=args.num_workers,
                   pin_memory=True)

dataset = LoadDataset(images_hdf5_path=args.data_root + args.images_path,
                      targets_hdf5_path=args.data_root + args.target_path)

# split train/test
n_test = int(len(dataset) * args.test_size)
n_train = len(dataset) - n_test
train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_set, shuffle=False)
test_loader = DataLoader(test_set, shuffle=False)


# model creation
unet = UNet3D(in_channels=1,
              out_channels=1,
              final_sigmoid=False)

# set up optimizer, the loss, learning rate and etc.
optimizer = torch.optim.RMSprop(unet.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
criterion = nn.CrossEntropyLoss()

# start training
for epoch in range(args.epochs):
    unet.train()
    for batch in train_loader:
        images = batch['image']#.permute(0, 1, 4, 3, 2)
        targets = batch['target']#.permute(0, 1, 4, 3, 2)

        prediction, pool_fea = unet(images)
        train_loss = criterion(prediction, targets) + dice_loss(prediction[:, 0, :, :, :], targets[:, 0, :, :, :])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    # evaluate
    if epoch % args.log_step:
        print("Epoch[{0}/{1}], Train loss:{2}".format(epoch + 1, args.epochs, train_loss))
        for batch in test_loader:
            images = batch['image']  # .permute(0, 1, 4, 3, 2)
            targets = batch['target']  # .permute(0, 1, 4, 3, 2)
            with torch.no_grad():
                # predict the mask
                prediction, pool_fea = unet(images)

                test_loss = criterion(prediction, targets) + dice_loss(prediction[:, 0, :, :, :],
                                                                        targets[:, 0, :, :, :])
                print("Epoch[{0}/{1}], Test loss:{2}".format(epoch + 1, args.epochs, test_loss))




