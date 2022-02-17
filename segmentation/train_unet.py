import argparse
import os
import json
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import warnings
warnings.filterwarnings('ignore')

from data_loader import LoadDataset
from attention_unet import UNet3D
from dice_score import dice_loss
#from torchgeometry.losses import dice_loss

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--data_root', type=str, default='data/', help='path to the root of data directory')
parser.add_argument('--images_path', type=str, default='apex_lime/correct_predictions_apex.hdf5', help='images path')
parser.add_argument('--target_path', type=str, default='apex_lime/masks_apex.hdf5', help='targets path')
parser.add_argument('--exp_dir', type=str, default='segmentation/experiments/', help='saved experiences directory')
parser.add_argument('--exp_no', type=str, default='00', help='experiment number')

# training
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--resume', type=str, help='load weights')
parser.add_argument('--test_size', type=int, default=0.2, help='percentage of test size')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay rate')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.exp_dir + args.exp_no):
    os.makedirs(args.exp_dir + args.exp_no)
# Initialize logging
training_configs = dict(exp=args.exp_no, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                              test_size=args.test_size, device=device.type)
json_path = args.exp_dir + args.exp_no + f"/{args.exp_no}.json"
with open(json_path, "w") as jf:
    json.dump(training_configs, jf)
print(training_configs)

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
train_loader = DataLoader(train_set, shuffle=False, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, **loader_args)

# model creation
unet = UNet3D(in_channels=1,
              out_channels=1,
              final_sigmoid=True)
unet.to(device=device)
if args.resume is not None:
    unet.load_state_dict(torch.load(args.resume))
    print(f"Model weights {args.resume} loaded!")

# set up optimizer, the loss, learning rate and etc.
optimizer = torch.optim.RMSprop(unet.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)

# start training
for epoch in range(args.epochs):
    criterion = nn.BCEWithLogitsLoss()
    epoch_train_loss_list = []
    epoch_test_loss_list = []
    epoch_train_dice_score_list = []
    epoch_test_dice_score_list = []
    unet.train()
    try:
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                targets = batch['target'].to(device=device, dtype=torch.float32)
                prediction, pool_fea = unet(images)

                probs = (torch.sigmoid(prediction) > 0.5).float()
                dl = dice_loss(probs[:, 0, :, :, :], targets[:, 0, :, :, :])
                dice_score = 1 - dl
                train_loss = criterion(prediction[:, 0, :, :, :], targets[:, 0, :, :, :]) + dl
                epoch_train_loss_list.append(train_loss)
                epoch_train_dice_score_list.append(dice_score)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                pbar.update(images.shape[0])

            avg_train_loss = torch.mean(torch.tensor(epoch_train_loss_list))
            avg_train_dice_score = torch.mean(torch.tensor(epoch_train_dice_score_list))

            # evaluate
            for batch in test_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                targets = batch['target'].to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    # predict the mask
                    prediction, pool_fea = unet(images)
                    probs = (torch.sigmoid(prediction) > 0.5).float()
                    dl = dice_loss(probs[:, 0, :, :, :], targets[:, 0, :, :, :])
                    dice_score = 1 - dl
                    test_loss = criterion(prediction, targets) + dl
                    epoch_test_loss_list.append(test_loss)
                    epoch_test_dice_score_list.append(dice_score)

            avg_test_loss = torch.mean(torch.tensor(epoch_test_loss_list))
            avg_test_dice_score = torch.mean(torch.tensor(epoch_test_dice_score_list))

            log_file_path = args.exp_dir + args.exp_no + f"/loss.txt"
            loss_log = f"Train loss: {avg_train_loss}, Test loss:{avg_test_loss}\n"
            dice_log = f"Train dice score: {avg_train_dice_score}, Test dice score:{avg_test_dice_score}\n\n"
            print(loss_log+dice_log)
            with open(log_file_path, "a") as f:
                f.write(loss_log+dice_log)
            f.close()
        print(f"Model saved at {args.exp_dir}{args.exp_no}/exp{args.exp_no}.pth")
        torch.save(unet.state_dict(), f"{args.exp_dir}{args.exp_no}/exp{args.exp_no}.pth")
    except KeyboardInterrupt:
        print(f"Model saved at {args.exp_dir}{args.exp_no}/exp{args.exp_no}.pth")
        torch.save(unet.state_dict(), f"{args.exp_dir}{args.exp_no}/exp{args.exp_no}.pth")




