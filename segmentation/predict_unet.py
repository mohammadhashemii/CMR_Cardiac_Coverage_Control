import argparse
import os
import torch
from PIL import Image
import numpy as np
from torch._C import dtype
from torch.utils.data import DataLoader, random_split

from data_loader import LoadDataset
from attention_unet import UNet3D
from utils import save_img_and_mask

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--data_root', type=str, default='data/', help='path to the root of data directory')
parser.add_argument('--images_path', type=str, default='apex_lime/correct_predictions_apex.hdf5', help='images path')
parser.add_argument('--target_path', type=str, default='apex_lime/masks_apex.hdf5', help='targets path')
parser.add_argument('--exp_dir', type=str, default='segmentation/experiments/', help='saved experiences directory')
parser.add_argument('--exp_no', type=str, default='00', help='experiment number')

# prediction
parser.add_argument('--save_preds_no', type=int, default=50, help='number of samples to save')
parser.add_argument('--out_threshold', type=float, default=0.5, help='threshold for mask prediction')
parser.add_argument('--test_size', type=int, default=0.2, help='percentage of test size')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

results_dir = args.exp_dir + args.exp_no + "/results/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# load the images and targets
loader_args = dict(batch_size=1,
                   num_workers=2)

dataset = LoadDataset(images_hdf5_path=args.data_root + args.images_path,
                      targets_hdf5_path=args.data_root + args.target_path)

# to get the samples used in the test set for prediction
n_test = int(len(dataset) * args.test_size)
n_train = len(dataset) - n_test
train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(0))
test_loader = DataLoader(test_set, shuffle=False, **loader_args)

# load model
unet = UNet3D(in_channels=1,
              out_channels=1,
              final_sigmoid=False)

unet.to(device=device)
saved_model_path = args.exp_dir + args.exp_no + f"/exp{args.exp_no}.pth"
unet.load_state_dict(torch.load(saved_model_path, map_location=device))
print(f"Model {saved_model_path} loaded!")


def mask_to_image(mask: np.ndarray):
    return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


# prediction
unet.eval()
idx = 0
with torch.no_grad():
    for batch in test_loader:  # each batch contains only 1 sample
        if idx > args.save_preds_no:
            print(f"results saved at {results_dir}")
            break
        image = batch['image'].to(device=device, dtype=torch.float32)
        ground_truth = batch['target'].to(device=device, dtype=torch.float32)

        pred, _ = unet(image)
        probs = torch.sigmoid(pred)[0, 0] > args.out_threshold

        image = image[0, 0].cpu().numpy()
        ground_truth = ground_truth[0, 0].cpu().numpy()
        mask = probs.cpu().numpy()
        save_img_and_mask(image, mask, ground_truth, filepath=results_dir + f"{idx}.jpg")
        idx += 1




