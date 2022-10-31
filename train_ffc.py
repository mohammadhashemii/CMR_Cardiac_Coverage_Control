import torch
from torch.utils.data import DataLoader
from dataset import MRIDataset
from model_ffc import FFC3D
from utils import load_config

# Loading configs
train_configs = load_config("training_settings.yaml")

# Creating datasets
train_dataset = MRIDataset(root=train_configs['data_path'],
                           input_size=train_configs['input_size'], fold_no=0, is_training=True)
test_dataset = MRIDataset(root=train_configs['data_path'],
                          input_size=train_configs['input_size'], fold_no=0, is_training=False)

print("{} TRAIN DATA loaded!".format(len(train_dataset)))
print("{} TEST DATA loaded!".format(len(test_dataset)))

train_loader = DataLoader(train_dataset, **train_configs['data_loader'])
test_loader = DataLoader(test_dataset, **train_configs['data_loader'])

# Model defining
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
init_conv_kwargs = {'ratio_gin': 0, 'ratio_gout': 0.5, 'enable_lfu': False}
internal_conv_kwargs = {'ratio_gin': 0.5, 'ratio_gout': 0.5, 'enable_lfu': False}
ffc3d = FFC3D(input_nc=1, init_conv_kwargs=init_conv_kwargs, internal_conv_kwargs=internal_conv_kwargs).to(device)

# optimization settings
optimizer = torch.optim.SGD(ffc3d.parameters(),
                            lr=train_configs['lr'],
                            momentum=train_configs['momentum'])

for epoch in range(train_configs['n_epochs']):
    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.to(device)
        batch_images = torch.unsqueeze(batch_images, dim=1)
        out = ffc3d(batch_images.permute(0, 1, 4, 2, 3))
        print(out)
