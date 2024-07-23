import torch
from torch.utils.data import DataLoader
from dataset import MRIDataset
from model_multiscale import CNN3D
from utils import load_config
from tqdm import tqdm

# Loading configs
train_configs = load_config("training_settings.yaml")

# Creating datasets
train_dataset = MRIDataset(root=train_configs['data_path'],
                           input_size=train_configs['input_size'], fold_no=0, is_training=True, retrurn_segmented=False)
test_dataset = MRIDataset(root=train_configs['data_path'],
                          input_size=train_configs['input_size'], fold_no=0, is_training=False, retrurn_segmented=False)

print("{} TRAIN DATA loaded!".format(len(train_dataset)))
print("{} TEST DATA loaded!".format(len(test_dataset)))

train_loader = DataLoader(train_dataset, **train_configs['data_loader'])
test_loader = DataLoader(test_dataset, **train_configs['data_loader'])

# Model defining
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
multiscale = CNN3D(is_training=True).to(device)
# ffc3d.load_state_dict(torch.load(train_configs['save_dir'] + "exp0_aug_100epochs.pth"))

# optimization settings
optimizer = torch.optim.Adam(multiscale.parameters(),
                            lr=train_configs['lr'])

criterion = torch.nn.BCEWithLogitsLoss().to(device)                             

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

for epoch in range(train_configs['n_epochs']):
    epoch_train_loss = 0
    epoch_train_acc = 0
    epoch_val_loss = 0
    epoch_val_acc = 0
    for batch_images, batch_labels in tqdm(train_loader):

        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        batch_images = torch.unsqueeze(batch_images, dim=1)
        optimizer.zero_grad()
        out = multiscale(batch_images.permute(0, 1, 4, 2, 3))

        train_loss = criterion(out, batch_labels.unsqueeze(1).float())
        train_acc = binary_acc(out, batch_labels.unsqueeze(1).float())
        
        train_loss.backward()
        optimizer.step()

        epoch_train_loss += train_loss.item()
        epoch_train_acc += train_acc.item()

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_images = torch.unsqueeze(batch_images, dim=1)

            out = multiscale(batch_images.permute(0, 1, 4, 2, 3))

            val_loss = criterion(out, batch_labels.unsqueeze(1).float())
            val_acc = binary_acc(out, batch_labels.unsqueeze(1).float())


            epoch_val_loss += val_loss.item()
            epoch_val_acc += val_acc.item()



    print(f'Epoch {epoch+0:03}: | Train Loss: {epoch_train_loss/len(train_loader):.5f} | Train Acc: {epoch_train_acc/len(train_loader):.3f}')
    print(f'Val Loss: {epoch_val_loss/len(test_loader):.5f} | Val Acc: {epoch_val_acc/len(test_loader):.3f}')

    torch.save(multiscale.state_dict(), train_configs['save_dir'] + "multiscale_exp00_50epochs_fold0.pth")


