import torch
from torch.utils.data import DataLoader
from dataset import MRIDataset
from model_ffcRNN import FFC3DRNN
from utils import load_config
from tqdm import tqdm

# Loading configs
train_configs = load_config("training_settings.yaml")

# Creating datasets
test_dataset = MRIDataset(root=train_configs['data_path'],
                          input_size=train_configs['input_size'], retrurn_segmented=True, fold_no=1, is_training=False)

print("{} TEST DATA loaded!".format(len(test_dataset)))
test_loader = DataLoader(test_dataset, **train_configs['data_loader'])

# Model defining
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
init_conv_kwargs = {'ratio_gin': 0, 'ratio_gout': 0.75, 'enable_lfu': False}
internal_conv_kwargs = {'ratio_gin': 0.75, 'ratio_gout': 0.75, 'enable_lfu': False}
ffc3d = FFC3DRNN(input_nc=1, init_conv_kwargs=init_conv_kwargs, internal_conv_kwargs=internal_conv_kwargs).to(device)
ffc3d.load_state_dict(torch.load(train_configs['save_dir'] + "exp17_apex_100epochs_fold1.pth"))

# optimization settings
optimizer = torch.optim.Adam(ffc3d.parameters(),
                            lr=train_configs['lr'])

criterion = torch.nn.BCEWithLogitsLoss().to(device)                     

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


correct_samples = 0
correct_samples_with_segmentaed = 0
TP = 0
TN = 0
FP = 0
FN = 0
TP_seg = 0
TN_seg = 0
FP_seg = 0
FN_seg = 0
# for this version batch size must be 1
with torch.no_grad():
    for batch_images, batch_labels, batch_segmented_images in tqdm(test_loader):

        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        batch_segmented_images = batch_segmented_images.to(device)
        batch_images = torch.unsqueeze(batch_images, dim=1)
        batch_segmented_images = torch.unsqueeze(batch_segmented_images, dim=1)

        out, hidden = ffc3d(batch_segmented_images.permute(0, 1, 4, 2, 3), torch.zeros(len(batch_segmented_images), 8).to(device))
        out, _ = ffc3d(batch_images.permute(0, 1, 4, 2, 3), hidden)
        y_pred_tag = torch.round(torch.sigmoid(out))

        
        if y_pred_tag.item() == 1:
            if y_pred_tag == batch_labels.unsqueeze(1).float():
                correct_samples += 1
                TP += 1
            else:
                FP += 1         
        elif y_pred_tag.item() == 0:
            if y_pred_tag == batch_labels.unsqueeze(1).float():
                correct_samples += 1
                TN += 1
            else:
                FN += 1 
    


print("Test accuracy is : {}".format( 100 * correct_samples / len(test_dataset)))

precision = 100 * TP / (TP + FP)
recall = 100 * TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1-score: {}".format(f1))
print()
print("TP: {}".format(TP))
print("TN: {}".format(TN))
print("FP: {}".format(FP))
print("FN: {}".format(FN))
