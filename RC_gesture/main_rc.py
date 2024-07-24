import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as tt
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from torchsummary import summary
import matplotlib.pyplot as plt
from torch import nn, optim

# %matplotlib inline
import time
from torch.autograd import Variable
import cv2
from PIL import Image
from torchvision import transforms
import os

from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
from models import CustomModel

from models import ReadMap

from torchsummary import summary
from torchvision import models
import matplotlib as mpl
from matplotlib.pyplot import MultipleLocator
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ARCHI', type=str, default='RC_CNN',
                        help='RC_CNN, RC_readout')
    
    parser.add_argument('--GPU', type=int, default=0,
                        help='GPU device to use')

    args = parser.parse_args()
    return args


options = parse_args()
print(options)



mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

os.environ['CUDA_VISIBLE_DEVICES'] = str(options.GPU)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(10)

DATA_ROOT = Path('/home/rram/RC_Computing_YPJJ/hand_action/data/leapgestrecog/leapGestRecog/')


checkpoints_directory = "./checkpoints"
if not os.path.exists(checkpoints_directory):
    os.makedirs(checkpoints_directory)

checkpoints_directory = "./figures"
if not os.path.exists(checkpoints_directory):
    os.makedirs(checkpoints_directory)


tmp_ds = ImageFolder(DATA_ROOT / '00')
CLASSES_NAME = tmp_ds.classes
print(f"class is {CLASSES_NAME}")

train_tfms = tt.Compose([
                         tt.Grayscale(num_output_channels=1), # Pictures black and white
                         tt.Resize([32, 32]),
                         # Settings for expanding the dataset
                         tt.RandomHorizontalFlip(),           # Random 90 degree rotations
                         tt.RandomRotation(30),               # Random 30 degree rotations
                         tt.ToTensor(),                      # Cast to tensor
                         ])                      

test_tfms = tt.Compose([
                        tt.Grayscale(num_output_channels=1),
                        tt.Resize([32, 32]),
                        tt.ToTensor(),
                        ])


train_ds_list = []
test_ds_list = []
for dir in os.listdir(DATA_ROOT):
    train_ds = ImageFolder(os.path.join(DATA_ROOT, dir), train_tfms)
    test_ds = ImageFolder(os.path.join(DATA_ROOT, dir), test_tfms)
    train_ds_list.append(train_ds)
    test_ds_list.append(test_ds)

train_ds = ConcatDataset(train_ds_list)
test_ds = ConcatDataset(test_ds_list)



def transformRate(data2D, N_ts, max_is_present_for, mapping, seed=0):
    np.random.seed(seed)
    data2D = data2D.cpu().numpy()
    data = []
    for trials in range(N_ts):
        trial = np.random.random(data2D.shape)
        data.append((data2D * max_is_present_for / N_ts > trial).astype(dtype=np.uint8))
    res = np.array(data, dtype=np.uint8)
    res = res.swapaxes(0, 1)

    # print(f"res shape is {res.shape}")

    # Apply the mapping
    res_shape = res.shape
    res_flat = res.reshape(-1, N_ts)
    res_bin_str = [''.join(map(str, row)) for row in res_flat]
    res_mapped = np.array([mapping[bin_str] for bin_str in res_bin_str])

    # Remove the time dimension and add a singleton dimension
    res_mapped = res_mapped.reshape(res_shape[0], *res_shape[2:])
    

    # print(f"res_mapped shape is {res_mapped.shape}")
    # print(f"res_mapped shape is {res_mapped}")

    # Remove the time dimension
    
    return res_mapped

# Read the mapping from the Excel file
mapping_df = pd.read_excel('./device_data/RC_Mapping.xlsx', header=None, dtype={0: str})

# Convert the DataFrame to a dictionary
mapping = pd.Series(mapping_df[1].values, index=mapping_df[0]).to_dict()

print(mapping)

class SpikingLeapGests(torch.utils.data.Dataset):
    def __init__(self, dataset, N_ts=4, max_is_present_for=5, seed=0):
        self.dataset = dataset
        self.N_ts = N_ts
        self.max_is_present_for = max_is_present_for
        self.seed = seed
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img, gest = self.dataset[idx]
        
        spiking_img = torch.from_numpy(transformRate(img, self.N_ts, self.max_is_present_for, mapping, seed=self.seed)).float()

        return spiking_img, gest

# creat spike data
spiking_train_ds = SpikingLeapGests(train_ds)
spiking_test_ds = SpikingLeapGests(test_ds)

print("Number of samples in spiking_train_ds:", len(spiking_train_ds))
print("Number of samples in spiking_test_ds:", len(spiking_test_ds))

batch_size=64
train_loader = DataLoader(spiking_train_ds, batch_size, shuffle=True)
valid_loader = DataLoader(spiking_test_ds, batch_size, shuffle=False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if options.ARCHI == "RC_CNN":
    model = CustomModel().to(device)
elif options.ARCHI == "RC_readout":
    model = ReadMap().to(device)

print(model)

summary(model.to(device), input_size=(3, 32, 32))

params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = torch.optim.Adam(params_to_update, lr=0.001)
criterion = nn.CrossEntropyLoss()


epochs = 100
epoch_losses = []
epoch_accuracies = []

best_test_accuracy = 0


for epoch in range(epochs):
    
    running_loss = 0.0
    epoch_loss = []
    correct = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss.append(loss.item())

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    test_running_loss = 0
    test_epoch_loss = []
    test_correct = 0
    test_total = 0
    for batch_idx, (data, labels) in enumerate(valid_loader):
        model.eval()
        data = data.to(device)
        labels = labels.to(device)
        
        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)
                
        test_running_loss += loss.item()
        test_epoch_loss.append(loss.item())

        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        checkpoint = {
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, f'./checkpoints/best_RC{options.ARCHI}_model_checkpoint.pth')
        
    print(f'Epoch {epoch+1}, loss: {np.mean(epoch_loss)}, train acc: {train_accuracy}%, test loss: {np.mean(test_epoch_loss)}, test acc: {test_accuracy}%')
    epoch_losses.append(np.mean(epoch_loss))
    epoch_accuracies.append((train_accuracy, test_accuracy))




# Save epoch_losses and epoch_accuracies to CSV
epoch_data = pd.DataFrame(list(zip(epoch_losses, epoch_accuracies)), columns=['Loss', 'Accuracy(train-test)'])
epoch_data.to_csv(f'./checkpoints/RC{options.ARCHI}_epoch_data.csv', index=False)

# Plot Loss vs Epoch
plt.figure()
plt.plot(range(1, epochs+1), epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.savefig(f'./figures/RC{options.ARCHI}_Loss_vs_Epoch.png')
plt.savefig(f'./figures/RC{options.ARCHI}_Loss_vs_Epoch.pdf')

# Plot Accuracy vs Epoch
train_accuracies, test_accuracies = zip(*epoch_accuracies)
plt.figure()
plt.plot(range(1, epochs+1), train_accuracies, label='Train')
plt.plot(range(1, epochs+1), test_accuracies, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.savefig(f'./figures/RC{options.ARCHI}_Accuracy_vs_Epoch.png')
plt.savefig(f'./figures/RC{options.ARCHI}_Accuracy_vs_Epoch.pdf')


### plot confusion matrix ###
def get_all_preds(model, loader):
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    for batch in loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, labels.float()), dim=0)
    return all_labels, all_preds

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer

# Load the best model from the checkpoint
best_model, _ = load_checkpoint(f'./checkpoints/best_RC{options.ARCHI}_model_checkpoint.pth')
best_model = best_model.to(device)


# Get all predictions on test data
all_labels, all_preds = get_all_preds(best_model, valid_loader)
_, predicted_labels = torch.max(all_preds, 1)

# # Compute the confusion matrix
# cm = confusion_matrix(all_labels.cpu().numpy(), predicted_labels.cpu().numpy())

# # Define the category labels
# cat = [str(i) for i in range(10)]  # Replace this with your own category labels if necessary

# # Plot the confusion matrix
# plt.figure(figsize=(10, 10))
# sn.heatmap(cm, annot=True, xticklabels=cat, yticklabels=cat)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.savefig('./figures/best_rc_model_confusion_matrix.png')
# plt.savefig('./figures/best_rc_model_confusion_matrix.pdf')

# Compute the confusion matrix
cm = confusion_matrix(all_labels.cpu().numpy(), predicted_labels.cpu().numpy())

# Normalize the confusion matrix by dividing each cell value by the sum of its row
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_percentage = cm_normalized * 100  # Convert to percentage

# Define the category labels
cat = [str(i) for i in range(10)]  # Replace this with your own category labels if necessary

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
sn.heatmap(cm_percentage, annot=True, fmt='.2f', xticklabels=cat, yticklabels=cat)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(f'./figures/RC{options.ARCHI}_confusion_matrix.png')
plt.savefig(f'./figures/RC{options.ARCHI}_confusion_matrix.pdf')



