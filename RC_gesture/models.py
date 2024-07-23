import torch
from torch import nn, optim
import torch.nn.functional as F




class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1)
        # self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2)
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2)
        # self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)  # Keep the input size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        x = F.relu(self.conv2(x))
        # x = self.pool2(x)
        x = F.relu(self.conv3(x))
        # x = self.pool3(x)

        # print(f"size after conv is {x.shape}")
        
        x = x.view(x.size(0), -1)  # Keep the shape parameter
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)



class ReadMap(nn.Module):
    def __init__(self):
        super(ReadMap, self).__init__()

        self.fc = nn.Linear(1024, 10)  # Keep the input size

    def forward(self, x):
        
        x = x.view(x.size(0), -1)  # Keep the shape parameter
        # print(f"x shape is {x.shape}")
        x = self.fc(x)
        return F.softmax(x, dim=1)