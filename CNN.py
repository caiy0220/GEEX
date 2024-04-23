import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from utils.log import *

# Defining a simple CNN for MNIST and FMNIST
class CNN(nn.Module):
    def __init__(self, input_size):
        chn, w, _ = input_size  # assuming a square input
        super().__init__()
        self.conv1 = nn.Conv2d(chn, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        d = ((w-4)//2-4)//2
        self.fc1 = nn.Linear(16 * d * d, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CNNSoftmax(CNN):
    def forward(self, x):
        return F.softmax(super().forward(x), dim=1)

class SoftmaxWrapping(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def parameters(self, recurse: bool = True):
        return self.m.parameters(recurse)

    def forward(self, x):
        return F.softmax(self.m(x), dim=1)