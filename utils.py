import os
import numpy as np
import PIL
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

PIX_MEAN_IMAGENET = torch.tensor((0.485, 0.456, 0.406))
PIX_STD_IMAGENET = torch.tensor((0.229, 0.224, 0.225))

# preprocessing for IMAGENET
preprocessing = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(PIX_MEAN_IMAGENET, PIX_STD_IMAGENET)
    ])

def load_img(file_path):
    img = PIL.Image.open(file_path)
    img = img.resize((299, 299))
    img = np.asarray(img)
    return img

def get_device(m):
    return next(m.parameters()).device

def denormalize_imagenet(img):
    return img * PIX_STD_IMAGENET + PIX_MEAN_IMAGENET

def idx2pos(idx, col=28):
    row_id = idx // col 
    col_id = idx % col 
    return row_id, col_id

class SoftmaxWrapping(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def parameters(self, recurse: bool = True):
        return self.m.parameters(recurse)

    def forward(self, x):
        return F.softmax(self.m(x), dim=1)

class RangeSampler(torch.utils.data.Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)
    