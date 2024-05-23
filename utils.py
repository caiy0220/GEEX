import os
import numpy as np
import PIL
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

def load_img(file_path):
    img = PIL.Image.open(file_path)
    img = img.resize((299, 299))
    img = np.asarray(img)
    return img

# def preprocessing(imgs):
#     imgs = np.array(imgs)
#     transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     imgs = imgs/255
#     imgs = np.transpose(imgs, (0,3,1,2))
#     imgs = torch.tensor(imgs, dtype=torch.float32)
#     imgs = transformer.forward(imgs)
#     # return images.requires_grad_(True)
#     return imgs

def get_device(m):
    return next(m.parameters()).device

PIX_MEAN_IMAGENET = torch.tensor((0.485, 0.456, 0.406))
PIX_STD_IMAGENET = torch.tensor((0.229, 0.224, 0.225))

preprocessing = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(PIX_MEAN_IMAGENET, PIX_STD_IMAGENET)
    ])

def denormalize_imagenet(img):
    return img * PIX_STD_IMAGENET + PIX_MEAN_IMAGENET

class SoftmaxWrapping(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def parameters(self, recurse: bool = True):
        return self.m.parameters(recurse)

    def forward(self, x):
        return F.softmax(self.m(x), dim=1)
