# coding: utf8
import torch.nn as nn
from torchvision import models

model_name = 'naive'

def get_model_name():
    return model_name

def build_model(num_classes, use_gpu):
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)

    if use_gpu:
        net = net.cuda()

    return net