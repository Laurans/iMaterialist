# coding: utf8
import inspect

import torch
import torch.nn as nn
from torchvision import models

model_name = 'on_apparel_model'



def get_model_name():
    return model_name


def build_model(num_classes, use_gpu):
    src_module = inspect.getsource(MyModule)
    net = MyModule(num_classes)

    if use_gpu:
        net = net.cuda()

    return net, src_module


class MyModule(nn.Module):
    def __init__(self, num_classes):
        super(MyModule, self).__init__()
        self.pretrained_model = models.resnet18(pretrained=True)
        self.pretrained_model.fc = nn.Linear(self.pretrained_model.fc.in_features, num_classes)

    def forward(self, x):
        out = self.pretrained_model(x)
        return out

    def get_parameters(self):
        return self.parameters()
