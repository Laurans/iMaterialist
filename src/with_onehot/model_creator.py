# coding: utf8
import inspect

import torch
import torch.nn as nn
from torchvision import models

model_name = 'onehot'



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

        self.num_apparels = 4
        self.num_attribute = 22

        self.net = torch.load('../models/on_apparel_model_0608-1606_entire')
        self.net.pretrained_model.fc = nn.Linear(self.net.pretrained_model.fc.in_features, num_classes)

        self.model_apparel =nn.Sequential(
            nn.Linear(self.num_apparels, num_classes),
            nn.ReLU(),
        )

        self.model_attribute = nn.Sequential(
            nn.Linear(self.num_attribute, num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        out1 = self.net(x[0])

        out2 = self.model_apparel(x[1][:, :self.num_apparels])
        out3 = self.model_attribute(x[1][:, self.num_apparels:])

        out4 = torch.add(out2, out3)
        out6 = torch.add(out1,out4)

        return out6

    def get_parameters(self):
        return self.parameters()