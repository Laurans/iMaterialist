# coding: utf8
import inspect

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np

model_name = 'by_attribute'



def get_model_name():
    return model_name


def build_model(num_classes, use_gpu):
    src_module = inspect.getsource(MyModule)
    net = MyModule(num_classes)

    if use_gpu:
        net = net.cuda()

    return net, src_module


class MyModule(nn.Module):
    def __init__(self, infos):
        super(MyModule, self).__init__()
        self.hidden_dim = 512*2
        #self.size = 512*7*7
        in_channels = 512
        self.size_a = 512*1*1
        self.size_b = 256 * 7 * 7
        self.task_dim = infos[1]
        init_prelu = torch.FloatTensor([0.01])


        pretrained = models.resnet18(pretrained=True)
        p = 0.25

        #self.features = pretrained.features
        children = [c for c in pretrained.children()]
        children.pop()
        self.pool = children.pop()

        self.features = nn.Sequential(*children)

        self.tasks_transform = nn.Sequential(
            nn.Linear(self.task_dim,self.task_dim-1),
            nn.BatchNorm1d(self.task_dim-1),
        )

        # ----------------------------
        self.weights = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(init_prelu),
            nn.Dropout(p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(init_prelu)
        )

        self.weights_task_apparel = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim+infos[2]),
            nn.Linear(self.hidden_dim+infos[2], self.hidden_dim),
            nn.PReLU(init_prelu),
        )

        self.weights_task_attribute = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim+infos[3]),
            nn.Linear(self.hidden_dim+infos[3], self.hidden_dim),
            nn.PReLU(init_prelu)
        )

        # ----------------------------

        self.linear_a = nn.Sequential(
            nn.BatchNorm1d(self.size_a),
            nn.Linear(self.size_a, self.hidden_dim),
            nn.PReLU(init_prelu),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(init_prelu)
        )

        self.linear_b = nn.Sequential(
            nn.Linear(self.size_b, self.hidden_dim),
            nn.PReLU(init_prelu),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(init_prelu)

        )
        self.linear_apparel = nn.Sequential(
            nn.BatchNorm1d(self.size_a),
            nn.Linear(self.size_a, self.hidden_dim),
            nn.PReLU(init_prelu),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, infos[2]),
            nn.LogSoftmax()
        )

        self.linear_attribute = nn.Sequential(
            nn.Linear(44, infos[3]),
            nn.LogSoftmax()
        )

        self.linear_task = nn.Sequential(
            nn.Linear(44, self.hidden_dim),
            nn.PReLU(init_prelu)
        )

        size = 6170
        self.final_linear = nn.Sequential(
            nn.BatchNorm1d(size),
            nn.Dropout(p),
            nn.Linear(size, self.hidden_dim*2),
            nn.PReLU(init_prelu),
            nn.BatchNorm1d(self.hidden_dim*2),
            nn.Dropout(p),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.PReLU(init_prelu),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, infos[0]),
            nn.LogSoftmax()
        )


    def forward(self, x):

        brut = self.features(x[0])
        task_brut = self.tasks_transform(x[1])

        pt = self.pool(brut)
        pt = pt.view(pt.size(0), self.size_a)


        apparel_out = self.linear_apparel(pt)
        feat = self.linear_a(pt)

        attribute_out = self.linear_attribute(task_brut)
        task_out = self.linear_task(task_brut)
        #import ipdb; ipdb.set_trace()


        weights_tapp = self.weights_task_apparel(torch.cat([task_out, apparel_out], 1))
        weights_tatt = self.weights_task_attribute(torch.cat([task_out, attribute_out], 1))
        weights = self.weights(feat * task_out*(weights_tatt+weights_tapp))

        out = self.final_linear(torch.cat([feat * task_out,
                                           feat * weights,
                                           feat,
                                           task_out,
                                           task_out * weights_tapp,
                                           task_out * weights_tatt,
                                           apparel_out,
                                           attribute_out], 1))
        return [out, apparel_out, attribute_out]

    def get_parameters(self):
        return self.parameters()
