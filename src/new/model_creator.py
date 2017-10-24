# coding: utf8
import inspect

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

model_name = 'new'

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
        self.size_a = 256*6*6
        self.task_dim = infos[1]
        self.last_dim = 4096
        init_prelu = torch.FloatTensor([0.25])


        pretrained = models.alexnet(pretrained=True)
        p = 0.5

        self.pretrained_branch_1 = pretrained.features

        # ----------------------------
        self.fc_branch_1 = nn.Sequential(
            nn.BatchNorm1d(self.size_a),
            nn.Dropout(p),
            nn.Linear(self.size_a, self.hidden_dim),
            nn.PReLU(init_prelu),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(init_prelu),
        )

        # ----------------------------
        # self.weight_branch_1 = nn.Sequential(
        #     nn.BatchNorm1d(self.hidden_dim),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.PReLU(init_prelu),
        #     nn.Dropout(p),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.PReLU(init_prelu),
        # )
        #
        # self.weight_task_apparel = nn.Sequential(
        #     nn.BatchNorm1d(self.hidden_dim+infos[2]),
        #     nn.Linear(self.hidden_dim+infos[2], self.hidden_dim),
        #     nn.PReLU(init_prelu),
        # )
        #
        # self.weight_task_attribute = nn.Sequential(
        #     nn.BatchNorm1d(self.hidden_dim+infos[3]),
        #     nn.Linear(self.hidden_dim+infos[3], self.hidden_dim),
        #     nn.PReLU(init_prelu),
        # )

        # ----------------------------
        # ----------------------------
        #self.attribute_embedding = nn.Linear(infos[0], infos[0]-10)

        self.tasks_transform = nn.Sequential(
            nn.Linear(self.task_dim, self.task_dim - 1),
            nn.BatchNorm1d(self.task_dim - 1),
        )

        self.linear_task = nn.Sequential(
            nn.Linear(44, self.hidden_dim),
            nn.PReLU(init_prelu)
        )

        # ----------------------------
        self.clist = nn.ModuleList()
        for _ in range(5):
            self.clist.append(nn.Sequential(*[nn.BatchNorm1d(self.size_a+5), nn.Linear(self.size_a+5,4096), nn.ReLU()]))
        # ----------------------------

        self.linear_apparel = nn.Sequential(
            nn.BatchNorm1d(256*6*6),
            nn.Linear(256*6*6, self.hidden_dim),
            nn.PReLU(init_prelu),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, infos[2]),
            nn.LogSoftmax()
        )

        self.reduction = nn.Linear(4096*5, 4096)

        self.final = nn.Sequential(
            nn.BatchNorm1d(5124),
            nn.Dropout(p),
            nn.Linear(5124, self.last_dim),
            nn.PReLU(init_prelu),
            nn.Dropout(p),
            nn.Linear(self.last_dim, self.hidden_dim),
            nn.PReLU(init_prelu),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, infos[0]),
            nn.LogSoftmax()
        )

        self.fusion = nn.Sequential(nn.Linear(self.hidden_dim*45, self.hidden_dim), nn.PReLU())

    def forward(self, x):
        task_brut = self.tasks_transform(x[3])
        task_out = self.linear_task(task_brut)

        # img -> pool
        pretrained_block_1 = self.pretrained_branch_1(x[0])
        size = pretrained_block_1.size(1)*pretrained_block_1.size(2)*pretrained_block_1.size(3)
        flatten = pretrained_block_1.view(pretrained_block_1.size(0), size)

        # fc for apparel prediction
        outc = []
        for j, c in enumerate(self.clist):
            cattie = torch.cat([flatten, x[3][:,j*5:j*5+5]],1)
            #import ipdb; ipdb.set_trace()
            outc.append(c.forward(cattie))

        apparel_pred = self.linear_apparel(flatten)


        grand_concat1 =torch.cat(outc,1)
        grand_concat1 = self.reduction(grand_concat1)

        grand_concat2 = torch.cat([grand_concat1, task_out, apparel_pred],1)
        out = self.final(grand_concat2)
        return [out, apparel_pred]

    def get_parameters(self):
        #import ipdb; ipdb.set_trace()
        return self.parameters()
