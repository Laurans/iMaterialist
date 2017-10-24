# coding: utf8
import inspect

import torch
import torch.nn as nn
from torchvision import models

model_name = 'with_embedding'


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
        num_apparels = 4
        num_attribute = 22
        inter_state = 1024
        hidden1 = 544

        self.model_apparel =nn.Sequential(
            nn.Embedding(num_apparels+1, num_apparels-1)
        )

        self.model_attribute = nn.Sequential(
            nn.Embedding(num_attribute+1, num_attribute-1)
        )

        self.categorical_classifier = nn.Sequential(
            nn.Linear((num_apparels+num_attribute-2),hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, inter_state),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            #nn.Linear(inter_state, 4096),
            #nn.ReLU(inplace=True),
            #nn.Linear(4096, 1024),
            #nn.ReLU(inplace=True),
            nn.Linear(inter_state, num_classes),
        )

        self.pretrained_model = models.resnet18(pretrained=True)
        self.pretrained_model.fc = nn.Linear(self.pretrained_model.fc.in_features, inter_state)

    def forward(self, x):
        out1 = self.pretrained_model(x[0])

        out1 = out1.view(out1.size(0), -1)
        out2 = self.model_apparel(x[1][:, 0])
        out3 = self.model_attribute(x[1][:, 1])

        x4 = torch.cat([out2,out3], 1)
        out4 = self.categorical_classifier(x4)
        cat = torch.add(out1, out4)
        out = self.classifier(cat)
        return out

    def get_parameters(self):
        # base_params = [
        #     {'params': self.model_apparel.parameters()},
        #     {'params': self.model_attribute.parameters()},
        #     {'params': self.pretrained_model.fc.parameters()},
        #     {'params': self.classifier.parameters()}
        # ]
        return self.parameters()

        #x = x.view(x.size(0), -1)

        # def build_model():
        #     vgg16 = VGG16(include_top=False, weights='imagenet')
        #     #vgg16.trainable = False
        #     inp = Input((3, 224, 224))
        #     x = vgg16(inp)
        #     x = Flatten(name='flatten')(x)
        #     x = Dense(4096, activation='relu', name='fc1')(x)
        #     x = Dense(4096, activation='relu', name='fc2')(x)
        #     out = Dense(209, activation='softmax', name='predictions')(x)
        #     model = Model(inp, out)
        #     model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-4), metrics=['accuracy'])
        #     return model
