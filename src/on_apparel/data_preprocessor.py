# coding: utf8
import os
import pickle

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

data_folder = '../data'
data_name = 'on_apparel'
preprocessing_model_path = '../preprocessing_models'


def load_serialized_data():
    dset_loaders, dset_sizes, dset_classes = pickle.load(
        open('{}/serialized/{}.pkl'.format(data_folder, data_name), 'rb'))
    return dset_loaders, dset_sizes, len(dset_classes)


def load_data_from_scratch(hyperparams, categorical_prepocess=True):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    dsets = {x: datasets.ImageFolder(os.path.join(data_folder, x+'_apparel'), transform=data_transforms[x]) for x in
             data_transforms.keys()}
    dset_loaders = {x: DataLoader(dsets[x], batch_size=hyperparams['batch_size'], shuffle=hyperparams['shuffle'],
                                  num_workers=hyperparams['num_workers']) for x in data_transforms.keys()}
    dset_sizes = {x: len(dsets[x]) for x in data_transforms.keys()}
    dset_classes = dsets['train'].classes

    pickle.dump([dset_loaders, dset_sizes, dset_classes],
                open('{}/serialized/{}.pkl'.format(data_folder, data_name), 'wb'))

    assert len(dset_classes) == 4
    return dset_loaders, dset_sizes, len(dset_classes)
