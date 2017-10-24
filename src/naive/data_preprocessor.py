# coding: utf8
import os
import pickle

from torch.utils.data import DataLoader
from torchvision import transforms

from naive.datasets_folders import ImageTaskFolder

data_folder = '../data'
data_name = 'val'


def load_serialized_data():
    dset_loaders, dset_sizes, dset_classes = pickle.load(open('{}/serialized/{}.pkl'.format(data_folder, data_name), 'rb'))
    return dset_loaders, dset_sizes, len(dset_classes)


def load_data_from_scratch(hyperprams):
    data_transforms = {
        'val': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    dsets = {x: ImageTaskFolder(os.path.join(data_folder, x), data_transforms[x]) for x in data_transforms.keys()}
    dset_loaders = {x: DataLoader(dsets[x], batch_size=hyperprams['batch_size'], shuffle=hyperprams['shuffle'],
                                  num_workers=hyperprams['num_workers']) for x in data_transforms.keys()}
    dset_sizes = {x: len(dsets[x]) for x in data_transforms.keys()}
    dset_classes = dsets['val'].classes

    pickle.dump([dset_loaders, dset_sizes, dset_classes],
                open('{}/serialized/{}.pkl'.format(data_folder, data_name), 'wb'))

    return dset_loaders, dset_sizes, len(dset_classes)
