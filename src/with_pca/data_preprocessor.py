# coding: utf8
import pickle

from torch.utils.data import DataLoader
from torchvision import transforms

from by_attribute.extractor import ImageTaskFolder

data_folder = '../data'
data_name = 'with_attribute'
preprocessing_model_path = '../preprocessing_models'

def load_serialized_data():
    dset_loaders, dset_sizes, dset_classes = pickle.load(
        open('{}/serialized/{}.pkl'.format(data_folder, data_name), 'rb'))
    return dset_loaders, dset_sizes, dset_classes


def load_data_from_scratch(hyperparams, categorical_prepocess=True):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale(256),
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
        ]),
        'test': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    dsets = {x: ImageTaskFolder("{}/{}_raw".format(data_folder, x), "{}/{}.pkl".format(data_folder,x), transform=data_transforms[x], mode=x) for x in
             data_transforms.keys()}
    dset_loaders = {x: DataLoader(dsets[x], batch_size=hyperparams['batch_size'], shuffle=hyperparams['shuffle'],
                                  num_workers=hyperparams['num_workers']) for x in data_transforms.keys()}
    dset_sizes = {x: len(dsets[x]) for x in data_transforms.keys()}
    dset_classes = dsets['train'].num_classes
    dset_tasks = dsets['train'].num_tasks
    dset_apparel = dsets['train'].num_apparel
    dset_attribute = dsets['train'].num_attribute

    pickle.dump(dsets['test'], open('{}/serialized/{}.pkl'.format(data_folder, 'test'), 'wb'))
    pickle.dump([dset_loaders, dset_sizes, [dset_classes, dset_tasks, dset_apparel, dset_attribute]],
                open('{}/serialized/{}.pkl'.format(data_folder, data_name), 'wb'))

    return dset_loaders, dset_sizes, [dset_classes, dset_tasks, dset_apparel, dset_attribute]
