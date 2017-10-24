# coding: utf8
import os
import pickle

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader
from torchvision import transforms

from naive.datasets_folders import ImageTaskFolder

data_folder = '../data'
data_name = 'with_embedding'
preprocessing_model_path = '../preprocessing_models'


def load_serialized_data():
    dset_loaders, dset_sizes, dset_classes = pickle.load(
        open('{}/serialized/{}.pkl'.format(data_folder, data_name), 'rb'))
    return dset_loaders, dset_sizes, len(dset_classes)


def load_data_from_scratch(hyperparams, categorical_prepocess=True):
    if categorical_prepocess:
        create_preprocess_models()

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

    categorical_models = get_preprocess_models()
    dsets = {x: ImageTaskFolder(os.path.join(data_folder, x), categorical_transform=get_task_features,
                                preprocess_models=categorical_models, transform=data_transforms[x]) for x in
             data_transforms.keys()}
    dset_loaders = {x: DataLoader(dsets[x], batch_size=hyperparams['batch_size'], shuffle=hyperparams['shuffle'],
                                  num_workers=hyperparams['num_workers']) for x in data_transforms.keys()}
    dset_sizes = {x: len(dsets[x]) for x in data_transforms.keys()}
    dset_classes = dsets['train'].classes

    pickle.dump([dset_loaders, dset_sizes, dset_classes],
                open('{}/serialized/{}.pkl'.format(data_folder, data_name), 'wb'))

    return dset_loaders, dset_sizes, len(dset_classes)


def get_preprocess_models():
    apparel_le = pickle.load(open(os.path.join(preprocessing_model_path, 'apparel_le.pkl'), 'rb'))
    attribute_le = pickle.load(open(os.path.join(preprocessing_model_path, 'attribute_le.pkl'), 'rb'))
    return [apparel_le, attribute_le]


def get_task_features(df, preprocess_models):
    features = []
    cols = ["apparel", "attribute"]
    for model, col in zip(preprocess_models, cols):
        ft = model.transform([df[col]])
        features.append(ft)
    feat = np.hstack(features)
    assert feat.shape[0] == 2
    return feat


def create_preprocess_models():
    df = pickle.load(open('{}/train/dataframe.pkl'.format(data_folder), 'rb'))
    apparel_le = LabelEncoder()
    attribute_le = LabelEncoder()

    df['apparelId'] = apparel_le.fit_transform(df.apparel)
    df['attributeId'] = attribute_le.fit_transform(df.attribute)

    apparel_ohe = OneHotEncoder()
    apparel_ohe.fit(np.array(df['apparelId'].values.reshape(-1, 1)))
    attribute_ohe = OneHotEncoder()
    attribute_ohe.fit(df['attributeId'].values.reshape(-1, 1))

    pickle.dump(apparel_le, open('{}/apparel_le.pkl'.format(preprocessing_model_path), 'wb'))
    pickle.dump(attribute_le, open('{}/attribute_le.pkl'.format(preprocessing_model_path), 'wb'))
    pickle.dump(apparel_ohe, open('{}/apparel_ohe.pkl'.format(preprocessing_model_path), 'wb'))
    pickle.dump(attribute_ohe, open('{}/attribute_ohe.pkl'.format(preprocessing_model_path), 'wb'))
