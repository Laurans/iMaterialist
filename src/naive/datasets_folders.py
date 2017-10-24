import os
import os.path
import pickle

import scipy.sparse
import numpy
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def get_task_features(df, apparel_ohe, attribute_ohe):
    apparel_feature = apparel_ohe.transform(df.apparelId.values.reshape(-1, 1))
    attribute_feature = attribute_ohe.transform(df.attributeId.values.reshape(-1, 1))
    feat = scipy.sparse.hstack([apparel_feature, attribute_feature])
    return feat


def make_dataset(dir, class_to_idx, dataframe_filename, get_task_features, preprocess_models):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            df = pickle.load(open(os.path.join(root, dataframe_filename), 'rb'))
            for fname in fnames:
                if is_image_file(fname):
                    df_fname = df[df.imageId == fname.split('.')[0]]
                    for i in range(len(df_fname)):
                        feat = get_task_features(df_fname.iloc[i], preprocess_models)
                        path = os.path.join(root, fname)
                        item = (path, feat, class_to_idx[target])
                        images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


def transform_feat(feat):
    if type(feat) == numpy.ndarray:
        return torch.from_numpy(feat)
    else:
        return torch.from_numpy(feat.toarray()[0]).float()

class ImageTaskFolder(data.Dataset):
    def __init__(self, root, categorical_transform, preprocess_models=[], transform=None,
                 dataframe_filename="imgtasklabel.pkl"):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, dataframe_filename, categorical_transform, preprocess_models)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        import random
        random.shuffle(imgs)
        self.imgs = imgs[:4000]
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = default_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, feat, target = self.imgs[index]
        img = self.loader(path)
        img = self.transform(img)
        feat = transform_feat(feat)
        return img, feat, target

    def __len__(self):
        return len(self.imgs)
