import os
import os.path
import pickle
import json
import random

import scipy.sparse
import numpy
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import collections
import copy

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

APPARELS = ['shoe', 'dress', 'pants', 'outerwear']
ATTRIBUTES = ['gender', 'age', 'color', 'up height', 'decoration', 'material', 'silhouette', 'type', 'closure type',
             'length', 'occasion', 'toe shape', 'heel type', 'collar', 'fit', 'pattern', 'sleeve length', 'flat type',
             'back counter type', 'pump type', 'rise type', 'boot type']

with open('../data/raw/fgvc4_iMat.task_map.json') as data_file:
    task_map = json.load(data_file)['taskInfo']
    task_map = {d['taskName']: d['taskId'] for d in task_map}

with open('../data/raw/fgvc4_iMat.label_map.json') as data_file:
    label_map = json.load(data_file)['labelInfo']
    label_map = {d['labelName']: d['labelId'] for d in label_map}

apparels_id_map = collections.defaultdict(list)
id_apparels_map = {}
attribute_id_map = collections.defaultdict(list)
id_attribute_map = {}

for k, v in task_map.items():
    split_ = k.split(":")

    apparels_id_map[split_[0]].append(v)
    id_apparels_map[v] = split_[0]
    attribute_id_map[split_[1]].append(v)
    id_attribute_map[v] = split_[1]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class ImageTaskFolder(data.Dataset):
    def __init__(self, root, df_path, transform=None, mode='train'):
        self.evals, self.evecs_mat = pickle.load(open('../data/fancyPCA.pkl', 'rb'))
        self.root = root
        self.transform = transform
        self.transform_pca = copy.copy(transform)
        self.transform_pca.transforms.insert(-1, Lighting(1, self.evals, self.evecs_mat))
        self.loader = default_loader
        self.mode = mode
        self.num_classes = len(label_map.keys()) + 1
        self.num_tasks = len(task_map.keys())
        self.num_apparel = len(APPARELS)
        self.num_attribute = len(ATTRIBUTES)

        self.imgs = self.make_dataset(df_path)
        print(len(self.imgs))

        if len(self.imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " + root + "\n Supported image extensions are: " + ",".join(
                        IMG_EXTENSIONS)))

            # self.imgs = self.imgs[:200]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, task, (target, apparel_id, fpca, attribute_id) = self.imgs[index]
        img = self.loader(path)
        if fpca:
            img = self.transform_pca(img)
        else:
            img = self.transform(img)

        task = self.transform_task(task)
        assert target >= 0 and target < self.num_classes
        return img, task, target, apparel_id, attribute_id

    def __len__(self):
        return len(self.imgs)

    def transform_task(self, t1):
        task = torch.zeros(self.num_tasks)
        task[t1 - 1] = 1
        # task = torch.sparse.FloatTensor(torch.LongTensor([[t1-1]]), torch.FloatTensor([1]), torch.Size([self.vocab]))
        return task

    def make_dataset(self, df_path):
        images = []
        df = pickle.load(open(df_path, 'rb'))
        groupby = df.groupby(by='path')
        for root, _, fnames in sorted(os.walk(self.root)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)

                    if self.mode != 'test':
                        gpp = groupby.get_group(path)[['taskId', 'labelId']].values
                        set1 = list(gpp[:, 0])
                        att_list = list(apparels_id_map.keys())

                        for t in set1:
                            apparel = id_apparels_map[t]
                            if apparel in att_list:
                                att_list.remove(apparel)

                        tasks_to_null = [i for att in att_list for i in apparels_id_map[att]]

                        apparel_id = APPARELS.index(apparel)

                        for task, label in (gpp):
                            attribute = id_attribute_map[task]
                            attribute_id = ATTRIBUTES.index(attribute)
                            images.append((path, int(task),
                                           (int(label), apparel_id, False, attribute_id)))
                            if self.mode == 'train':
                                images.append((path, int(task),
                                               (int(label), apparel_id, True, attribute_id)))
                            tasknull = random.choice(tasks_to_null)
                            images.append((path, int(tasknull),(0, apparel_id, False, ATTRIBUTES.index(id_attribute_map[tasknull]))))

                    else:
                        for task in groupby.get_group(path)[['taskId']].values:
                            images.append((path, int(task[0]),
                                           (0, -1, False, -1)))
        return images


class Lighting(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.from_numpy(eigval).float()
        self.eigvec = torch.from_numpy(eigvec).float()

    def __call__(self, tensor):
        alpha = torch.normal(means=torch.zeros(3), std=self.alphastd)
        rgb = self.eigvec.clone(). \
            mul(alpha.view(1, 3).expand(3, 3)). \
            mul(self.eigval.view(1, 3).expand(3, 3)). \
            sum(1).squeeze()

        tensor = tensor.clone()
        for i in range(3):
            tensor[i].add(rgb[i])

        return tensor
