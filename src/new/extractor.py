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
from sklearn.externals import joblib
import cv2
from torchvision import transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

APPARELS = ['shoe', 'dress', 'pants', 'outerwear']
ATTRIBUTES = ['gender', 'age', 'color', 'up height', 'decoration', 'material', 'silhouette', 'type', 'closure type',
             'length', 'occasion', 'toe shape', 'heel type', 'collar', 'fit', 'pattern', 'sleeve length', 'flat type',
             'back counter type', 'pump type', 'rise type', 'boot type']

ATTRIBUTES_LABEL = pickle.load(open('../data/attribute_label.pkl','rb'))

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

cluster_model = joblib.load('../data/vbow.ml')
gmm_model = joblib.load('../data/gmm.ml')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def pil_loader(path, mode):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            if mode == 'train':
                if np.random.random() < 1/2:
                    rotation = np.random.uniform(-5,5)
                    img = img.rotate(rotation)

    return img



class ImageTaskFolder(data.Dataset):
    def __init__(self, root, df_path, trans=None, mode='train'):
        self.evals, self.evecs_mat = pickle.load(open('../data/fancyPCA.pkl', 'rb'))
        self.root = root
        self.transform = trans
        self.transform_pca = Lighting(1, self.evals, self.evecs_mat)
        self.transform2 = transforms.Compose([
            transforms.Scale(356), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.loader = pil_loader
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

        #self.imgs = self.imgs[:1000]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, task_app, task_att, task_task, (target, apparel_id, mode, attribute_id) = self.imgs[index]
        img = self.loader(path, self.mode)

        if mode == 1 or mode ==2:
            img = self.transform(img)

            if mode == 2:
                img = self.transform_pca(img)

        elif mode == 3 or mode ==4:
            img = self.transform2(img)
            if mode ==4:
                img = self.transform_pca(img)

        else:
            img = self.transform(img)

        task_app, task_att, task_task = self.transform_task([task_app, task_att, task_task])
        assert target >= 0 and target < self.num_classes
        return [img, task_app, task_att, task_task], [target, apparel_id]

    def __len__(self):
        return len(self.imgs)

    def transform_task(self, t1):
        task_app = torch.zeros(self.num_apparel)
        task_att = torch.zeros(self.num_classes)
        task_task = torch.zeros(self.num_tasks)
        task_app[t1[0]] = 1
        for i in ATTRIBUTES_LABEL[t1[1]]:
            task_att[i] = 1
        task_att[0] = 1
        task_task[t1[2]-1] = 1
        return task_app, task_att, task_task

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

                            if self.mode == 'train':
                                for l in range(1, 5):
                                    images.append((path, apparel_id, attribute_id, int(task),
                                                   (int(label), apparel_id, l, attribute_id)))

                                for _ in range(2):
                                    tasknull = random.choice(tasks_to_null)
                                    attribute_id_null = ATTRIBUTES.index(id_attribute_map[tasknull])
                                    apparel_id_null = APPARELS.index(id_apparels_map[tasknull])
                                    images.append((path, apparel_id_null, attribute_id_null, int(tasknull),
                                                   (0, apparel_id, np.random.randint(3), attribute_id_null)))

                            else:
                                images.append((path, apparel_id, attribute_id, int(task),
                                               (int(label), apparel_id, 0, attribute_id)))

                                tasknull = random.choice(tasks_to_null)
                                attribute_id_null = ATTRIBUTES.index(id_attribute_map[tasknull])
                                apparel_id_null = APPARELS.index(id_apparels_map[tasknull])
                                images.append((path, apparel_id_null, attribute_id_null, int(tasknull),
                                               (0, apparel_id, 0, attribute_id_null)))


                    else:
                        for task in groupby.get_group(path)[['taskId']].values:
                            attribute_id = ATTRIBUTES.index(id_attribute_map[task[0]])
                            apparel_id = APPARELS.index(id_apparels_map[task[0]])
                            images.append((path, apparel_id, attribute_id, int(task[0]),
                                           (0, -1, 0, 0)))
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

def img_to_vect(pil_img, cluster_model):
    img = numpy.array(pil_img)
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    _, desc = orb.detectAndCompute(gray, None)



    try:
        clustered_desc = cluster_model.predict(desc)
        img_bow_hist = np.bincount(clustered_desc, minlength=cluster_model.n_clusters)
        fv = fisher_vector(desc, gmm_model)
    except:
        img_bow_hist = np.zeros((100,))
        fv = np.zeros((500,))

    return  img_bow_hist.astype(np.float32), fv.astype(np.float32)

def fisher_vector(xx, gmm):
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict(xx)  # NxK

    P = np.pad(Q, (0, 500-Q.shape[0]), 'constant', constant_values=(0,0))
    return P
