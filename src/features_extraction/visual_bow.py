import cv2
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
import os
import tqdm
from sklearn.externals import joblib

def generate_gmm(number):
    pass

def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (512,512))
    if img is None:
        raise IOError("Unable to open '%s'. Are you sure it's a valid image path?")
    return img

def gen_sift_features(img_paths='../data/train_raw/'):
    img_descs = np.zeros((0,32))
    imnames = os.listdir(img_paths)
    imnames = ['../data/train_raw/{}'.format(i) for i in imnames[:1000]]
    for img_path in tqdm.tqdm(imnames, desc='Create sift descriptor'):
        try:
            desc = img_desc(img_path)
            img_descs = np.vstack((img_descs, desc))
        except:
            pass
    return img_descs


def img_desc(img_path, type='sjift'):
    img = read_image(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if type=='sift':
        sift = cv2.xfeatures2d.SIFT_create()
        _, desc = sift.detectAndCompute(gray, None)
    else:
        orb = cv2.ORB_create()
        _, desc = orb.detectAndCompute(gray, None)
    return desc

def cluster_descriptors(img_descs):
    all_train_descriptors = img_descs
    cluster_model = KMeans(100, verbose=1, n_jobs=-1)
    cluster_model.fit(all_train_descriptors)
    return cluster_model

def predict_descriptors(img_path, model):
    desc = img_desc(img_path)
    clustered_desc = model.predict(desc)
    img_bow_hist = np.bincount(clustered_desc, minlength=model.n_clusters)
    return img_bow_hist.reshape(1, -1)

def train():
    pass


feat = gen_sift_features()
#model = cluster_descriptors(feat)
#joblib.dump(model, '../data/vbow.ml')

#model2 = GaussianMixture(n_components=256, covariance_type='diag')
#model2.fit(feat)
#joblib.dump(model2, '../data/gmm.ml')
model2 = joblib.load('../data/gmm.ml')
feat = model2.predict(feat)
#print('PCA')
#model3 = PCA(n_components=128)
#model3.fit(feat)
#joblib.dump(model3, '../data/pca.ml')

