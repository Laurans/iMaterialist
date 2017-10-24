import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from numpy import linalg as LA
import os
import pickle
import tqdm

imnames = os.listdir('../data/val_raw')
imnames = ['../data/val_raw/{}'.format(i) for i in imnames]
# load list of images
imlist = (io.imread_collection(imnames[:3000]))

res = np.zeros(shape=(1, 3))
for i in tqdm.trange(len(imlist)):
    try:
        # re-size all images to 256 x 256 x 3
        ms = transform.resize(imlist[i], (256, 256, 3), mode='constant')
        # re-shape to make list of RGB vectors.
        arr = ms.reshape((256 * 256), 3)
        # consolidate RGB vectors of all images
        res = np.concatenate((res, arr), axis=0)
    except:
        continue
res = np.delete(res, (0), axis=0)

# subtracting the mean from each dimension
ms = res.mean(axis=0)
res = res - ms

R = np.cov(res, rowvar=False)

evals, evecs = LA.eigh(R)

idx = np.argsort(evals)[::-1]
evecs = evecs[:, idx]
# sort eigenvectors according to same index

evals = evals[idx]
# select the first 3 eigenvectors (3 is desired dimension
# of rescaled data array)

evecs = evecs[:, :3]
evecs_mat = np.column_stack((evecs))

pickle.dump([evals, evecs_mat], open('../data/fancyPCA.pkl', 'wb'))

