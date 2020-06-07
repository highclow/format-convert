##############################################################
# Copyright (c) 2018-present, JD.COM.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Xinyao Wang
# Email: xinyao.wang3@jd.com
##############################################################
import os
import glob
import shutil
import numpy as np
import cv2
from tqdm import tqdm
import skimage.io as sio
from skimage.morphology import convex_hull_image
from sklearn import cluster
import pdb

image_root = "rgbs/"
mask_root = "segs/"

save_image_root = "rgbs_filtered"
save_mask_root = "segs_filtered"

if not os.path.exists(save_image_root):
    os.makedirs(save_image_root)
if not os.path.exists(save_mask_root):
    os.makedirs(save_mask_root)

image_paths = glob.glob(image_root+"*.png")

for image_path in tqdm(image_paths):
    image_basename = os.path.basename(image_path)
    image_gray = cv2.imread(os.path.join(image_root, image_basename), cv2.IMREAD_GRAYSCALE)
    X = image_gray.reshape((-1, 1))
    k_means = cluster.KMeans(n_clusters=2)
    k_means.fit(X)
    X_clustered = k_means.labels_ * 255
    X_clustered = X_clustered.reshape(image_gray.shape).astype('uint8')
    image_mask = sio.imread(os.path.join(mask_root, image_basename))
    #image_mask[image_mask>0] = 1
    shape_area = (image_mask>0).sum()
    #hull = convex_hull_image(image_mask)
    if shape_area > 600:# and shape_area / hull.sum() > 0.0:
        shutil.copy(image_path, save_image_root)
#        sio.imsave(os.path.join(save_mask_root, image_basename), X_clustered)
#        import pdb; pdb.set_trace()
        shutil.copy(os.path.join(mask_root, image_basename), save_mask_root)
#        if os.path.exists(os.path.join(mask_root, image_basename[:-4]+".npy")):
#            shutil.copy(os.path.join(mask_root, image_basename[:-4]+".npy"), save_mask_root)
