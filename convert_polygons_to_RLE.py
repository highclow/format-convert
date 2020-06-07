##############################################################
# Copyright (c) 2018-present, JD.COM.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Xinyao Wang
# Email: xinyao.wang3@jd.com
##############################################################
from pycocotools import mask as cocomask
import numpy as np
import glob
import os
import json
import cv2
from tqdm import tqdm

# source_json_path = "/mnt/hdd2/wangxiny/rough_sorting/20200304_train_val/20200304_texture_aug_human_v2.json"
# save_json_path = "/mnt/hdd2/wangxiny/rough_sorting/20200304_train_val/20200304_texture_aug_human_v3.json"
source_json_path = "/mnt/hdd2/wangxiny/rough_sorting/20200325_train_val/20200325_texture_aug_human.json"
save_json_path = "/mnt/hdd2/wangxiny/rough_sorting/20200325_train_val/20200325_texture_aug_human_rle.json"
# source_json_path = "/mnt/hdd2/wangxiny/rough_sorting/20200304_train_val/20200304_texture_aug_human_val_small_v2.json"
# save_json_path = "/mnt/hdd2/wangxiny/rough_sorting/20200304_train_val/20200304_texture_aug_human_val_small_v3.json"

image_height = 700
image_width = 1440
print("Loading json file...")
with open(source_json_path, 'r') as f:
    source_data = json.load(f)
print("Done!")

for anno in tqdm(source_data['annotations']):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for segment in anno['segmentation']:
        seg_pts = np.array(segment).reshape(1,-1,2)
        mask = cv2.fillPoly(mask, seg_pts.astype(int), 1)

    mask_b = np.array(mask, dtype=np.uint8, order="F")
    mask_encod = cocomask.encode(mask_b)
    mask_encod['counts'] = mask_encod['counts'].decode('utf-8')
    anno['segmentation'] = mask_encod
    # if anno['category_id'] == 2:
    #     anno.pop('num_keypoints')
    #     anno.pop('keypoints')

with open(save_json_path, 'w') as f:
    json.dump(source_data, f)
