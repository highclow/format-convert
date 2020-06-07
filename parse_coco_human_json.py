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
import json
import glob
import numpy as np
from tqdm import tqdm
import cv2
import skimage.io as sio
from joblib import Parallel, delayed
import pdb


def parse_image_meta(image_meta_raw: list) -> dict:
    image_meta = {}
    for image_info in image_meta_raw:
        image_meta[image_info['id']] = image_info['file_name']
    return image_meta



def parse_annotations(annotation: dict, image_meta: dict):
    global min_size
    global max_size
    keypoints = np.array(annotation['keypoints']).reshape([-1, 3])
    if annotation['iscrowd'] == 1:
        return
    if np.sum(keypoints[:, 2] != 0) < 5:
        return
    image_id = annotation['image_id']
    image_basename = image_meta[image_id]
    image = sio.imread(os.path.join(image_paths, image_basename))
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=2)
    anno_idx = annotation['id']
    segment_mask = np.zeros_like(image[:,:,0])
    segments = annotation['segmentation']
    # Get segmentation data
    for segment in segments:
        points_segment = segment
        try:
            points_segmentNP = np.asarray(points_segment, dtype=np.int32).reshape((-1, 2))
        except:
            print(annotation)
            print(points_segment)
        segment_mask = cv2.fillPoly(segment_mask, [points_segmentNP], 1)
    mask_indices_y, mask_indices_x = np.where(segment_mask > 0)
    min_x, max_x, min_y, max_y = np.min(mask_indices_x), np.max(mask_indices_x), \
        np.min(mask_indices_y), np.max(mask_indices_y)
    image_patch = image[min_y:max_y+1, min_x:max_x+1, :]
    mask_patch = segment_mask[min_y:max_y+1, min_x:max_x+1]

    # Get keypoints data
    offset = [min_x, min_y, 0]
    keypoints[keypoints[:,2] != 0] -= offset

    try:
        sio.imsave(os.path.join(save_rgb_path, image_basename[:-4] + '_{0:04d}.png').format(anno_idx), image_patch)
    except:
        print("Saving image failed!")
        pdb.set_trace()
    sio.imsave(os.path.join(save_seg_path, image_basename[:-4] + '_{0:04d}.png').format(anno_idx), mask_patch, check_contrast=False)
    np.save(os.path.join(save_seg_path, image_basename[:-4] + '_{0:04d}.npy').format(anno_idx), keypoints)
    min_size = min(min_size, image_patch.shape[0] * image_patch.shape[1])
    max_size = max(max_size, image_patch.shape[0] * image_patch.shape[1])
    return


json_path = '/mnt/hdd2/wangxiny/common_datasets/COCO/annotations/person_keypoints_val2017.json'
image_paths = '/mnt/hdd2/wangxiny/common_datasets/COCO/val2017/'
save_rgb_path ='/mnt/hdd2/wangxiny/rough_sorting/data_augmentation/coco_person_val/'
save_seg_path ='/mnt/hdd2/wangxiny/rough_sorting/data_augmentation/coco_person_mask_val/'
if not os.path.exists(save_rgb_path):
    os.makedirs(save_rgb_path)
if not os.path.exists(save_seg_path):
    os.makedirs(save_seg_path)

min_size = float('inf')
max_size = 0

with open(json_path, 'r') as f:
    json_data = json.load(f)

image_meta = parse_image_meta(json_data['images'])
Parallel(n_jobs=10, backend='threading', verbose=10)(delayed(parse_annotations)(annotation, image_meta) for annotation in json_data['annotations'])
# for image_meta in json_data['images']:
#     process_json(image_meta)
print('Min size: {}, max size: {}'.format(min_size, max_size))
