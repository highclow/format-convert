##############################################################
# Copyright (c) 2018-present, JD.COM.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Xinyao Wang
# Email: xinyao.wang3@jd.com
##############################################################
import json
import os
import tqdm
import glob
import random

if __name__ == "__main__":
    json_paths = glob.glob("/mnt/hdd2/wangxiny/rough_sorting/20200304_train_val/20200304_texture_motion_kpts_seg/*.json")
#    json_paths.extend(glob.glob("/mnt/hdd2/wangxiny/rough_sorting/20200304_train_val/20200304_bg_motion_kpts_seg/*.json"))
    # json_paths = glob.glob("/mnt/hdd2/wangxiny/rough_sorting/20200304_train_val/20200304_texture_motion_kpts_val_seg/*.json")
    # json_paths.extend(glob.glob("/mnt/hdd2/wangxiny/rough_sorting/20200304_train_val/20200304_bg_motion_kpts_val_seg/*.json"))
    random.seed(1024)
    random.shuffle(json_paths)
    random.shuffle(json_paths)
    random.shuffle(json_paths)
    # json_paths = json_paths[:1000]
    save_file = "/mnt/hdd2/wangxiny/rough_sorting/20200325_train_val/20200325_texture_aug_human.json"
    json_data = {}
    json_data["categories"] = [
        {"supercategory": "person",
         "id": 1,
         "name": "person",
         'keypoints': ['nose',
                       'left_eye',
                       'right_eye',
                       'left_ear',
                       'right_ear',
                       'left_shoulder',
                       'right_shoulder',
                       'left_elbow',
                       'right_elbow',
                       'left_wrist',
                       'right_wrist',
                       'left_hip',
                       'right_hip',
                       'left_knee',
                       'right_knee',
                       'left_ankle',
                       'right_ankle'],
         'skeleton': [[16, 14],
                      [14, 12],
                      [17, 15],
                      [15, 13],
                      [12, 13],
                      [6, 12],
                      [7, 13],
                      [6, 7],
                      [6, 8],
                      [7, 9],
                      [8, 10],
                      [9, 11],
                      [2, 3],
                      [1, 2],
                      [1, 3],
                      [2, 4],
                      [3, 5],
                      [4, 6],
                      [5, 7]]},
        {"supercategory": "package", "id": 2, "name": "package",
         'keypoints': ['nose',
                       'left_eye',
                       'right_eye',
                       'left_ear',
                       'right_ear',
                       'left_shoulder',
                       'right_shoulder',
                       'left_elbow',
                       'right_elbow',
                       'left_wrist',
                       'right_wrist',
                       'left_hip',
                       'right_hip',
                       'left_knee',
                       'right_knee',
                       'left_ankle',
                       'right_ankle'],
         'skeleton': [[16, 14],
                      [14, 12],
                      [17, 15],
                      [15, 13],
                      [12, 13],
                      [6, 12],
                      [7, 13],
                      [6, 7],
                      [6, 8],
                      [7, 9],
                      [8, 10],
                      [9, 11],
                      [2, 3],
                      [1, 2],
                      [1, 3],
                      [2, 4],
                      [3, 5],
                      [4, 6],
                      [5, 7]]},
    ]

    anno_objs = []
    image_objs = []
    cur_image_id = 0
    cur_anno_id = 0

    for json_file in tqdm.tqdm(json_paths):
        with open(json_file, "r") as f:
            inst_data = json.load(f)
            image_obj = {
                "license": 1,
                "url": "None",
                "file_name": inst_data["file_name"],
                "height": inst_data["height"],
                "width": inst_data["width"],
                "date_captured": "None",
                "id": cur_image_id,
            }
            image_objs.append(image_obj)

            for anno_obj in inst_data["annotations"]:
                if "segmentation" not in anno_obj.keys():
                    import pdb
                    pdb.set_trace()
                anno_obj["image_id"] = cur_image_id
                anno_obj["id"] = cur_anno_id
                if anno_obj['category_id'] == 2:
                    anno_obj['num_keypoints'] = 0
                    anno_obj['keypoints'] = [0.0] * 51
                anno_objs.append(anno_obj)
                cur_anno_id += 1

            cur_image_id += 1

    json_data["images"] = image_objs
    json_data["annotations"] = anno_objs
    with open(save_file, 'w') as f:
        json.dump(json_data, f)
