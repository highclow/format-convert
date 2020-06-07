import os
import json
import glob
import numpy as np
from tqdm import tqdm
import cv2
import skimage.io as sio
import pdb
import xml.dom.minidom

def process_shape(pointsNP, image):
    mask = np.zeros_like(image)[:,:,0]
    mask = cv2.fillPoly(mask, [pointsNP], 1)
    min_x, max_x, min_y, max_y = np.min(pointsNP[:, 1]), np.max(pointsNP[:, 1]), \
        np.min(pointsNP[:, 0]), np.max(pointsNP[:, 0])
    image_patch = image[min_x:max_x+1, min_y:max_y+1, :]
    mask_patch = mask.get()[min_x:max_x+1, min_y:max_y+1]
    return image_patch, mask_patch


#labelmap = {u"管制弓弩": 1,
#            u"管制刀具": 2,
#            u"冲锋枪": 3,
#            u"手枪": 3,
#            u'机枪': 3,
#            u'步枪': 3,
#            u'手雷': 4,
#            u'地雷': 4,
#            u'火箭弹': 4,
#            u'厨用刀': 5,
#            u'玩具枪': 6,
#            u'玩具弓箭': 7 }

def process_json(xml_file):
    global min_size
    global max_size
    dom = xml.dom.minidom.parse(xml_file)
    root = dom.documentElement
    img_file = dom.getElementsByTagName("path")[0].childNodes[0].data.split("\\")[-1]
    img_name = os.path.splitext(img_file)[0]
#    label = labelmap[img_name.split('_')[0]]
    objects=dom.getElementsByTagName("object")
    image = sio.imread(os.path.join(os.path.dirname(xml_file), img_file))
    for k, object in enumerate(objects):
        polygon = root.getElementsByTagName('polygon')[k]
        polygon = polygon.childNodes[0].data
        polygon = np.array(json.loads(polygon)).astype(int)
        image_patch, mask_patch = process_shape(polygon, image)
        
#        mask_patch[mask_patch == 1] = label
        mask_patch[mask_patch == 1] = 255

        if image_patch is not None and mask_patch is not None:
            sio.imsave(os.path.join(save_rgb_path, img_name + '_{0:04d}.png').format(k), image_patch)
            sio.imsave(os.path.join(save_seg_path, img_name + '_{0:04d}.png').format(k), mask_patch, check_contrast=False)
            min_size = min(min_size, image_patch.shape[0] * image_patch.shape[1])
            max_size = max(max_size, image_patch.shape[0] * image_patch.shape[1])



xml_paths = 'weapons/'
image_paths = 'weapons/'
save_rgb_path ='rgbs/'
save_seg_path ='segs/'

if not os.path.exists(save_rgb_path):
    os.makedirs(save_rgb_path)
if not os.path.exists(save_seg_path):
    os.makedirs(save_seg_path)

xml_files = glob.glob(xml_paths + '*.xml')

min_size = float('inf')
max_size = 0

for xml_file in tqdm(xml_files):
    process_json(xml_file)
print('Min size: {}, max size: {}'.format(min_size, max_size))
