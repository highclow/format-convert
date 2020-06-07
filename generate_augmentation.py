import os
import glob
import json
import cv2
from tqdm import tqdm
import random
import numpy as np
import skimage.io as sio
from PIL import Image
from imantics import Polygons, Mask
from skimage import transform as ski_transform
from skimage.color import rgb2gray
from joblib import Parallel, delayed
from imgaug import augmenters as iaa
from torchvision.transforms import Lambda, Compose
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue
import pdb

class RandomColor(object):
    def __init__(self, brightness=0.3, contrast=0.15, saturation=0.15, hue=0.0):
        self.brightness = brightness
        self.saturation = saturation
        self.contrast = contrast
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        image = sample['image']
        if np.random.uniform() > 0.2:
            image = Image.fromarray(image)
            transform = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
            image = transform(image)
            image = np.array(image)
            sample['image'] = image
        return sample

class ResizeCrop(object):
    def __call__(self, sample):
        image = sample['image']
        ori_height = image.shape[0]
        ori_width = image.shape[1]
        image_height, image_width = image.shape[:2]
        max_ratio = min(image_height / ori_height,
                        image_width / ori_width)
        ratio = np.random.uniform(max_ratio*0.7, max_ratio)
        output_height = int(ori_height * ratio)
        output_width = int(ori_width * ratio)
        crop_y = np.random.randint(0, image_height - output_height)
        crop_x = np.random.randint(0, image_width - output_width)
        crop_image = cv2.resize(image[crop_y:crop_y+output_height+1,
                                      crop_x:crop_x+output_width+1].copy(),
                                (ori_width, ori_height),
                                interpolation=cv2.INTER_AREA)
        sample['image'] = crop_image
        return sample

class RandomRotation(object):

    def __init__(self, rt_range, scale_range,
                 min_size, max_size):
        self.rt_range = rt_range
        self.scale_range = scale_range
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['fg_mask']
        scale = np.clip(np.random.randn() * self.scale_range + 1.0,
                        1 - self.scale_range, 1 + self.scale_range)
        rot = np.random.uniform(-self.rt_range, self.rt_range)
        if np.random.uniform() <= 0.2:
            rot = 0.0
            scale = 1.0
        (h, w) = image.shape[:2]
        if h * scale * w * scale < self.min_size:
            scale = np.sqrt(self.min_size / (h * w))
        if h * scale * w * scale > self.max_size:
            scale = np.sqrt(self.max_size / (h * w))
        if rot != 0.0 or scale != 1.0:
            image = self._rotate_and_scale(image, rot, scale, is_mask=False)
            if mask is not None:
                mask = self._rotate_and_scale(mask, rot, scale, is_mask=True)
        # (h, w) = image.shape[:2]
        # print('scale:{} {} {}'.format(h, w, scale))
        sample['image'] = image
        sample['fg_mask'] = mask
        return sample

    @staticmethod
    def _rotate_and_scale(image, angle, scale, is_mask):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        if is_mask:
            inter_method = cv2.INTER_NEAREST
        else:
            inter_method = cv2.INTER_AREA
        return cv2.warpAffine(image, M, (nW, nH), flags=inter_method)

class PadTexture(object):
    def __call__(self, sample):
        image = sample['image']
        ori_height = image.shape[0]
        ori_width = image.shape[1]
        resize_ratio = np.random.uniform(0.2, 1.5)
        image = cv2.resize(image, (min((int(image.shape[1]*resize_ratio), 450)),
                                   min((int(image.shape[1]*resize_ratio)), 450)),
                           cv2.INTER_AREA)
        patch_height, patch_width = image.shape[:2]
        top = (ori_height - patch_height) // 2
        bottom = ori_height - patch_height - top
        left = (ori_width - patch_width) // 2
        right = ori_width - patch_width - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_WRAP)
        sample['image'] = image

        return sample

class RandomFlip(object):
    def __init__(self, flip_rate=0.5):
        self.flip_rate=0.5

    def __call__(self, sample):
        image, mask = sample['image'], sample['fg_mask']
        if np.random.uniform() > self.flip_rate:
            image = np.fliplr(image).copy()
            if mask is not None:
                mask = np.fliplr(mask).copy()
        if np.random.uniform() > self.flip_rate:
            image = np.flipud(image).copy()
            if mask is not None:
                mask = np.flipud(mask).copy()
        sample['image'] = image
        sample['fg_mask'] = mask
        return sample

class RandomJpegEffect(object):
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, sample):
        image = sample['image']
        if np.random.uniform() < self.prob:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            result, encimg = cv2.imencode('.jpg', image, encode_param)
            image = cv2.imdecode(encimg, 1)
        sample['image'] = image
        return sample

class RandomGaussianBlur(object):
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, sample):
        image = sample['image']
        if np.random.uniform() < self.prob:
            size = np.random.randint(1, 5) * 2 + 1
            image = cv2.GaussianBlur(image, (size, size), 0)
        sample['image'] = image
        return sample

class RandomGaussianNoise(object):
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, sample):
        image = sample['image']
        if np.random.uniform() < self.prob:
            div = np.random.uniform(0, 20)
            add_noise = iaa.AdditiveGaussianNoise(scale=div)
            image = add_noise.augment_image(image)
        sample['image'] = image
        return sample


#class MotionBlur(object):
#    def __init__(self,
#                 angle_range,
#                 blur_range,
#                 motion_min,
#                 motion_max,
#                 probability):
#        self.angle_range = angle_range
#        self.blur_range = blur_range
#        self.motion_min = motion_min
#        self.motion_max = motion_max
#        self.probability = probability
#
#    def motion_kernel(self, angle, d, sz=65):
#        kern = np.ones((1, d), np.float32)
#        c, s = np.cos(angle), np.sin(angle)
#        A = np.float32([[c, -s, 0], [s, c, 0]])
#        sz2 = sz // 2
#        A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
#        kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
#        return kern
#
#    def __call__(self, fg_sample, bg_sample, fg_cur_idx,
#                 mask_indices, aug_pos_x, aug_pos_y):
#        angle = np.random.uniform(-self.angle_range, self.angle_range)
#        motion_size = np.random.randint(self.motion_min, self.motion_max+1)
#        blur_size = np.random.randint(0, self.blur_range+1)
#        psf = self.motion_kernel(angle, motion_size)
#        if blur_size > 0:
#            psf = cv2.blur(psf, (blur_size, blur_size))
#        psf /= psf.sum()
#
#        fg_image, fg_mask = fg_sample['image'], fg_sample['fg_mask']
#        bg_image, _ = bg_sample['image'], bg_sample['fg_mask']
#
#        motion_padding = max(blur_size, motion_size)
#        fg_image = cv2.copyMakeBorder(fg_image, motion_padding, motion_padding,
#                                        motion_padding, motion_padding, cv2.BORDER_CONSTANT,
#                                        value=0)
#        fg_mask = cv2.copyMakeBorder(fg_mask, motion_padding, motion_padding,
#                                        motion_padding, motion_padding, cv2.BORDER_CONSTANT,
#                                        value=0)
#        fg_height, fg_width = fg_mask.shape
#        bg_padding = 0
#        if aug_pos_y + fg_height >= bg_image.shape[0]:
#            bg_padding = aug_pos_y + fg_height - bg_image.shape[0] + 1
#        if aug_pos_x + fg_width >= bg_image.shape[1]:
#            bg_padding = max(bg_padding, aug_pos_x + fg_width - bg_image.shape[1] + 1)
#        if bg_padding > 0:
#            bg_image = cv2.copyMakeBorder(bg_image, bg_padding, bg_padding,
#                                        bg_padding, bg_padding, cv2.BORDER_CONSTANT,
#                                        value=0)
#        bg_patch = bg_image[aug_pos_y+bg_padding:aug_pos_y+bg_padding+fg_height,
#                            aug_pos_x+bg_padding:aug_pos_x+bg_padding+fg_width, :]
#        fg_mask[fg_mask!=fg_cur_idx] = 0
#        fg_mask[fg_mask==fg_cur_idx] = 1
#        fg_mask_3 = np.stack([fg_mask,fg_mask,fg_mask], axis=2)
#        fg = np.where(fg_mask_3==0, bg_patch, fg_image)
#        fg_blr = cv2.filter2D(fg, cv2.CV_32F, psf, borderType=cv2.BORDER_CONSTANT)
#        fg_mask_blr = cv2.filter2D(fg_mask, cv2.CV_32F, psf, borderType=cv2.BORDER_CONSTANT)
#        fg_mask_blr[fg_mask_blr<0.0] = 0.0
#        fg_mask_blr[fg_mask_blr>1.0] = 1.0
#        fg_mask_blr3 = np.stack([fg_mask_blr, fg_mask_blr, fg_mask_blr], axis=2)
#        fg_sample['image'] = (fg_blr * fg_mask_blr3 + bg_patch * (1 - fg_mask_blr3)).astype(np.uint8)
#        fg_mask_blr_cp = fg_mask_blr.copy()
#        fg_mask_blr[fg_mask_blr>0.01] = fg_cur_idx # use 0.01 for floating precision issue
#        fg_mask_blr = fg_mask_blr.astype(np.uint8)
#        adj_mask_indices = np.where(fg_mask_blr == fg_cur_idx)
#        fg_mask_blr[fg_mask_blr_cp<0.6] = 0
#        fg_sample['fg_mask'] = fg_mask_blr
#        return adj_mask_indices

class DataGenerator(object):
    def __init__(self,
                 bg_path_list,
                 fg_path_list,
                 fg_mask_path,
                 save_rgb_path='.',
                 save_seg_path='.',
                 bg_is_texture=False,
                 max_groups=3,
                 min_fg=3,
                 max_fg=10,
                 rotation=0.0,
                 scale=1.0,
                 brightness=0.0,
                 contrast=0.0,
                 saturation=0.0,
                 jpeg_effect=False,
                 gaussian_blur=False,
                 gaussian_noise=False,
                 flip=False,
                 motion_blur=False,
                 random_color=False,
                 prefix='img'):

        self.bg_path_list = bg_path_list
        self.bg_is_texture = bg_is_texture
        self.fg_path_list = fg_path_list
        self.fg_mask_path = fg_mask_path
        self.max_groups = max_groups
        self.min_fg = min_fg
        self.max_fg = max_fg
        self.save_rgb_path = save_rgb_path
        self.save_seg_path = save_seg_path
        self.rotation = rotation
        self.scale = scale
        self.jpeg_effect = jpeg_effect
        self.gaussian_blur = gaussian_blur
        self.gaussian_noise = gaussian_noise
        self.flip = flip
        self.motion_blur = motion_blur
        self.random_color = random_color
        self.prefix = prefix
        self._build_fg_pipe()
        self._build_bg_pipe()

        if self.motion_blur:
            self.apply_motion_blur = MotionBlur(180, 5, 20, 40, 0.1)


    def _build_fg_pipe(self):
        self.fg_pipe = []
        if self.rotation != 0 or self.scale != 0.0:
            rotater = RandomRotation(self.rotation, self.scale,
                                     # min_size=20000, max_size=40000)
                                     min_size=2000, max_size=160000)
            self.fg_pipe.append(rotater)
        if self.jpeg_effect:
            jpeg_effector = RandomJpegEffect(0.1)
            self.fg_pipe.append(jpeg_effector)
        if self.gaussian_noise:
            noiser = RandomGaussianNoise(0.1)
            self.fg_pipe.append(noiser)
        if self.gaussian_blur:
            blurer = RandomGaussianBlur(0.1)
            self.fg_pipe.append(blurer)
        if self.flip:
            fliper = RandomFlip()
            self.fg_pipe.append(fliper)
        if self.random_color:
            random_colorer = RandomColor()
            self.fg_pipe.append(random_colorer)

    def _build_bg_pipe(self):
        self.bg_pipe = []
        if self.bg_is_texture:
            texture_padder = PadTexture()
            self.bg_pipe.append(texture_padder)
        else:
            resizer = ResizeCrop()
            self.bg_pipe.append(resizer)
        if self.jpeg_effect:
            jpeg_effector = RandomJpegEffect(0.2)
            self.bg_pipe.append(jpeg_effector)
        if self.gaussian_noise:
            noiser = RandomGaussianNoise(0.2)
            self.bg_pipe.append(noiser)
        if self.gaussian_blur:
            blurer = RandomGaussianBlur(0.2)
            self.bg_pipe.append(blurer)
        if self.flip:
            fliper = RandomFlip()
            self.bg_pipe.append(fliper)
        if self.random_color:
            random_colorer = RandomColor()
            self.bg_pipe.append(random_colorer)

    def apply_augmentation(self, sample, pipeline):
        for func in pipeline:
            sample = func(sample)
        return sample

    def apply_single_fg(self, bg_sample,
                        group_bbox_pos_x,
                        group_bbox_pos_y,
                        group_bbox_width,
                        group_bbox_height):
        fg_img_path = random.choice(self.fg_path_list)
        fg_mask_path = os.path.join(self.fg_mask_path,
                                    os.path.basename(fg_img_path)[:-3] + 'png')
        ori_width = bg_sample['width']
        ori_height = bg_sample['height']
        fg_img = sio.imread(fg_img_path)
        fg_mask = sio.imread(fg_mask_path)
        ratio = np.random.uniform() * 2 + 1
        height, width = fg_mask.shape
        fg_img = cv2.resize(fg_img, (int(width * ratio), int(height*ratio)),
                            interpolation=cv2.INTER_AREA)
        fg_mask = cv2.resize(fg_mask, (int(width * ratio), int(height*ratio)),
                             interpolation=cv2.INTER_NEAREST)

        height, width = fg_mask.shape
        fg_sample = {}
        label = np.sort(np.unique(fg_mask))[-1]
        if np.sum(bg_sample['fg_mask'] == label) > 0:
           print(label, np.unique(bg_sample['fg_mask']), np.sum(bg_sample['fg_mask'] == label))
           return bg_sample
        fg_sample['image'] = fg_img.copy()
        fg_sample['fg_mask'] = fg_mask.copy()
        fg_sample = self.apply_augmentation(fg_sample, self.fg_pipe)
        mask_indices = np.where((fg_sample['fg_mask'] != 0))
        y_min = np.min(mask_indices[0])
        y_max = np.max(mask_indices[0])
        x_min = np.min(mask_indices[1])
        x_max = np.max(mask_indices[1])
        if x_max - x_min > group_bbox_width or y_max - y_min > group_bbox_height:
            if (ori_width - x_max + x_min - 2 <= 0) or (ori_height - y_max + y_min - 2 <= 0):
                return bg_sample
            aug_pos_x = np.random.randint(0, ori_width - x_max + x_min - 2)
            aug_pos_y = np.random.randint(0, ori_height - y_max + y_min - 2)
        else:
            aug_pos_x = np.random.randint(group_bbox_pos_x,
                                          min(max((ori_width - x_max + x_min - 2), group_bbox_pos_x+1),
                                              group_bbox_pos_x + group_bbox_width))
            aug_pos_y = np.random.randint(group_bbox_pos_y,
                                          min(max((ori_height - y_max + y_min - 2), group_bbox_pos_y+1),
                                              group_bbox_pos_y + group_bbox_height))

#        if self.motion_blur and \
#           np.random.uniform(0, 1) < self.apply_motion_blur.probability and \
#           y_max + aug_pos_y + 80 < ori_height and \
#           x_max + aug_pos_x + 80 < ori_width:
#            rgb_mask_indices = self.apply_motion_blur(fg_sample, bg_sample, fg_cur_idx,
#                                                      mask_indices, aug_pos_x, aug_pos_y)
#            
#            bg_sample['image'][rgb_mask_indices[0]+aug_pos_y,
#                               rgb_mask_indices[1]+aug_pos_x] = \
#                                   fg_sample['image'][rgb_mask_indices[0],
#                                                      rgb_mask_indices[1]].copy()
#            mask_indices = np.where(fg_sample['fg_mask'] == fg_cur_idx)
#            bg_sample['fg_mask'][mask_indices[0]+aug_pos_y,
#                                mask_indices[1]+aug_pos_x] = \
#                                    fg_sample['fg_mask'][mask_indices[0],
#                                                         mask_indices[1]].copy()
#            
#        else:
        indices = (mask_indices[0]+aug_pos_y, mask_indices[1]+aug_pos_x) 
        bg_sample['image'][indices] = fg_sample['image'][mask_indices[0],
                                                         mask_indices[1]].copy()
        bg_sample['fg_mask'][indices] = fg_sample['fg_mask'][mask_indices[0],
                                                            mask_indices[1]].copy()
#        bg_sample['fg_mask'][indices] = 255
        return bg_sample

    def generate_json(self, bg_sample, image_index):
        raw_mask = bg_sample['fg_mask']
        json_data = {}
        json_data['file_name'] = self.prefix + '_' + str(image_index) + '.jpg'
        json_data['height'] = bg_sample['height']
        json_data['width'] = bg_sample['width']
        json_data['annotations'] = []
        inst_ids = np.unique(raw_mask)
        inst_ids = inst_ids[inst_ids >= 1]
        for inst_id in inst_ids:
            inst_anno = {}
            inst_mask = np.zeros_like(raw_mask)
            inst_mask[raw_mask == inst_id] = inst_id
            if inst_mask.sum() < 200:
                continue
            inst_poly = Mask(inst_mask).polygons()
            inst_anno['bbox'] = list(inst_poly.bbox())
            inst_anno['segmentation'] = [(np.array(poly) + 0.5).tolist() for \
                                             poly in inst_poly.segmentation \
                                             if len(poly) % 2 == 0 and len(poly) >= 6]

            # Adjust bbox for removed parts
            if len(inst_anno['segmentation']) != len(inst_poly.segmentation):
                min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
                for poly in inst_anno['segmentation']:
                    min_x = min(min_x, min(poly[0::2]))
                    max_x = max(max_x, max(poly[0::2]))
                    min_y = min(min_y, min(poly[1::2]))
                    max_y = max(max_y, max(poly[1::2]))
                    inst_anno['bbox'] = [int(min_x - 0.5), int(min_y - 0.5), \
                                         int(max_x - 0.5), int(max_y - 0.5)]

            inst_anno['category_id'] = 0
            inst_anno['iscrowd'] = 0
            json_data['annotations'].append(inst_anno)

        return json_data




    def generate_sample(self, index):
        bg_img_path = random.choice(self.bg_path_list)
        bg_img = sio.imread(bg_img_path)
        ori_height, ori_width = bg_img.shape[:2]
        if len(bg_img.shape) == 2:
            bg_img = np.stack((bg_img, bg_img, bg_img), axis=2)
        bg_sample = {'image': bg_img, 'fg_mask':None, 'width':ori_width, 'height':ori_height}

        bg_sample = self.apply_augmentation(bg_sample, self.bg_pipe)
        bg_sample['fg_mask'] = np.zeros_like(bg_sample['image'][:,:,0])
        max_size, min_size = max(ori_height, ori_width), min(ori_height, ori_width)
        bbox_min = min_size * 0.5
        bbox_max = min(max_size * 0.9, min_size)

        group_bbox_height = np.random.randint(bbox_min, bbox_max)
        group_bbox_width = np.random.randint(bbox_min, bbox_max)
        group_bbox_pos_x = np.random.randint(0, ori_width - \
                                            group_bbox_width - 2)
        group_bbox_pos_y = np.random.randint(0, ori_height - \
                                            group_bbox_height - 2)
        num_fgs = np.random.randint(self.min_fg, self.max_fg + 1)
        for _ in range(num_fgs):
          try:
            bg_sample = self.apply_single_fg(bg_sample,
                                             group_bbox_pos_x,
                                             group_bbox_pos_y,
                                             group_bbox_width,
                                             group_bbox_height)
          except:
            continue

        if not os.path.exists(self.save_rgb_path):
            os.makedirs(self.save_rgb_path)
        if not os.path.exists(self.save_seg_path):
            os.makedirs(self.save_seg_path)
        sio.imsave(os.path.join(self.save_rgb_path, self.prefix + '_' + \
                                str(index) + '.jpg'), bg_sample['image'],
                   quality=100)
        sio.imsave(os.path.join(self.save_seg_path, self.prefix + '_' + \
                                str(index) + '.png'), bg_sample['fg_mask'],
                   check_contrast=False)
        # Writing json meta data
        json_data = self.generate_json(bg_sample, index)
        with open(os.path.join(self.save_seg_path, self.prefix + '_' + \
                                str(index) + '.json'), 'w') as f:
            json.dump(json_data, f)
        return

    def generate_all(self, num_samples):
#        Parallel(n_jobs=10, backend='threading', verbose=10)(delayed(self.generate_sample)(index) for index in range(num_samples))
    
        [self.generate_sample(index) for index in range(num_samples)]


def process_background_blur():
    bg_image_path = 'background/'
    bg_path_list = glob.glob(bg_image_path+'*.jpg')
    fg_image_path = 'rgbs_filtered/'
    fg_path_list = glob.glob(fg_image_path+'*.png')
    fg_mask_path = 'segs_filtered/'
    generator_texture = DataGenerator(bg_path_list=bg_path_list,
                                      fg_path_list=fg_path_list,
                                      fg_mask_path=fg_mask_path,
                                      save_rgb_path='aug_rgbs/',
                                      save_seg_path='aug_masks/',
                                      bg_is_texture=False,
                                      max_groups=3,
                                      min_fg=1,
                                      max_fg=3,
                                      rotation=60.0,
                                      scale=0.5,
                                      brightness=0.50,
                                      contrast=0.15,
                                      saturation=0.15,
                                      jpeg_effect=True,
                                      gaussian_blur=False,
                                      gaussian_noise=True,
                                      flip=True,
                                      motion_blur=False,
                                      prefix='train_0')
    generator_texture.generate_all(10000)
if __name__ == '__main__':
    process_background_blur()
