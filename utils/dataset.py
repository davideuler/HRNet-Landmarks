import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch

MATCHED_PARTS = {
    "300W": ([1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
             [18, 27], [19, 26], [20, 25], [21, 24], [22, 23], [32, 36], [33, 35],
             [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47], [49, 55],
             [50, 54], [51, 53], [62, 64], [61, 65], [68, 66], [59, 57], [60, 56])}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_train=True, transform=None):
        if is_train:
            self.csv_file = cfg.dataset['train']
        else:
            self.csv_file = cfg.dataset['test']

        self.is_train = is_train
        self.transform = transform
        self.sigma = cfg.model['sigma']
        self.scf = cfg.dataset['scale_factor']
        self.data_root = cfg.dataset['data_dir']
        self.input_size = cfg.model['image_size']
        self.rot_factor = cfg.dataset['rot_factor']
        self.output_size = cfg.model['heatmap_size']

        # load annotations
        self.lmk_frame = pd.read_csv(self.csv_file)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.lmk_frame)

    def __getitem__(self, idx):
        row = self.lmk_frame.iloc[idx]
        name = row['image_name']
        scale = row['scale']
        center_w = row['center_w']
        center_h = row['center_h']
        pts = row[4:].values.astype('float').reshape(-1, 2)

        img = Image.open(os.path.join(self.data_root, name))
        img = np.array(img.convert('RGB'), dtype=np.float32)
        center = torch.Tensor([center_w, center_h])
        scale *= 1.25

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scf, 1 + self.scf))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5:
                img = np.fliplr(img)
                pts = self.flip_lr(pts, width=img.shape[1])
                center[0] = img.shape[1] - center[0]
        img = self.crop(img, center, scale, self.input_size, rot=r)
        img = ((img / 255.0 - self.mean) / self.std).astype(np.float32)
        img = img.transpose([2, 0, 1])

        t_pts = pts.copy()
        target = np.zeros((pts.shape[0], self.output_size[0], self.output_size[1]))

        for i in range(pts.shape[0]):
            if t_pts[i, 1] > 0:
                t_pts[i, 0:2] = transform_pixel(t_pts[i, 0:2] + 1, center,
                                                scale, self.output_size, rot=r)
                target[i] = self.generate_target(target[i], t_pts[i] - 1)

        target = torch.Tensor(target)
        t_pts = torch.Tensor(t_pts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': t_pts}
        return img, target, meta

    @staticmethod
    def flip_lr(x, width):
        matched_parts = MATCHED_PARTS['300W']
        x[:, 0] = width - x[:, 0]
        for pair in matched_parts:
            tmp = x[pair[0] - 1, :].copy()
            x[pair[0] - 1, :] = x[pair[1] - 1, :]
            x[pair[1] - 1, :] = tmp
        return x

    @staticmethod
    def crop(img, center, scale, output_size, rot=0):
        center_new = center.clone()

        h, w = img.shape[:2]
        sf = scale * 200.0 / output_size[0]
        if sf < 2:
            sf = 1
        else:
            new_size = int(np.floor(max(h, w) / sf))
            new_ht = int(np.floor(h / sf))
            new_wd = int(np.floor(w / sf))
            if new_size < 2:
                if len(img.shape) > 2:
                    return torch.zeros(output_size[0], output_size[1], img.shape[2])
                else:
                    return torch.zeros(output_size[0], output_size[1])
            else:
                # Resize image
                img = cv2.resize(img, (new_wd, new_ht), interpolation=cv2.INTER_LINEAR)
                center_new[0] = center_new[0] * 1.0 / sf
                center_new[1] = center_new[1] * 1.0 / sf
                scale = scale / sf

        # Upper left point
        ul = np.array(transform_pixel([0, 0], center_new, scale, output_size, invert=1))
        # Bottom right point
        br = np.array(transform_pixel(output_size, center_new, scale, output_size, invert=1))

        # Padding for rotation
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
        if not rot == 0:
            ul -= pad
            br += pad

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]

        # Ensure new_shape has positive dimensions
        if new_shape[0] <= 0 or new_shape[1] <= 0:
            return torch.zeros(output_size[0], output_size[1], img.shape[2]) \
                if len(img.shape) > 2 else torch.zeros(output_size[0], output_size[1])

        new_img = np.zeros(new_shape, dtype=np.float32)

        # Compute ranges for cropping
        new_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        old_x = max(0, ul[0]), min(img.shape[1], br[0])
        old_y = max(0, ul[1]), min(img.shape[0], br[1])

        # Ensure slices are valid
        if old_y[1] <= old_y[0] or old_x[1] <= old_x[0] or new_y[1] <= new_y[0] or new_x[1] <= new_x[0]:
            return torch.zeros(output_size[0], output_size[1], img.shape[2]) \
                if len(img.shape) > 2 else torch.zeros(output_size[0], output_size[1])

        # Perform cropping
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

        if not rot == 0:
            # Rotate image
            (h, w) = new_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rot, 1.0)
            new_img = cv2.warpAffine(new_img, M, (w, h), flags=cv2.INTER_LINEAR)
            new_img = new_img[pad:-pad, pad:-pad]

        new_img = cv2.resize(new_img, output_size, interpolation=cv2.INTER_LINEAR)
        return new_img

    @staticmethod
    def generate_target(img, pt):
        ul = [int(pt[0] - 3.0), int(pt[1] - 3.0)]
        br = [int(pt[0] + 3.0 + 1), int(pt[1] + 4.0)]
        if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
                br[0] < 0 or br[1] < 0):
            return img

        x = np.arange(0, (2 * 3.0 + 1), 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = (2 * 3.0 + 1) // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / 2)

        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return img

def resize(image, input_size):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(1.0, input_size / shape[0], input_size / shape[1])

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:
        image = cv2.resize(image,
                           dsize=pad,
                           interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    return image


def transform_pixel(pt, center, scale, output_size, invert=0, rot=0):
    t = get_transform(center, scale, output_size, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    points = [pt[0] - 1, pt[1] - 1, 1.]
    new_pt = np.dot(t, np.array(points).T)
    return new_pt[:2].astype(int) + 1


def get_transform(center, scale, output_size, rot=0):
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + .5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1] / 2
        t_mat[1, 2] = -output_size[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t
