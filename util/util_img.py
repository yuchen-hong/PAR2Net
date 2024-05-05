import os
import os.path as osp

import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_dir(path):
    if not osp.isdir(path):
        os.makedirs(path)


def normalization(img):
    return img.astype(np.float32) / 255.0


def quantization(img):  # in (0,1) out(0,255)
    img = np.clip(img, 0, 1)
    img_oct = img * 255
    img_oct = img_oct.astype(np.uint8)
    return img_oct