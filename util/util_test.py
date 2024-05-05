from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict
from util.util_img import *

import torch
import math
import os
import cv2

import os.path as osp
import numpy as np


def MatrixToImage(data):
    data = data * 255
    data = data.clip(0, 255)
    new_im = data.astype(np.uint8)
    return new_im


def tensor2np(tensor, is_feature=False):
    batch_size, channel, H, W = tensor.size()
    if not is_feature:
        img = MatrixToImage(tensor.data.cpu().numpy().reshape(channel, H, W).transpose(1, 2, 0))
        return img
    else:
        feature = tensor.data.cpu().numpy().reshape(channel, H, W).transpose(1, 2, 0)
        return feature


def tensor2np_PIL2cv2(tensor):
    batch_size, channel, H, W = tensor.size()
    img = MatrixToImage(tensor.data.cpu().numpy().reshape(channel, H, W).transpose(1, 2, 0))
    img = img[:, :, ::-1]
    return img


def np2tensor(img, cuda=False):
    img = normalization(img)
    tensor = torch.from_numpy(np.expand_dims(img, axis=0).transpose((0, 3, 1, 2)))
    if cuda:
        tensor = tensor.cuda()
    return tensor.float()


def np2tensor_wo_norm(img, cuda=False):
    _, _, C = img.shape
    img = img.astype(np.float32)
    tensor = torch.from_numpy(np.expand_dims(img, axis=0).transpose((0, 3, 1, 2)))
    if cuda:
        tensor = tensor.cuda()
    return tensor.float()
