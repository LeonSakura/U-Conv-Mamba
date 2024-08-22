# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time
import sys

import cv2
import numpy as np
# import numpy.ma as ma
from tqdm import tqdm
from PIL import Image
from scipy import ndimage

import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix

from lib.utils.utils import AverageMeter
from lib.utils.utils import get_confusion_matrix
from lib.utils.utils import adjust_learning_rate
from lib.utils.utils import Map16, Vedio

from lib.core.criterion import DistillationLoss1, DistillationLoss2, freematch_ce_loss, MultiClassFocalLoss
import random
from lib.utils import transformsgpu
from lib.core.criterion import PrototypeContrastiveLoss

# from utils.DenseCRF import DenseCRF

# from apex import amp

import lib.utils.distributed as dist

vedioCap = Vedio('./output/cdOffice.mp4')
map16 = Map16(vedioCap)


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size

