# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

import _init_paths
import lib.models
import lib.datasets
from lib.config import config
from lib.config import update_config
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, speed_test

import math
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # ------------------------LSUI-----------------------------
                        default="../experiments/LSUI/UWMamba.yaml",


                        # ------------------------UIEB-----------------------------
                        # default= "../experiments/UIEB/UWMamba.yaml",

                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('lib.models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('lib.models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    pretrained_state = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_state = {k: v for k, v in pretrained_state.items() if
                        (k in model_dict and v.shape == model_dict[k].shape)}

    for k, _ in pretrained_state.items():
        print('=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_state)
    model.load_state_dict(model_dict, strict=False)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('lib.datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    PSNR_list = []
    SSIM_list = []
    sv_dir = './enhanced_output/'
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)
    start = timeit.default_timer()

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, name = batch
            image = image.cuda()
            label = label.cuda()
            pred = model(image)
            if pred.shape[2] != label.shape[2] or pred.shape[3] != label.shape[3]:
                pred = F.interpolate(pred, size=(label.shape[2], label.shape[3]), mode='bilinear',align_corners=True)
            pred_ = torch.squeeze(pred,dim=0).permute(1,2,0).cpu().numpy()
            label_ = torch.squeeze(label,dim=0).permute(1,2,0).cpu().numpy()

            score_psnr = cal_psnr(pred_, label_, data_range=1)
            score_ssim = cal_ssim(pred_, label_, channel_axis=2, data_range=1)
            PSNR_list.append(score_psnr)
            SSIM_list.append(score_ssim)
            pred_vis = pred_*255
            pred_vis = pred_vis[:, :, ::-1]
            save_name = os.path.join(sv_dir, name[0] + '.jpg')
            cv2.imwrite(save_name,pred_vis)

            if idx % 10 == 0:
                print(idx)
    PSNR_list = np.array(PSNR_list)
    PSNR = PSNR_list.mean()
    SSIM_list = np.array(SSIM_list)
    SSIM = SSIM_list.mean()

    msg = 'PSNR: {: 4.4f}, SSIM: {: 4.4f}'.format(PSNR, SSIM)
    logging.info(msg)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int64((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
