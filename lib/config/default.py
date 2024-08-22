
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.PRETRAINED_1 = ''

_C.MODEL.PRETRAINED_2 = ''
_C.MODEL.STUBBORNNESS_TEAHCER_BN_ADAPT = False
_C.MODEL.STUBBORNNESS_TEAHCER_BN_MOM_PRE = 0.1
_C.MODEL.STUBBORNNESS_TEAHCER_BN_DECAY_FACTOR = 0.94
_C.MODEL.STUBBORNNESS_TEAHCER_BN_MIN_MOM_CONSTANT = 0.005
_C.MODEL.LABELAWARE_PSEUDO = False

_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.NUM_OUTPUTS = 2
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.CHOOSE = "vit_b"

_C.MODEL.OCR = CN()
_C.MODEL.OCR.MID_CHANNELS = 512
_C.MODEL.OCR.KEY_CHANNELS = 256
_C.MODEL.OCR.DROPOUT = 0.05
_C.MODEL.OCR.SCALE = 1

_C.LOSS = CN()
_C.LOSS.USE_OHEM = False
_C.LOSS.USE_SEMI_FOCAL_LOSS = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = False
_C.LOSS.BALANCE_WEIGHTS = [0.5, 0.5]
_C.LOSS.EDGE_LOSS_WEIGHTS = 0.1
_C.LOSS.KD_LOGIT_LOSS_WEIGHTS = 0.5
_C.LOSS.KD_FEATURE_LOSS_WEIGHTS = 0.00005

# DATASET related params
_C.DATASET = CN()
_C.DATASET.MODEL = 'train'
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'cityscapes'
_C.DATASET.DATASET_VAL = 'NC'
_C.DATASET.TRAIN = 'train'
_C.DATASET.VAL = 'val'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SET = 'list/cityscapes/train.lst'
_C.DATASET.EXTRA_TRAIN_SET = ''
_C.DATASET.TEST_SET = 'list/cityscapes/val.lst'

_C.DATASET.UNSUPDATASET = 'NC_Semi'
_C.DATASET.UNSUPROOT = ''
_C.DATASET.UNSUPTRAIN_SET = 'list/unsuptrain.lst'

_C.DATASET.ENH_DATASET= 'LSUI'
_C.DATASET.ENH_ROOT= ''
_C.DATASET.ENH_TEST_SET= 'list/val.lst'
_C.DATASET.ENH_TRAIN_SET= 'list/train.lst'

# training
_C.TRAIN = CN()

_C.TRAIN.FREEZE_LAYERS = ''
_C.TRAIN.FREEZE_EPOCHS = -1
_C.TRAIN.NONBACKBONE_KEYWORDS = []
_C.TRAIN.NONBACKBONE_MULT = 10

_C.TRAIN.IMAGE_SIZE = [1024, 512]  # width * height
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16

_C.TRAIN.RANDOM_BRIGHTNESS = False
_C.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [60, 80]
# _C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.DOUBLE_LR = False
_C.TRAIN.LR = 0.01
_C.TRAIN.ENH_LR = 0.01
_C.TRAIN.LR_IMAGENET_BRANCH = 0.005
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = False
_C.TRAIN.FINETUNE = False
_C.TRAIN.FINETUNE_FILE = "../pretrained_models/DDRNet39.pth"
_C.TRAIN.FINETUNE_LR = 0.0005
_C.TRAIN.DISTILLATION = False
_C.TRAIN.KD_LOGIT = False
_C.TRAIN.KD_FEATURE = False
_C.TRAIN.KD_LOWFEATURE = False
_C.TRAIN.ADAIN = False
_C.TRAIN.TEACHER_DATASET = 'cityscapes'
_C.TRAIN.TEACHER_DATASET_ROOT = 'D:/dataset/cityscape/'
_C.TRAIN.TEACHER_SET = 'list/train.lst'
_C.TRAIN.KDA = "../pretrained_models/ddrnet23_citys_finetune_ACDC_5216.pth"

_C.TRAIN.SEMI = False
_C.TRAIN.PROTOTYPE = False
_C.TRAIN.USE_STUDENT2 = False
_C.TRAIN.STUDENT2_WEIGHT = "../pretrained_models/rtformer_NC_20fewshot_3037.pth"
# Cutmix Config
_C.TRAIN.cutmix_mask_prop_range = (0.25, 0.5)
_C.TRAIN.cutmix_boxmask_n_boxes = 3
_C.TRAIN.cutmix_boxmask_fixed_aspect_ratio = False
_C.TRAIN.cutmix_boxmask_by_size = False
_C.TRAIN.cutmix_boxmask_outside_bounds = False
_C.TRAIN.cutmix_boxmask_no_invert = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
# only using some training samples
_C.TRAIN.NUM_SAMPLES = 0

# testing
_C.TEST = CN()

_C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
_C.TEST.BASE_SIZE = 2048

_C.TEST.BATCH_SIZE_PER_GPU = 32
# only testing some samples
_C.TEST.NUM_SAMPLES = 0

_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False
_C.TEST.SCALE_LIST = [1]

_C.TEST.OUTPUT_INDEX = -1

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

