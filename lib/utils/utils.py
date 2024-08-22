# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.core.criterion import DistillationLoss1, DistillationLoss2, freematch_ce_loss, MultiClassFocalLoss
import random
from lib.utils import transformsgpu


class FullModel(nn.Module):
    """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """

    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def pixel_acc(self, pred, label):
        if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
            pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, labels, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        loss = self.loss(outputs, labels)
        if isinstance(outputs, list):
            # fanhui wei duo gezhi
            acc = self.pixel_acc(outputs[1], labels)
        # besinetv1
        #     acc = self.pixel_acc(outputs[3], labels)
        else:
            # fanhui wei 1 gezhi
            acc = self.pixel_acc(outputs, labels)
        return torch.unsqueeze(loss, 0), outputs, acc


# ---------------------------引入无监督低光照增强损失--------------------------------
class FullModel_enhance(nn.Module):
    """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """

    # def __init__(self, model, loss, L_spa, L_exp, L_color, L_TV):
    def __init__(self, model, loss, L_spa, L_TV):
        super(FullModel_enhance, self).__init__()
        self.model = model
        self.loss = loss

        self.L_spa = L_spa
        # self.L_color = L_color
        # self.L_exp = L_exp
        self.L_TV = L_TV

    def pixel_acc(self, pred, label):
        if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
            pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, labels, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        # 原图
        ori = outputs[0]
        # ZeroDCE增强后 3通道图
        enhanced_image = outputs[1]
        # ZeroDCE增强所需的A
        A = outputs[2]
        # 计算增强的无监督损失
        loss_spa = torch.mean(self.L_spa(enhanced_image, ori))
        # loss_exp = 10 * torch.mean(self.L_exp(enhanced_image))
        # loss_col = 5 * torch.mean(self.L_color(enhanced_image))
        loss_TV = 200 * self.L_TV(A)

        # Zero_BiSeNetV1后4个输出
        outputs = outputs[3:]
        loss = self.loss(outputs, labels)
        # fanhui wei 1 gezhi
        # acc = self.pixel_acc(outputs, labels)
        # fanhui wei duo gezhi
        acc = self.pixel_acc(outputs[3], labels)

        # return torch.unsqueeze(loss, 0), outputs, acc, loss_spa, loss_exp, loss_col, loss_TV
        return torch.unsqueeze(loss, 0), outputs, acc, loss_spa, loss_TV


# ---------------------------引入边缘损失--------------------------------
class FullModel_loss(nn.Module):
    """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """

    def __init__(self, model, loss, edge_loss):
        super(FullModel_loss, self).__init__()
        self.model = model
        self.loss = loss
        self.edge_loss = edge_loss

    def pixel_acc(self, pred, label):
        if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
            pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, labels, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        # 引出的边缘信息
        edge = outputs[2]
        # 计算边缘损失
        eloss = self.edge_loss(edge, labels)
        # ddrnet23s原来的两个输出
        outputs = outputs[:2]
        loss = self.loss(outputs, labels)
        # fanhui wei 1 gezhi
        # acc = self.pixel_acc(outputs, labels)
        # fanhui wei duo gezhi
        acc = self.pixel_acc(outputs[1], labels)
        return torch.unsqueeze(loss, 0), outputs, acc, eloss


# kd  lianhexunlian
class FullModel_kd(nn.Module):
    """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """

    def __init__(self, model, loss, teacher_model, featurekd):
        super(FullModel_kd, self).__init__()
        self.model = model
        self.loss = loss
        self.teacher_model = teacher_model
        self.featurekd = featurekd

    def pixel_acc(self, pred, label):
        if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
            pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, labels, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        teacher_outs = self.teacher_model(inputs)
        kd_feature_loss = 0
        if self.featurekd:
            # logit loss
            # criterion_distill1 = DistillationLoss1(self.teacher_model, 'soft', alpha=1, tau=5.0)
            # kd_loss1 = criterion_distill1(inputs, outputs)
            criterion_distill2 = DistillationLoss2(self.teacher_model, 'soft', alpha=1, tau=5.0)
            kd_loss2 = criterion_distill2(inputs, outputs)
            # kd_logit = kd_loss1 + kd_loss2
            kd_logit = kd_loss2

            # feature KD loss
            # s_seghead_feature = outputs[2]
            # s_final_feature = outputs[3]
            # t_seghead_feature = teacher_outs[2]
            # t_final_feature = teacher_outs[3]
            s_final_feature = outputs[2]
            t_final_feature = teacher_outs[2]
            kd_feature_loss = torch.dist(s_final_feature, t_final_feature, p=2)
        else:
            criterion_distill2 = DistillationLoss2(self.teacher_model, 'soft', alpha=1, tau=3.0)
            kd_logit = criterion_distill2(inputs, outputs)
        # ddrnet23s原来的两个输出
        outputs = outputs[:2]
        loss = self.loss(outputs, labels)
        # fanhui wei 1 gezhi
        # acc = self.pixel_acc(outputs, labels)
        # fanhui wei duo gezhi
        acc = self.pixel_acc(outputs[1], labels)
        return torch.unsqueeze(loss, 0), outputs, acc, kd_logit, kd_feature_loss


# kd  solo train logiy
class FullModel_kd1(nn.Module):
    """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """

    def __init__(self, model, loss, criterion_distill2):
        super(FullModel_kd1, self).__init__()
        self.model = model
        self.loss = loss
        self.criterion_distill2 = criterion_distill2

    def pixel_acc(self, pred, label):
        if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
            pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, labels, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        kd_logit = self.criterion_distill2(inputs, outputs)
        kd_feature_loss = 0
        # ddrnet23s原来的两个输出
        outputs = outputs[:2]
        loss = self.loss(outputs, labels)
        # fanhui wei 1 gezhi
        # acc = self.pixel_acc(outputs, labels)
        # fanhui wei duo gezhi
        acc = self.pixel_acc(outputs[1], labels)
        return torch.unsqueeze(loss, 0), outputs, acc, kd_logit, kd_feature_loss


class FullModel_kd_edge(nn.Module):
    """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """

    def __init__(self, model, loss, teacher_model, featurekd, edge_loss):
        super(FullModel_kd_edge, self).__init__()
        self.model = model
        self.loss = loss
        self.edge_loss = edge_loss
        self.teacher_model = teacher_model
        self.featurekd = featurekd

    def pixel_acc(self, pred, label):
        if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
            pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, labels, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        # 引出的边缘信息
        edge = outputs[2]
        # 计算边缘损失
        eloss = self.edge_loss(edge, labels)
        # ddrnet23s原来的两个输出
        outputs = outputs[:2]
        teacher_outs = self.teacher_model(inputs)
        kd_feature_loss = 0
        if self.featurekd:
            # logit loss
            # criterion_distill1 = DistillationLoss1(self.teacher_model, 'soft', alpha=1, tau=5.0)
            # kd_loss1 = criterion_distill1(inputs, outputs)
            criterion_distill2 = DistillationLoss2(self.teacher_model, 'soft', alpha=1, tau=5.0)
            kd_loss2 = criterion_distill2(inputs, outputs)
            # kd_logit = kd_loss1 + kd_loss2
            kd_logit = kd_loss2

            # feature KD loss
            # s_seghead_feature = outputs[2]
            # s_final_feature = outputs[3]
            # t_seghead_feature = teacher_outs[2]
            # t_final_feature = teacher_outs[3]
            s_final_feature = outputs[2]
            t_final_feature = teacher_outs[2]
            kd_feature_loss = torch.dist(s_final_feature, t_final_feature, p=2)
        else:
            criterion_distill2 = DistillationLoss2(self.teacher_model, 'soft', alpha=1, tau=3.0)
            kd_logit = criterion_distill2(inputs, outputs)

        loss = self.loss(outputs, labels)
        # fanhui wei 1 gezhi
        # acc = self.pixel_acc(outputs, labels)
        # fanhui wei duo gezhi
        acc = self.pixel_acc(outputs[1], labels)
        return torch.unsqueeze(loss, 0), outputs, acc, kd_logit, kd_feature_loss, eloss


class FullModel_Semi(nn.Module):
    """
  Distribute the loss on multi-gpu to reduce
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """

    def __init__(self, model, loss):
        super(FullModel_Semi, self).__init__()
        self.model = model
        self.loss = loss

    def pixel_acc(self, pred, label):
        if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
            pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, labels, sup=True, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        if sup == True:
            loss = self.loss(outputs, labels)
            return torch.unsqueeze(loss, 0), outputs
        else:
            return outputs


class FullModel_kd_Semi(nn.Module):
    """
     Distribute the loss on multi-gpu to reduce
     the memory cost in the main gpu.
     You can check the following discussion.
     https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
     """

    def __init__(self, model, loss, teacher_model, criterion2):
        super(FullModel_kd_Semi, self).__init__()
        self.model = model
        self.loss = loss
        self.teacher_model = teacher_model
        self.semi_loss = criterion2

        # freematch - Threshold
        self.m = 0.999
        self.class_threshold = torch.ones(19) / 19
        self.class_threshold = self.class_threshold.cuda()
        self.global_threshold = self.class_threshold.mean()
        self.global_threshold = self.global_threshold.cuda()

    def pixel_acc(self, pred, label):
        if pred.shape[2] != label.shape[1] and pred.shape[3] != label.shape[2]:
            pred = F.interpolate(pred, (label.shape[1:]), mode="bilinear")
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def update_threshold(self, softmax_teacher_outs):
        max_probs, teacher_pse_label = torch.max(softmax_teacher_outs, dim=1)
        self.global_threshold = self.global_threshold * self.m + (1 - self.m) * max_probs.mean()
        gap = torch.nn.AdaptiveAvgPool2d(1)
        softmax_teacher_outs = gap(softmax_teacher_outs).squeeze(dim=2)
        softmax_teacher_outs = softmax_teacher_outs.squeeze(dim=2)
        # a = softmax_teacher_outs.view(softmax_teacher_outs.shape[0], 19, -1).mean(dim=2).mean(dim=0)
        self.class_threshold = self.class_threshold * self.m + (1 - self.m) * softmax_teacher_outs.mean(dim=0)

    def forward(self, inputs, labels, sup=True, *args, **kwargs):
        if sup == True:

            # 进行弱数据增广
            # inputs_aug, labels_aug, _, _ = augment_samples(inputs, labels, None, do_classmix=False,
            #                                                batch_size=inputs.shape[0], ignore_label=255, weak=True)
            # outputs = self.model(inputs_aug, *args, **kwargs)
            # teacher_outs = self.teacher_model(inputs_aug)
            # outputs = outputs[1]
            # teacher_outs = teacher_outs[1].detach()
            # loss = self.loss(outputs, labels_aug)

            # 未进行数据增广
            outputs = self.model(inputs, *args, **kwargs)
            teacher_outs = self.teacher_model(inputs)
            # outputs = [outputs[0], outputs[2]]
            # teacher_outs = [teacher_outs[0], teacher_outs[2]]
            outputs = outputs[1]
            teacher_outs = teacher_outs[1].detach()
            loss = self.loss(outputs, labels)

            return torch.unsqueeze(loss, 0), outputs, teacher_outs
        else:
            # 用 teacher model 生成伪标签
            teacher_outs = self.teacher_model(inputs)
            teacher_outs_1 = teacher_outs[1].detach()
            teacher_outs_1 = F.interpolate(teacher_outs_1, size=[inputs.shape[2], inputs.shape[3]], mode='bilinear')
            softmax_teacher_outs = torch.softmax(teacher_outs_1, dim=1)
            max_probs, teacher_pse_label = torch.max(softmax_teacher_outs, dim=1)
            outputs = self.model(inputs, *args, **kwargs)

            feature_loss = 0
            # feature_cos loss
            # cos_criterion = nn.CosineEmbeddingLoss(reduction='mean')
            # cos_loss_flag = torch.ones([outputs[0].shape[0]]).cuda()
            # cos_loss = cos_criterion(outputs[0].view(outputs[0].shape[0], -1), teacher_outs[0].detach().view(teacher_outs[0].shape[0], -1), cos_loss_flag)
            # cos_loss = cos_loss.mean()
            # feature_loss = cos_loss

            # kd loss
            # criterion_distill = DistillationLoss2(self.teacher_model, 'soft', alpha=2, tau=5.0)
            # kd_loss = criterion_distill(inputs, outputs)
            # feature_loss = kd_loss

            outputs = outputs[1]
            teacher_outs = teacher_outs[1].detach()

            # freematch-Threshold Pseudo
            self.update_threshold(softmax_teacher_outs)
            mod = self.class_threshold / torch.max(self.class_threshold, dim=-1)[0]
            # mask的维度 [B,H,W]
            mask = max_probs.ge(self.global_threshold * mod[teacher_pse_label]).to(max_probs.dtype)
            outputs = F.interpolate(outputs, size=[inputs.shape[2], inputs.shape[3]], mode='bilinear')
            pse_loss = freematch_ce_loss(outputs, teacher_pse_label)
            pse_loss = pse_loss * mask
            pse_loss = pse_loss.mean()

            # fix-Threshold Pseudo
            # max_probs, teacher_pse_label = torch.max(softmax_teacher_outs, dim=1)
            # teacher_pse_label[max_probs < 0.8] = 255
            # pse_loss = self.semi_loss(outputs, teacher_pse_label)

            # 一致性损失
            # 进行弱数据增广
            inputs_weak_aug, _, _, _ = augment_samples(inputs, None, None, do_classmix=False,
                                                           batch_size=inputs.shape[0], ignore_label=255, weak=True)
            # 进行强数据增广
            inputs_strong_aug, _, _, _ = augment_samples(inputs, None, None, do_classmix=False,
                                                         batch_size=inputs.shape[0], ignore_label=255)
            outputs_strong_aug = self.model(inputs_strong_aug, *args, **kwargs)
            outputs_weak_aug = self.teacher_model(inputs_weak_aug)
            # 用 KD LOSS 来计算一致性损失
            # criterion_distill = DistillationLoss2(self.teacher_model, 'soft', alpha=2, tau=5.0)
            # consist_loss = criterion_distill(inputs_weak_aug, outputs_strong_aug)
            # 用余弦相似度loss 来计算一致性损失
            cos_criterion = nn.CosineEmbeddingLoss(reduction='mean')
            cos_loss_flag = torch.ones([outputs_strong_aug[1].shape[0]]).cuda()
            cos_loss = cos_criterion(outputs_strong_aug[1].view(outputs_strong_aug[1].shape[0], -1), outputs_weak_aug[1].detach().view(outputs_weak_aug[1].shape[0], -1), cos_loss_flag)
            consist_loss = 0.5*cos_loss.mean()

            for ema_param in self.teacher_model.named_parameters():
                if ema_param[0] == 'conv1.conv1_1.0.weight':
                    b = ema_param[1].data
            return feature_loss, pse_loss, consist_loss, outputs, teacher_outs


def augmentationTransform(parameters, data=None, target=None, probs=None, jitter_vale=0.4, min_sigma=0.2, max_sigma=2.,
                          ignore_label=255):
    """

    Args:
        parameters: dictionary with the augmentation configuration
        data: BxCxWxH input data to augment
        target: BxWxH labels to augment
        probs: BxWxH probability map to augment
        jitter_vale:  jitter augmentation value
        min_sigma: min sigma value for blur
        max_sigma: max sigma value for blur
        ignore_label: value for ignore class

    Returns:
            augmented data, target, probs
    """
    assert ((data is not None) or (target is not None))
    if "Mix" in parameters:
        data, target, probs = transformsgpu.mix(mask=parameters["Mix"], data=data, target=target, probs=probs)

    # if "RandomScaleCrop" in parameters:
    #     data, target, probs = transformsgpu.random_scale_crop(scale=parameters["RandomScaleCrop"], data=data,
    #                                                           target=target, probs=probs, ignore_label=ignore_label)
    if "flip" in parameters:
        data, target, probs = transformsgpu.flip(flip=parameters["flip"], data=data, target=target, probs=probs)

    if "ColorJitter" in parameters:
        data, target, probs = transformsgpu.colorJitter(colorJitter=parameters["ColorJitter"], data=data, target=target,
                                                        probs=probs, s=jitter_vale)
    if "GaussianBlur" in parameters:
        data, target, probs = transformsgpu.gaussian_blur(blur=parameters["GaussianBlur"], data=data, target=target,
                                                          probs=probs, min_sigma=min_sigma, max_sigma=max_sigma)

    if "Grayscale" in parameters:
        data, target, probs = transformsgpu.grayscale(grayscale=parameters["Grayscale"], data=data, target=target,
                                                      probs=probs)
    if "Solarize" in parameters:
        data, target, probs = transformsgpu.solarize(solarize=parameters["Solarize"], data=data, target=target,
                                                     probs=probs)

    return data, target, probs


def augment_samples(images, labels, probs, do_classmix, batch_size, ignore_label, weak=False):
    """
    Perform data augmentation

    Args:
        images: BxCxWxH images to augment
        labels:  BxWxH labels to augment
        probs:  BxWxH probability maps to augment
        do_classmix: whether to apply classmix augmentation
        batch_size: batch size
        ignore_label: ignore class value
        weak: whether to perform weak or strong augmentation

    Returns:
        augmented data, augmented labels, augmented probs

    """

    if do_classmix:
        # ClassMix: Get mask for image A
        for image_i in range(batch_size):  # for each image
            classes = torch.unique(labels[image_i])  # get unique classes in pseudolabel A
            nclasses = classes.shape[0]

            # remove ignore class
            if ignore_label in classes and len(classes) > 1 and nclasses > 1:
                classes = classes[classes != ignore_label]
                nclasses = nclasses - 1

            # if dataset == 'pascal_voc':  # if voc dataaset, remove class 0, background
            #     if 0 in classes and len(classes) > 1 and nclasses > 1:
            #         classes = classes[classes != 0]
            #         nclasses = nclasses - 1

            # pick half of the classes randomly
            classes = (classes[torch.Tensor(
                np.random.choice(nclasses, int(((nclasses - nclasses % 2) / 2) + 1), replace=False)).long()]).cuda()

            # acumulate masks
            if image_i == 0:
                MixMask = transformsgpu.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
            else:
                MixMask = torch.cat(
                    (MixMask, transformsgpu.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()))

        params = {"Mix": MixMask}
    else:
        params = {}

    if weak:
        params["flip"] = random.random() < 0.5
        params["ColorJitter"] = random.random() < 0.2
        params["GaussianBlur"] = random.random() < 0.
        params["Grayscale"] = random.random() < 0.0
        params["Solarize"] = random.random() < 0.0
        # if random.random() < 0.5:
        #     scale = random.uniform(0.75, 1.5)
        # else:
        #     scale = 1
        # params["RandomScaleCrop"] = scale

        # Apply strong augmentations to unlabeled images
        image_aug, labels_aug, probs_aug = augmentationTransform(params,
                                                                 data=images, target=labels,
                                                                 probs=probs, jitter_vale=0.125,
                                                                 min_sigma=0.1, max_sigma=1.5,
                                                                 ignore_label=ignore_label)
    else:
        params["flip"] = random.random() < 0.5
        params["ColorJitter"] = random.random() < 0.8
        params["GaussianBlur"] = random.random() < 0.2
        params["Grayscale"] = random.random() < 0.0
        params["Solarize"] = random.random() < 0.0
        # if random.random() < 0.80:
        #     scale = random.uniform(1, 1.5)
        # else:
        #     scale = 1
        # params["RandomScaleCrop"] = scale

        # Apply strong augmentations to unlabeled images
        image_aug, labels_aug, probs_aug = augmentationTransform(params,
                                                                 data=images, target=labels,
                                                                 probs=probs, jitter_vale=0.25,
                                                                 min_sigma=0.1, max_sigma=1.5,
                                                                 ignore_label=ignore_label)

    return image_aug, labels_aug, probs_aug, params


class teacherforward(nn.Module):
    def __init__(self, model):
        super(teacherforward, self).__init__()
        self.model = model

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs


# ----------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                          (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
        label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


# def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
#     """
#     Calcute the confusion matrix by given label and pred
#     """
#     label = label.cpu().numpy()
#     pred = pred.cpu().numpy().transpose(0, 2, 3, 1)
#     pred = np.asarray(np.argmax(pred, axis=3), dtype=np.uint8)
#     mask = (label >= 0) & (label < num_class)
#     label = num_class * label[mask].astype('int') + pred[mask]
#     count = np.bincount(label, minlength=num_class ** 2)
#     confusion_matrix = count.reshape(num_class, num_class)
#
#     return confusion_matrix


def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr


import cv2
from PIL import Image


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * np.tile(colors[label],
                                                                        (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


class Vedio(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 480))

    def addImage(self, img, colorMask):
        img = img[:, :, ::-1]
        colorMask = colorMask[:, :, ::-1]  # shape:
        img = np.concatenate([img, colorMask], axis=1)
        self.cap.write(img)

    def releaseCap(self):
        self.cap.release()


class Map16(object):
    def __init__(self, vedioCap, visualpoint=True):
        self.names = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                      "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck",
                      "bus", "train", "motorcycle", "bicycle")
        self.colors = np.array([[128, 64, 128],
                                [244, 35, 232],
                                [70, 70, 70],
                                [102, 102, 156],
                                [190, 153, 153],
                                [153, 153, 153],
                                [250, 170, 30],
                                [220, 220, 0],
                                [107, 142, 35],
                                [152, 251, 152],
                                [70, 130, 180],
                                [220, 20, 60],
                                [255, 0, 0],
                                [0, 0, 142],
                                [0, 0, 70],
                                [0, 60, 100],
                                [0, 80, 100],
                                [0, 0, 230],
                                [119, 11, 32]
                                ], dtype=np.uint8)
        self.outDir = "output/map16"
        self.vedioCap = vedioCap
        self.visualpoint = visualpoint

    def visualize_result(self, data, pred, dir, img_name=None):
        # img = data

        # pred = np.int32(pred)
        # pixs = pred.size
        # uniques, counts = np.unique(pred, return_counts=True)
        # for idx in np.argsort(counts)[::-1]:
        #     name = self.names[uniques[idx]]
        #     ratio = counts[idx] / pixs * 100
        #     if ratio > 0.1:
        #         print("  {}: {:.2f}%".format(name, ratio))

        # calculate point
        # if self.visualpoint:
        #     #转化为灰度float32类型进行处理
        #     img = img.copy()
        #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     img_gray = np.float32(img_gray)
        #     #得到角点坐标向量
        #     goodfeatures_corners = cv2.goodFeaturesToTrack(img_gray, 400, 0.01, 10)
        #     goodfeatures_corners = np.int0(goodfeatures_corners)
        #     # 注意学习这种遍历的方法（写法）
        #     for i in goodfeatures_corners:
        #         #注意到i 是以列表为元素的列表，所以需要flatten或者ravel一下。
        #         x,y = i.flatten()
        #         cv2.circle(img,(x,y), 3, [0,255,], -1)

        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(np.uint8)

        # im_vis = img * 0.7 + pred_color * 0.3
        im_vis = pred_color
        im_vis = im_vis.astype(np.uint8)

        # for vedio result show
        # self.vedioCap.addImage(im_vis, pred_color)

        img_name = img_name
        if not os.path.exists(dir):
            os.makedirs(dir)
        Image.fromarray(im_vis).save(os.path.join(dir, img_name))


def speed_test(model, size=896, iteration=100):
    input_t = torch.Tensor(1, 3, size, size).cuda()
    feed_dict = {}
    feed_dict['img_data'] = input_t

    print("start warm up")

    for i in range(10):
        model(feed_dict, segSize=(size, size))

    print("warm up done")
    start_ts = time.time()
    for i in range(iteration):
        model(feed_dict, segSize=(size, size))

    torch.cuda.synchronize()
    end_ts = time.time()

    t_cnt = end_ts - start_ts
    print("=======================================")
    print("FPS: %f" % (100 / t_cnt))
    print(f"Inference time {t_cnt / 100 * 1000} ms")
