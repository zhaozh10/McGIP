# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torchvision.transforms
import torchvision.transforms.functional as F
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose
import numpy as np
from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy
# import cv2
import pdb


class CCompose(Compose):
    def __call__(self, x):  # x: [sample, box]
        sample,box=x
        img = self.transforms[0](sample,box)
        for t in self.transforms[1:]:
            img = t(img)
        return img

class MultiViewTransform:
    """Create multiple views of the same image"""
    def __init__(self, transform, num_views=2):
        if not isinstance(transform, (list, tuple)):
            transform = [transform for _ in range(num_views)]
        self.transforms = transform

    def __call__(self, x):
        views = [t(x) for t in self.transforms]
        return views


def get_trans(transform_list):
    transform = CCompose(transform_list[0])
    transform = MultiViewTransform(transform, num_views=2)
    return transform

@DATASETS.register_module()
class MammoCCrop(BaseDataset):

    def __init__(self, data_source, num_views, pipelines,  eval=False, prefetch=False):
        self.eval=eval
        self.data_source = build_datasource(data_source)
        ccrop_list = []
        ccrop_list.append([build_from_cfg(p, PIPELINES) for p in pipelines[0]])
        rcrop_list=[]
        rcrop_list.append([build_from_cfg(p, PIPELINES) for p in pipelines[1]])
        self.transform_ccrop=get_trans(ccrop_list)
        self.transform_rcrop =get_trans(rcrop_list)

        self.boxes = torch.tensor([0., 0., 1., 1.]).repeat(self.__len__(), 1)
        self.use_box = True


        self.prefetch = prefetch



    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        img = F.resize(img, [224, 224])
        if self.eval:
            temp_trans=torchvision.transforms.ToTensor()
            img=temp_trans(img)
            return dict(img=img, idx=idx)

        if self.use_box:
            box = self.boxes[idx].float().tolist()
            img = self.transform_ccrop([img, box])
        else:
            img = self.transform_rcrop(img)
        return dict(img=img, idx=idx)


    def evaluate(self, results, logger=None):
        return NotImplemented

