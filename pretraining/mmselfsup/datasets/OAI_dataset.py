# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torchvision.transforms.functional as F
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose
import numpy as np
from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy
# import cv2
import pdb


@DATASETS.register_module()
class OAIset(BaseDataset):

    def __init__(self, data_source, num_views, pipelines, sup=True, all_gt=True, prefetch=False):
        assert len(num_views) == len(pipelines)
        self.data_source = build_datasource(data_source)

        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)
        self.prefetch = prefetch
        self.sup = sup
        self.all_gt = all_gt

        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipelines[i]] * num_views[i])
        self.trans = trans

    def __getitem__(self, idx):
        
        img = self.data_source.get_img(idx)
        labels = self.data_source.get_cat_ids(idx)
        all_labels = self.data_source.get_gt_labels()
        
        multi_views = list(map(lambda trans: trans(img), self.trans))
        
        if self.prefetch:
            multi_views = [
                torch.from_numpy(to_numpy(img)) for img in multi_views
            ]

        
        if self.sup:
            if self.all_gt:
                return dict(img=multi_views, idx=idx, gt=all_labels)
            else:
                return dict(img=multi_views, idx=idx, gt=labels)

        else:
            return dict(img=multi_views, idx=idx)

    def evaluate(self, results, logger=None):
        return NotImplemented