# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torchvision.transforms.functional as F
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose
import numpy as np
from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy
import cv2
import pdb


@DATASETS.register_module()
class dentalSet(BaseDataset):

    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        assert len(num_views) == len(pipelines)
        self.data_source = build_datasource(data_source)
        # self.repo=np.load(repoInfo,allow_pickle=True)
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)
        self.prefetch = prefetch

        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipelines[i]] * num_views[i])
        self.trans = trans
        # writing your code here


    def __getitem__(self, idx):
        # pdb.set_trace()
        img = self.data_source.get_img(idx)
        gt=self.data_source.get_cat_ids(idx)
        all_gt=self.data_source.get_gt_labels()
        # print("===========================")
        # print(gt)
        img = F.resize(img, [300, 600])
        # img = F.resize(img, [800, 1600])
        # print(img.shape)
        # img = cv2.imread(item_name, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (224, 224))
        multi_views = list(map(lambda trans: trans(img), self.trans))
        # idx_list=[idx,idx]
        # prefetch将预加载数据到GPU上
        if self.prefetch:
            multi_views = [
                torch.from_numpy(to_numpy(img)) for img in multi_views
            ]

        # print(multi_views[0].shape)
        return dict(img=multi_views, idx=idx, gt=gt)
        # return dict(img=multi_views, idx=idx, gt=all_gt)
        # img = img.astype(np.float32) / 255

        # img = np.expand_dims(img, axis=2)
        # img = np.tile(img, (1, 1, 3))
        # Generator = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(224), n_views=2)
        # imgpair = Generator(img)
        #
        # return imgpair, idx

    def evaluate(self, results, logger=None):
        return NotImplemented