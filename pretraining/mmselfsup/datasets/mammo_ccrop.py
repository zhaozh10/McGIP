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
        # print("==============")
        # print(x[0])
        # print(len(x[0]))
        # print(x[2])
        # print(self.transforms[0])
        sample,box=x
        # self.transforms(sample,box)
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
    # print(transform_list[0])
    transform = CCompose(transform_list[0])
    # print("=======check transform type=============")
    # print(type(transform))
    transform = MultiViewTransform(transform, num_views=2)
    return transform

@DATASETS.register_module()
class MammoCCrop(BaseDataset):

    def __init__(self, data_source, num_views, pipelines,  eval=False, prefetch=False):
        # assert len(num_views) == len(pipelines_ccrop)
        self.eval=eval
        # print("==============================")
        # print(len(pipelines))
        # print(len(pipelines[0]))
        self.data_source = build_datasource(data_source)
        ccrop_list = []
        ccrop_list.append([build_from_cfg(p, PIPELINES) for p in pipelines[0]])
        rcrop_list=[]
        rcrop_list.append([build_from_cfg(p, PIPELINES) for p in pipelines[1]])
        self.transform_ccrop=get_trans(ccrop_list)
        # print("=====here=============")
        # print(type(self.transform_ccrop))
        self.transform_rcrop =get_trans(rcrop_list)

        self.boxes = torch.tensor([0., 0., 1., 1.]).repeat(self.__len__(), 1)
        self.use_box = True



        # ccrop_list = []
        # for pipe in pipelines_ccrop:
        #     pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
        #     ccrop_list.append(pipeline)
        # rcrop_list = []
        # for pipe in pipelines_rcrop:
        #     pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
        #     rcrop_list.append(pipeline)
        self.prefetch = prefetch

        # trans_ccrop = []
        # assert isinstance(num_views, list)
        # for i in range(len(num_views)):
        #     trans_ccrop.extend([self.pipelines[i]] * num_views[i])
        # trans_rcrop = []
        # assert isinstance(num_views, list)
        # for i in range(len(num_views)):
        #     trans_rcrop.extend([self.pipelines[i]] * num_views[i])
        # # self.trans = trans
        # self.transform_rcrop = trans_rcrop
        # self.transform_ccrop = trans_ccrop

        # writing your code here



    def __getitem__(self, idx):
        # pdb.set_trace()
        img = self.data_source.get_img(idx)
        # labels = self.data_source.get_cat_ids(idx)
        # all_labels = self.data_source.get_gt_labels()
        img = F.resize(img, [224, 224])
        # multi_views = list(map(lambda trans: trans(img), self.trans))
        if self.eval:
            temp_trans=torchvision.transforms.ToTensor()
            img=temp_trans(img)
            # print(img)
            return dict(img=img, idx=idx)

        if self.use_box:
            box = self.boxes[idx].float().tolist()
            # print("we've here")# box=[h_min, w_min, h_max, w_max]
            img = self.transform_ccrop([img, box])
        else:
            img = self.transform_rcrop(img)
        # idx_list=[idx,idx]
        # prefetch将预加载数据到GPU上
        # if self.prefetch:
        #     multi_views = [
        #         torch.from_numpy(to_numpy(img)) for img in multi_views
        #     ]



        return dict(img=img, idx=idx)

        # img = img.astype(np.float32) / 255

        # img = np.expand_dims(img, axis=2)
        # img = np.tile(img, (1, 1, 3))
        # Generator = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(224), n_views=2)
        # imgpair = Generator(img)
        #
        # return imgpair, idx

    def evaluate(self, results, logger=None):
        return NotImplemented

