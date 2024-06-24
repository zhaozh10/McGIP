# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class BYOL_sup(BaseModel):
    """BYOL.

    Implementation of `Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning <https://arxiv.org/abs/2006.07733>`_.
    The momentum adjustment is in `core/hooks/byol_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        base_momentum (float): The base momentum coefficient for the target
            network. Defaults to 0.996.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 base_momentum=0.996,
                 init_cfg=None,
                 relation='relation.npy',
                 **kwargs):
        super(BYOL_sup, self).__init__(init_cfg)
        assert neck is not None
        self.online_net = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.target_net = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False

        self.backbone = self.online_net[0]
        self.neck = self.online_net[1]
        assert head is not None
        self.head = build_head(head)

        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.relation = np.load(relation)

    # def _create_buffer(self, N, idx_list):
    #     """
    #
    #     Args:
    #         N: batchsize
    #         idx_list:
    #
    #     Returns:
    #         labels generated according to similarity between gaze patterns, with shape[N,N]
    #
    #     """
    #     weights = np.array(self.weights.split('-')).astype(float)
    #     # labels = torch.zeros([2*self.args.batch_size,2*self.args.batch_size])
    #     labels = torch.zeros([N,N],dtype=torch.long)
    #     for i in range(N):
    #         idx = int(idx_list[i].item())
    #         for j in range(i,N):
    #             jdx = int(idx_list[j].item())
    #             if (i == j):
    #                 labels[i][j]=1
    #             else:
    #                 sim = self.relation[idx][jdx]
    #                 sim = np.array(sim)
    #                 score = np.dot(sim, weights)
    #                 if (score > self.threshold):
    #                     labels[i][j] = 1
    #                     # labels[j][i] = 1
    #
    #     labels = labels.cuda()
    #     return labels

    def _create_buffer(self, N, gt_list):
        """

        Args:
            N: batchsize
            idx_list:

        Returns:
            labels generated according to similarity between gaze patterns, with shape[N,N]

        """
        # labels = torch.zeros([2*self.args.batch_size,2*self.args.batch_size])
        labels = torch.zeros([N,N],dtype=torch.long)
        for i in range(N):
            for j in range(i,N):
                if (i == j):
                    labels[i][j]=1
                elif (gt_list[i] == gt_list[j]):
                        labels[i][j] = 1
                        labels[j][i] = 1

        labels = labels.cuda()
        return labels

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img,idx,gt,**kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list)
        idx_list = idx
        gt_list=gt
        img_v1 = img[0]
        img_v2 = img[1]
        labels = self._create_buffer(img_v1.shape[0], gt_list)
        # print(labels)
        num_pair = torch.sum(labels)
        # print("==========================")
        # print("Num of pair")
        # print(num_pair)
        # print("==========================")
        # compute online features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        # compute target features
        with torch.no_grad():
            proj_target_v1 = self.target_net(img_v1)[0]
            proj_target_v2 = self.target_net(img_v2)[0]

        losses = 2. * (
            self.head(proj_online_v1, proj_target_v2,labels)['loss'] +
            self.head(proj_online_v2, proj_target_v1,labels)['loss'])
        return dict(loss=losses)
