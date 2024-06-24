# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class BYOL_GzPT(BaseModel):
    """BYOL_GzPT.

    Implementation of BYOL+GzPT

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        base_momentum (float): The base momentum coefficient for the target
            network. Defaults to 0.996.
        threshold (float): threshold for the construction of positive pairs. Defaults to 0.7
        relation (string): The file containing gaze simialrity. 
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 base_momentum=0.996,
                 init_cfg=None,
                 p=0.5,
                 threshold=0.9, relation='correlation_tima.npy',
                 **kwargs):
        super(BYOL_GzPT, self).__init__(init_cfg)
        assert neck is not None
        self.p=p
        self.threshold=threshold
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

    def _create_buffer(self, N, idx_list):
        """

        Args:
            N: batchsize
            idx_list:

        Returns:
            labels generated according to similarity between gaze patterns, with shape[N,N]

        """
        labels = torch.zeros([N,N],dtype=torch.long)
        for i in range(N):
            idx = int(idx_list[i].item())
            for j in range(i,N):
                jdx = int(idx_list[j].item())
                if (i == j):
                    labels[i][j]=1
                else:
                    sim = self.relation[idx][jdx]
                    if (sim > self.threshold):
                        if (torch.rand(1)>self.p):
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

    def forward_train(self, img,idx, **kwargs):
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
        img_v1 = img[0]
        img_v2 = img[1]
        labels = self._create_buffer(img_v1.shape[0], idx_list)
        num_pair = torch.sum(labels)
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
