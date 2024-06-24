# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel
import numpy as np
import torch
@ALGORITHMS.register_module()
class SimSiam_GzPT(BaseModel):
    """SimSiam_gaze.

    Implementation of SimSiam+McGIP
    The operation of fixing learning rate of predictor is in
    `core/hooks/simsiam_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 p=0.5,
                 threshold=0.7, relation='correlation_TIMA.npy',
                 **kwargs):
        super(SimSiam_GzPT, self).__init__(init_cfg)
        assert neck is not None
        self.p=p
        self.threshold = threshold
        self.relation = np.load(relation)
        self.encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.backbone = self.encoder[0]
        self.neck = self.encoder[1]
        assert head is not None
        self.head = build_head(head)

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


    def forward_train(self, img,idx):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
        Returns:
            loss[str, Tensor]: A dictionary of loss components
        """
        assert isinstance(img, list)
        # print('yes')
        idx_list=idx
        img_v1 = img[0]
        img_v2 = img[1]
        labels = self._create_buffer(img_v1.shape[0], idx_list)
        num_pair = torch.sum(labels)

        z1 = self.encoder(img_v1)[0]  # NxC
        z2 = self.encoder(img_v2)[0]  # NxC

        losses = 0.5 * (self.head(z1, z2,labels)['loss'] + self.head(z2, z1,labels)['loss'])
        return dict(loss=losses)
