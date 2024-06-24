# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import HEADS


@HEADS.register_module()
class GazeHead(BaseModule):
    """Head for contrastive learning.

    The contrastive loss is implemented in this head and is used in SimCLR,
    MoCo, DenseCL, etc.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 0.1.
    """

    def __init__(self, temperature=0.07):
        super(GazeHead, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, z, labels):
        """Forward function to compute contrastive loss.

        Args:
            # pos (Tensor): Nx1 positive similarity.
            # neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_pair = torch.sum(labels)
        z = z/ self.temperature
        logits_matrix = nn.functional.log_softmax(z, dim=1)

        res = -torch.mul(logits_matrix, labels)
        res = torch.sum(res)
        loss = res / num_pair
        losses = dict()
        losses['loss']=loss
        # N = pos.size(0)
        # logits = torch.cat((pos, neg), dim=1)
        # logits /= self.temperature
        # labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)
        # losses = dict()
        # losses['loss'] = self.criterion(logits, labels)
        return losses
