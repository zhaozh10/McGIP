# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import HEADS


@HEADS.register_module()
class MoCoGazeHead(BaseModule):
    """Head for contrastive learning.

    The contrastive loss is implemented in this head and is used in SimCLR,
    MoCo, DenseCL, etc.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 0.1.
    """

    def __init__(self, temperature=0.07):
        super(MoCoGazeHead, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, l_pos,l_neg, labels):
        """Forward function to compute contrastive loss.

        Args:
            # pos (Tensor): Nx1 positive similarity.
            # neg (Tensor): Nxk negative similarity, where k is the length of queue
            # labels (Tensor): Nxk denotes pos pair according to gaze info


        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N=l_pos.shape[0]
        # print(f"batchsize {N}")
        z=torch.cat((l_pos,l_neg),dim=1)
        additional_pos_pair=torch.sum(labels)
        # print(labels.shape)
        labels = torch.concat((torch.ones([N, 1],dtype=torch.long).cuda(), labels),dim=1)
        num_pair = torch.sum(labels)
        # print(f'This iter has {additional_pos_pair} additional pos pair，{num_pair} in total')
        z = z / self.temperature
        logits_matrix = nn.functional.log_softmax(z, dim=1)

        res = -torch.mul(logits_matrix, labels)
        res = torch.sum(res)
        loss = res / num_pair
        losses = dict()
        losses['loss'] = loss


        return losses
